"""The repair agent (Pydantic AI), investigates a flagged provision with the
dossier's evidence and *proposes* a typed EditPlan. It never mutates the
document; the workflow applies the plan transactionally (edit_ops), so the
agent's job is judgement, not surgery.

The evidence boundary is injected (`EvidenceStore`) so this module stays free
of the MinIO/DBOS I/O that apps/api supplies, and so the agent is unit-testable
offline with a stub store + a Pydantic AI TestModel/FunctionModel.

The run ends when the model calls the `submit_plan` output tool. Submission is
a tool call, not prompted text, because Gemini flash via the gateway reliably
emits tool calls and reliably does not emit a final text turn while tools are
declared (it looped an accepted preview 72 times without ever answering). The
output validator re-runs the sandbox on every submitted plan: a rejected plan
comes back to the model as a ModelRetry carrying the structured reasons, so
revision happens in-loop. `retries` is the revision budget.
"""

from __future__ import annotations

import hashlib

from pydantic import ValidationError
from pydantic_ai import Agent, ModelRetry, RunContext, ToolOutput
from pydantic_ai.models import Model

from codify.repair.deps import RepairDeps
from codify.repair.edit_ops import validate_body_bluebell
from codify.repair.evidence import EvidenceStore
from codify.repair.ops import EditPlan, SetBody
from codify.repair.sandbox import preview_plan
from codify.repair.tools import register_tools

__all__ = ["RepairDeps", "build_agent"]

REVISION_BUDGET = 3  # output retries: sandbox rejections, bad JSON and text-only turns


def submit_plan(plan_json: str) -> EditPlan:
    """Submit your final answer and end the investigation.

    Args:
        plan_json: the EditPlan as a JSON string:
            {"ops": [...], "reasoning": "..."}, {"ops": []} to abstain.
    """
    # A string argument by design: the discriminated op union cannot ride a
    # tool schema through the gateway (see preview_plan).
    try:
        return EditPlan.model_validate_json(plan_json)
    except ValidationError as exc:
        raise ModelRetry(f"could not parse the plan: {exc}") from exc


_INSTRUCTIONS = """\
You repair exactly one structural defect in a legal document's Akoma Ntoso XML.

You have an investigation toolkit: the document outline (show_outline), any
element (show_subtree), full-text search over the document (search_akn) and the
OCR'd source (search_source), any source page's text (read_page) or scan
(view_source_page), both engines' reads of a page (compare_reads), and a
cropped block render (view_region). Investigate as far as the defect requires —
evidence on adjacent pages or in sibling provisions counts — then finish by
calling submit_plan with a minimal EditPlan of typed ops. submit_plan is the
ONLY way to finish; an investigation without it is wasted work.

Match your effort to the defect. A structural fix (move, renumber, split,
money numeral) needs the outline or siblings, ONE look at the evidence, one
preview, and submit_plan — a handful of calls. Spend deep investigation
(multiple pages, scans, read comparisons) only on body restoration, where the
text itself is at stake. Never re-read evidence you have already seen.

Rules, in order of importance:
- Ground every body you write in what the source actually says. Never invent,
  paraphrase-shorten, or summarise legal text. Verbatim is correct.
- Bodies are Bluebell plain text (one line per point/paragraph), never raw XML
  and never a structural-keyword line (ARTICLE/SECTION/(1)/(a)). Validate a
  body with check_bluebell before you commit to it.
- Test your plan with preview_plan ONCE: pass the SAME JSON you will give
  submit_plan, as a string. The moment the verdict is accepted, call
  submit_plan with that exact JSON — never preview an accepted plan again.
  A submitted plan is checked the same way; a rejection costs you a retry.

Per finding:
- empty_article: the article's number is already in the structure but its body
  text was not captured. Find that article on the source page (view_source_page,
  and read_page on the next page if it continues there) and set_body with its
  full verbatim text — every paragraph and point. NEVER delete an empty
  article. Even when its text restates or amends another law ("Article (7) of
  ... is amended to read ..."), that amending clause IS the article's body —
  transcribe it. Deleting real law is the worst outcome. Only if the source
  genuinely has no text for this article, submit an empty plan.
- orphan_articles / hierarchy_coherence: move the unit under its correct parent
  (show_outline to find it).
- duplicate_number: usually the OCR collapsed a numbered sequence — sibling
  points/articles that should read 1, 2, 3, 4 came out as 1,1,1,1 or 1,1,2,2.
  ALWAYS call show_siblings first to see the WHOLE run under the parent (colliding
  groups often overlap, so fixing one group in isolation just moves the clash).
  Then renumber_sequence(parent_eid, kind, nums) with the true number for EVERY
  sibling in order — read the page for the real sequence, else number them in
  document order. Write each num exactly as printed on the page, keeping its
  enumerator punctuation ("1.", "(a)", "2)"), not the bare digit. Only when
  two units are EXACT duplicates (identical text, a stray repeated copy) delete
  the copy. Avoid merge.
  MIS-PARSE case: if the "duplicate" is a bare inline reference (e.g. an annex
  "(3)" inside prose) split off as a false point — not a collapsed sequence —
  restore the truncated point's full body from the page with set_body, move any
  real nested unit out with move, then delete the false wrapper. This
  restore-and-remove queues for approval, since it edits real text.
- swallowed_enumerator: split the point that swallowed the marker. Pass the
  finding's `matched` surface as the marker, exactly as it appears in the body.
  One split per plan: if an earlier repair moved the marker into a new sibling,
  search_akn for the marker and split the element that holds it now.
- money_words_mismatch: digits and written-out words disagree. Read the amount
  on the page (view_region on the block if the scan is hard to read), then
  rewrite ONLY the wrong numeral with set_money_numeral, `old` exactly as it
  appears and `new` in the same digit script. The words are usually right, but
  confirm on the page before choosing.
- number_gap: annotate ONLY when the page confirms the missing numbers are
  repealed or never existed — the note must state what you verified, never a
  placeholder. If the page shows the articles exist but were not captured,
  that is an upstream extraction miss you cannot fix: submit an empty plan
  immediately rather than spending revisions.
- displaced_terminal_material: signature/promulgation lines stuck inside the
  last provision. move_to_conclusions on the flagged element; the apply layer
  re-detects the attestation itself, so nothing else moves.
- empty_article / ocr_garble where read_page shows the text IS in the source:
  prefer restore_from_source over set_body — it splices the source span
  verbatim and cannot paraphrase. Fall back to set_body when the OCR text is
  too garbled to splice and you are transcribing from the scan image.
  restore_from_source is ONLY for missing/garbled bodies; it is never a
  numbering repair.
- A delete op queues for human approval rather than applying; prefer any
  repair that preserves text when one exists.
- If you cannot ground a fix on the source, abstain: call submit_plan with
  {"ops": []}. Abstention is never penalised.
"""


# Content-derived, so a prompt edit can never ship with a stale version stamp.
PROMPT_VERSION = hashlib.sha256(_INSTRUCTIONS.encode("utf-8")).hexdigest()[:8]


def build_agent(model: Model, store: EvidenceStore) -> Agent[RepairDeps, EditPlan]:
    agent = Agent(
        model,
        deps_type=RepairDeps,
        # Tool (not prompted) output: the model finishes by CALLING submit_plan,
        # never by emitting text, so the run cannot stall on a model that only
        # produces tool calls, and tool_choice becomes 'required' on the wire.
        output_type=ToolOutput(submit_plan, name="submit_plan"),
        instructions=_INSTRUCTIONS,
        name="akn-repair",
        retries=REVISION_BUDGET,
        # Extra tool calls beside submit_plan are skipped, not run.
        end_strategy="early",
    )

    @agent.instructions
    def _finding_context(ctx: RunContext[RepairDeps]) -> str:
        f = ctx.deps.finding
        return (
            f"Finding to repair: check={f.get('check')!r} on eId={ctx.deps.eid!r}. "
            f"Message: {f.get('message', '')}"
        )

    register_tools(agent, store)

    @agent.output_validator
    def _validate_plan(ctx: RunContext[RepairDeps], plan: EditPlan) -> EditPlan:
        if not plan.ops:
            return plan  # abstention is always available
        if not ctx.deps.akn_xml:
            # No document workspace (offline/stub deps): keep the subtree-local
            # body check so a bad SetBody still self-corrects in-loop.
            for op in plan.ops:
                if isinstance(op, SetBody) and op.eid == ctx.deps.eid:
                    err = validate_body_bluebell(
                        ctx.deps.subtree_xml,
                        op.bluebell,
                        country=ctx.deps.country,
                        doctype=ctx.deps.doctype,
                    )
                    if err:
                        raise ModelRetry(f"set_body for {op.eid} won't apply: {err}")
            return plan
        report = preview_plan(
            ctx.deps.akn_xml,
            plan,
            country=ctx.deps.country,
            doctype=ctx.deps.doctype,
            target_finding=ctx.deps.finding,
            source_text=ctx.deps.evidence_window(),
            evidence=ctx.deps.source_evidence(),
            source_text_mismatch=ctx.deps.source_text_mismatch,
        )
        if not report.ok:
            raise ModelRetry(
                f"plan rejected: {report.reason_text()}. Revise the plan or return "
                "an empty plan if the source cannot support a repair."
            )
        return plan

    return agent
