"""Repair orchestration, offline with a stub agent runner (no LLM, no DBOS)."""

from __future__ import annotations

import asyncio

from codify.pipeline.enrich.bluebell import parse_to_akn
from codify.pipeline.enrich.validator import validate_akn
from codify.repair.agent import RepairDeps
from codify.repair.loop import repair_document
from codify.repair.ops import Delete, EditPlan, SetBody, SetMoneyNumeral


def _fixture() -> str:
    bb = "BODY\n  SECTION 1\n    ARTICLE 1\n      Body of article one.\n    ARTICLE 2\n"
    return parse_to_akn(bb, country="gb", doctype="act", number="1", date="2020-01-01")


_PAGE = "ARTICLE 2\nBody of the second article."
_SPANS = [{"page": 1, "method": "t", "start": 0, "end": len(_PAGE)}]


def test_repair_document_applies_and_improves() -> None:
    xml = _fixture()

    async def agent_run(deps: RepairDeps) -> EditPlan:
        return EditPlan(ops=[SetBody(eid=deps.eid, bluebell="Body of the second article.")])

    outcome = asyncio.run(
        repair_document(
            xml,
            validate_akn(xml),
            agent_run=agent_run,
            country="gb",
            object_key="k",
            page_text=_PAGE,
            raw_spans=_SPANS,
        )
    )
    assert outcome.improved
    assert outcome.after < outcome.before
    assert any(o["disposition"] == "cleared" for o in outcome.findings)
    applied = [p for p in outcome.proposals if p["status"] == "applied"]
    assert applied and applied[0]["risk"] == "medium"


def test_ungrounded_set_body_queues_for_approval() -> None:
    """No text evidence anywhere: the plan is valid but beyond machine
    authority, queued, and the document untouched."""
    xml = _fixture()

    async def agent_run(deps: RepairDeps) -> EditPlan:
        return EditPlan(ops=[SetBody(eid=deps.eid, bluebell="Text read off the image.")])

    outcome = asyncio.run(
        repair_document(xml, validate_akn(xml), agent_run=agent_run, country="gb", object_key="k")
    )
    assert outcome.akn == xml
    assert any(o["disposition"] == "queued_for_approval" for o in outcome.findings)
    pending = [p for p in outcome.proposals if p["status"] == "pending"]
    assert pending and pending[0]["risk"] == "high"
    assert pending[0]["audit"]["escalated"]


def test_repair_document_rejects_bad_plan_no_improvement() -> None:
    # Agent proposes a Delete for an empty_article, policy forbids it, so
    # nothing applies and the document is unchanged.
    xml = _fixture()

    async def agent_run(deps: RepairDeps) -> EditPlan:
        return EditPlan(ops=[Delete(eid=deps.eid)])

    outcome = asyncio.run(
        repair_document(xml, validate_akn(xml), agent_run=agent_run, country="gb", object_key="k")
    )
    assert not outcome.improved
    assert outcome.akn == xml
    assert all(o["disposition"] != "cleared" for o in outcome.findings)


def test_money_finding_reaches_the_agent_and_is_repaired() -> None:
    """The loop skips any check absent from _OP_POLICY."""
    xml = parse_to_akn(
        "BODY\n  ARTICLE 8\n    not less than (٥٠٠,٠٠٠) خمسين ألف دينار زرزوري.\n",
        country="xz",
        doctype="act",
        number="12",
        date="1994-01-01",
    )
    findings = [f for f in validate_akn(xml) if f["check"] == "money_words_mismatch"]
    assert findings, "fixture no longer reproduces the money defect"
    seen: list[str] = []

    async def _agent(deps: RepairDeps) -> EditPlan:
        seen.append(str(deps.finding["check"]))
        return EditPlan(ops=[SetMoneyNumeral(eid=deps.eid, old="٥٠٠,٠٠٠", new="٥٠,٠٠٠")])

    outcome = asyncio.run(
        repair_document(xml, findings, agent_run=_agent, country="xz", object_key="k")
    )
    assert seen == ["money_words_mismatch"], "the loop skipped the finding"
    assert outcome.findings[0]["disposition"] == "cleared", outcome.findings
    assert "(٥٠,٠٠٠)" in outcome.akn


def test_budget_exhaustion_is_its_own_disposition() -> None:
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    xml = _fixture()

    async def exhausted(deps: RepairDeps) -> EditPlan:
        raise UnexpectedModelBehavior("Exceeded maximum retries (3) for output validation")

    outcome = asyncio.run(
        repair_document(xml, validate_akn(xml), agent_run=exhausted, country="gb", object_key="k")
    )
    assert outcome.akn == xml
    assert any(o["disposition"] == "revision_budget_exhausted" for o in outcome.findings)


def test_deps_carry_the_document_workspace() -> None:
    xml = _fixture()
    seen: list[RepairDeps] = []

    async def capture(deps: RepairDeps) -> EditPlan:
        seen.append(deps)
        return EditPlan()

    asyncio.run(
        repair_document(
            xml,
            validate_akn(xml),
            agent_run=capture,
            country="gb",
            object_key="k",
            version_id="v1",
        )
    )
    assert seen and seen[0].akn_xml == xml
    assert seen[0].version_id == "v1"
    assert seen[0].findings, "document-wide findings ride the deps"


def test_non_actionable_and_dissolved_targets_get_their_dispositions() -> None:
    xml = _fixture()
    real = validate_akn(xml)
    findings = [
        {"check": "ref_resolution", "eid": "x", "severity": "info", "message": ""},
        {"check": "empty_article", "severity": "error", "message": "no target"},
        *[f for f in real if f["check"] == "empty_article"],
        # Same finding again: once cleared, its eId no longer resolves the same way.
    ]

    async def agent_run(deps: RepairDeps) -> EditPlan:
        return EditPlan(ops=[SetBody(eid=deps.eid, bluebell="Body of the second article.")])

    outcome = asyncio.run(
        repair_document(
            xml,
            findings,
            agent_run=agent_run,
            country="gb",
            object_key="k",
            page_text=_PAGE,
            raw_spans=_SPANS,
        )
    )
    dispositions = [o["disposition"] for o in outcome.findings]
    assert dispositions[0] == "skipped_non_actionable"  # check not in policy
    assert dispositions[1] == "skipped_non_actionable"  # no target eid
    assert "cleared" in dispositions


def test_source_mismatch_queues_even_a_grounded_repair() -> None:
    xml = _fixture()

    async def agent_run(deps: RepairDeps) -> EditPlan:
        return EditPlan(ops=[SetBody(eid=deps.eid, bluebell="Body of the second article.")])

    outcome = asyncio.run(
        repair_document(
            xml,
            validate_akn(xml),
            agent_run=agent_run,
            country="gb",
            object_key="k",
            page_text=_PAGE,
            raw_spans=_SPANS,
            source_text_mismatch=True,
        )
    )
    assert outcome.akn == xml
    assert any(o["disposition"] == "queued_for_approval" for o in outcome.findings)


def test_defect_class_number_gap_never_reaches_the_agent() -> None:
    """A defect-class gap is an upstream extraction miss; annotating it would
    claim a repeal the page does not show, so the agent must not be asked."""
    xml = _fixture()
    findings = [
        {
            "check": "number_gap",
            "eid": "art_1",
            "likely": "defect",
            "severity": "warning",
            "message": "",
            "missing": [2, 3, 4, 5],
        }
    ]
    seen: list[str] = []

    async def agent_run(deps: RepairDeps) -> EditPlan:
        seen.append(deps.eid)
        return EditPlan()

    outcome = asyncio.run(
        repair_document(xml, findings, agent_run=agent_run, country="gb", object_key="k")
    )
    assert seen == []
    assert outcome.findings[0]["disposition"] == "skipped_non_actionable"


def test_usage_limit_gets_its_own_disposition() -> None:
    from pydantic_ai.exceptions import UsageLimitExceeded

    xml = _fixture()

    async def capped(deps: RepairDeps) -> EditPlan:
        raise UsageLimitExceeded("The next request would exceed the request_limit of 80")

    outcome = asyncio.run(
        repair_document(xml, validate_akn(xml), agent_run=capped, country="gb", object_key="k")
    )
    assert outcome.akn == xml
    assert any(o["disposition"] == "investigation_budget_exhausted" for o in outcome.findings)


def test_budget_death_salvages_an_accepted_preview() -> None:
    """The observed failure: the model held a sandbox-accepted plan and burned
    the budget re-previewing it. The loop adopts the proven plan (through the
    same gates) instead of discarding it."""
    import json

    from pydantic_ai.exceptions import UsageLimitExceeded

    xml = _fixture()

    async def capped(deps: RepairDeps) -> EditPlan:
        deps.preview_state["accepted_plan_json"] = json.dumps(
            {
                "ops": [
                    {"op": "set_body", "eid": deps.eid, "bluebell": "Body of the second article."}
                ],
                "reasoning": "recorded by preview before the budget died",
            }
        )
        raise UsageLimitExceeded("The next request would exceed the request_limit of 80")

    outcome = asyncio.run(
        repair_document(
            xml,
            validate_akn(xml),
            agent_run=capped,
            country="gb",
            object_key="k",
            page_text=_PAGE,
            raw_spans=_SPANS,
        )
    )
    assert outcome.improved
    salvaged = [o for o in outcome.findings if o.get("salvaged")]
    assert salvaged and salvaged[0]["disposition"] == "cleared"
    # Provenance survives into the persisted proposal audit, not just the log.
    applied = [p for p in outcome.proposals if p["status"] == "applied"]
    assert applied and applied[0]["audit"]["salvaged"] is True


def test_salvaged_plan_still_faces_the_sandbox() -> None:
    """Salvage adopts the plan, not its verdict: a recorded plan the loop's own
    gates refuse lands ops_rejected, document untouched."""
    from pydantic_ai.exceptions import UsageLimitExceeded

    xml = _fixture()

    async def capped(deps: RepairDeps) -> EditPlan:
        # Delete is not permitted for empty_article, whatever preview said.
        deps.preview_state["accepted_plan_json"] = (
            '{"ops": [{"op": "delete", "eid": "%s"}], "reasoning": ""}' % deps.eid
        )
        raise UsageLimitExceeded("The next request would exceed the request_limit of 80")

    outcome = asyncio.run(
        repair_document(xml, validate_akn(xml), agent_run=capped, country="gb", object_key="k")
    )
    assert outcome.akn == xml
    assert any(o["disposition"] == "ops_rejected" and o.get("salvaged") for o in outcome.findings)


def test_budget_death_salvages_an_accepted_abstention() -> None:
    from pydantic_ai.exceptions import UsageLimitExceeded

    xml = _fixture()

    async def capped(deps: RepairDeps) -> EditPlan:
        deps.preview_state["accepted_plan_json"] = '{"ops": [], "reasoning": ""}'
        raise UsageLimitExceeded("The next request would exceed the request_limit of 80")

    outcome = asyncio.run(
        repair_document(xml, validate_akn(xml), agent_run=capped, country="gb", object_key="k")
    )
    assert outcome.akn == xml
    assert any(o["disposition"] == "empty_plan" and o.get("salvaged") for o in outcome.findings)


def test_revision_exhaustion_salvages_an_accepted_preview() -> None:
    """The revision budget also burns on text-only turns and malformed
    submissions, so its death is no evidence against a recorded acceptance:
    an accepted plan cannot go stale against an immutable snapshot."""
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    xml = _fixture()

    async def stubborn(deps: RepairDeps) -> EditPlan:
        deps.preview_state["accepted_plan_json"] = '{"ops": [], "reasoning": ""}'
        raise UnexpectedModelBehavior("Exceeded maximum output retries (3)")

    outcome = asyncio.run(
        repair_document(xml, validate_akn(xml), agent_run=stubborn, country="gb", object_key="k")
    )
    assert any(o["disposition"] == "empty_plan" and o.get("salvaged") for o in outcome.findings)


def test_non_budget_model_failure_never_salvages() -> None:
    """An unexplained model failure is not a budget death: the run may be
    broken in ways the recorded acceptance cannot vouch for."""
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    xml = _fixture()

    async def broken(deps: RepairDeps) -> EditPlan:
        deps.preview_state["accepted_plan_json"] = '{"ops": [], "reasoning": ""}'
        raise UnexpectedModelBehavior("Received empty model response")

    outcome = asyncio.run(
        repair_document(xml, validate_akn(xml), agent_run=broken, country="gb", object_key="k")
    )
    assert any(o["disposition"] == "errored" for o in outcome.findings)
    assert not any(o.get("salvaged") for o in outcome.findings)


def test_corrupt_recorded_plan_degrades_to_exhaustion_not_a_crash() -> None:
    """_salvage runs inside the per-finding except block: a raise there would
    escape repair_document and lose every previously applied edit."""
    from pydantic_ai.exceptions import UsageLimitExceeded

    xml = _fixture()

    async def capped(deps: RepairDeps) -> EditPlan:
        deps.preview_state["accepted_plan_json"] = "{broken"
        raise UsageLimitExceeded("The next request would exceed the request_limit of 80")

    outcome = asyncio.run(
        repair_document(xml, validate_akn(xml), agent_run=capped, country="gb", object_key="k")
    )
    assert outcome.akn == xml
    assert any(o["disposition"] == "investigation_budget_exhausted" for o in outcome.findings)


def test_salvage_end_to_end_through_a_real_agent_run() -> None:
    """The full chain, no hand-written state: the real preview tool records the
    acceptance, the real agent run dies on UsageLimits, the loop salvages. Pins
    the cross-module contract (the preview_state key and the deps-object
    identity between the tool's write and the loop's read)."""
    import json

    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.usage import UsageLimits

    from codify.repair.agent import build_agent

    xml = _fixture()

    class _Store:
        async def render_page(self, object_key: str, page_no: int) -> bytes:
            return b"png"

        async def render_region(self, v: str, k: str, p: int, b: int) -> bytes:
            return b"png"

        async def page_read(self, v: str, p: int) -> None:
            return None

    def loops_on_preview(messages: list, info: AgentInfo) -> ModelResponse:
        # Previews an acceptable plan, then previews it again, the observed
        # loop shape, and the request cap kills the run before any submission.
        plan = json.dumps(
            {
                "ops": [
                    {
                        "op": "set_body",
                        "eid": "sec_1__art_2",
                        "bluebell": "Body of the second article.",
                    }
                ],
                "reasoning": "",
            }
        )
        return ModelResponse(
            parts=[
                ToolCallPart(tool_name="preview_plan", args={"plan_json": plan}, tool_call_id="c")
            ]
        )

    agent = build_agent(FunctionModel(loops_on_preview), _Store())

    async def agent_run(deps: RepairDeps) -> EditPlan:
        result = await agent.run(
            "Repair the flagged finding.", deps=deps, usage_limits=UsageLimits(request_limit=2)
        )
        return result.output

    outcome = asyncio.run(
        repair_document(
            xml,
            validate_akn(xml),
            agent_run=agent_run,
            country="gb",
            object_key="k",
            page_text=_PAGE,
            raw_spans=_SPANS,
        )
    )
    assert outcome.improved
    assert any(o["disposition"] == "cleared" and o.get("salvaged") for o in outcome.findings)


def test_hooks_fire_once_per_finding_in_order() -> None:
    """Every finding gets exactly one progress event and one settle event,
    including the early-skip paths, and risk rides the sandbox-visited ones."""
    xml = _fixture()
    real = [f for f in validate_akn(xml) if f["check"] == "empty_article"]
    findings = [
        {"check": "ref_resolution", "eid": "x", "severity": "info", "message": ""},
        *real,
    ]
    progressed: list[tuple[int, int, str]] = []
    settled: list[tuple[int, str, str | None]] = []

    async def agent_run(deps: RepairDeps) -> EditPlan:
        return EditPlan(ops=[SetBody(eid=deps.eid, bluebell="Body of the second article.")])

    async def on_progress(done: int, total: int, finding: dict) -> None:
        progressed.append((done, total, str(finding.get("check", ""))))

    async def on_settled(i: int, entry: dict, risk: str | None) -> None:
        settled.append((i, str(entry["disposition"]), risk))

    asyncio.run(
        repair_document(
            xml,
            findings,
            agent_run=agent_run,
            country="gb",
            object_key="k",
            page_text=_PAGE,
            raw_spans=_SPANS,
            on_progress=on_progress,
            on_settled=on_settled,
        )
    )
    assert progressed == [(0, 2, "ref_resolution"), (1, 2, "empty_article")]
    assert settled[0] == (0, "skipped_non_actionable", None)
    assert settled[1][0] == 1 and settled[1][1] == "cleared"
    assert settled[1][2] in ("low", "medium")  # sandbox-visited settles carry risk


def test_absent_hooks_change_nothing() -> None:
    xml = _fixture()

    async def agent_run(deps: RepairDeps) -> EditPlan:
        return EditPlan()

    outcome = asyncio.run(
        repair_document(xml, validate_akn(xml), agent_run=agent_run, country="gb", object_key="k")
    )
    assert any(o["disposition"] == "empty_plan" for o in outcome.findings)


def test_hooks_survive_the_agent_exception_path() -> None:
    """Every finding settles exactly once even when the agent dies, a live
    progress UI desynchronises on precisely the failure paths otherwise."""
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    xml = _fixture()
    settled: list[tuple[int, str, str | None]] = []

    async def broken(deps: RepairDeps) -> EditPlan:
        raise UnexpectedModelBehavior("Received empty model response")

    async def on_settled(i: int, entry: dict, risk: str | None) -> None:
        settled.append((i, str(entry["disposition"]), risk))

    outcome = asyncio.run(
        repair_document(
            xml,
            validate_akn(xml),
            agent_run=broken,
            country="gb",
            object_key="k",
            on_settled=on_settled,
        )
    )
    assert len(settled) == len(outcome.findings)
    assert any(d == "errored" and r is None for _, d, r in settled)
