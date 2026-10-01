"""Repair agent wiring and the in-loop revision boundary, offline, no live LLM."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from codify.pipeline.enrich.bluebell import parse_to_akn
from codify.pipeline.enrich.ocr import PageResult
from codify.pipeline.enrich.validator import validate_akn
from codify.repair.agent import RepairDeps, build_agent
from codify.repair.grounding import combine_with_spans, spans_to_json
from codify.repair.ops import EditPlan


class _Store:
    """EvidenceStore stub recording renders."""

    def __init__(self) -> None:
        self.rendered: list[tuple[str, int]] = []

    async def render_page(self, object_key: str, page_no: int) -> bytes:
        self.rendered.append((object_key, page_no))
        return b"\x89PNG\r\n\x1a\n-fake-page"

    async def render_region(
        self, version_id: str, object_key: str, page_no: int, block_index: int
    ) -> bytes:
        return b"\x89PNG\r\n\x1a\n-fake-region"

    async def page_read(self, version_id: str, page_no: int) -> dict[str, Any] | None:
        return {
            "engine": "mistral_ocr",
            "model": "m",
            "text": "primary read",
            "rival_text": "rival read",
            "divergence": 0.4,
            "metrics": {"verdict": "reread", "reasons": ["divergence:high"]},
            "layout": None,
        }


def _deps(page_no: int | None = 3) -> RepairDeps:
    return RepairDeps(
        finding={"check": "empty_article", "message": "article 2 has a num but no body"},
        eid="sec_1__art_2",
        subtree_xml="<article eId='sec_1__art_2'><num>2</num></article>",
        source_text="Article 2. The real body text of the second article.",
        page_no=page_no,
        object_key="uploads/gb/abc.pdf",
        country="gb",
    )


_ABSTAIN = json.dumps({"ops": [], "reasoning": ""})


def _test_model() -> TestModel:
    # TestModel invents junk for the output tool's plan_json otherwise, and a
    # junk submission burns the whole retry budget on parse failures.
    return TestModel(custom_output_args={"plan_json": _ABSTAIN})


def _submit(plan_json: str, call_id: str = "s1") -> ModelResponse:
    """The model's terminal move: submission is a tool call, never text."""
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name="submit_plan", args={"plan_json": plan_json}, tool_call_id=call_id
            )
        ]
    )


def test_agent_produces_edit_plan_and_exercises_every_tool() -> None:
    store = _Store()
    agent = build_agent(_test_model(), store)
    result = agent.run_sync("Repair the flagged finding.", deps=_deps())

    assert isinstance(result.output, EditPlan)
    # TestModel exercises every tool once, so the injected renderer was called.
    assert ("uploads/gb/abc.pdf", 3) in store.rendered


def test_tools_never_mutate_the_workspace() -> None:
    deps, _ = _workspace_deps()
    before = (deps.akn_xml, deps.doc_text, list(deps.spans))
    agent = build_agent(_test_model(), _Store())
    agent.run_sync("Repair the flagged finding.", deps=deps)
    assert (deps.akn_xml, deps.doc_text, list(deps.spans)) == before
    # preview_state is the ONE sanctioned write-back channel; TestModel's junk
    # preview args parse to nothing, so even it stays empty here.
    assert deps.preview_state == {}


def test_view_source_page_handles_unmapped_page() -> None:
    class _NoRender(_Store):
        async def render_page(self, object_key: str, page_no: int) -> bytes:
            raise AssertionError("must not render when no page is mapped")

    agent = build_agent(_test_model(), _NoRender())
    result = agent.run_sync("Repair the flagged finding.", deps=_deps(page_no=None))
    assert isinstance(result.output, EditPlan)  # no crash; renderer never called


def _workspace_deps() -> tuple[RepairDeps, str]:
    """A two-page workspace where article 2's body sits on the page AFTER the
    page its anchor mapped to, the cross-page evidence case."""
    bb = "BODY\n  SECTION 1\n    ARTICLE 1\n      Body of article one.\n    ARTICLE 2\n"
    akn = parse_to_akn(bb, country="gb", doctype="act", number="1", date="2020-01-01")
    finding = next(f for f in validate_akn(akn) if f["check"] == "empty_article")
    pages = [
        PageResult(page_number=2, text="ARTICLE 2", method="vision_ocr"),
        PageResult(
            page_number=3,
            text="continued: the restored body of the second article.",
            method="vision_ocr",
        ),
    ]
    doc_text, spans = combine_with_spans(pages)
    eid = str(finding["eid"])
    deps = RepairDeps(
        finding=finding,
        eid=eid,
        subtree_xml=f"<article eId='{eid}'><num>2</num></article>",
        source_text="ARTICLE 2",
        page_no=2,
        object_key="uploads/gb/abc.pdf",
        country="gb",
        version_id="v1",
        akn_xml=akn,
        doc_text=doc_text,
        spans=spans_to_json(spans),
        findings=[finding],
    )
    return deps, eid


def _plan_json(eid: str, body: str) -> str:
    return json.dumps({"ops": [{"op": "set_body", "eid": eid, "bluebell": body}], "reasoning": ""})


@pytest.mark.asyncio
async def test_cross_page_repair_reads_the_next_page_then_grounds() -> None:
    """The agent pulls evidence from a page other than the finding's own, and
    the sandbox accepts the plan because grounding is document-wide."""
    deps, eid = _workspace_deps()
    calls: list[int] = []

    def drive(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="read_page", args={"page": 3}, tool_call_id="c1")]
            )
        return _submit(_plan_json(eid, "The restored body of the second article."))

    agent = build_agent(FunctionModel(drive), _Store())
    result = await agent.run("Repair the flagged finding.", deps=deps)

    assert result.output.ops, "plan should survive the sandbox validator"
    returns = [
        p
        for m in result.all_messages()
        for p in getattr(m, "parts", ())
        if isinstance(p, ToolReturnPart) and p.tool_name == "read_page"
    ]
    assert returns and "restored body" in str(returns[0].content)


@pytest.mark.asyncio
async def test_sandbox_rejection_comes_back_as_a_retry_and_self_corrects() -> None:
    deps, eid = _workspace_deps()
    calls: list[int] = []

    def drive(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 1:  # fabricated text: fails the grounding floor
            return _submit(_plan_json(eid, "Wholly invented unrelated words here."), "s1")
        return _submit(_plan_json(eid, "The restored body of the second article."), "s2")

    agent = build_agent(FunctionModel(drive), _Store())
    result = await agent.run("Repair the flagged finding.", deps=deps)

    assert len(calls) == 2, "the rejection must reach the model as a retry"
    assert result.output.ops


@pytest.mark.asyncio
async def test_revision_budget_exhaustion_raises() -> None:
    deps, eid = _workspace_deps()

    def stubborn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return _submit(_plan_json(eid, "Wholly invented unrelated words here."))

    agent = build_agent(FunctionModel(stubborn), _Store())
    with pytest.raises(UnexpectedModelBehavior, match="retries"):
        await agent.run("Repair the flagged finding.", deps=deps)


@pytest.mark.asyncio
async def test_abstention_always_passes_the_validator() -> None:
    deps, _ = _workspace_deps()

    def refuse(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return _submit(json.dumps({"ops": [], "reasoning": "no evidence"}))

    agent = build_agent(FunctionModel(refuse), _Store())
    result = await agent.run("Repair the flagged finding.", deps=deps)
    assert result.output.ops == []


def test_page_access_is_bounded_to_evidence_pages() -> None:
    deps, _ = _workspace_deps()
    store = _Store()

    def ask(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(m, ModelResponse) for m in messages):
            return ModelResponse(
                parts=[ToolCallPart(tool_name="read_page", args={"page": 99}, tool_call_id="c1")]
            )
        return _submit(_ABSTAIN)

    agent = build_agent(FunctionModel(ask), store)
    result = agent.run_sync("Repair the flagged finding.", deps=deps)
    returns = [
        p
        for m in result.all_messages()
        for p in getattr(m, "parts", ())
        if isinstance(p, ToolReturnPart) and p.tool_name == "read_page"
    ]
    assert returns and "not in evidence" in str(returns[0].content)


@pytest.mark.asyncio
async def test_preview_tool_takes_a_json_string_and_reports_the_verdict() -> None:
    """The op union cannot ride a tool schema through the gateway (the demo
    failure mode: every structured-arg call failed validation and burned the
    tool retries), so the tool takes the plan as a JSON string."""
    deps, eid = _workspace_deps()
    calls: list[int] = []

    def drive(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="preview_plan",
                        args={
                            "plan_json": _plan_json(eid, "The restored body of the second article.")
                        },
                        tool_call_id="c1",
                    )
                ]
            )
        return _submit(_plan_json(eid, "The restored body of the second article."))

    agent = build_agent(FunctionModel(drive), _Store())
    result = await agent.run("Repair the flagged finding.", deps=deps)
    returns = [
        p
        for m in result.all_messages()
        for p in getattr(m, "parts", ())
        if isinstance(p, ToolReturnPart) and p.tool_name == "preview_plan"
    ]
    assert returns and "verdict: accepted" in str(returns[0].content)
    # Acceptance steers to the exit and records the plan for budget salvage.
    assert "call submit_plan" in str(returns[0].content)
    assert deps.preview_state["accepted_plan_json"] == _plan_json(
        eid, "The restored body of the second article."
    )
    assert result.output.ops


@pytest.mark.asyncio
async def test_preview_tool_reports_a_bad_payload_instead_of_raising() -> None:
    deps, _ = _workspace_deps()
    calls: list[int] = []

    def drive(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="preview_plan",
                        args={"plan_json": "{not valid json"},
                        tool_call_id="c1",
                    )
                ]
            )
        return _submit(_ABSTAIN)

    agent = build_agent(FunctionModel(drive), _Store())
    result = await agent.run("Repair the flagged finding.", deps=deps)
    returns = [
        p
        for m in result.all_messages()
        for p in getattr(m, "parts", ())
        if isinstance(p, ToolReturnPart) and p.tool_name == "preview_plan"
    ]
    # Feedback as a tool RETURN, never a retry-consuming exception.
    assert returns and "could not parse the plan" in str(returns[0].content)


def _preview(plan_json: str, call_id: str) -> ModelResponse:
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name="preview_plan", args={"plan_json": plan_json}, tool_call_id=call_id
            )
        ]
    )


@pytest.mark.asyncio
async def test_bad_submit_json_retries_then_succeeds() -> None:
    """A malformed submission is a ModelRetry, not a crash, the model gets to
    correct it inside the revision budget."""
    deps, _ = _workspace_deps()
    calls: list[int] = []

    def drive(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 1:
            return _submit("{not valid json", "s1")
        return _submit(_ABSTAIN, "s2")

    agent = build_agent(FunctionModel(drive), _Store())
    result = await agent.run("Repair the flagged finding.", deps=deps)
    assert len(calls) == 2
    assert result.output.ops == []


@pytest.mark.asyncio
async def test_text_only_response_is_nudged_into_a_tool_call() -> None:
    """Tool-mode output turns a text-only turn into a retry prompt (the framework's
    built-in continue-nudge) instead of ending the run with unparsed prose."""
    deps, _ = _workspace_deps()
    calls: list[int] = []

    def drive(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 1:
            return ModelResponse(parts=[TextPart("I believe the document needs no repair.")])
        return _submit(_ABSTAIN)

    agent = build_agent(FunctionModel(drive), _Store())
    result = await agent.run("Repair the flagged finding.", deps=deps)
    assert len(calls) == 2, "the text turn must come back as a retry, not an answer"
    assert result.output.ops == []


@pytest.mark.asyncio
async def test_repeated_accepted_preview_short_circuits() -> None:
    """Re-previewing an accepted plan (the observed loop: 72 identical calls)
    skips the sandbox and points at submit_plan instead."""
    deps, eid = _workspace_deps()
    plan = _plan_json(eid, "The restored body of the second article.")
    calls: list[int] = []

    def drive(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 1:
            return _preview(plan, "c1")
        if len(calls) == 2:
            return _preview(plan, "c2")
        return _submit(plan, "s1")

    agent = build_agent(FunctionModel(drive), _Store())
    result = await agent.run("Repair the flagged finding.", deps=deps)
    returns = [
        str(p.content)
        for m in result.all_messages()
        for p in getattr(m, "parts", ())
        if isinstance(p, ToolReturnPart) and p.tool_name == "preview_plan"
    ]
    assert len(returns) == 2
    assert "verdict: accepted" in returns[0]
    assert "already accepted" in returns[1] and "verdict" not in returns[1]
    assert result.output.ops


@pytest.mark.asyncio
async def test_rejected_preview_records_nothing() -> None:
    """Only ACCEPTED plans are salvage material, a rejected preview must not
    leave a plan behind for the budget-death path to adopt."""
    deps, eid = _workspace_deps()
    calls: list[int] = []

    def drive(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 1:  # fabricated body: fails the grounding floor
            return _preview(_plan_json(eid, "Wholly invented unrelated words here."), "c1")
        return _submit(_ABSTAIN, "s1")

    agent = build_agent(FunctionModel(drive), _Store())
    await agent.run("Repair the flagged finding.", deps=deps)
    assert "accepted_plan_json" not in deps.preview_state


@pytest.mark.asyncio
async def test_abstention_preview_records_for_salvage() -> None:
    deps, _ = _workspace_deps()
    calls: list[int] = []

    def drive(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        if len(calls) == 1:
            return _preview(_ABSTAIN, "c1")
        return _submit(_ABSTAIN, "s1")

    agent = build_agent(FunctionModel(drive), _Store())
    result = await agent.run("Repair the flagged finding.", deps=deps)
    assert deps.preview_state["accepted_plan_json"] == _ABSTAIN
    assert result.output.ops == []


@pytest.mark.asyncio
async def test_tool_call_beside_submit_plan_is_skipped() -> None:
    deps, eid = _workspace_deps()
    plan = _plan_json(eid, "The restored body of the second article.")

    def drive(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(tool_name="submit_plan", args={"plan_json": plan}, tool_call_id="s"),
                ToolCallPart(tool_name="preview_plan", args={"plan_json": plan}, tool_call_id="p"),
            ]
        )

    agent = build_agent(FunctionModel(drive), _Store())
    result = await agent.run("Repair the flagged finding.", deps=deps)

    assert dict(deps.preview_state) == {}, "preview_plan ran beside submit_plan"
    assert [op.eid for op in result.output.ops] == [eid]
