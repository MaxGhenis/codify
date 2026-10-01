"""The adjudicator, exercised offline.

The guarantee this module sells is not that the model is right, it is that a
wrong answer is a wrong choice from a declared set rather than an invented
hierarchy. These tests are about that boundary. Whether the model is any good is
a separate question, answered by measuring it against a hand-marked sample of
real spans rather than by anything here.
"""

from __future__ import annotations

import pytest
from pydantic_ai import capture_run_messages
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from codify.adjudicate import (
    Adjudication,
    AdjudicationRequest,
    build_request,
    candidate_kinds,
    is_adjudicable,
    validate_choice,
)
from codify.adjudicate.agent import build_adjudicator
from codify.jurisdictions import JurisdictionConfig, load_config
from codify.quality.invariants import AmbiguitySpan


# Called per test, not at import: a tree without the config must skip, and a
# module-level call raises during collection where no hook can convert it.
def xz_config() -> JurisdictionConfig:
    return load_config("xz")


def _request(**over: object) -> AdjudicationRequest:
    base: dict[str, object] = {
        "line": "مادة (5)",
        "reason": "marker_shaped_line_unclaimed",
        "candidate_kinds": ("part", "chapter", "article"),
    }
    return AdjudicationRequest(**{**base, **over})  # type: ignore[arg-type]


async def _no_page(object_key: str, page_no: int) -> bytes:
    raise AssertionError("no test here should reach the page renderer")


class TestTheClosedSet:
    def test_a_kind_outside_the_offered_set_is_rejected(self) -> None:
        """The whole guarantee. `subsection` is a real AKN element, so the shape
        is valid and the invariant gate downstream would accept it; only the
        offered set knows this document class never declares one."""
        with pytest.raises(ValueError, match="was not offered"):
            validate_choice(Adjudication(kind="subsection", reason="looks like one"), _request())

    def test_refusing_is_always_allowed(self) -> None:
        verdict = Adjudication(kind=None, reason="a cross-reference inside a sentence")
        assert validate_choice(verdict, _request()).kind is None

    def test_an_offered_kind_passes(self) -> None:
        verdict = Adjudication(kind="article", number="5", reason="opens the body below it")
        assert validate_choice(verdict, _request()).number == "5"

    def test_a_number_without_a_kind_is_not_a_reading(self) -> None:
        with pytest.raises(ValueError, match="not a reading"):
            Adjudication(kind=None, number="5", reason="there is a five here")

    def test_a_class_with_no_declared_hierarchy_can_only_be_refused(self) -> None:
        """Offering nothing is how an unknown document class stays unanswerable
        rather than becoming answerable with a guess."""
        assert candidate_kinds(None, "act") == ()
        assert candidate_kinds(xz_config(), "not-a-real-class") == ()


class TestWhatGetsAsked:
    def test_only_the_unsettled_readings_are_adjudicable(self) -> None:
        """A `duplicate_number` is a decision the scanner made and can defend;
        sending it to a model would re-open a settled question."""
        unclaimed = AmbiguitySpan(
            kind="unmatched_marker",
            start=0,
            end=5,
            detail={"reason": "marker_shaped_line_unclaimed"},
        )
        settled = AmbiguitySpan(
            kind="duplicate_number", start=0, end=5, detail={"reason": "toc_twin"}
        )
        assert is_adjudicable(unclaimed) is True
        assert is_adjudicable(settled) is False

    def test_the_request_carries_the_line_and_its_neighbours(self) -> None:
        text = "أولاً\nثانياً\nمادة (3)\nنص الحكم.\nمادة (4)\n"
        span = AmbiguitySpan(
            kind="unmatched_marker",
            start=text.index("مادة (3)"),
            end=text.index("مادة (3)") + 8,
            detail={"reason": "marker_shaped_line_unclaimed"},
        )
        req = build_request(span, text, config=xz_config(), country="xz", doctype="act")
        assert req.line == "مادة (3)"
        assert req.before[-1] == "ثانياً"
        assert req.after[0] == "نص الحكم."
        assert "article" in req.candidate_kinds


class TestTheAgentBoundary:
    @pytest.mark.asyncio
    async def test_an_unoffered_kind_is_sent_back_for_another_try(self) -> None:
        """The validator has to reach the model as a retry, not surface as a
        crash: a hallucinated level is an ordinary thing for it to do once."""
        calls: list[int] = []

        def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(1)
            kind = "subsection" if len(calls) == 1 else "article"
            return ModelResponse(
                parts=[
                    TextPart(f'{{"kind": "{kind}", "number": "5", "reason": "it opens a unit"}}')
                ]
            )

        agent = build_adjudicator(FunctionModel(answer), _no_page)
        with capture_run_messages():
            result = await agent.run("Decide this line.", deps=_request())
        assert result.output.kind == "article"
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_a_refusal_survives_the_validator(self) -> None:
        def refuse(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[TextPart('{"kind": null, "number": null, "reason": "prose, mid-sentence"}')]
            )

        agent = build_adjudicator(FunctionModel(refuse), _no_page)
        result = await agent.run("Decide this line.", deps=_request())
        assert result.output.kind is None
