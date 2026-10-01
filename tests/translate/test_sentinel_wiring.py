"""End-to-end sentinel round-trip through translate_bodies with a stub LLM.

The identity stub echoes sentinel-encoded body text verbatim; the wiring
should decode sentinels back to their expected target surfaces (digits
ASCII-folded; labelled refs keep only the digit portion). A "dropping"
stub omits a sentinel; the wiring must record the missing sentinel ID
against the eid and emit a flag.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from codify.translate.anchors import SourceUnit
from codify.translate.translate_bodies import translate_bodies


class _IdentitySentinelLLM:
    """Echoes each entry's body back verbatim. Sentinels round-trip
    cleanly and decode to their target surfaces."""

    async def chat_schema(
        self,
        prompt: str,
        schema: Any,
        *,
        system: str | None = None,
        model: str | None = None,
    ) -> Any:
        batch_start = prompt.rfind("Batch:\n")
        payload = json.loads(prompt[batch_start + len("Batch:\n") :])
        return schema.model_validate(
            {
                "blocks": [
                    {
                        "eid": e["eid"],
                        "heading": e["heading"],
                        "lines": e["body"].split("\n") if e["body"] else [],
                    }
                    for e in payload
                ]
            }
        )


class _DroppingSentinelLLM:
    """Echoes but strips every sentinel, simulates an LLM that
    paraphrased a numeric token instead of copying its marker."""

    async def chat_schema(
        self,
        prompt: str,
        schema: Any,
        *,
        system: str | None = None,
        model: str | None = None,
    ) -> Any:
        import re

        batch_start = prompt.rfind("Batch:\n")
        payload = json.loads(prompt[batch_start + len("Batch:\n") :])
        blocks = []
        for e in payload:
            body = e["body"] or ""
            # Strip every sentinel marker from the echo
            body = re.sub(r"[⟨‹]N\d{3,}[⟩›]", "", body)
            blocks.append(
                {
                    "eid": e["eid"],
                    "heading": e["heading"],
                    "lines": body.split("\n") if body else [],
                }
            )
        return schema.model_validate({"blocks": blocks})


class _PartialDropLLM:
    """Preserves the first sentinel in each body, drops the second."""

    async def chat_schema(
        self,
        prompt: str,
        schema: Any,
        *,
        system: str | None = None,
        model: str | None = None,
    ) -> Any:
        import re

        batch_start = prompt.rfind("Batch:\n")
        payload = json.loads(prompt[batch_start + len("Batch:\n") :])
        blocks = []
        for e in payload:
            body = e["body"] or ""
            # Drop the second sentinel occurrence, keep the first
            markers = re.findall(r"[⟨‹]N\d{3,}[⟩›]", body)
            if len(markers) >= 2:
                body = body.replace(markers[1], "", 1)
            blocks.append(
                {
                    "eid": e["eid"],
                    "heading": e["heading"],
                    "lines": body.split("\n") if body else [],
                }
            )
        return schema.model_validate({"blocks": blocks})


def _unit(eid: str, body: str) -> SourceUnit:
    return SourceUnit(
        akn_eid=eid,
        kind="article",
        akn_type="article",
        number="1",
        depth=1,
        heading=None,
        body_text=body,
    )


def test_fallback_block_marks_lines_for_retry_not_source() -> None:
    # An omitted or batch-failed unit must enter the repair loop rather than ship
    # its verbatim source (which the body-alien gate rejects and no retry sees).
    from codify.translate.translate_bodies import _UNTRANSLATED_MARKER, _fallback_block

    block = _fallback_block(_unit("art_1", "سطر أول\nسطر ثان"))
    assert block.lines == [_UNTRANSLATED_MARKER, _UNTRANSLATED_MARKER]
    assert block.eid == "art_1"


def test_fallback_block_empty_body_stays_empty() -> None:
    from codify.translate.translate_bodies import _fallback_block

    assert _fallback_block(_unit("art_1", "")).lines == []


class TestSentinelRoundTrip:
    async def test_identity_translation_decodes_to_expected_surfaces(self):
        units = [
            _unit(
                "art_62",
                "Any person who violates the provisions of Article (12) shall be "
                "punished with a fine of not less than 1000 Zerzuran Dinars.",
            )
        ]
        outcome = await translate_bodies(
            units,
            target_language="Identity",
            notes_block="",
            llm=_IdentitySentinelLLM(),
            concurrency=1,
            batch_size=8,
        )
        # Should have extracted at least 2 tokens (article ref + money)
        assert outcome.sentinels_expected >= 2
        assert outcome.sentinels_missing == 0
        assert outcome.stray_sentinels == 0

        # Decoded output preserves the digits
        block = outcome.response.bodies[0]
        joined = "\n".join(block.lines)
        assert "12" in joined
        assert "1000" in joined
        # Sentinels do not leak into the persisted output
        assert "⟨" not in joined
        assert "‹" not in joined

    async def test_dropped_sentinels_flag_the_eid(self):
        units = [_unit("art_62", "See Article 5 within 30 days.")]
        outcome = await translate_bodies(
            units,
            target_language="Identity",
            notes_block="",
            llm=_DroppingSentinelLLM(),
            concurrency=1,
            batch_size=8,
        )
        # Both sentinels expected; both dropped by the stub
        assert outcome.sentinels_expected == 2
        assert outcome.sentinels_missing == 2
        assert "art_62" in outcome.sentinels_missing_by_eid
        # Flag emitted
        sentinel_flags = [f for f in outcome.flags if f["code"] == "translate_sentinel_missing"]
        assert len(sentinel_flags) == 1
        assert "art_62" == sentinel_flags[0]["location"]

    async def test_partial_drop_reports_only_the_missing_one(self):
        units = [
            _unit(
                "art_62",
                "Any person who violates Article (12) shall be punished with 1000 Dinars.",
            )
        ]
        outcome = await translate_bodies(
            units,
            target_language="Identity",
            notes_block="",
            llm=_PartialDropLLM(),
            concurrency=1,
            batch_size=8,
        )
        # One preserved, one dropped
        assert outcome.sentinels_missing == 1

    async def test_no_numeric_tokens_no_sentinel_overhead(self):
        """A body with no digit-carrying tokens should extract zero
        sentinels, the wiring should short-circuit cleanly."""
        units = [_unit("art_1", "The Minister shall issue Regulations.")]
        outcome = await translate_bodies(
            units,
            target_language="Identity",
            notes_block="",
            llm=_IdentitySentinelLLM(),
            concurrency=1,
            batch_size=8,
        )
        assert outcome.sentinels_expected == 0
        assert outcome.sentinels_missing == 0
        # Output identical to source
        assert outcome.response.bodies[0].lines == ["The Minister shall issue Regulations."]


class TestGloballyUniqueIds:
    async def test_multi_batch_ids_are_unique_across_batches(self):
        """When two batches translate, sentinel IDs must not collide
        across batches so the audit's attribution is unambiguous."""
        # Force multiple batches by using batch_size=1 and 2 units
        units = [
            _unit("art_a", "See Article 5."),
            _unit("art_b", "See Article 10."),
        ]
        outcome = await translate_bodies(
            units,
            target_language="Identity",
            notes_block="",
            llm=_IdentitySentinelLLM(),
            concurrency=2,
            batch_size=1,
        )
        assert outcome.sentinels_missing == 0
        # Both bodies decoded cleanly
        assert "5" in outcome.response.bodies[0].lines[0]
        assert "10" in outcome.response.bodies[1].lines[0]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])


class _MarkerInventingLLM:
    """Emits a marker on a synthetic source unit without numerals."""

    async def chat_schema(
        self,
        prompt: str,
        schema: Any,
        *,
        system: str | None = None,
        model: str | None = None,
    ) -> Any:
        batch_start = prompt.rfind("Batch:\n")
        payload = json.loads(prompt[batch_start + len("Batch:\n") :])
        return schema.model_validate(
            {
                "blocks": [
                    {
                        "eid": e["eid"],
                        "heading": e["heading"],
                        "lines": ["a term of ⟨N001⟩ years"],
                    }
                    for e in payload
                ]
            }
        )


class TestStraySweepIsUnconditional:
    async def test_marker_on_a_token_free_unit_is_caught(self):
        units = [_unit("art_13", "imprisonment for a term of no less than six months.")]
        outcome = await translate_bodies(
            units,
            target_language="English",
            notes_block="",
            llm=_MarkerInventingLLM(),
            concurrency=1,
            batch_size=8,
        )
        assert outcome.sentinels_expected == 0
        assert outcome.stray_sentinels == 1
        assert outcome.stray_sentinels_by_eid == {"art_13": ["⟨N001⟩"]}


class _EchoPlusStrayLLM:
    """Echoes the body and appends a marker the manifest never issued."""

    async def chat_schema(
        self,
        prompt: str,
        schema: Any,
        *,
        system: str | None = None,
        model: str | None = None,
    ) -> Any:
        batch_start = prompt.rfind("Batch:\n")
        payload = json.loads(prompt[batch_start + len("Batch:\n") :])
        return schema.model_validate(
            {
                "blocks": [
                    {
                        "eid": e["eid"],
                        "heading": e["heading"],
                        "lines": [(e["body"] or "") + " ⟨N999⟩"],
                    }
                    for e in payload
                ]
            }
        )


class TestStraySweepCountsOnce:
    async def test_token_bearing_unit_counts_a_stray_once(self):
        """A unit with tokens of its own must not be swept twice."""
        outcome = await translate_bodies(
            [_unit("art_5", "a fine of 1000 dinars")],
            target_language="English",
            notes_block="",
            llm=_EchoPlusStrayLLM(),
            concurrency=1,
            batch_size=8,
        )
        assert outcome.stray_sentinels == 1
        assert outcome.stray_sentinels_by_eid == {"art_5": ["⟨N999⟩"]}
        assert sum(1 for f in outcome.flags if f["code"] == "translate_sentinel_stray") == 1
