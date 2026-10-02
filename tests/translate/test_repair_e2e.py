"""End-to-end integration tests for the Phase 1 translate-document flow, covering
the seams between primitives that unit tests exercise in isolation.

`TestRepairLoopEndToEnd`: when the first LLM pass drops sentinels or ships source
script, the repair loop restores them and the audit ends at
`numeric_slot_recall == 1.0` with `total_repairs > 0`; a digit-free document
takes the fast path at `repair_pass_count == 0` with no repair call.
`TestOCRHeaderBeltAndBraces`: `strip_ocr_headers` before translation and the
audit's header-bleed sweep after it, on gazette running headers.
`TestHebrewCoverage` and `TestMoneyGroupingSeparators`: alien-script `<num>`
detection against a Hebrew target, and comma-grouped and decimal amounts
surviving the sentinel round trip.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from lxml import etree

from codify.pipeline.enrich.arabic_normalise import strip_ocr_headers
from codify.translate import translate_document
from codify.translate.audit import audit_translation
from codify.translate.numeric_extract import (
    decode_sentinels,
    extract_tokens,
)

_AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"


# ── Stub LLMs ──────────────────────────────────────────────────────────────


class _DropThenRepairLLM:
    """First body-fill call for each batch strips sentinels; the repair
    call (which sees a different prompt shape) restores the encoded
    source. Simulates the "LLM sometimes drops markers but can be
    prompted to restore" pattern the repair loop was designed for.
    """

    def __init__(self) -> None:
        self.body_fill_calls = 0
        self.repair_calls = 0

    async def chat_schema_stream(
        self,
        prompt: str,
        schema: Any,
        *,
        on_partial: Any = None,
        system: str | None = None,
        model: str | None = None,
    ) -> Any:
        return schema.model_validate(
            {
                "target_language": "Identity",
                "defined_terms": [],
                "terms_of_art": [],
                "named_entities": [],
                "deontic_conventions": "",
                "structural_conventions": "",
                "ambiguities": [],
            }
        )

    async def chat_schema(
        self,
        prompt: str,
        schema: Any,
        *,
        system: str | None = None,
        model: str | None = None,
    ) -> Any:
        import re

        # Repair prompt starts with "Target language: X\n\nRepair the
        # translation", treat that as the repair path.
        if "Repair the translation" in prompt:
            self.repair_calls += 1
            start = prompt.find("Provision:\n")
            payload = json.loads(prompt[start + len("Provision:\n") :])
            return schema.model_validate(
                {
                    "eid": payload["eid"],
                    "heading": None,
                    "lines": [payload["body_source_with_sentinels"]],
                }
            )

        # Body-fill: strip every sentinel from the echoed body.
        self.body_fill_calls += 1
        batch_start = prompt.rfind("Batch:\n")
        payload = json.loads(prompt[batch_start + len("Batch:\n") :])
        blocks = []
        for e in payload:
            body = re.sub(r"[⟨‹]N\d{3,}[⟩›]", "", e["body"] or "")
            blocks.append(
                {
                    "eid": e["eid"],
                    "heading": e["heading"],
                    "lines": body.split("\n") if body else [],
                }
            )
        return schema.model_validate({"blocks": blocks})

    async def chat(
        self, prompt: str, system: str | None = None, model: str | None = None, **kw: Any
    ) -> str:
        return prompt.split("Title to translate:\n", 1)[-1].strip()


class _CleanIdentityLLM:
    """Echoes body-fill batches verbatim (sentinels round-trip). No
    repair calls expected, used to prove the sentinel-empty fast path.
    """

    def __init__(self) -> None:
        self.body_fill_calls = 0
        self.repair_calls = 0

    async def chat_schema_stream(
        self,
        prompt: str,
        schema: Any,
        *,
        on_partial: Any = None,
        system: str | None = None,
        model: str | None = None,
    ) -> Any:
        return schema.model_validate(
            {
                "target_language": "Identity",
                "defined_terms": [],
                "terms_of_art": [],
                "named_entities": [],
                "deontic_conventions": "",
                "structural_conventions": "",
                "ambiguities": [],
            }
        )

    async def chat_schema(
        self,
        prompt: str,
        schema: Any,
        *,
        system: str | None = None,
        model: str | None = None,
    ) -> Any:
        if "Repair the translation" in prompt:
            self.repair_calls += 1
            raise AssertionError("repair should not fire on clean document")
        self.body_fill_calls += 1
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

    async def chat(
        self, prompt: str, system: str | None = None, model: str | None = None, **kw: Any
    ) -> str:
        return prompt.split("Title to translate:\n", 1)[-1].strip()


class _OmitThenRepairLLM:
    """Body-fill omits every block (an LLM dropping a whole provision); the
    repair call then translates it. Proves an omitted unit routes through the
    repair loop (via the marker _fallback_block now emits) and is recovered,
    with no residual fallback-to-source.
    """

    def __init__(self) -> None:
        self.body_fill_calls = 0
        self.repair_calls = 0

    async def chat_schema_stream(
        self,
        prompt: str,
        schema: Any,
        *,
        on_partial: Any = None,
        system: str | None = None,
        model: str | None = None,
    ) -> Any:
        return schema.model_validate(
            {
                "target_language": "Identity",
                "defined_terms": [],
                "terms_of_art": [],
                "named_entities": [],
                "deontic_conventions": "",
                "structural_conventions": "",
                "ambiguities": [],
            }
        )

    async def chat_schema(
        self,
        prompt: str,
        schema: Any,
        *,
        system: str | None = None,
        model: str | None = None,
    ) -> Any:
        if "Repair the translation" in prompt:
            self.repair_calls += 1
            start = prompt.find("Provision:\n")
            payload = json.loads(prompt[start + len("Provision:\n") :])
            return schema.model_validate(
                {"eid": payload["eid"], "heading": None, "lines": ["Translated body."]}
            )
        # Body-fill: omit every block so each unit hits the fallback path.
        self.body_fill_calls += 1
        return schema.model_validate({"blocks": []})

    async def chat(
        self, prompt: str, system: str | None = None, model: str | None = None, **kw: Any
    ) -> str:
        return prompt.split("Title to translate:\n", 1)[-1].strip()


class _EchoArabicThenRepairLLM:
    """Body-fill returns the provision with its source-script text left in
    place (present, no marker), the dominant body_alien failure, where the
    model echoes the source instead of translating it. The alien detector must
    route it to repair, which translates it.
    """

    def __init__(self) -> None:
        self.body_fill_calls = 0
        self.repair_calls = 0

    async def chat_schema_stream(
        self,
        prompt: str,
        schema: Any,
        *,
        on_partial: Any = None,
        system: str | None = None,
        model: str | None = None,
    ) -> Any:
        return schema.model_validate(
            {
                "target_language": "English",
                "defined_terms": [],
                "terms_of_art": [],
                "named_entities": [],
                "deontic_conventions": "",
                "structural_conventions": "",
                "ambiguities": [],
            }
        )

    async def chat_schema(
        self,
        prompt: str,
        schema: Any,
        *,
        system: str | None = None,
        model: str | None = None,
    ) -> Any:
        if "Repair the translation" in prompt:
            self.repair_calls += 1
            start = prompt.find("Provision:\n")
            payload = json.loads(prompt[start + len("Provision:\n") :])
            return schema.model_validate(
                {"eid": payload["eid"], "heading": None, "lines": ["Fully translated body."]}
            )
        # Body-fill: echo the source body verbatim (left in the source script).
        self.body_fill_calls += 1
        batch_start = prompt.rfind("Batch:\n")
        payload = json.loads(prompt[batch_start + len("Batch:\n") :])
        return schema.model_validate(
            {
                "blocks": [
                    {
                        "eid": e["eid"],
                        "heading": e["heading"],
                        "lines": (e["body"] or "").split("\n") if e["body"] else [],
                    }
                    for e in payload
                ]
            }
        )

    async def chat(
        self, prompt: str, system: str | None = None, model: str | None = None, **kw: Any
    ) -> str:
        return prompt.split("Title to translate:\n", 1)[-1].strip()


class _StubbornAlienLLM(_EchoArabicThenRepairLLM):
    """Body-fill and the FIRST repair both leave the source script in place; the
    second repair translates. Proves source-script residue keeps the provision
    outstanding across passes rather than dropping after one failed repair.
    """

    async def chat_schema(
        self,
        prompt: str,
        schema: Any,
        *,
        system: str | None = None,
        model: str | None = None,
    ) -> Any:
        if "Repair the translation" in prompt:
            self.repair_calls += 1
            start = prompt.find("Provision:\n")
            payload = json.loads(prompt[start + len("Provision:\n") :])
            if self.repair_calls >= 2:
                return schema.model_validate(
                    {"eid": payload["eid"], "heading": None, "lines": ["Fully translated body."]}
                )
            # First repair still echoes the (Arabic) source.
            return schema.model_validate(
                {
                    "eid": payload["eid"],
                    "heading": None,
                    "lines": [payload["body_source_with_sentinels"]],
                }
            )
        return await super().chat_schema(prompt, schema, system=system, model=model)


# ── Fixtures ───────────────────────────────────────────────────────────────


def _akn_with_one_article(body_text: str) -> str:
    """Minimal AKN fixture with one article carrying the given body. Includes
    the ``<meta>`` block ``parse_akn`` requires so the fixture round-trips
    through the real parser."""
    return (
        f"<akomaNtoso xmlns='{_AKN_NS}'>"
        "<act>"
        "<meta>"
        "<identification source='#test'>"
        "<FRBRWork>"
        "<FRBRthis value='/akn/xx/act/2020/1/main'/>"
        "<FRBRuri value='/akn/xx/act/2020/1'/>"
        "<FRBRdate date='2020-01-01' name='Generation'/>"
        "<FRBRauthor href='#test'/>"
        "<FRBRcountry value='xx'/>"
        "</FRBRWork>"
        "<FRBRExpression>"
        "<FRBRthis value='/akn/xx/act/2020/1/eng@2020-01-01/main'/>"
        "<FRBRuri value='/akn/xx/act/2020/1/eng@2020-01-01'/>"
        "<FRBRdate date='2020-01-01' name='Generation'/>"
        "<FRBRauthor href='#test'/>"
        "<FRBRlanguage language='eng'/>"
        "</FRBRExpression>"
        "<FRBRManifestation>"
        "<FRBRthis value='/akn/xx/act/2020/1/eng@2020-01-01/main.xml'/>"
        "<FRBRuri value='/akn/xx/act/2020/1/eng@2020-01-01.akn'/>"
        "<FRBRdate date='2020-01-01' name='Generation'/>"
        "<FRBRauthor href='#test'/>"
        "</FRBRManifestation>"
        "</identification>"
        "</meta>"
        "<preface></preface>"
        "<body>"
        "<article eId='art_1'>"
        "<num>1</num>"
        "<content>"
        f"<p>{body_text}</p>"
        "</content>"
        "</article>"
        "</body>"
        "</act>"
        "</akomaNtoso>"
    )


class TestRepairLoopEndToEnd:
    async def test_dropped_sentinels_get_restored_by_repair(self):
        """LLM drops sentinels on first pass; repair loop restores them.
        Final audit reflects perfect recall + repair activity."""
        source = _akn_with_one_article(
            "Any person who violates Article (12) shall pay 1000 Zerzuran Dinars."
        )
        llm = _DropThenRepairLLM()
        result = await translate_document(
            source,
            target_language="Identity",
            llm=llm,
        )
        # Repair fired
        assert llm.repair_calls > 0
        assert result.audit["repair_pass_count"] > 0
        assert result.audit["total_repairs"] > 0
        # Every sentinel got restored
        assert result.audit["numeric_slot_recall"] == 1.0
        assert result.audit["sentinels_missing"] == 0

    async def test_omitted_block_is_repaired_not_left_as_source(self):
        """The LLM omits a whole provision; _fallback_block marks it, so the
        repair loop translates it. The provision ships translated, not as source
        Arabic, and a recovered fallback does not count toward the grade ratio."""
        source = _akn_with_one_article("نص عربي غير مترجم.")
        llm = _OmitThenRepairLLM()
        result = await translate_document(source, target_language="Identity", llm=llm)
        assert llm.repair_calls > 0
        assert result.audit["repair_pass_count"] > 0
        assert "Translated body." in result.akn_xml
        assert "نص عربي" not in result.akn_xml  # source Arabic did not ship

    async def test_source_script_line_is_repaired_not_shipped(self):
        """Body-fill returns the article body still in the source script (present,
        no marker). The alien detector routes it to repair, which translates it,
        so the delivered body carries no source-script residue."""
        source = _akn_with_one_article("هذا نص عربي غير مترجم بالكامل.")
        llm = _EchoArabicThenRepairLLM()
        result = await translate_document(source, target_language="English", llm=llm)
        assert llm.repair_calls > 0
        assert result.audit["repair_pass_count"] > 0
        assert result.audit["body_alien_script_hits"] == 0  # residue cleared
        assert "Fully translated body." in result.akn_xml
        assert "نص عربي" not in result.akn_xml

    async def test_source_script_residue_retried_until_it_clears(self):
        """A repair that still echoes the source must not drop the provision after
        one attempt: source-script residue is recomputed each pass and keeps the
        eid outstanding until it clears (or the iteration budget runs out)."""
        source = _akn_with_one_article("هذا نص عربي غير مترجم بالكامل.")
        llm = _StubbornAlienLLM()
        result = await translate_document(source, target_language="English", llm=llm)
        assert llm.repair_calls >= 2  # retried past the first failed repair
        assert result.audit["body_alien_script_hits"] == 0
        assert "Fully translated body." in result.akn_xml

    async def test_clean_document_no_repair_calls(self):
        """No numeric tokens → no sentinels → no repair calls."""
        source = _akn_with_one_article(
            "The Minister shall issue Regulations governing this matter."
        )
        llm = _CleanIdentityLLM()
        result = await translate_document(
            source,
            target_language="Identity",
            llm=llm,
        )
        assert llm.repair_calls == 0
        assert result.audit["repair_pass_count"] == 0
        assert result.audit["total_repairs"] == 0

    async def test_clean_document_with_digits_fast_path(self):
        """Body has digits but the LLM echoes cleanly, sentinels
        round-trip on first pass, no repair fires."""
        source = _akn_with_one_article("A term of 30 days shall apply.")
        llm = _CleanIdentityLLM()
        result = await translate_document(
            source,
            target_language="Identity",
            llm=llm,
        )
        assert llm.repair_calls == 0
        assert result.audit["repair_pass_count"] == 0
        assert result.audit["numeric_slot_recall"] == 1.0
        assert result.audit["sentinels_expected"] >= 1


class TestOCRHeaderBeltAndBraces:
    def test_strip_removes_bleed_before_translation(self):
        """The pre-translation strip pass removes header bleed from
        the source tree; the audit's post-translation sweep finds
        zero hits because nothing was there to translate."""
        xml = (
            f"<root xmlns='{_AKN_NS}'>"
            "<p>الوقائع الزرزورية header</p>"
            "<p>substantive prose without any header</p>"
            "</root>"
        )
        root = etree.fromstring(xml.encode())
        patterns = [r"الوقائع\s+الزرزورية"]
        n = strip_ocr_headers(root, patterns)
        assert n == 1
        # After strip, first paragraph no longer carries the masthead.
        paras = list(root.iter(f"{{{_AKN_NS}}}p"))
        assert "الوقائع" not in (paras[0].text or "")

    def test_audit_sweep_catches_survival_when_strip_missed(self):
        """If a translation carries text matching an OCR-header pattern,
        the audit's post-translation sweep catches it. This is the
        'braces' half of the belt-and-braces design."""
        translated_prose = (
            "Introduction to the Environmental Law\n"
            "Al-Waqa'i Al-Zerzuriyya\n"
            "The Minister shall issue Regulations."
        )
        out = audit_translation(
            "source",
            translated_prose,
            {},
            header_patterns=[r"Al\s*-?\s*Waqa'?i\s+Al\s*-?\s*Zerzuriyya"],
        )
        assert out["header_bleed_hits"]

    def test_belt_holds_no_bleed_reaches_audit(self):
        """Realistic flow: strip runs on the source AKN, translation
        proceeds on the cleaned bodies, audit sweep finds no bleed."""
        xml = (
            f"<root xmlns='{_AKN_NS}'><p>الوقائع الزرزورية prefix substantive prose text</p></root>"
        )
        root = etree.fromstring(xml.encode())
        patterns = [r"الوقائع\s+الزرزورية"]
        strip_ocr_headers(root, patterns)
        # After strip, a downstream translation of the cleaned p.text
        # would not carry the masthead. The audit sees no bleed.
        translated_text = "substantive prose text"
        out = audit_translation("source", translated_text, {}, header_patterns=patterns)
        assert out["header_bleed_hits"] == {}


class TestHebrewCoverage:
    def test_alien_arabic_in_num_flagged_for_heb_target(self):
        """Arabic script in <num> for a Hebrew-target translation is
        contamination, the audit must catch it."""
        translated_xml = (
            "<akn:akn xmlns:akn='http://docs.oasis-open.org/legaldocml/ns/akn/3.0'>"
            "<akn:article><akn:num>مادة 5</akn:num><akn:p>text</akn:p></akn:article>"
            "</akn:akn>"
        )
        out = audit_translation("src", translated_xml, {}, target_language="heb")
        assert out["num_alien_script_hits"] == 1

    def test_clean_hebrew_num_no_hits(self):
        translated_xml = (
            "<akn:akn xmlns:akn='http://docs.oasis-open.org/legaldocml/ns/akn/3.0'>"
            "<akn:article><akn:num>סעיף 5</akn:num><akn:p>טקסט</akn:p></akn:article>"
            "</akn:akn>"
        )
        out = audit_translation("src", translated_xml, {}, target_language="heb")
        # Hebrew letters in <num> are fine for a Hebrew target.
        assert out["num_alien_script_hits"] == 0

    def test_hebrew_article_ref_extracted(self):
        """The extractor must recognise Hebrew article labels
        (סעיף, הסעיף, פסקה, הפסקה) and produce article-ref tokens
        with the digit as the target surface."""
        m = extract_tokens("על פי סעיף 14 לחוק")
        art_tokens = [t for t in m.tokens if t.kind == "article_ref"]
        assert len(art_tokens) == 1
        assert art_tokens[0].expected_target_surface == "14"

    def test_hebrew_paragraph_ref_extracted(self):
        m = extract_tokens("פסקה 3 קובעת")
        art_tokens = [t for t in m.tokens if t.kind == "article_ref"]
        assert len(art_tokens) == 1
        assert art_tokens[0].expected_target_surface == "3"


class TestMoneyGroupingSeparators:
    def test_comma_grouped_amount_extracted(self):
        m = extract_tokens("a fine of 1,000 Zerzuran Dinars")
        money_tokens = [t for t in m.tokens if t.kind == "money"]
        assert len(money_tokens) == 1
        # Grouping preserved as-is in target surface (digits + . + ,)
        assert money_tokens[0].expected_target_surface == "1,000"

    def test_decimal_amount_extracted(self):
        m = extract_tokens("500.50 Dinars per day")
        money_tokens = [t for t in m.tokens if t.kind == "money"]
        assert len(money_tokens) == 1
        assert money_tokens[0].expected_target_surface == "500.50"

    def test_grouped_amount_round_trips_via_sentinel(self):
        source = "a fine of 1,000.50 Zerzuran Dinars"
        m = extract_tokens(source)
        from codify.translate.numeric_extract import (
            encode_sentinels,
        )

        encoded = encode_sentinels(source, m)
        decoded, missing = decode_sentinels(encoded, m)
        assert missing == []
        assert "1,000.50" in decoded


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
