# ruff: noqa: E501  # AKN XML test fixtures: line wraps would change tested whitespace
"""Preamble + longTitle translation: extractors, patcher, fallback flags."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from lxml import etree

from codify.akn import AKN_NS
from codify.translate.translate import (
    _extract_preamble_lines,
    _extract_preface_lines,
    _retry_backfilled_lines,
    _translate_preamble,
)
from codify.translate.translate_bodies import _UNTRANSLATED_MARKER
from codify.translate.write import (
    apply_translation_to_akn,
    preamble_p_slots,
    preface_p_slots,
)

_AKN = f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act>
    <meta><identification source="#codify"/></meta>
    <preface>
      <longTitle><p>قانون المراعي رقم 7 لسنة 1999</p></longTitle>
      <p>جمهورية زرزورة</p>
      <p></p>
    </preface>
    <preamble>
      <formula name="enactingFormula"><p>أصدرنا القانون الآتي:</p></formula>
      <recitals>
        <recital><p>بعد الاطلاع على القانون الأساسي،</p></recital>
        <recital><p>وبناءً على تنسيب مجلس الوزراء،</p></recital>
      </recitals>
      <p>وباسم الشعب،</p>
    </preamble>
    <body><article eId="art_1"><num>1</num><content><p>نص المادة.</p></content></article></body>
  </act>
</akomaNtoso>
'''


class TestSlotExtraction:
    def test_preface_slots_include_long_title_and_skip_empty(self):
        lines = _extract_preface_lines(_AKN)
        assert lines == ["قانون المراعي رقم 7 لسنة 1999", "جمهورية زرزورة"]

    def test_preamble_slots_cover_formula_recitals_and_direct_p(self):
        lines = _extract_preamble_lines(_AKN)
        assert lines == [
            "أصدرنا القانون الآتي:",
            "بعد الاطلاع على القانون الأساسي،",
            "وبناءً على تنسيب مجلس الوزراء،",
            "وباسم الشعب،",
        ]

    def test_extract_and_patch_share_slot_order(self):
        root = etree.fromstring(_AKN.encode("utf-8"))
        assert len(preface_p_slots(root)) == len(_extract_preface_lines(_AKN))
        assert len(preamble_p_slots(root)) == len(_extract_preamble_lines(_AKN))

    def test_missing_regions_yield_empty(self):
        bare = f'<akomaNtoso xmlns="{AKN_NS}"><act><body/></act></akomaNtoso>'
        assert _extract_preface_lines(bare) == []
        assert _extract_preamble_lines(bare) == []


class TestPatchPreamble:
    def test_patches_all_slots(self):
        out = apply_translation_to_akn(
            _AKN,
            [],
            ["Waterways Law No. 7 of 1902", "State of Zerzura"],
            target_language="English",
            preamble_lines=[
                "We have promulgated the following law:",
                "Having reviewed the Basic Law,",
                "And upon the recommendation of the Council of Ministers,",
                "And in the name of the people,",
            ],
        )
        root = etree.fromstring(out.encode("utf-8"))
        texts = ["".join(p.itertext()) for p in preamble_p_slots(root)]
        assert texts == [
            "We have promulgated the following law:",
            "Having reviewed the Basic Law,",
            "And upon the recommendation of the Council of Ministers,",
            "And in the name of the people,",
        ]
        preface_texts = ["".join(p.itertext()) for p in preface_p_slots(root)]
        assert preface_texts == ["Waterways Law No. 7 of 1902", "State of Zerzura"]

    def test_none_preamble_lines_leave_source_untouched(self):
        out = apply_translation_to_akn(_AKN, [], [], target_language="English")
        root = etree.fromstring(out.encode("utf-8"))
        texts = ["".join(p.itertext()) for p in preamble_p_slots(root)]
        assert texts[0] == "أصدرنا القانون الآتي:"


class _RaisingLLM:
    async def chat(self, *a: Any, **kw: Any) -> str:
        raise RuntimeError("boom")


class _ShortLLM:
    """Answers without slot markers at all, so no slot can be matched."""

    async def chat(self, *a: Any, **kw: Any) -> str:
        return "Only one line back."


@pytest.mark.asyncio
async def test_preamble_llm_failure_falls_back_with_flag():
    src = ["سطر أول", "سطر ثان"]
    lines, flags = await _translate_preamble(src, target_language="English", llm=_RaisingLLM())
    assert lines == src
    assert flags[0]["code"] == "translate_preamble_fallback"
    assert flags[0]["location"] == "preamble"


@pytest.mark.asyncio
async def test_marker_less_response_falls_back_wholesale():
    # Nothing parseable came back, so the region keeps its source rather than
    # spending a retry per slot on a model that is ignoring the protocol.
    src = ["سطر أول", "سطر ثان", "سطر ثالث"]
    lines, flags = await _translate_preamble(src, target_language="English", llm=_ShortLLM())
    assert lines == src
    assert any(f["code"] == "translate_preamble_fallback" for f in flags)


class _ShortThenRaisingLLM:
    """Returns one keyed slot, then fails every per-slot retry."""

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, *a: Any, **kw: Any) -> str:
        self.calls += 1
        if self.calls == 1:
            return "[[0]] Only one line back."
        raise RuntimeError("retry boom")


@pytest.mark.asyncio
async def test_preamble_retry_failure_falls_back_to_source_not_sentinel():
    src = ["سطر أول", "سطر ثان", "سطر ثالث"]
    lines, flags = await _translate_preamble(
        src, target_language="English", llm=_ShortThenRaisingLLM()
    )
    assert len(lines) == 3
    # The literal placeholder must never survive; failed slots keep source.
    assert all("[UNTRANSLATED]" not in line for line in lines)
    assert lines[1] == src[1] and lines[2] == src[2]
    assert any(f["code"] == "translate_preamble_line_fallback_source" for f in flags)


@pytest.mark.asyncio
async def test_empty_preamble_is_noop():
    lines, flags = await _translate_preamble([], target_language="English", llm=_RaisingLLM())
    assert lines == [] and flags == []


class _EchoThenTranslateLLM:
    """Echoes the Arabic source once, then translates: the persistent retry must
    refuse the echo and re-ask before it accepts a real translation."""

    def __init__(self, translation: str) -> None:
        self.translation = translation
        self.calls = 0

    async def chat(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        self.calls += 1
        source = prompt.strip().splitlines()[-1]
        return source if self.calls == 1 else self.translation


class _AlwaysEchoLLM:
    """Always echoes the source line, so no attempt is a real translation."""

    async def chat(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        return prompt.strip().splitlines()[-1]


@pytest.mark.asyncio
async def test_retry_refuses_source_echo_then_accepts_translation():
    src = ["قانون المراعي رقم 7 لسنة 1999"]
    llm = _EchoThenTranslateLLM("Pastures Law No. 7 of 1999")
    out, fallback = await _retry_backfilled_lines(
        [_UNTRANSLATED_MARKER],
        src,
        location="preface",
        system_prompt="",
        target_language="English",
        llm=llm,
    )
    assert fallback == 0
    assert out[0] == "Pastures Law No. 7 of 1999"
    assert llm.calls >= 2  # the Arabic echo was refused and it re-asked


@pytest.mark.asyncio
async def test_retry_falls_back_to_source_only_after_every_attempt_echoes():
    src = ["قانون المراعي رقم 7 لسنة 1999"]
    out, fallback = await _retry_backfilled_lines(
        [_UNTRANSLATED_MARKER],
        src,
        location="preface",
        system_prompt="",
        target_language="English",
        llm=_AlwaysEchoLLM(),
    )
    assert fallback == 1
    assert out[0] == src[0]  # source retained only after all attempts failed


class _FixedReplyLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def chat(self, *a: Any, **kw: Any) -> str:
        return self.reply


@pytest.mark.asyncio
async def test_retry_keeps_translation_that_retains_one_source_token():
    # A mostly-English translation that keeps a single Arabic proper noun is a
    # good result; it must not be reverted to the fully-untranslated source.
    src = ["قرار رئيس سلطة المراعي رقم 5 لسنة 2014"]
    reply = "Decision of the Head of the السلطة for Environment Quality No. 5 of 2014"
    out, fallback = await _retry_backfilled_lines(
        [_UNTRANSLATED_MARKER],
        src,
        location="preface",
        system_prompt="",
        target_language="English",
        llm=_FixedReplyLLM(reply),
    )
    assert fallback == 0
    assert out[0] == reply


@pytest.mark.asyncio
async def test_retry_rejects_exact_echo_even_with_no_alien_table():
    # A target with no alien-script table (patterns fold to empty) must still
    # refuse an exact echo rather than accept it on length alone.
    src = ["قانون المراعي رقم 7 لسنة 1999"]
    out, fallback = await _retry_backfilled_lines(
        [_UNTRANSLATED_MARKER],
        src,
        location="preface",
        system_prompt="",
        target_language="French",
        llm=_AlwaysEchoLLM(),
    )
    assert fallback == 1
    assert out[0] == src[0]


class _BatchEchoesSlotThenRetryTranslatesLLM:
    """Batch echoes the Arabic source for slot 0 and translates slot 1; the
    per-slot retry then translates slot 0. Exercises batch-level echo rejection:
    an echoed batch slot must not be accepted on length alone."""

    def __init__(self, sources: list[str]) -> None:
        self.sources = sources
        self.calls = 0

    async def chat(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        self.calls += 1
        if self.calls == 1:  # the batch call
            return f"[[0]] {self.sources[0]}\n[[1]] English one"
        src = prompt.strip().splitlines()[-1]  # per-slot retry
        return "ENGLISH " + str(self.sources.index(src))


@pytest.mark.asyncio
async def test_batch_source_echo_is_rejected_and_retried():
    src = ["قانون رقم 6 لسنة 1998", "بشأن مراكز الإصلاح"]
    llm = _BatchEchoesSlotThenRetryTranslatesLLM(src)
    lines, flags = await _translate_preamble(src, target_language="English", llm=llm)
    # Slot 0's batch echo was refused and retried; slot 1's batch translation stood.
    assert lines[0] == "ENGLISH 0"
    assert lines[1] == "English one"
    assert all("[UNTRANSLATED]" not in line for line in lines)
    assert any(f["code"] == "translate_preamble_slot_echoed" for f in flags)


class _BatchEchoesAllThenRetryTranslatesLLM:
    """Batch echoes every slot (so nothing is accepted), then the retry
    translates each. Guards the all-echoed case from short-circuiting to source."""

    def __init__(self, sources: list[str]) -> None:
        self.sources = sources
        self.calls = 0

    async def chat(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        self.calls += 1
        if self.calls == 1:
            return "\n".join(f"[[{i}]] {s}" for i, s in enumerate(self.sources))
        src = prompt.strip().splitlines()[-1]
        return "ENGLISH " + str(self.sources.index(src))


@pytest.mark.asyncio
async def test_all_slots_echoed_are_retried_not_returned_as_source():
    # When every batch slot is an echo, translations is empty; the region must
    # still route each slot into the retry rather than short-circuiting to source.
    src = ["قانون رقم 6 لسنة 1998", "بشأن مراكز الإصلاح"]
    lines, _flags = await _translate_preamble(
        src, target_language="English", llm=_BatchEchoesAllThenRetryTranslatesLLM(src)
    )
    assert lines == ["ENGLISH 0", "ENGLISH 1"]
    assert all("[UNTRANSLATED]" not in line for line in lines)


_EU_SHAPES = f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act>
    <meta><identification source="#codify"/></meta>
    <preface>
      <longTitle><p><docType>Directive</docType> <docNumber>2014/24/EU</docNumber> on public procurement</p></longTitle>
    </preface>
    <preamble>
      <citations><citation eId="cit_1"><p>Having regard to the Treaty,</p></citation></citations>
      <recitals><recital eId="rec_1"><p>Whereas public procurement matters,</p></recital></recitals>
    </preamble>
    <body><article eId="art_1"><num>1</num><content><p>Scope.</p></content></article></body>
  </act>
</akomaNtoso>
'''


class TestEuShapes:
    def test_semantic_long_title_is_not_a_slot(self):
        # <docType>/<docNumber> children would be destroyed by _set_p_text;
        # mixed-content long titles are excluded from the patch slots.
        root = etree.fromstring(_EU_SHAPES.encode("utf-8"))
        assert preface_p_slots(root) == []

    def test_citations_are_preamble_slots(self):
        root = etree.fromstring(_EU_SHAPES.encode("utf-8"))
        texts = ["".join(p.itertext()) for p in preamble_p_slots(root)]
        assert texts == [
            "Having regard to the Treaty,",
            "Whereas public procurement matters,",
        ]
        lines = _extract_preamble_lines(_EU_SHAPES)
        assert lines == texts


class _ShortThenChattyLLM:
    """Under-returns, then answers retries with chatter plus the line."""

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, *a: Any, **kw: Any) -> str:
        self.calls += 1
        if self.calls == 1:
            return "[[0]] Only one line back."
        return "Here is the translation:\nThe actual legal line."


@pytest.mark.asyncio
async def test_retry_multi_line_output_kept_whole():
    src = ["سطر أول", "سطر ثان"]
    lines, _ = await _translate_preamble(src, target_language="English", llm=_ShortThenChattyLLM())
    assert len(lines) == 2
    # Multi-line retry output is joined, never silently truncated to line 1.
    assert "actual legal line" in lines[1]


class TestFurniturePassthrough:
    """Non-linguistic lines never reach the model."""

    def test_furniture_detector(self) -> None:
        from codify.translate.translate import _is_furniture_line

        for line in ("30062000", "-٥-", "00033/000", "12/4/2004", "٢٠٠٤"):
            assert _is_furniture_line(line), line
        for line in (
            "الوقائع الزرزورية",
            "The Council of Ministers",
            "قرار رقم (39) لسنة 2004",
        ):
            assert not _is_furniture_line(line), line

    @pytest.mark.asyncio
    async def test_furniture_slots_copied_verbatim_and_kept_out_of_prompt(self) -> None:
        from codify.translate.translate import _translate_flat_lines

        llm = AsyncMock()
        # Keys are absolute source indices, so the furniture slot at 1 is
        # simply absent from the exchange.
        llm.chat.return_value = "[[0]] Line one.\n\n[[2]] Line two."
        lines, flags = await _translate_flat_lines(
            ["سطر أول", "30062000", "سطر ثاني"],
            location="preface",
            system_prompt="s",
            target_language="English",
            llm=llm,
        )
        assert lines == ["Line one.", "30062000", "Line two."]
        assert any(f["code"] == "translate_preface_furniture_passthrough" for f in flags)
        assert "30062000" not in llm.chat.call_args_list[0].args[0]

    @pytest.mark.asyncio
    async def test_all_furniture_block_makes_no_llm_call(self) -> None:
        from codify.translate.translate import _translate_flat_lines

        llm = AsyncMock()
        lines, _flags = await _translate_flat_lines(
            ["123/456", "-٥-"],
            location="preface",
            system_prompt="s",
            target_language="English",
            llm=llm,
        )
        assert lines == ["123/456", "-٥-"]
        assert llm.chat.call_count == 0


class TestFrontMatterParityRepair:
    """Silent duplication and empty slots are retried."""

    @pytest.mark.asyncio
    async def test_duplicated_slot_retried_individually(self) -> None:
        from codify.translate.translate import _translate_flat_lines

        llm = AsyncMock()
        llm.chat.side_effect = [
            "[[0]] Dup line.\n\n[[1]] Dup line.",
            "Second line fixed.",
        ]
        lines, flags = await _translate_flat_lines(
            ["سطر أول", "سطر ثاني مختلف"],
            location="preamble",
            system_prompt="s",
            target_language="English",
            llm=llm,
        )
        assert lines == ["Dup line.", "Second line fixed."]
        assert any(f["code"] == "translate_preamble_parity_repair" for f in flags)

    @pytest.mark.asyncio
    async def test_duplicate_faithful_to_duplicate_source_not_retried(self) -> None:
        from codify.translate.translate import _translate_flat_lines

        llm = AsyncMock()
        llm.chat.return_value = "[[0]] Same title.\n\n[[1]] Same title."
        lines, flags = await _translate_flat_lines(
            ["نفس اللقب", "نفس اللقب"],
            location="preface",
            system_prompt="s",
            target_language="English",
            llm=llm,
        )
        assert lines == ["Same title.", "Same title."]
        assert llm.chat.call_count == 1
        assert not any(f["code"] == "translate_preface_parity_repair" for f in flags)

    @pytest.mark.asyncio
    async def test_parity_retry_falls_back_to_source(self) -> None:
        from codify.translate.translate import _translate_flat_lines

        llm = AsyncMock()
        llm.chat.side_effect = ["[[0]] Dup.\n\n[[1]] Dup.", RuntimeError("down")]
        lines, flags = await _translate_flat_lines(
            ["سطر أول", "سطر ثاني مختلف"],
            location="preamble",
            system_prompt="s",
            target_language="English",
            llm=llm,
        )
        assert lines == ["Dup.", "سطر ثاني مختلف"]
        assert any(f["code"] == "translate_preamble_line_fallback_source" for f in flags)


class TestFrontMatterSlotDefects:
    """The audit-side counter feeding the persist gate."""

    def _doc(self, preface_ps: list[str]) -> str:
        from codify.akn import AKN_NS

        paras = "".join(f'<p eId="preface__p_{i + 1}">{t}</p>' for i, t in enumerate(preface_ps))
        return (
            f'<akomaNtoso xmlns="{AKN_NS}"><act>'
            f'<meta><identification source="#codify"/></meta>'
            f"<preface>{paras}</preface>"
            f'<body><section eId="sec_1"><content><p>x</p></content></section></body>'
            f"</act></akomaNtoso>"
        )

    def test_counts_empty_and_duplicate_slots(self) -> None:
        from codify.translate.audit import front_matter_slot_defects

        src = self._doc(["سطر أول", "سطر ثاني", "سطر ثالث"])
        tgt = self._doc(["Line one", "", "Line one"])
        defects = front_matter_slot_defects(src, tgt)
        empty, duplicate = defects.empty_slots, defects.duplicate_slots
        assert empty == 1
        # slot 3 duplicates slot 1, not its neighbour; adjacency rule means
        # only ADJACENT duplicates count, and slot 2 is empty so no pair.
        assert duplicate == 0
        tgt2 = self._doc(["Line one", "Line one", "Line three"])
        second = front_matter_slot_defects(src, tgt2)
        assert (second.empty_slots, second.duplicate_slots) == (0, 1)

    def test_clean_translation_counts_zero(self) -> None:
        from codify.translate.audit import front_matter_slot_defects

        src = self._doc(["سطر أول", "سطر ثاني"])
        tgt = self._doc(["Line one", "Line two"])
        clean = front_matter_slot_defects(src, tgt)
        assert (clean.empty_slots, clean.duplicate_slots) == (0, 0)


class TestReviewHardening:
    """Review-driven cases: script coverage, triple collapse, alignment."""

    def test_hebrew_and_cyrillic_are_linguistic(self) -> None:
        from codify.translate.translate import _is_furniture_line

        for line in (
            "חוק המים",
            "Закон про водні ресурси:",
            "Νόμος για τα ύδατα",
        ):
            assert not _is_furniture_line(line), line
        for line in ("٣٠٠٦٢٠٠٠", "123-456", "١٢/٤/٢٠٠٤"):
            assert _is_furniture_line(line), line

    @pytest.mark.asyncio
    async def test_triple_collapse_repairs_both_extras(self) -> None:
        from codify.translate.translate import _translate_flat_lines

        llm = AsyncMock()
        llm.chat.side_effect = [
            "[[0]] Dup.\n\n[[1]] Dup.\n\n[[2]] Dup.",
            "Fixed two.",
            "Fixed three.",
        ]
        lines, flags = await _translate_flat_lines(
            ["سطر أول", "سطر ثاني", "سطر ثالث"],
            location="preamble",
            system_prompt="s",
            target_language="English",
            llm=llm,
        )
        assert lines == ["Dup.", "Fixed two.", "Fixed three."]
        assert llm.chat.call_count == 3

    @pytest.mark.asyncio
    async def test_backfill_retry_uses_full_slot_indices_past_furniture(self) -> None:
        # LLM under-returns for the two linguistic slots; the retry for the
        # backfilled slot must translate source slot 3 (after the furniture
        # slot), not slot 2.
        from codify.translate.translate import _translate_flat_lines

        llm = AsyncMock()
        llm.chat.side_effect = ["[[0]] Only one line.", "Second linguistic fixed."]
        lines, _flags = await _translate_flat_lines(
            ["سطر أول", "30062000", "سطر ثاني مميز"],
            location="preface",
            system_prompt="s",
            target_language="English",
            llm=llm,
        )
        assert lines == ["Only one line.", "30062000", "Second linguistic fixed."]
        retry_prompt = llm.chat.call_args_list[1].args[0]
        assert "سطر ثاني مميز" in retry_prompt
        assert "30062000" not in retry_prompt

    def test_slot_defects_unparseable_returns_zero(self) -> None:
        from codify.translate.audit import front_matter_slot_defects

        unreadable = front_matter_slot_defects("<broken", "<also-broken")
        assert (unreadable.empty_slots, unreadable.duplicate_slots) == (0, 0)


class TestSlotKeyedContract:
    """Front-matter slots are addressed by key, never by arrival order, so a
    slot the model drops cannot move its neighbours."""

    def test_parses_latin_and_arabic_indic_markers(self) -> None:
        from codify.translate.translate import _parse_slots

        assert _parse_slots("[[0]] First\n\n[[1]] Second")[0] == {
            0: "First",
            1: "Second",
        }
        # An Arabic-target response may return the marker in Arabic-Indic digits.
        assert _parse_slots("[[٠]] أول\n\n[[١]] ثان")[0] == {0: "أول", 1: "ثان"}
        # A single newline between slots is as good as a blank one.
        assert _parse_slots("[[2]] Only one line\n[[3]] and the next")[0] == {
            2: "Only one line",
            3: "and the next",
        }
        # A repeated key keeps the first copy rather than letting a duplicate
        # overwrite good text.
        assert _parse_slots("[[1]] Good\n\n[[1]] Duplicate")[0] == {1: "Good"}

    @pytest.mark.asyncio
    async def test_dropped_middle_slot_does_not_shift_its_successors(self) -> None:
        # A paragraph mid-list comes back
        # missing. Under positional backfill every following slot moved up one
        # and stayed shifted for 42 paragraphs; keyed slots retry in place.
        from codify.translate.translate import _translate_flat_lines

        llm = AsyncMock()
        llm.chat.side_effect = [
            "[[0]] Article 44.\n\n[[1]] Article 45.\n\n[[3]] Article 47.\n\n[[4]] Article 48.",
            "Article 46.",
        ]
        lines, flags = await _translate_flat_lines(
            ["مادة ٤٤", "مادة ٤٥", "مادة ٤٦", "مادة ٤٧", "مادة ٤٨"],
            location="preface",
            system_prompt="s",
            target_language="English",
            llm=llm,
        )
        assert lines == [
            "Article 44.",
            "Article 45.",
            "Article 46.",
            "Article 47.",
            "Article 48.",
        ]
        assert any(f["code"] == "translate_preface_slot_unmatched" for f in flags)
        # The retry asked for the dropped slot's own source line, not a neighbour.
        assert "مادة ٤٦" in llm.chat.call_args_list[1].args[0]

    @pytest.mark.asyncio
    async def test_reordered_response_lands_on_the_right_slots(self) -> None:
        from codify.translate.translate import _translate_flat_lines

        llm = AsyncMock()
        llm.chat.return_value = "[[2]] Third.\n\n[[0]] First.\n\n[[1]] Second."
        lines, _flags = await _translate_flat_lines(
            ["أول", "ثان", "ثالث"],
            location="preamble",
            system_prompt="s",
            target_language="English",
            llm=llm,
        )
        assert lines == ["First.", "Second.", "Third."]

    @pytest.mark.asyncio
    async def test_invented_keys_are_ignored_not_written(self) -> None:
        from codify.translate.translate import _translate_flat_lines

        llm = AsyncMock()
        llm.chat.side_effect = [
            "[[0]] First.\n\n[[9]] Text for a slot that was never sent.",
            "Second.",
        ]
        lines, _flags = await _translate_flat_lines(
            ["أول", "ثان"],
            location="preface",
            system_prompt="s",
            target_language="English",
            llm=llm,
        )
        assert lines == ["First.", "Second."]

    @pytest.mark.asyncio
    async def test_truncated_tail_is_retried_in_batches(self) -> None:
        # A response that stops early leaves its tail unmatched; those slots
        # retry individually instead of the surplus merging into one slot.
        from codify.translate.translate import _translate_flat_lines

        src = [f"سطر {i}" for i in range(45)]
        first = "\n\n".join(f"[[{i}]] Line {i}." for i in range(30))
        second = "\n\n".join(f"[[{i}]] Line {i}." for i in range(40, 45))
        llm = AsyncMock()
        llm.chat.side_effect = [first, second] + [f"Retried {i}." for i in range(30, 40)]
        lines, flags = await _translate_flat_lines(
            src,
            location="preface",
            system_prompt="s",
            target_language="English",
            llm=llm,
        )
        assert lines[:30] == [f"Line {i}." for i in range(30)]
        assert lines[30:40] == [f"Retried {i}." for i in range(30, 40)]
        assert lines[40:] == [f"Line {i}." for i in range(40, 45)]
        assert any(f["code"] == "translate_preface_slot_unmatched" for f in flags)

    @pytest.mark.asyncio
    async def test_one_failed_batch_leaves_other_batches_aligned(self) -> None:
        from codify.translate.translate import _translate_flat_lines

        src = [f"سطر {i}" for i in range(45)]
        second = "\n\n".join(f"[[{i}]] Line {i}." for i in range(40, 45))
        llm = AsyncMock()
        llm.chat.side_effect = [RuntimeError("batch down"), second] + [
            f"Retried {i}." for i in range(40)
        ]
        lines, flags = await _translate_flat_lines(
            src,
            location="preface",
            system_prompt="s",
            target_language="English",
            llm=llm,
        )
        assert lines[40:] == [f"Line {i}." for i in range(40, 45)]
        assert lines[:40] == [f"Retried {i}." for i in range(40)]
        assert any(f["code"] == "translate_preface_batch_failed" for f in flags)


class TestSlotContractHardening:
    """Failure shapes found reviewing the first cut of the slot contract. Each
    one placed real text under the wrong identifier or shipped source language
    behind a passing gate, which is the class the contract exists to remove."""

    @pytest.mark.asyncio
    async def test_a_later_batch_renumbering_cannot_overwrite_an_earlier_one(
        self,
    ) -> None:
        # Renumbering from zero is the likeliest malformation of a batched
        # keyed prompt. Those keys belong to batch 1, so accepting them would
        # put batch 2's paragraphs under batch 1's identifiers.
        from codify.translate.translate import _translate_flat_lines

        src = [f"سطر {i}" for i in range(45)]
        first = "\n\n".join(f"[[{i}]] Correct {i}." for i in range(40))
        renumbered = "\n\n".join(f"[[{i}]] Wrong {i}." for i in range(5))
        llm = AsyncMock()
        llm.chat.side_effect = [first, renumbered] + [f"Retried {i}." for i in range(40, 45)]
        lines, flags = await _translate_flat_lines(
            src,
            location="preface",
            system_prompt="s",
            target_language="English",
            llm=llm,
        )
        assert lines[:40] == [f"Correct {i}." for i in range(40)]
        assert lines[40:] == [f"Retried {i}." for i in range(40, 45)]
        assert any(f["code"] == "translate_preface_slot_malformed" for f in flags)

    @pytest.mark.asyncio
    async def test_marker_inside_a_translation_does_not_split_a_slot(self) -> None:
        # An unanchored marker would truncate slot 0 here and hand its tail to
        # slot 1, discarding the real slot 1 that follows.
        from codify.translate.translate import _parse_slots

        parsed, repeats = _parse_slots(
            "[[0]] See annex [[1]] of the schedule and the rest.\n\n[[1]] Real slot one."
        )
        assert parsed[0] == "See annex [[1]] of the schedule and the rest."
        assert parsed[1] == "Real slot one."
        assert repeats == 0

    def test_repeated_keys_are_counted_not_silently_dropped(self) -> None:
        from codify.translate.translate import _parse_slots

        parsed, repeats = _parse_slots("[[1]] Good\n\n[[1]] Duplicate")
        assert parsed == {1: "Good"}
        assert repeats == 1

    @pytest.mark.asyncio
    async def test_trailing_commentary_is_refused_rather_than_delivered(self) -> None:
        # Text after the last marker lands in the last slot; a slot far longer
        # than its source is retried instead of shipped as legal text.
        from codify.translate.translate import _translate_flat_lines

        src = ["نص قصير", "نص قصير آخر ولكنه أطول قليلاً من الأول لأغراض القياس"]
        commentary = (
            "[[0]] Short text.\n\n[[1]] Another short text, slightly longer than the first.\n\n"
            "Note: I could not translate the second paragraph because it appears to be a "
            "page footer rather than legislative text, and I have therefore left it out of "
            "the translation entirely for your review and consideration."
        )
        llm = AsyncMock()
        llm.chat.side_effect = [commentary, "Another short text, slightly longer."]
        lines, flags = await _translate_flat_lines(
            src,
            location="preface",
            system_prompt="s",
            target_language="English",
            llm=llm,
        )
        assert "page footer" not in lines[1]
        assert any(f["code"] == "translate_preface_slot_malformed" for f in flags)

    @pytest.mark.asyncio
    async def test_duplicate_separated_by_furniture_is_still_caught(self) -> None:
        from codify.translate.translate import _translate_flat_lines

        llm = AsyncMock()
        llm.chat.side_effect = ["[[0]] Alpha.\n\n[[2]] Alpha.", "Beta."]
        lines, flags = await _translate_flat_lines(
            ["أول", "00033/000", "ثان"],
            location="preface",
            system_prompt="s",
            target_language="English",
            llm=llm,
        )
        assert lines == ["Alpha.", "00033/000", "Beta."]
        assert any(f["code"] == "translate_preface_parity_repair" for f in flags)

    @pytest.mark.asyncio
    async def test_total_failure_reports_how_much_kept_its_source(self) -> None:
        from codify.translate.translate import _translate_flat_lines

        src = [f"سطر {i}" for i in range(45)]
        llm = AsyncMock()
        llm.chat.side_effect = RuntimeError("gateway 503")
        lines, flags = await _translate_flat_lines(
            src,
            location="preface",
            system_prompt="s",
            target_language="English",
            llm=llm,
        )
        assert lines == src
        issue = next(f["issue"] for f in flags if f["code"] == "translate_preface_fallback")
        assert "45" in issue and "2" in issue

    @pytest.mark.asyncio
    async def test_a_retry_that_repeats_commentary_falls_back_to_source(self) -> None:
        # The retry is the last line of defence; an unvalidated one ships
        # whatever the model repeats.
        from codify.translate.translate import _translate_flat_lines

        src = ["نص قصير", "نص قصير آخر ولكنه أطول قليلاً من الأول لأغراض القياس"]
        commentary = (
            "Another short text. Note: I could not translate the second paragraph because "
            "it appears to be a page footer rather than legislative text, and I have "
            "therefore left it out of the translation entirely for your consideration."
        )
        llm = AsyncMock()
        llm.chat.side_effect = [f"[[0]] Short text.\n\n[[1]] {commentary}", commentary]
        lines, flags = await _translate_flat_lines(
            src,
            location="preface",
            system_prompt="s",
            target_language="English",
            llm=llm,
        )
        assert lines[1] == src[1]
        assert any(f["code"] == "translate_preface_line_fallback_source" for f in flags)


class TestSourceRetentionIsScriptAware:
    """Equality with the source is not proof of a fallback: a slot blocks
    delivery, so a legitimately unchanged paragraph must not count."""

    def _doc(self, paras: list[str]) -> str:
        inner = "".join(f'<p eId="preface__p_{i + 1}">{t}</p>' for i, t in enumerate(paras))
        return (
            f'<akomaNtoso xmlns="{AKN_NS}"><act>'
            f'<meta><identification source="#codify"/></meta>'
            f"<preface>{inner}</preface>"
            f'<body><section eId="sec_1"><content><p>x</p></content></section></body>'
            f"</act></akomaNtoso>"
        )

    def test_only_source_script_counts(self) -> None:
        from codify.translate.audit import front_matter_source_retained

        # An acronym, a citation and stamp digits all survive translation
        # unchanged; only the Arabic paragraph is a retained fallback.
        text = ["نص عربي غير مترجم", "UNESCO", "12/1999", "00033/000"]
        doc = self._doc(text)
        assert front_matter_source_retained(doc, doc, "eng") == 1

    def test_unknown_target_language_counts_nothing(self) -> None:
        from codify.translate.audit import front_matter_source_retained

        doc = self._doc(["نص عربي غير مترجم"])
        assert front_matter_source_retained(doc, doc, None) == 0

    def test_a_translated_slot_is_not_retention(self) -> None:
        from codify.translate.audit import front_matter_source_retained

        src = self._doc(["نص عربي"])
        tgt = self._doc(["Arabic text"])
        assert front_matter_source_retained(src, tgt, "eng") == 0

    def test_arabic_indic_digit_furniture_is_not_retention(self) -> None:
        from codify.translate.audit import front_matter_source_retained

        # Arabic-Indic numerals (a section number, a year) are passed through
        # verbatim as furniture; a number is not untranslated text and must not
        # trip the gate, even though the digits sit in the Arabic Unicode block.
        doc = self._doc(["١. ٦٠", "١٩٤٠"])
        assert front_matter_source_retained(doc, doc, "eng") == 0

    def test_arabic_letters_still_count_beside_digits(self) -> None:
        from codify.translate.audit import front_matter_source_retained

        # A genuinely untranslated Arabic line is still caught when it carries
        # letters, digits alongside them do not mask the retention.
        doc = self._doc(["قرار رقم ٢٩ لسنة ١٩٤٠"])
        assert front_matter_source_retained(doc, doc, "eng") == 1


class TestRepeatedFurniture:
    """A gazette repeats its running masthead down the front matter. The
    translator renders the title once and leaves the repeats blank, which the
    empty-slot counter cannot tell from a dropped paragraph on its own."""

    @staticmethod
    def _doc(lines: list[str]) -> str:
        paras = "".join(f'<p eId="preface__p_{i}">{t}</p>' for i, t in enumerate(lines))
        return (
            '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0"><act>'
            f"<preface>{paras}</preface>"
            '<body><section eId="sec_1"><content><p>x</p></content></section></body>'
            "</act></akomaNtoso>"
        )

    _MASTHEAD = "قانون رقم (7) لسنة 1999 بشأن المراعي"

    def test_a_repeat_of_a_translated_line_is_not_a_drop(self) -> None:
        from codify.translate.audit import front_matter_slot_defects

        src = self._doc([self._MASTHEAD, "نص آخر", self._MASTHEAD])
        tgt = self._doc(["Law No. (7) of 1999 on the Environment", "Other text", ""])
        defects = front_matter_slot_defects(src, tgt)
        assert defects.empty_slots == 0
        assert defects.repeated_furniture_by_eid == {"preface__p_2": self._MASTHEAD}

    def test_a_genuine_drop_is_still_a_drop(self) -> None:
        from codify.translate.audit import front_matter_slot_defects

        src = self._doc([self._MASTHEAD, "فقرة فريدة", self._MASTHEAD])
        tgt = self._doc(["Law No. (7) of 1999 on the Environment", "", ""])
        defects = front_matter_slot_defects(src, tgt)
        assert defects.empty_by_eid == {"preface__p_1": "فقرة فريدة"}
        assert defects.repeated_furniture_slots == 1

    def test_a_repeat_whose_first_occurrence_was_also_blank_is_a_drop(self) -> None:
        """The text has to have reached the reader somewhere. If every copy is
        blank, the line is gone however often the source repeats it."""
        from codify.translate.audit import front_matter_slot_defects

        src = self._doc([self._MASTHEAD, "نص آخر", self._MASTHEAD])
        tgt = self._doc(["", "Other text", ""])
        defects = front_matter_slot_defects(src, tgt)
        assert sorted(defects.empty_by_eid) == ["preface__p_0", "preface__p_2"]
        assert defects.repeated_furniture_slots == 0

    def test_a_repeat_differing_only_by_orthography_still_counts(self) -> None:
        """Sources compare folded, so a tatweel in one copy of the masthead
        does not turn a benign repeat into a blocking drop."""
        from codify.translate.audit import front_matter_slot_defects

        src = self._doc([self._MASTHEAD, "نص آخر", self._MASTHEAD.replace("قانون", "قــانون")])
        tgt = self._doc(["Law No. (7) of 1999 on the Environment", "Other text", ""])
        assert front_matter_slot_defects(src, tgt).empty_slots == 0

    def test_a_source_slot_with_no_target_counterpart_is_a_drop(self) -> None:
        """A paragraph absent from the target reads the same to a reader as one
        left blank, so it is counted the same way."""
        from codify.translate.audit import front_matter_slot_defects

        src = self._doc(["نص أول", "نص ثاني"])
        tgt = self._doc(["Text one"])
        assert front_matter_slot_defects(src, tgt).empty_by_eid == {"preface__p_1": "نص ثاني"}


class TestRepeatEcho:
    """A blank slot whose source was translated higher up gets that text, so
    the reader sees the masthead the gazette prints rather than a gap."""

    @staticmethod
    def _echo(sources: list[str], lines: list[str]) -> tuple[int, list[str]]:
        from codify.translate.translate import _echo_repeated_lines

        out = list(lines)
        return _echo_repeated_lines(sources, out), out

    def test_a_repeat_is_filled_from_the_first_occurrence(self) -> None:
        filled, out = self._echo(["عنوان", "نص", "عنوان"], ["Title", "Body", ""])
        assert filled == 1
        assert out == ["Title", "Body", "Title"]

    def test_a_unique_blank_line_is_left_alone(self) -> None:
        """Only a repeat can be filled without inventing text; a unique blank
        stays blank so the delivery gate still sees the drop."""
        filled, out = self._echo(["عنوان", "نص فريد"], ["Title", ""])
        assert (filled, out) == (0, ["Title", ""])

    def test_orthographic_variants_count_as_the_same_line(self) -> None:
        filled, out = self._echo(
            ["قانون المراعي", "نص", "قــانون المراعي"], ["Pastures Law", "Body", ""]
        )
        assert (filled, out) == (1, ["Pastures Law", "Body", "Pastures Law"])

    def test_a_repeat_before_any_translation_stays_blank(self) -> None:
        """Nothing to copy from yet, and guessing forward would put text in a
        slot no translation has produced."""
        filled, out = self._echo(["عنوان", "عنوان"], ["", "Title"])
        assert (filled, out) == (0, ["", "Title"])
