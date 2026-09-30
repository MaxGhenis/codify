"""Tests for OCR text extraction."""

import asyncio
import io
import unicodedata

import pytest
from pypdf import PdfWriter

from codify.pipeline.enrich import ocr
from codify.pipeline.enrich.ocr import (
    LAYOUT_MODEL,
    PageResult,
    _looks_garbled,
    _promote_rival_when_empty,
    combine_page_texts,
    extract_text_from_pdf,
)


def test_rival_promoted_when_the_authoritative_read_is_empty() -> None:
    """A RECITATION refusal returns nothing; the Mistral layout pass already
    transcribed the page, so its rival text is kept rather than shipping a blank."""
    page = PageResult(
        page_number=69,
        text="",
        method="vision_ocr",
        rival_text="المادة ٣٧٢ نص عربي",
        divert_reason="garbled",
    )
    out = _promote_rival_when_empty(page)
    assert out.text == "المادة ٣٧٢ نص عربي"
    assert out.method == "mistral_ocr"
    assert out.model == LAYOUT_MODEL


def test_a_real_read_is_never_overridden_by_its_rival() -> None:
    """Vision wins on quality wherever both engines read (the engine A/B, #170),
    so a non-empty authoritative read is left untouched even when a rival exists."""
    page = PageResult(
        page_number=70, text="vision text", method="vision_ocr", rival_text="mistral text"
    )
    assert _promote_rival_when_empty(page) is page


def test_an_empty_page_with_no_rival_stays_empty() -> None:
    page = PageResult(page_number=1, text="", method="vision_ocr", rival_text="")
    assert _promote_rival_when_empty(page) is page


def test_a_blank_leaf_is_not_promoted_even_with_a_rival() -> None:
    """A page with no ink read empty correctly; a rival that still emitted body
    text is noise, so it must not be manufactured into content."""
    page = PageResult(page_number=5, text="", method="vision_ocr", rival_text="spurious", ink=0.004)
    assert _promote_rival_when_empty(page) is page


def test_promotion_keeps_the_empty_primary_read_as_the_rival() -> None:
    """The trace must still show the primary engine returned nothing: the empty
    read moves into rival_text so page_reads records both attempts."""
    page = PageResult(
        page_number=1, text="", method="vision_ocr", rival_text="from mistral", ink=0.3
    )
    out = _promote_rival_when_empty(page)
    assert out.text == "from mistral"
    assert out.method == "mistral_ocr"
    assert out.rival_text == ""  # the empty vision read, preserved


@pytest.mark.asyncio
async def test_extract_promotes_the_rival_when_vision_reads_empty(monkeypatch) -> None:
    """End to end, not just the helper: a page vision content-filters to empty
    while the layout pass transcribed it is served from the rival, so removing
    the promotion pass from extract_text_from_pdf would fail this."""
    import codify.pipeline.enrich.ocr as ocr_mod

    rival = "المادة التي قرأها مسار التخطيط بينما رفضت الرؤية إعادتها"

    class _Page:
        def extract_text(self) -> str:
            return ""  # no text layer, so the page goes to the OCR path

    class _Reader:
        def __init__(self, *_a: object, **_k: object) -> None:
            self.pages = [_Page()]

    async def _fake_layout(*_a: object, **_k: object) -> tuple[dict, dict]:
        return {}, {1: rival}

    async def _fake_ocr(*_a: object, **_k: object) -> str:
        return ""  # vision refuses (content_filter), as on act/1863 pages 69/71

    async def _fake_window(_pdf: bytes, first: int, last: int, _dpi: int) -> dict[int, bytes]:
        return {n: b"\x89PNG" for n in range(first, last + 1)}

    monkeypatch.setattr(ocr_mod, "PdfReader", _Reader)
    monkeypatch.setattr(ocr_mod, "_layout_pass", _fake_layout)
    monkeypatch.setattr(ocr_mod, "_ocr_page_with_context", _fake_ocr)
    monkeypatch.setattr(ocr_mod, "_render_window", _fake_window)
    monkeypatch.setattr(ocr_mod, "ink_ratio", lambda _b: 0.3)

    pages = await extract_text_from_pdf(b"%PDF-fake", client=object())

    assert len(pages) == 1
    assert pages[0].text == rival
    assert pages[0].method == "mistral_ocr"
    assert pages[0].rival_text == ""  # the empty vision read is preserved as the rival


# Arabic prose: dense in the definite article and common particles.
_CLEAN_AR = (
    "يجوز للسلطة أن تصدر قرارا بوقف العمل بالتسجيل متى ثبت لديها أن صاحب "
    "الأداة قد أخل بأحد الشروط الواردة في النظام، ويبلغ القرار إلى ذوي الشأن "
    "خلال المدة التي تحددها اللائحة، ولمن صدر ضده القرار أن يتظلم منه أمام "
    "لجنة التظلمات، ويكون قرارها في التظلم نهائيا من تاريخ صدوره بالجريدة "
)
# CMap-garble shape: Arabic letters scrambled into non-words, no article density.
_GARBLED_AR = "أاي نشع ألدانشعته يانشع لأده يان سييديا ههيين والقا ير والمحييوت سيي " * 4
# A subtler CMap garble: article density stays intact (so it slips the al-prefix
# test) but alef-maqsura (ى) is injected mid-word, "تسىري" for "تسري", the way
# "مادة" arrives as "ما ع" and structure detection then finds no anchors.
_MAQSURA_GARBLE = (
    "تسىري أحكىام هىذا القىانون علىى الأعمىال التجاريىة والتجىار بمىا لا "
    "يتعىارض مىع النظىام العىام والقىوانيش الخىاصة التي تحكىم المسىألة " * 4
)


def test_looks_garbled_flags_scrambled_arabic():
    assert _looks_garbled(_GARBLED_AR) is True


def test_looks_garbled_flags_midword_maqsura_injection():
    """A CMap garble that keeps article density but speckles ى mid-word must
    still route to OCR, else structure detection sees no `مادة` anchors."""
    assert _looks_garbled(_MAQSURA_GARBLE) is True


def test_looks_garbled_passes_real_arabic():
    assert _looks_garbled(_CLEAN_AR) is False


def test_looks_garbled_ignores_short_pages():
    """Under the Arabic-char floor there's too little to judge; never flag."""
    assert _looks_garbled("القانون الأساسي") is False


def test_looks_garbled_ignores_non_arabic():
    assert _looks_garbled("The quick brown fox jumps over the lazy dog. " * 8) is False


def test_looks_garbled_normalization_invariant():
    """Decomposed and composed forms of the same clean text agree."""
    decomposed = unicodedata.normalize("NFD", _CLEAN_AR)
    assert decomposed != _CLEAN_AR  # the corpus has composable sequences
    assert _looks_garbled(decomposed) == _looks_garbled(_CLEAN_AR) is False


def _blank_pdf(pages: int) -> bytes:
    """A text-less PDF; every page falls below MIN_TEXT_LENGTH so each is
    diverted to the OCR path."""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_extract_keeps_pages_when_ocr_unavailable():
    """With no LLM client the OCR pass can't run, but diverted pages must still
    appear in the output rather than being silently dropped."""
    results = await extract_text_from_pdf(_blank_pdf(3), client=None)
    assert [r.page_number for r in results] == [1, 2, 3]
    assert all(r.method == "text_extraction" for r in results)


def test_combine_page_texts():
    pages = [
        PageResult(page_number=1, text="First page.", method="text_extraction"),
        PageResult(page_number=2, text="", method="text_extraction"),
        PageResult(page_number=3, text="Third page.", method="text_extraction"),
    ]
    combined = combine_page_texts(pages)
    assert combined == "First page.\n\nThird page."


def test_combine_empty_pages():
    pages = [
        PageResult(page_number=1, text="", method="text_extraction"),
    ]
    assert combine_page_texts(pages) == ""


def test_combine_strips_control_chars():
    """Vision OCR can emit NUL/control bytes that lxml's parse rejects; the
    combined text must be XML-safe while preserving tabs/newlines."""
    pages = [
        PageResult(page_number=1, text="Clean\x00 text\x07.", method="vision_ocr"),
        PageResult(page_number=2, text="Tab\tand\nnewline kept.", method="vision_ocr"),
    ]
    combined = combine_page_texts(pages)
    assert "\x00" not in combined and "\x07" not in combined
    assert combined == "Clean text.\n\nTab\tand\nnewline kept."


def test_a_diverted_page_that_read_to_nothing_leaves_a_marker():
    """The loss a silent drop hid: a page routed to OCR that came back empty now
    marks its place in the combined text."""
    pages = [
        PageResult(page_number=1, text="First page.", method="text_extraction"),
        PageResult(page_number=2, text="", method="vision_ocr", divert_reason="too_short"),
        PageResult(page_number=3, text="Third page.", method="text_extraction"),
    ]
    assert combine_page_texts(pages) == "First page.\n\n⟦page 2 unreadable⟧\n\nThird page."


def test_a_genuinely_blank_page_is_still_dropped():
    """No divert means nobody read it and failed; an empty page is just empty."""
    pages = [
        PageResult(page_number=1, text="First page.", method="text_extraction"),
        PageResult(page_number=2, text="", method="text_extraction"),
    ]
    assert "unreadable" not in combine_page_texts(pages)


def test_a_blank_leaf_with_no_ink_is_not_called_unreadable():
    """A diverted empty page that carried no ink is a blank leaf, not a failed
    read: omitted, never marked."""
    pages = [
        PageResult(page_number=1, text="First page.", method="text_extraction"),
        PageResult(
            page_number=2, text="", method="vision_ocr", divert_reason="too_short", ink=0.004
        ),
    ]
    assert "unreadable" not in combine_page_texts(pages)


def test_both_combiners_agree_on_the_marker_and_span_it():
    """`combine_with_spans` must stay identical to `combine_page_texts`, so the
    repair loop's eId→page map holds, and the degraded page gets a span."""
    from codify.repair.grounding import combine_with_spans

    pages = [
        PageResult(page_number=1, text="First page.", method="text_extraction"),
        PageResult(page_number=2, text="", method="vision_ocr", divert_reason="too_short"),
    ]
    combined = combine_page_texts(pages)
    with_spans, spans = combine_with_spans(pages)

    assert combined == with_spans
    assert [s.page for s in spans] == [1, 2]


def test_the_marker_reaches_the_structurer():
    """The structurer lifts the marker into an editorial remark, so the region
    filter must hand it over rather than strip it."""
    from codify.pipeline.enrich.region_text import combine_text_for_structure

    text = "First page.\n\n⟦page 2 unreadable⟧\n\nThird page."

    assert combine_text_for_structure(text, {}) == text


@pytest.mark.asyncio
async def test_extract_text_from_invalid_pdf():
    """Should handle invalid PDF bytes gracefully."""
    with pytest.raises(Exception):
        await extract_text_from_pdf(b"not a pdf", client=None)


@pytest.mark.asyncio
async def test_a_divergent_text_layer_is_read_by_vision_not_kept(monkeypatch) -> None:
    """The acceptance path end to end, not just the helpers: a page carrying a
    clean-looking but divergent embedded layer is diverted to vision, tagged
    text_layer_divergent, and its rejected layer is not fed back as the anchor.
    PdfReader is faked so the page has a real text layer without a fixture; the
    rival and vision reads are stubbed (poppler is absent in CI)."""
    import codify.pipeline.enrich.ocr as ocr_mod

    # Arabic prose: clears the single-read garble check, so the ONLY reason it
    # diverts is disagreement with the rival.
    layer = "تتولى الدائرة المختصة الإشراف على سجل الأدوات وتضع القواعد التي تحدد شروط القيد فيه"
    rival = "an entirely disjoint rival transcription sharing no tokens with the layer at all"

    class _Page:
        def extract_text(self) -> str:
            return layer

    class _Reader:
        def __init__(self, *_a: object, **_k: object) -> None:
            self.pages = [_Page()]

    captured: dict[str, str] = {}

    async def _fake_layout(*_a: object, **_k: object) -> tuple[dict, dict]:
        return {}, {1: rival}

    async def _fake_ocr(*_a: object, page_text_layer: str = "", **_k: object) -> str:
        captured["anchor"] = page_text_layer
        return "المادة البديلة من مسار الرؤية النظيف"

    async def _fake_window(_pdf: bytes, first: int, last: int, _dpi: int) -> dict[int, bytes]:
        return {n: b"\x89PNG" for n in range(first, last + 1)}

    monkeypatch.setattr(ocr_mod, "PdfReader", _Reader)
    monkeypatch.setattr(ocr_mod, "_layout_pass", _fake_layout)
    monkeypatch.setattr(ocr_mod, "_ocr_page_with_context", _fake_ocr)
    monkeypatch.setattr(ocr_mod, "_render_window", _fake_window)
    monkeypatch.setattr(ocr_mod, "ink_ratio", lambda _b: 0.1)

    pages = await extract_text_from_pdf(b"%PDF-fake", client=object())

    assert len(pages) == 1
    assert pages[0].method == "vision_ocr"
    assert pages[0].divert_reason == "text_layer_divergent"
    assert pages[0].text == "المادة البديلة من مسار الرؤية النظيف"
    # The rejected layer must not anchor its own replacement.
    assert captured["anchor"] == ""


@pytest.mark.asyncio
async def test_a_layer_that_splits_words_the_rival_keeps_whole_goes_to_vision(monkeypatch):
    """The MK 132 case. The layer encodes a tracked-out label as literal spaces,
    which whole-page divergence cannot see: the two reads share almost every
    token. Only the split words fault the layer, and the layer must not anchor
    the read that replaces it."""
    import codify.pipeline.enrich.ocr as ocr_mod

    body = (
        " Bahwa para Pemohon mendalilkan ketentuan a quo bertentangan dengan "
        "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945 sepanjang dimaknai "
        "sebagaimana diuraikan dalam permohonan a quo."
    )
    layer = "N a m a : Budi Santoso\nA l a m a t : Dulang, RT. 001/RW. 000" + body
    rival = "Nama : Budi Santoso\nAlamat : Dulang, RT. 001/RW. 000" + body

    class _Page:
        def extract_text(self) -> str:
            return layer

    class _Reader:
        def __init__(self, *_a: object, **_k: object) -> None:
            self.pages = [_Page()]

    captured: dict[str, str] = {}

    async def _fake_layout(*_a: object, **_k: object) -> tuple[dict, dict]:
        return {}, {1: rival}

    async def _fake_ocr(*_a: object, page_text_layer: str = "", **_k: object) -> str:
        captured["anchor"] = page_text_layer
        return "Nama : Budi Santoso Alamat : Dulang, RT. 001/RW. 000" + body

    async def _fake_window(_pdf: bytes, first: int, last: int, _dpi: int) -> dict[int, bytes]:
        return {n: b"\x89PNG" for n in range(first, last + 1)}

    monkeypatch.setattr(ocr_mod, "PdfReader", _Reader)
    monkeypatch.setattr(ocr_mod, "_layout_pass", _fake_layout)
    monkeypatch.setattr(ocr_mod, "_ocr_page_with_context", _fake_ocr)
    monkeypatch.setattr(ocr_mod, "_render_window", _fake_window)
    monkeypatch.setattr(ocr_mod, "ink_ratio", lambda _b: 0.1)

    pages = await extract_text_from_pdf(b"%PDF-fake", client=object())

    assert len(pages) == 1
    assert pages[0].divert_reason == "letter_spaced"
    assert pages[0].method == "vision_ocr"
    assert "A l a m a t" not in pages[0].text
    # Divergence alone would have kept this page: the reads share nearly every token.
    assert pages[0].divergence is not None
    assert pages[0].divergence < ocr_mod._TEXT_LAYER_MAX_DIVERGENCE
    assert captured["anchor"] == ""


@pytest.mark.asyncio
async def test_a_page_whose_rival_splits_the_same_words_keeps_its_layer(monkeypatch):
    """A deliberately letter-spaced heading reads the same way to both engines,
    so nothing faults the layer and the exact extraction is kept."""
    import codify.pipeline.enrich.ocr as ocr_mod

    layer = (
        "K E T E N T U A N U M U M\nPasal 1 Dalam Peraturan Pemerintah ini yang "
        "dimaksud dengan konservasi sumber daya alam hayati dan ekosistemnya adalah "
        "pengelolaan sumber daya alam hayati yang pemanfaatannya dilakukan secara bijaksana."
    )

    class _Page:
        def extract_text(self) -> str:
            return layer

    class _Reader:
        def __init__(self, *_a: object, **_k: object) -> None:
            self.pages = [_Page()]

    async def _fake_layout(*_a: object, **_k: object) -> tuple[dict, dict]:
        # An independent transcription that also reads the heading spaced.
        return {}, {1: "## " + layer.replace("Pasal 1", "Pasal 1\n\n")}

    monkeypatch.setattr(ocr_mod, "PdfReader", _Reader)
    monkeypatch.setattr(ocr_mod, "_layout_pass", _fake_layout)

    pages = await extract_text_from_pdf(b"%PDF-fake", client=object())

    assert len(pages) == 1
    assert pages[0].method == "text_extraction"
    assert pages[0].divert_reason == ""


@pytest.mark.asyncio
async def test_a_diverted_layer_page_is_never_dropped_when_vision_returns_nothing(monkeypatch):
    """A page pulled for its content had a text layer over the length floor, so it
    is not a blank leaf however little ink the render finds. Before the rival was
    exempted from the ink test, such a page left the combined text with no marker:
    the divert turned two wrong words into a missing page."""
    import codify.pipeline.enrich.ocr as ocr_mod

    body = (
        " Bahwa para Pemohon mendalilkan ketentuan a quo bertentangan dengan "
        "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945 sepanjang dimaknai."
    )
    layer = "A l a m a t : Dulang, RT. 001/RW. 000" + body
    rival = "Alamat : Dulang, RT. 001/RW. 000" + body

    class _Page:
        def extract_text(self) -> str:
            return layer

    class _Reader:
        def __init__(self, *_a: object, **_k: object) -> None:
            self.pages = [_Page()]

    async def _fake_layout(*_a: object, **_k: object) -> tuple[dict, dict]:
        return {}, {1: rival}

    async def _fake_ocr(*_a: object, **_k: object) -> str:
        return ""  # vision refuses, as on act/1863

    async def _fake_window(_pdf: bytes, first: int, last: int, _dpi: int) -> dict[int, bytes]:
        return {n: b"\x89PNG" for n in range(first, last + 1)}

    monkeypatch.setattr(ocr_mod, "PdfReader", _Reader)
    monkeypatch.setattr(ocr_mod, "_layout_pass", _fake_layout)
    monkeypatch.setattr(ocr_mod, "_ocr_page_with_context", _fake_ocr)
    monkeypatch.setattr(ocr_mod, "_render_window", _fake_window)
    # Under the blank-leaf cutoff, which is where a sparse identity page sits.
    monkeypatch.setattr(ocr_mod, "ink_ratio", lambda _b: 0.001)

    pages = await extract_text_from_pdf(b"%PDF-fake", client=object())

    assert len(pages) == 1
    # The clean rival that proved the layer was damaged is what the page keeps.
    assert "Alamat : Dulang" in pages[0].text
    assert "A l a m a t" not in pages[0].text
    assert ocr_mod._page_body_for_combine(pages[0]) is not None


def _synth_pdf(n_pages: int) -> bytes:
    """Blank multi-page PDF, sufficient for chunk-boundary tests."""
    w = PdfWriter()
    for _ in range(n_pages):
        w.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_mistral_ocr_chunking_splits_over_30_pages_and_rewrites_indexes():
    """Azure AI Foundry's Mistral OCR endpoint rejects PDFs >30 pages with
    error 3730. The chunker must split the input into ≤25-page chunks and
    remap each chunk's local page ``index`` back to the document's absolute
    page number, so downstream reassembly is transparent to callers."""
    from pypdf import PdfReader

    from codify.pipeline.enrich.ocr import (
        MISTRAL_OCR_MAX_PAGES_PER_CALL,
        _mistral_ocr_pdf_chunked,
    )

    total_pages = 60
    pdf_bytes = _synth_pdf(total_pages)
    reader = PdfReader(io.BytesIO(pdf_bytes))

    calls: list[int] = []

    class StubClient:
        async def ocr(self, *, pdf: bytes, model: str) -> list[dict]:
            chunk_reader = PdfReader(io.BytesIO(pdf))
            n = len(chunk_reader.pages)
            assert n <= MISTRAL_OCR_MAX_PAGES_PER_CALL
            calls.append(n)
            return [{"index": i, "markdown": f"page {i}"} for i in range(n)]

    pages = await _mistral_ocr_pdf_chunked(
        client=StubClient(),
        pdf_bytes=pdf_bytes,
        reader=reader,
        total_pages=total_pages,
        model="mistral-ocr-4-0",
        on_progress=None,
    )

    assert len(calls) == 3
    assert sum(calls) == total_pages
    assert [p["index"] for p in pages] == list(range(total_pages))


@pytest.mark.asyncio
async def test_mistral_ocr_chunking_passthrough_when_under_limit():
    """A PDF at or below the per-call cap must go in one request, unmodified,
    so the fast path stays fast for the common case."""
    from pypdf import PdfReader

    from codify.pipeline.enrich.ocr import (
        MISTRAL_OCR_MAX_PAGES_PER_CALL,
        _mistral_ocr_pdf_chunked,
    )

    pdf_bytes = _synth_pdf(MISTRAL_OCR_MAX_PAGES_PER_CALL)
    reader = PdfReader(io.BytesIO(pdf_bytes))
    call_count = 0

    class StubClient:
        async def ocr(self, *, pdf: bytes, model: str) -> list[dict]:
            nonlocal call_count
            call_count += 1
            assert pdf == pdf_bytes  # not re-encoded
            return [{"index": i, "markdown": ""} for i in range(MISTRAL_OCR_MAX_PAGES_PER_CALL)]

    pages = await _mistral_ocr_pdf_chunked(
        client=StubClient(),
        pdf_bytes=pdf_bytes,
        reader=reader,
        total_pages=MISTRAL_OCR_MAX_PAGES_PER_CALL,
        model="mistral-ocr-4-0",
        on_progress=None,
    )
    assert call_count == 1
    assert len(pages) == MISTRAL_OCR_MAX_PAGES_PER_CALL


class TestMarkdownFurniture:
    """OCR markdown furniture is stripped before anchoring, and both
    combiners stay byte-identical so page spans keep working."""

    def test_strip_heading_prefixes_and_latex(self) -> None:
        from codify.pipeline.enrich.ocr import _strip_markdown_furniture

        out = _strip_markdown_furniture(
            "## قرار مجلس الوزراء رقم (٣٩)\nعنوان$^{١}$\n###\nنص عادي.\n"
        )
        assert "#" not in out and "$^{" not in out
        assert "قرار مجلس الوزراء" in out and "نص عادي." in out

    def test_combiners_identical_with_markdown_pages(self) -> None:
        from codify.pipeline.enrich.ocr import PageResult, combine_page_texts
        from codify.repair.grounding import combine_with_spans

        pages = [
            PageResult(
                page_number=1, text="## عنوان\nنص الصفحة الأولى الكامل هنا.", method="mistral_ocr"
            ),
            PageResult(
                page_number=2, text="نص الصفحة الثانية الكامل هنا أيضاً.", method="mistral_ocr"
            ),
        ]
        combined = combine_page_texts(pages)
        with_spans, spans = combine_with_spans(pages)
        assert combined == with_spans
        assert [s.page for s in spans] == [1, 2]

    def test_low_text_pages_flagged(self) -> None:
        from codify.pipeline.enrich.ocr import PageResult, low_text_pages

        pages = [
            PageResult(
                page_number=1,
                text="نص طويل بما يكفي لتجاوز حد الفحص الأدنى بوضوح.",
                method="mistral_ocr",
            ),
            PageResult(page_number=2, text="", method="mistral_ocr"),
            PageResult(page_number=3, text="قصير", method="mistral_ocr"),
        ]
        assert low_text_pages(pages) == [2, 3]


class TestStructuredOcrResponse:
    """Headers, footers and typed blocks stay out of markdown without being discarded."""

    @staticmethod
    def _pages(page: dict) -> list:
        """Drive the Mistral fast path over one stubbed page dict."""

        class StubClient:
            async def ocr(self, *, pdf: bytes, model: str) -> list[dict]:
                return [page]

        return asyncio.run(
            extract_text_from_pdf(_synth_pdf(1), client=StubClient(), ocr_model="mistral-ocr-4-0")
        )

    @staticmethod
    def _stub_page_route(monkeypatch) -> None:
        """Stand in for the per-page vision route. It renders pages through
        poppler, which CI does not install, and these tests are about which
        route is chosen rather than what that route produces."""
        import codify.pipeline.enrich.ocr as ocr_mod

        async def _fake(*args: object, **kwargs: object) -> str:
            return "المادة البديلة - نص من مسار الرؤية لكل صفحة."

        async def _fake_window(_pdf: bytes, first: int, last: int, _dpi: int) -> dict[int, bytes]:
            return {n: b"\x89PNG" for n in range(first, last + 1)}

        monkeypatch.setattr(ocr_mod, "_ocr_page_with_context", _fake)
        # Rendering moved out of the page helper into the fan-out, so stubbing
        # the helper alone now leaves poppler on the path. CI has no poppler.
        monkeypatch.setattr(ocr_mod, "_render_window", _fake_window)

    def test_furniture_and_blocks_land_on_the_page_result(self) -> None:
        blocks = [{"type": "header", "content": "مصنف"}, {"type": "title", "content": "قانون"}]
        body = "المادة ١ - نص المادة الأولى الكامل هنا بوضوح تام."
        pages = self._pages(
            {
                "index": 0,
                "markdown": body,
                "header": "منظومة القضاء والتشريع في فلسطين",
                "footer": "الوقائع الفلسطينية العدد ١٣٨٠",
                "blocks": blocks,
            }
        )
        # The line the defect lived on: `text` is the markdown and nothing else.
        # Concatenating either furniture field back on is what put a masthead at
        # anchor zero, and it passes every other assertion in this class.
        assert pages[0].text == body
        assert pages[0].header == "منظومة القضاء والتشريع في فلسطين"
        assert pages[0].footer == "الوقائع الفلسطينية العدد ١٣٨٠"
        assert pages[0].layout is not None
        assert [b.type for b in pages[0].layout.blocks] == ["header", "title"]
        assert pages[0].layout.blocks[0].content == "مصنف"

    def test_a_nul_byte_in_furniture_cannot_reach_the_artifact(self) -> None:
        """Header and footer go into a jsonb column, which rejects NUL. The body
        is scrubbed by `clean_page_text`; these bypass it."""
        pages = self._pages(
            {
                "index": 0,
                "markdown": "المادة ١ - نص المادة الأولى الكامل هنا بوضوح تام.",
                "header": "ترويسة\x00الجريدة",
                "footer": "\x07تذييل",
            }
        )
        assert pages[0].header == "ترويسةالجريدة"
        assert pages[0].footer == "تذييل"

    def test_a_page_without_them_is_absent_safe(self) -> None:
        """A page dict missing the keys defaults to empty strings, not None."""
        pages = self._pages({"index": 0, "markdown": "نص الصفحة الكامل هنا بوضوح."})
        assert pages[0].header == "" and pages[0].footer == ""
        # Layout is present because Mistral read the page; it is simply empty.
        assert pages[0].layout is not None
        assert pages[0].layout.blocks == []

    def test_furniture_never_re_enters_the_body_text(self) -> None:
        """`combine_page_texts` feeds the anchor scanner; furniture must not re-enter it."""
        from codify.pipeline.enrich.ocr import PageResult, combine_page_texts

        body = "المادة ١ - نص المادة الأولى الكامل هنا بوضوح تام."
        bare = PageResult(page_number=1, text=body, method="mistral_ocr")
        dressed = PageResult(
            page_number=1,
            text=body,
            method="mistral_ocr",
            header="منظومة القضاء والتشريع",
            footer="العدد ١٣٨٠",
            layout=ocr.PageLayout(engine="mistral_ocr"),
        )
        assert combine_page_texts([dressed]) == combine_page_texts([bare])

    def test_combiners_stay_identical_when_furniture_is_present(self) -> None:
        """The repair loop's page spans are built by a second combiner that must
        reproduce the first byte-for-byte."""
        from codify.pipeline.enrich.ocr import PageResult, combine_page_texts
        from codify.repair.grounding import combine_with_spans

        pages = [
            PageResult(
                page_number=1,
                text="المادة ١ - نص المادة الأولى الكامل هنا بوضوح تام.",
                method="mistral_ocr",
                header="ترويسة",
                footer="تذييل",
            ),
            PageResult(
                page_number=2,
                text="المادة ٢ - نص المادة الثانية الكامل هنا أيضاً.",
                method="mistral_ocr",
                header="ترويسة",
            ),
        ]
        combined, spans = combine_with_spans(pages)
        assert combined == combine_page_texts(pages)
        assert [s.page for s in spans] == [1, 2]

    def test_an_all_furniture_page_drops_out_of_the_span_table(self) -> None:
        """Routine now that furniture no longer pads a masthead-only page over the
        length floor. The combiners must still agree and the spans stay contiguous."""
        from codify.pipeline.enrich.ocr import PageResult, combine_page_texts
        from codify.repair.grounding import combine_with_spans

        pages = [
            PageResult(page_number=1, text="المادة ١ - نص كامل هنا.", method="mistral_ocr"),
            PageResult(page_number=2, text="", method="mistral_ocr", header="ترويسة"),
            PageResult(page_number=3, text="المادة ٢ - نص كامل هنا.", method="mistral_ocr"),
        ]
        combined, spans = combine_with_spans(pages)
        assert combined == combine_page_texts(pages)
        assert [s.page for s in spans] == [1, 3]

    def test_a_page_that_is_all_furniture_is_reported(self) -> None:
        """Empty markdown with populated furniture: the classifier took the whole page."""
        from structlog.testing import capture_logs

        with capture_logs() as captured:
            pages = self._pages({"index": 0, "markdown": "", "header": "ترويسة الجريدة"})
        assert pages[0].text == ""
        events = [c for c in captured if c["event"] == "ocr_page_all_furniture"]
        assert len(events) == 1 and events[0]["pages"] == [1]

    def test_furniture_survives_the_multi_chunk_merge(self) -> None:
        """A PS gazette scan runs past the 25-page cap, so it is served by several
        calls whose pages are merged and re-indexed. A `_one` that rebuilt the page
        dicts instead of mutating them would drop furniture for exactly this class
        of document and pass every other test here."""

        class StubClient:
            async def ocr(self, *, pdf: bytes, model: str) -> list[dict]:
                from pypdf import PdfReader

                n = len(PdfReader(io.BytesIO(pdf)).pages)
                return [
                    {
                        "index": i,
                        "markdown": f"المادة {i} - نص كامل هنا بوضوح تام.",
                        "header": f"ترويسة {i}",
                        "footer": f"تذييل {i}",
                        "blocks": [{"type": "text", "content": str(i)}],
                    }
                    for i in range(n)
                ]

        pages = asyncio.run(
            extract_text_from_pdf(_synth_pdf(60), client=StubClient(), ocr_model="mistral-ocr-4-0")
        )
        assert [p.page_number for p in pages] == list(range(1, 61))
        # Page 30 is chunk-local index 4 of the second call; furniture and index
        # rewrite have to land on the same page or the merge is silently lossy.
        assert pages[29].header == "ترويسة 4"
        assert pages[59].footer == "تذييل 9"
        assert pages[59].layout is not None
        assert [b.content for b in pages[59].layout.blocks] == ["9"]

    def test_an_incomplete_read_falls_through_instead_of_committing_a_short_law(
        self, monkeypatch
    ) -> None:
        """Returning the partial set would put a law in the corpus shorter than its
        source, with a log line as the only trace. The per-page route reads every
        page, so an incomplete fast path is treated as a failed one."""
        self._stub_page_route(monkeypatch)
        from structlog.testing import capture_logs

        class StubClient:
            async def ocr(self, *, pdf: bytes, model: str) -> list[dict]:
                return [
                    {"index": 0, "markdown": "المادة ١ - نص كامل هنا بوضوح تام."},
                    {"index": 2, "markdown": "المادة ٣ - نص كامل هنا بوضوح تام."},
                ]

        with capture_logs() as captured:
            pages = asyncio.run(
                extract_text_from_pdf(
                    _synth_pdf(3), client=StubClient(), ocr_model="mistral-ocr-4-0"
                )
            )
        events = [c for c in captured if c["event"] == "mistral_ocr_page_set_mismatch"]
        assert len(events) == 1
        assert events[0]["log_level"] == "error"
        assert events[0]["expected"] == 3 and events[0]["returned"] == 2
        assert events[0]["missing"] == [2]
        # Fell through, so every page is present and none claims the OCR engine
        # that did not read it.
        assert [p.page_number for p in pages] == [1, 2, 3]
        assert not any(p.method == "mistral_ocr" for p in pages)

    def test_a_duplicate_index_is_caught_even_though_the_count_matches(self, monkeypatch) -> None:
        """The failure a count comparison cannot see: page 2 twice and page 3 gone
        totals three either way, and would commit page 2's text twice."""
        self._stub_page_route(monkeypatch)
        from structlog.testing import capture_logs

        class StubClient:
            async def ocr(self, *, pdf: bytes, model: str) -> list[dict]:
                return [
                    {"index": 0, "markdown": "المادة ١ - نص كامل هنا بوضوح تام."},
                    {"index": 1, "markdown": "المادة ٢ - نص كامل هنا بوضوح تام."},
                    {"index": 1, "markdown": "المادة ٢ - نص كامل هنا بوضوح تام."},
                ]

        with capture_logs() as captured:
            asyncio.run(
                extract_text_from_pdf(
                    _synth_pdf(3), client=StubClient(), ocr_model="mistral-ocr-4-0"
                )
            )
        events = [c for c in captured if c["event"] == "mistral_ocr_page_set_mismatch"]
        assert len(events) == 1
        assert events[0]["duplicated"] == [2] and events[0]["missing"] == [3]

    def test_a_complete_read_reports_no_mismatch(self) -> None:
        from structlog.testing import capture_logs

        class StubClient:
            async def ocr(self, *, pdf: bytes, model: str) -> list[dict]:
                return [{"index": i, "markdown": f"المادة {i} - نص كامل هنا."} for i in range(3)]

        with capture_logs() as captured:
            pages = asyncio.run(
                extract_text_from_pdf(
                    _synth_pdf(3), client=StubClient(), ocr_model="mistral-ocr-4-0"
                )
            )
        assert not [c for c in captured if c["event"] == "mistral_ocr_page_set_mismatch"]
        assert all(p.method == "mistral_ocr" for p in pages)

    def test_a_page_with_no_index_is_named_not_dropped_in_silence(self, monkeypatch) -> None:
        """An unplaceable page used to raise, which the caller caught and turned
        into a full re-read on the other engine: loud and lossless. Dropping it
        quietly instead would lose a page of law with no trace."""
        self._stub_page_route(monkeypatch)
        from structlog.testing import capture_logs

        class StubClient:
            async def ocr(self, *, pdf: bytes, model: str) -> list[dict]:
                return [
                    {"index": 0, "markdown": "المادة ١ - نص كامل هنا بوضوح تام."},
                    {"index": None, "markdown": "نص بلا رقم صفحة."},
                ]

        with capture_logs() as captured:
            pages = asyncio.run(
                extract_text_from_pdf(
                    _synth_pdf(2), client=StubClient(), ocr_model="mistral-ocr-4-0"
                )
            )
        assert [c["event"] for c in captured].count("ocr_page_missing_index") == 1
        # And it is not merely logged: the page set is short, so the document is
        # re-read rather than committed with a page missing.
        assert [p.page_number for p in pages] == [1, 2]
        assert not any(p.method == "mistral_ocr" for p in pages)

    def test_many_all_furniture_pages_produce_one_event_listing_them_all(self) -> None:
        """One event per document: a 900-page gazette must not emit 900 warnings."""
        from structlog.testing import capture_logs

        class StubClient:
            async def ocr(self, *, pdf: bytes, model: str) -> list[dict]:
                return [
                    {"index": 0, "markdown": "المادة ١ - نص كامل هنا بوضوح تام."},
                    {"index": 1, "markdown": "", "header": "ترويسة الجريدة"},
                    {"index": 2, "markdown": "", "footer": "الوقائع الفلسطينية ١٣٨٠"},
                ]

        with capture_logs() as captured:
            asyncio.run(
                extract_text_from_pdf(
                    _synth_pdf(3), client=StubClient(), ocr_model="mistral-ocr-4-0"
                )
            )
        events = [c for c in captured if c["event"] == "ocr_page_all_furniture"]
        assert len(events) == 1
        assert events[0]["pages"] == [2, 3]

    def test_the_bbox_is_four_ordinates_not_a_list(self) -> None:
        """The endpoint returns `top_left_x` and friends. An earlier fixture
        asserted a `bbox` list, which no response has ever contained."""
        block = ocr.OcrBlock.model_validate(
            {
                "type": "text",
                "content": "المادة",
                "top_left_x": 60,
                "top_left_y": 3,
                "bottom_right_x": 670,
                "bottom_right_y": 92,
            }
        )

        assert (block.top_left_x, block.bottom_right_y) == (60, 92)
        assert not hasattr(block, "bbox")

    def test_layout_is_absent_rather_than_empty_when_nobody_looked(self) -> None:
        """A vision page separates no furniture. None says so; an empty layout
        would read as a page that genuinely carries none."""
        assert PageResult(page_number=1, text="x", method="vision_ocr").layout is None

    def test_blocks_are_excluded_from_every_serialisation(self) -> None:
        """In-process only. A future caller that dumps a PageResult must not carry
        a bounding box per paragraph into an artifact row or a trace payload."""
        page = PageResult(
            page_number=1,
            text="نص",
            method="mistral_ocr",
            header="ترويسة",
            layout=ocr.PageLayout(
                engine="mistral_ocr",
                blocks=[ocr.OcrBlock(type="header", content="مصنف")],
            ),
        )
        assert page.layout is not None
        assert "layout" not in page.model_dump()
        assert "layout" not in page.model_dump_json()
        # Targeted, not a blanket exclusion.
        assert "header" in page.model_dump()

    def test_a_null_index_in_a_chunk_does_not_sink_the_document(self) -> None:
        """int(None) raises, `extract_text_from_pdf` swallows it, and the document
        silently downgrades to a different OCR engine while the run reports success."""

        class StubClient:
            async def ocr(self, *, pdf: bytes, model: str) -> list[dict]:
                from pypdf import PdfReader

                n = len(PdfReader(io.BytesIO(pdf)).pages)
                out: list[dict] = [{"index": None, "markdown": "junk"}]
                out += [{"index": i, "markdown": f"المادة {i} - نص كامل."} for i in range(n)]
                return out

        pages = asyncio.run(
            extract_text_from_pdf(_synth_pdf(60), client=StubClient(), ocr_model="mistral-ocr-4-0")
        )
        assert [p.page_number for p in pages] == list(range(1, 61))
        assert all(p.method == "mistral_ocr" for p in pages)

    def test_whitespace_only_furniture_is_not_furniture(self) -> None:
        """The strip is load-bearing twice over: it gates this warning and the
        artifact's truthiness filter."""
        from structlog.testing import capture_logs

        with capture_logs() as captured:
            pages = self._pages({"index": 0, "markdown": "", "header": "  \n "})
        assert pages[0].header == ""
        assert not [c for c in captured if c["event"] == "ocr_page_all_furniture"]

    def test_a_page_with_body_text_is_not_reported(self) -> None:
        from structlog.testing import capture_logs

        with capture_logs() as captured:
            self._pages(
                {
                    "index": 0,
                    "markdown": "المادة ١ - نص المادة الأولى الكامل هنا بوضوح تام.",
                    "header": "ترويسة الجريدة",
                }
            )
        assert not [c for c in captured if c["event"] == "ocr_page_all_furniture"]


class TestVisionFanOut:
    """The vision route handles degraded scans across multi-page documents."""

    @staticmethod
    def _run(monkeypatch, *, pages: int, fail_on: set[int] | None = None) -> list:
        """Drive the vision route with rendering stubbed. Poppler is absent on
        CI, and these tests are about fan-out control flow, not rasterisation."""
        import codify.pipeline.enrich.ocr as ocr_mod

        renders: list[tuple[int, int]] = []

        async def _fake_window(_pdf: bytes, first: int, last: int, _dpi: int) -> dict[int, bytes]:
            renders.append((first, last))
            return {n: b"\x89PNG" for n in range(first, last + 1)}

        async def _fake_page(*args: object, **kwargs: object) -> str:
            page_num = int(args[1])  # type: ignore[call-overload]
            if fail_on and page_num in fail_on:
                raise RuntimeError(f"page {page_num} will not read")
            return f"المادة {page_num} - نص كامل من مسار الرؤية لهذه الصفحة."

        monkeypatch.setattr(ocr_mod, "_render_window", _fake_window)
        monkeypatch.setattr(ocr_mod, "_ocr_page_with_context", _fake_page)

        class StubClient:
            model = "gemini-vision"

        results = asyncio.run(extract_text_from_pdf(_synth_pdf(pages), client=StubClient()))
        return [results, renders]

    def test_a_sparse_divert_set_does_not_render_the_whole_span(self) -> None:
        """Chunking by count let pages 1, 50 and 900 render as one 900-page
        span: an OOM on a 16 GB host, to read three pages. Windows are bounded
        by page distance instead."""
        windows = ocr._render_windows([1, 2, 3, 50, 900])

        assert windows == [[1, 2, 3], [50], [900]]
        assert all(w[-1] - w[0] < ocr.VISION_RENDER_WINDOW for w in windows)
        # Contiguous input is where a count-based grouping still looks right:
        # measuring the span from the window's last page instead of its first
        # yields one 20-page render and passes every other assertion here.
        contiguous = ocr._render_windows(list(range(1, 21)))
        assert all(w[-1] - w[0] < ocr.VISION_RENDER_WINDOW for w in contiguous)

    def test_a_page_that_cannot_be_rendered_fails_rather_than_reads_empty(self) -> None:
        """An unrenderable page returning "" is a successful empty page:
        `combine_page_texts` filters it and the corpus loses it in silence."""

        async def _no_image() -> str:
            return await ocr._ocr_page_with_context(
                b"%PDF-1.4 not a real pdf",
                1,
                1,
                None,  # type: ignore[arg-type]
                "",
                None,
                300,
                image_bytes=None,
                render_if_missing=False,
            )

        with pytest.raises(RuntimeError, match="could not be rendered"):
            asyncio.run(_no_image())

    def test_a_page_that_will_not_read_still_fails_the_run(self, monkeypatch) -> None:
        """Per-page degradation must remain visible: without a
        visible trace it stamps a wholly-degraded document as verbatim digital
        text and drops empty pages out of the combined output unremarked."""
        with pytest.raises(RuntimeError):
            self._run(monkeypatch, pages=4, fail_on={2})

    def test_the_pdf_is_parsed_per_window_not_per_page(self, monkeypatch) -> None:
        """`convert_from_bytes` was called once per page with the whole
        document, so a 900-page file parsed itself 900 times."""
        _, renders = self._run(monkeypatch, pages=20)

        # Absolute, not derived from VISION_RENDER_WINDOW: an expectation
        # computed from the constant under test passes at a window of 1, which
        # is the per-page parse this exists to prevent.
        assert len(renders) < 20, "rendering is still one parse per page"
        assert renders[0][1] > renders[0][0], "a window must span more than one page"
        assert sum(last - first + 1 for first, last in renders) == 20

    def test_the_vision_budget_is_shared_across_documents(self, monkeypatch) -> None:
        """The semaphore was built inside the call, so N concurrent documents
        each got the full budget: at `ingest_queue_concurrency` 25 the real
        ceiling was 25 x MAX_CONCURRENT against one provider quota."""
        monkeypatch.setattr(ocr, "MAX_CONCURRENT", 2)
        monkeypatch.setattr(ocr, "VISION_RENDER_WINDOW", 8)
        ocr._VISION_GATES.clear()
        live = 0
        peak = 0

        async def _fake_window(_pdf, first, last, _dpi):
            return {n: b"\x89PNG" for n in range(first, last + 1)}

        async def _fake_page(*args, **kwargs) -> str:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1
            return "المادة ١ - نص كامل من مسار الرؤية لهذه الصفحة."

        monkeypatch.setattr(ocr, "_render_window", _fake_window)
        monkeypatch.setattr(ocr, "_ocr_page_with_context", _fake_page)

        class StubClient:
            model = "gemini-vision"

        async def _two_documents() -> None:
            await asyncio.gather(
                extract_text_from_pdf(_synth_pdf(6), client=StubClient()),
                extract_text_from_pdf(_synth_pdf(6), client=StubClient()),
            )

        asyncio.run(_two_documents())

        assert peak <= 2, f"two documents ran {peak} vision calls against a budget of 2"


class TestImagePlaceholders:
    """A figure is a region, not a sentence of the law."""

    def test_an_image_reference_never_reaches_the_text(self) -> None:
        """Observed first-line on a 1938 gazette, directly above the law title,
        which is where the masthead defect used to sit."""
        page = "![img-0.jpeg](img-0.jpeg)\n\nالملحق رقم ١\n\n# قانون تمدين الزيت"

        cleaned = ocr._clean_ocr_output(page)

        assert "img-0.jpeg" not in cleaned
        assert cleaned.startswith("الملحق رقم ١")

    def test_the_surrounding_text_is_untouched(self) -> None:
        body = "المادة ١ تسري أحكام هذا القانون على الأعمال التجارية والتجار."

        assert ocr._clean_ocr_output(body) == body

    def test_a_page_that_is_only_a_figure_reads_empty(self) -> None:
        assert ocr._clean_ocr_output("![img-0.jpeg](img-0.jpeg)") == ""


class TestAnchoredPrompt:
    """The vision prompt carries the page's own text layer."""

    @staticmethod
    def _prompt_for(page_text_layer: str) -> str:
        """Run one page through the real prompt builder, capturing what it sent."""
        seen: dict[str, str] = {}

        class StubClient:
            model = "gemini-vision"

            async def vision(self, *, prompt: str, images: list, system: str, model=None) -> str:
                seen["prompt"] = prompt
                return "المادة ١ - نص كامل."

        asyncio.run(
            ocr._ocr_page_with_context(
                b"",
                3,
                9,
                StubClient(),  # type: ignore[arg-type]
                "Document: قانون",
                None,
                300,
                image_bytes=b"\x89PNG",
                render_if_missing=False,
                page_text_layer=page_text_layer,
            )
        )
        return seen["prompt"]

    def test_the_pages_own_text_reaches_the_prompt(self) -> None:
        layer = "المادة ١ - غرامة لا تقل عن ٥٠ دينارا"
        prompt = self._prompt_for(layer)

        assert layer in prompt
        # An anchor presented as the answer invites the model to echo the
        # garbled layer back.
        assert "may be partial or garbled" in prompt
        assert "This is page 3 of 9." in prompt

    def test_a_page_with_no_text_layer_still_gets_a_clean_prompt(self) -> None:
        """A pure scan extracts nothing. The anchor block must vanish whole
        rather than leave its heading over an empty body."""
        prompt = self._prompt_for("")

        assert "Text layer extracted" not in prompt
        assert prompt.endswith("Extract all text from this page.")

    def test_whitespace_only_text_counts_as_no_text_layer(self) -> None:
        assert "Text layer extracted" not in self._prompt_for("   \n\t  ")

    def test_a_long_anchor_is_truncated(self) -> None:
        """A dense page would otherwise crowd the image out of the context."""
        prompt = self._prompt_for("ب" * (ocr.ANCHOR_CHAR_LIMIT * 3))

        assert prompt.count("ب") == ocr.ANCHOR_CHAR_LIMIT

    def test_the_diverted_pages_own_text_is_what_gets_threaded(self, monkeypatch) -> None:
        """End to end: the text the divert gate rejected is the text the prompt
        receives, for that page and not another."""
        from pypdf._page import PageObject

        import codify.pipeline.enrich.ocr as ocr_mod

        # Short enough to divert (`too_short`), and per-page so a swap between
        # pages fails rather than passes.
        layers = {0: "المادة ١ غرامة", 1: "المادة ٢ عقوبة"}
        counter = {"n": -1}

        def _fake_extract(self, *args: object, **kwargs: object) -> str:
            counter["n"] += 1
            return layers[counter["n"] % 2]

        anchors: dict[int, str] = {}

        async def _fake_window(_pdf: bytes, first: int, last: int, _dpi: int) -> dict[int, bytes]:
            return {n: b"\x89PNG" for n in range(first, last + 1)}

        async def _fake_page(*args: object, **kwargs: object) -> str:
            anchors[int(args[1])] = str(kwargs["page_text_layer"])  # type: ignore[call-overload]
            return "المادة ١ - نص كامل من مسار الرؤية لهذه الصفحة."

        monkeypatch.setattr(PageObject, "extract_text", _fake_extract)
        monkeypatch.setattr(ocr_mod, "_render_window", _fake_window)
        monkeypatch.setattr(ocr_mod, "_ocr_page_with_context", _fake_page)

        class StubClient:
            model = "gemini-vision"

        asyncio.run(extract_text_from_pdf(_synth_pdf(2), client=StubClient()))

        assert anchors == {1: layers[0], 2: layers[1]}


class TestEngineResolution:
    """Which engine reads a page is a code decision with config as an override.

    A line in `demo.sky.yaml` outlived the measurement that retired it, and
    nothing in the codebase said so."""

    def test_no_configuration_means_the_measured_default(self) -> None:
        assert ocr.resolve_ocr_model("") is None
        assert ocr.resolve_ocr_model(None) is None

    def test_an_override_is_honoured(self) -> None:
        assert ocr.resolve_ocr_model("mistral-ocr-4-0") == "mistral-ocr-4-0"

    def test_an_override_announces_itself(self, capsys) -> None:
        """Silence is the failure being fixed. A deviation from a measured
        default has to be visible without reading deploy config."""
        ocr.resolve_ocr_model("mistral-ocr-4-0")

        logged = capsys.readouterr().out
        assert "ocr_engine_overridden" in logged
        assert "mistral-ocr-4-0" in logged

    def test_the_default_stays_quiet(self, capsys) -> None:
        ocr.resolve_ocr_model("")

        assert "ocr_engine_overridden" not in capsys.readouterr().out


class TestComposedRead:
    """Layout from the engine that separates it, text from the engine that reads
    it best, and each recorded as its own."""

    @staticmethod
    def _client(*, ocr_pages=None, ocr_raises=False):
        class StubClient:
            model = "gemini-vision"

            async def ocr(self, *, pdf: bytes, model: str) -> list[dict]:
                if ocr_raises:
                    raise RuntimeError("foundry down")
                return list(ocr_pages or [])

        return StubClient()

    @staticmethod
    def _page(index: int, markdown: str = "نص من محرك التخطيط") -> dict:
        return {
            "index": index,
            "markdown": markdown,
            "header": "منظومة القضاء",
            "footer": "الوقائع الفلسطينية",
            "blocks": [
                {
                    "type": "header",
                    "content": "منظومة القضاء",
                    "top_left_x": 0,
                    "top_left_y": 3,
                    "bottom_right_x": 670,
                    "bottom_right_y": 40,
                }
            ],
            "dimensions": {"dpi": 87, "width": 720, "height": 1018},
            "confidence_scores": {
                "average_page_confidence_score": 0.93,
                "minimum_page_confidence_score": 0.26,
                "word_confidence_scores": [{"text": "نص", "confidence": 0.99, "start_index": 0}],
            },
        }

    def _run(self, monkeypatch, client, pages: int = 2):
        import codify.pipeline.enrich.ocr as ocr_mod

        async def _fake_window(_pdf, first, last, _dpi):
            return {n: b"\x89PNG" for n in range(first, last + 1)}

        async def _fake_page(*args, **kwargs):
            return "المادة ١ - نص كامل من مسار الرؤية لهذه الصفحة."

        monkeypatch.setattr(ocr_mod, "_render_window", _fake_window)
        monkeypatch.setattr(ocr_mod, "_ocr_page_with_context", _fake_page)
        return asyncio.run(extract_text_from_pdf(_synth_pdf(pages), client=client))

    def test_layout_comes_from_one_engine_and_text_from_the_other(self, monkeypatch) -> None:
        client = self._client(ocr_pages=[self._page(0), self._page(1)])

        results = self._run(monkeypatch, client)

        assert [r.method for r in results] == ["vision_ocr", "vision_ocr"]
        assert all(r.text.startswith("المادة ١") for r in results)
        assert all(r.layout is not None for r in results)
        assert results[0].layout.engine == "mistral_ocr"
        assert results[0].layout.min_word_confidence == 0.26
        assert results[0].layout.dimensions.dpi == 87

    def test_a_failed_layout_pass_does_not_fail_the_run(self, monkeypatch) -> None:
        """Transcription is load-bearing, layout is not. The distinction is the
        module's existing rule, applied to a second engine."""
        results = self._run(monkeypatch, self._client(ocr_raises=True))

        assert [r.text.startswith("المادة ١") for r in results] == [True, True]
        assert all(r.layout is None for r in results)

    def test_the_rival_read_is_kept_and_measured_but_never_authoritative(self, monkeypatch) -> None:
        """Relative, not a magic number: the same page read two ways scores far
        apart when the engines disagree and zero when they agree."""
        vision = "المادة ١ - نص كامل من مسار الرؤية لهذه الصفحة."

        disagree = self._run(
            monkeypatch, self._client(ocr_pages=[self._page(0, "لا شيء مشترك")]), pages=1
        )[0]
        agree = self._run(monkeypatch, self._client(ocr_pages=[self._page(0, vision)]), pages=1)[0]

        assert disagree.text == agree.text == vision, "vision stays authoritative either way"
        assert agree.rival_text == vision
        assert agree.divergence == 0.0
        assert disagree.divergence is not None
        assert disagree.divergence > 0.8

    def test_a_page_the_layout_engine_missed_has_no_layout(self, monkeypatch) -> None:
        """Absence is per page, not per document: one short read must not strip
        layout from the pages that did come back."""
        results = self._run(monkeypatch, self._client(ocr_pages=[self._page(0)]), pages=2)

        assert results[0].layout is not None
        assert results[1].layout is None
        assert results[1].text.startswith("المادة ١")


def test_a_refused_page_is_unreadable_whatever_its_ink():
    """A content-filter refusal says the page had text to refuse, so a low ink
    reading does not turn it into a blank leaf."""
    from codify.pipeline.enrich.ocr import unreadable_pages

    pages = [
        PageResult(page_number=1, text="First page.", method="text_extraction"),
        PageResult(
            page_number=2,
            text="",
            method="vision_ocr",
            divert_reason="too_short",
            ink=0.004,
            finish_reason="content_filter: RECITATION",
        ),
        PageResult(
            page_number=3, text="", method="vision_ocr", divert_reason="too_short", ink=0.004
        ),
        PageResult(page_number=4, text="", method="vision_ocr", divert_reason="too_short"),
    ]
    assert unreadable_pages(pages) == {2: "content_filter: RECITATION", 4: "empty_read"}
    assert combine_page_texts(pages) == (
        "First page.\n\n⟦page 2 unreadable⟧\n\n⟦page 4 unreadable⟧"
    )


def test_a_page_the_rival_recovered_is_not_unreadable():
    from codify.pipeline.enrich.ocr import unreadable_pages

    page = PageResult(
        page_number=1,
        text="from the rival",
        method="mistral_ocr",
        finish_reason="content_filter: RECITATION",
    )
    assert unreadable_pages([page]) == {}


@pytest.mark.asyncio
async def test_extraction_keeps_the_vision_finish_reason(monkeypatch) -> None:
    """End to end through the fan-out: the reason the model gave rides the page."""
    import codify.pipeline.enrich.ocr as ocr_mod
    from codify.core.llm import VisionText
    from codify.pipeline.enrich.ocr import unreadable_pages

    class _Page:
        def extract_text(self) -> str:
            return ""

    class _Reader:
        def __init__(self, *_a: object, **_k: object) -> None:
            self.pages = [_Page()]

    async def _no_layout(*_a: object, **_k: object) -> tuple[dict, dict]:
        return {}, {}

    async def _refused(*_a: object, **_k: object) -> str:
        return VisionText("", "content_filter: RECITATION")

    async def _fake_window(_pdf: bytes, first: int, last: int, _dpi: int) -> dict[int, bytes]:
        return {n: b"\x89PNG" for n in range(first, last + 1)}

    monkeypatch.setattr(ocr_mod, "PdfReader", _Reader)
    monkeypatch.setattr(ocr_mod, "_layout_pass", _no_layout)
    monkeypatch.setattr(ocr_mod, "_ocr_page_with_context", _refused)
    monkeypatch.setattr(ocr_mod, "_render_window", _fake_window)
    monkeypatch.setattr(ocr_mod, "ink_ratio", lambda _b: 0.3)

    pages = await extract_text_from_pdf(b"%PDF-fake", client=object())

    assert [p.finish_reason for p in pages] == ["content_filter: RECITATION"]
    assert unreadable_pages(pages) == {1: "content_filter: RECITATION"}


@pytest.mark.asyncio
async def test_the_page_reader_passes_the_finish_reason_through_cleaning() -> None:
    from codify.core.llm import VisionText
    from codify.pipeline.enrich import ocr

    class _Client:
        async def vision(self, **_k: object) -> str:
            return VisionText("## Heading\nBody", "stop")

    out = await ocr._ocr_page_with_context(
        b"%PDF",
        1,
        1,
        _Client(),
        "",
        image_bytes=b"\x89PNG",
        render_if_missing=False,  # type: ignore[arg-type]
    )
    assert (getattr(out, "finish_reason", None), "Body" in out) == ("stop", True)
