"""Which pages the ingest path trusts, and which it sends to OCR.

The corruption that motivated this is a PDF whose glyph mapping does not
round-trip: the page renders correctly, carries a text layer, and extracts as
Arabic-looking text with the interior letters wrong. One civil code
arrived that way and 182 of its 219 pages were accepted, because the particle
test that had spotted it was ANDed with a definite-article test the corruption
leaves untouched.
"""

from __future__ import annotations

from codify.pipeline.enrich.ocr import (
    _TEXT_LAYER_MAX_DIVERGENCE,
    _divert_decision,
    _layer_split_words,
    _looks_garbled,
    _text_layer_untrusted,
    _vision_anchor,
)
from codify.quality.page_read import divergence

# Legal prose: particles dense, articles dense.
CLEAN = (
    "تتولى سلطة الموانئ الأطلسية الإشراف على سجل الأدوات الميكانيكية، وتضع "
    "القواعد التي تحدد شروط القيد فيه ومدة سريانه. وعلى كل من يرغب في القيد أن "
    "يقدم طلبا إلى الدائرة المختصة مرفقا به البيانات التي تطلبها اللائحة، "
    "وللدائرة أن ترفض الطلب متى تبين لها أن الأداة لا تستوفي المواصفات المقررة."
)

# The same page after the glyph mapping fails: interior letters doubled or
# swapped, so the particles stop being words while "ال" survives untouched.
DAMAGED = (
    "تتوليى سيلطة الميوانئ الأطلسية الإشيراف عليى سيجل الأدوات الميكانيكية، وتضيع "
    "القواعيد التيي تحيدد شيروط القييد فييه وميدة سيريانه. وعليى كيل مين ييرغب "
    "فيي القييد أني يقيدم طلبيا إليى الدائيرة المختصية مرفقيا بيه البيانيات التيي "
    "تطلبهيا اللائحية، وللدائيرة أني تيرفض الطليب متيى تبيين لهيا أني الأدايه ليا "
    "تسيتوفي المواصفيات المقيررة."
)


def test_divert_decision_accepts_clean_prose() -> None:
    """Long clean text with a trusted layer and no split words stays as text."""
    assert _divert_decision(CLEAN, layer_diverges=False, split_words=set()) == (False, "")


# A born-digital layer mis-extracted as Arabic Presentation Forms (U+FB50+): long
# enough to clear too_short, no base-block Arabic so not garbled.
PRESENTATION_FORMS = "ﭐ" * 60


def test_divert_decision_labels_each_reason() -> None:
    """One evaluation drives both the divert and its label, in this order."""
    assert _divert_decision("Art 5", layer_diverges=False, split_words=set()) == (True, "too_short")
    # DAMAGED is long enough to clear too_short, so garbled is what fires.
    assert _divert_decision(DAMAGED, layer_diverges=False, split_words=set()) == (True, "garbled")
    assert _divert_decision(PRESENTATION_FORMS, layer_diverges=False, split_words=set()) == (
        True,
        "presentation_forms",
    )
    assert _divert_decision(CLEAN, layer_diverges=True, split_words=set()) == (
        True,
        "text_layer_divergent",
    )
    assert _divert_decision(CLEAN, layer_diverges=False, split_words={"foo"}) == (
        True,
        "letter_spaced",
    )


def test_divert_decision_precedence_is_first_predicate_wins() -> None:
    """When several predicates fire, the earliest in the chain names the reason."""
    # too_short outranks everything, even with a divergent layer and split words.
    assert _divert_decision("hi", layer_diverges=True, split_words={"x"}) == (True, "too_short")
    # garbled outranks the later layer/letter-spacing arms.
    assert _divert_decision(DAMAGED, layer_diverges=True, split_words={"x"}) == (True, "garbled")
    # presentation_forms outranks a divergent layer.
    assert _divert_decision(PRESENTATION_FORMS, layer_diverges=True, split_words=set()) == (
        True,
        "presentation_forms",
    )
    # a divergent layer outranks letter-spacing.
    assert _divert_decision(CLEAN, layer_diverges=True, split_words={"x"}) == (
        True,
        "text_layer_divergent",
    )


def test_clean_arabic_prose_is_trusted() -> None:
    assert _looks_garbled(CLEAN) is False


def test_a_page_whose_glyph_mapping_failed_goes_to_ocr() -> None:
    """The regression. Under the old conjunction this page was accepted: its
    definite articles are intact, so the particle collapse never got to vote."""
    assert _looks_garbled(DAMAGED) is True


def test_the_definite_article_survives_the_corruption() -> None:
    """Why the conjunction was the bug rather than the threshold. If "ال" had
    collapsed too, the old predicate would have caught this and there would be
    nothing to fix."""
    words = DAMAGED.split()
    al_prefixed = sum(1 for w in words if w.startswith("ال")) / len(words)
    assert al_prefixed > 0.15


def test_non_arabic_text_is_left_to_the_normal_path() -> None:
    """The gate reads one script. English extracts cleanly or not at all, and a
    density test tuned for Arabic particles would flag it either way."""
    assert _looks_garbled("The Interpretation Act 1945 applies to every " * 12) is False


def test_a_clean_list_page_keeps_its_text_layer() -> None:
    """The gate over-corrected once already. An index or a fee schedule has no
    particles either, and re-OCRing one trades an exact extraction for a vision
    transcription of a table of law numbers."""
    index = "\n".join(f"قانون رقم ({i}) لسنة 2010 بشأن تنظيم المهن" for i in range(1, 16))
    assert _looks_garbled(index) is False


def test_the_same_garble_is_caught_however_the_extract_is_spaced() -> None:
    """Arabic density used to be measured against total length including
    whitespace, so a generously leaded two-column extract of the same corrupt
    page fell under the floor and was trusted."""
    assert _looks_garbled(DAMAGED) is True
    assert _looks_garbled(DAMAGED.replace(" ", "   \n  ")) is True


# The divergence backstop: a text layer that clears the single-read garble check
# but disagrees sharply with an independent engine's read is untrusted and sent
# to vision OCR. This catches corpora where injected
# same-script glyphs slipped the garble test but a clean vision rival exposed them.


def test_no_rival_never_faults_the_layer() -> None:
    """None divergence means there was no second read to compare against. Absence
    of a rival is not evidence against the layer, so it stays trusted."""
    assert _text_layer_untrusted(None, _TEXT_LAYER_MAX_DIVERGENCE) is False


def test_a_layer_that_agrees_with_the_rival_stays_trusted() -> None:
    assert _text_layer_untrusted(0.2, _TEXT_LAYER_MAX_DIVERGENCE) is False


def test_a_layer_that_disagrees_beyond_the_threshold_is_untrusted() -> None:
    assert _text_layer_untrusted(0.8, _TEXT_LAYER_MAX_DIVERGENCE) is True


def test_a_corrupt_layer_diverges_from_a_clean_rival_past_the_threshold() -> None:
    """End to end through the real divergence metric: the damaged extract and a
    clean read of the same page share almost no tokens, so the divert fires even
    when the corruption is same-script and subtle enough to pass the garble test."""
    div = divergence(DAMAGED, CLEAN)
    assert div is not None and div > _TEXT_LAYER_MAX_DIVERGENCE
    assert _text_layer_untrusted(div, _TEXT_LAYER_MAX_DIVERGENCE) is True


def test_a_divergent_layer_does_not_anchor_its_own_vision_reread() -> None:
    """The corruption that got the page diverted must not be fed back to the model
    as a hint. A page pulled for divergence anchors on nothing; the fresh read is
    the image alone."""
    assert _vision_anchor("text_layer_divergent", DAMAGED) == ""


def test_a_blank_or_garbled_divert_still_anchors_on_its_raw_text() -> None:
    """The other diverts are not distrusting the text's content, only its
    sufficiency, so the little text there was still helps disambiguate."""
    assert _vision_anchor("too_short", "Art 5") == "Art 5"
    assert _vision_anchor("garbled", "x") == "x"


def test_a_letter_spaced_layer_does_not_anchor_its_own_vision_reread() -> None:
    """Same reason as divergence: the split words are the defect, so feeding them
    back as a hint invites the model to reproduce them."""
    assert _vision_anchor("letter_spaced", _SPACED_LAYER) == ""


# Word splitting: a born-digital layer that encodes a tracked-out label as literal
# spaces. Invisible to divergence, which scores whole-page token overlap: on the
# document that motivated this the affected pages measured 0.02 to 0.44, none of
# them near the 0.5 divert threshold, because the damage is a few words on a page
# of hundreds. Only a rival that keeps the word whole faults the layer.

_SPACED_LAYER = "N a m a : Budi Santoso\nA l a m a t : Dulang, RT. 001/RW. 000, Desa Madandan"
_CLEAN_RIVAL = "Nama : Budi Santoso\nAlamat : Dulang, RT. 001/RW. 000, Desa Madandan"


def test_a_layer_that_splits_words_the_rival_keeps_whole_is_untrusted() -> None:
    assert _layer_split_words(_SPACED_LAYER, _CLEAN_RIVAL) == {"nama", "alamat"}


def test_the_divergence_score_cannot_see_this() -> None:
    """Why the predicate and not the threshold. The two reads share almost every
    token, so no threshold that keeps clean pages could ever pull this page."""
    div = divergence(_SPACED_LAYER, _CLEAN_RIVAL)
    assert div is not None and div < _TEXT_LAYER_MAX_DIVERGENCE
    assert _text_layer_untrusted(div, _TEXT_LAYER_MAX_DIVERGENCE) is False


def test_a_deliberately_spaced_heading_keeps_its_layer() -> None:
    """Typographic letter-spacing is the same shape as damage, and the corpus has
    both. The rival is an independent transcription that also reads the heading
    spaced, which is what PP 28/2011 page 26 does with `U M U M`."""
    layer = "K E T E N T U A N U M U M\nPasal 1 Dalam Peraturan ini yang dimaksud dengan:"
    rival = "## K E T E N T U A N U M U M\n\nPasal 1\n\nDalam Peraturan ini yang dimaksud dengan"
    assert _layer_split_words(layer, rival) == set()


def test_a_heading_the_rival_normalises_is_faulted() -> None:
    """The other half of the same case, named rather than left to be discovered:
    if the second engine ever stops reproducing tracked-out text, the layer does
    split a word the rival keeps whole and the page diverts. One page of exact
    extraction for a vision read is the accepted cost."""
    layer = "K E T E N T U A N U M U M\nPasal 1 Dalam Peraturan ini"
    assert _layer_split_words(layer, "KETENTUAN UMUM\nPasal 1 Dalam Peraturan ini") == {
        "ketentuanumum"
    }


def test_damage_is_not_cancelled_by_unrelated_noise_in_the_rival() -> None:
    """Why words and not counts. The layout pass renders a table row as spaced
    letters often enough that a count comparison hides real damage on the same
    page, which is precisely the front matter of a court judgment."""
    layer = "A l a m a t : Dulang\nNo Pasal Ayat"
    rival = "Alamat : Dulang\n| P a s a l | A y a t |"
    assert _layer_split_words(layer, rival) == {"alamat"}


def test_a_run_spelling_several_words_is_matched_whole() -> None:
    """`S e r t i f i k a t H a k M i l i k` is one run and three words."""
    layer = "bukti S e r t i f i k a t H a k M i l i k nomor 12"
    assert _layer_split_words(layer, "bukti Sertifikat Hak Milik nomor 12") == {
        "sertifikathakmilik"
    }


def test_a_long_run_starting_mid_word_still_matches() -> None:
    """The split does not always begin at a word boundary, and a run can outrun
    the join window. Past a length where a coincidental substring is not a real
    risk, the rival is matched with its spaces stripped as well."""
    layer = "n t u k a n A l i a n s i M a s y a r a k a t"
    assert _layer_split_words(layer, "membentukan Aliansi Masyarakat") == {
        "ntukanaliansimasyarakat"
    }


def test_a_rival_table_does_not_supply_a_word_across_its_cell_walls() -> None:
    """A layer whose column padding collapses to `A B C D` reads as one run. The
    rival renders the same four columns as cells, and joining across the walls
    would manufacture the word the layer is accused of splitting."""
    assert _layer_split_words("A B C D", "| A | B | C | D |") == set()


def test_a_short_run_does_not_match_inside_a_longer_rival_word() -> None:
    """Joined whole words, not the rival with its spaces stripped: otherwise
    `d a r i` matches inside `daripada` and faults a layer nothing is wrong with."""
    assert _layer_split_words("d a r i sumber lain", "daripada sumber lain") == set()


def test_a_rival_that_read_only_part_of_the_page_has_no_opinion() -> None:
    """A truncated layout read must not condemn body words it never saw."""
    layer = "N a m a : Budi\nA l a m a t : Dulang, RT 001, Desa Madandan"
    assert _layer_split_words(layer, "Nama : Budi") == {"nama"}


def test_no_rival_never_faults_the_layer_for_splitting() -> None:
    """Same rule as divergence: one read is not evidence against another read
    that does not exist."""
    assert _layer_split_words(_SPACED_LAYER, "") == set()
    assert _layer_split_words(_SPACED_LAYER, "   ") == set()


def test_a_clean_layer_is_not_faulted_by_a_split_rival() -> None:
    """Directional. A vision read that mangles a word must not cost the layer
    that read it correctly."""
    assert _layer_split_words(_CLEAN_RIVAL, _SPACED_LAYER) == set()


def test_the_predicate_reads_latin_only() -> None:
    """The run pattern is Latin, so an Arabic or Cyrillic layer that splits its
    letters is invisible here. Recorded so a zero on those corpora is not read as
    a measurement rather than as the check never having run."""
    assert _layer_split_words("ا ل ق ا ن و ن", "القانون") == set()
