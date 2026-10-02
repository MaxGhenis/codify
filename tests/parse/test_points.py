# ruff: noqa: E501  # Arabic test strings render shorter visually than their byte length
"""Tests for the deterministic sub-point nester."""

from __future__ import annotations

from codify.pipeline.enrich.points import nest_enumerated_lines


def test_numbered_then_lettered_nests_and_restarts():
    out = nest_enumerated_lines(["(1) first", "(2) second", "(a) sub a", "(b) sub b", "(3) third"])
    assert out == [
        "POINT (1)",
        "  first",
        "POINT (2)",
        "  second",
        "  POINT (a)",
        "    sub a",
        "  POINT (b)",
        "    sub b",
        "POINT (3)",  # restarts to the parent (num) level, not nested under (b)
        "  third",
    ]


def test_lead_in_text_stays_above_points():
    out = nest_enumerated_lines(["The Authority shall:", "(1) act", "(2) report"])
    assert out[0] == "The Authority shall:"
    assert out[1] == "POINT (1)"


def test_arabic_indic_and_arabic_alpha():
    out = nest_enumerated_lines(["(١) الأول", "(٢) الثاني", "(أ) فرع", "(ب) فرع"])
    assert out[0] == "POINT (١)"
    assert "  POINT (أ)" in out  # arabic-alpha nests one level under the digits


def test_roman_nests_under_letter():
    out = nest_enumerated_lines(["(a) alpha", "(i) roman one", "(ii) roman two"])
    # (i)/(ii) open a deeper level under (a), not continue the letter sequence.
    assert "  POINT (i)" in out
    assert "  POINT (ii)" in out


def test_lone_enumerator_not_nested():
    lines = ["Some prose.", "(1) the only point"]
    assert nest_enumerated_lines(lines) == lines


def test_plain_prose_unchanged():
    lines = ["A plain paragraph.", "Another sentence."]
    assert nest_enumerated_lines(lines) == lines


def test_inline_points_on_one_line_are_split():
    out = nest_enumerated_lines(["The Authority shall (1) act (2) report (3) review."])
    assert out[0] == "The Authority shall"
    assert [ln for ln in out if ln.startswith("POINT")] == ["POINT (1)", "POINT (2)", "POINT (3)"]


def test_colon_introduced_inline_sublist_is_split():
    """A lettered lead-in and its first numbered item run together on one line
    ("أ. … التالية: ١. …") must split so the first sub-point isn't absorbed into
    the parent's body. The split is cross-family, gated on the colon cue."""
    out = nest_enumerated_lines(
        ["(a) the committee does the following: (1) first task", "(2) second task", "(b) other"]
    )
    points = [ln for ln in out if ln.lstrip().startswith("POINT")]
    assert "POINT (a)" in [p.strip() for p in points]
    assert "POINT (1)" in [p.strip() for p in points]  # first sub-point recovered
    assert "POINT (2)" in [p.strip() for p in points]
    assert "POINT (b)" in [p.strip() for p in points]


def test_inline_single_reference_not_split():
    # A lone parenthesised number in prose is a reference, not a list.
    lines = ["This is governed by paragraph (2) of the preceding article."]
    assert nest_enumerated_lines(lines) == lines


def test_abbreviation_not_treated_as_marker():
    lines = ["The company, Inc. shall comply, etc. as required by law."]
    assert nest_enumerated_lines(lines) == lines


def test_dot_paragraphs_with_bracketed_subpoints_nest_correctly():
    """`1.` paragraphs followed by `1) 2) 3)` points must nest the points under
    the paragraphs, not flatten to one sibling level. UA zakon dominant failure
    mode (W10h), bracket-form and dot-form must be recognised as different
    levels even though both carry numeric tokens."""
    out = nest_enumerated_lines(
        [
            "1. First paragraph text.",
            "1) point one of paragraph 1",
            "2) point two of paragraph 1",
            "2. Second paragraph text.",
            "1) point one of paragraph 2",
        ]
    )
    # Paragraphs stay at level 0; points indent one level under them.
    paragraphs_at_root = [ln for ln in out if ln.startswith("POINT ") and not ln.startswith("  ")]
    nested = [ln for ln in out if ln.startswith("  POINT ")]
    assert paragraphs_at_root == ["POINT 1.", "POINT 2."]
    assert nested == ["  POINT 1)", "  POINT 2)", "  POINT 1)"]


def test_paren_both_and_paren_right_are_different_levels():
    """`(1)` (paren-both) and `1)` (paren-right) are different marker styles
    even though both carry numeric tokens. Treat them as different levels:
    a paren-both paragraph followed by paren-right points should nest."""
    out = nest_enumerated_lines(
        [
            "(1) paragraph one",
            "1) point one",
            "2) point two",
            "(2) paragraph two",
        ]
    )
    paragraphs = [ln for ln in out if ln.startswith("POINT (")]
    nested = [ln for ln in out if ln.startswith("  POINT ")]
    assert paragraphs == ["POINT (1)", "POINT (2)"]
    assert nested == ["  POINT 1)", "  POINT 2)"]


def test_arabic_paren_paragraphs_with_alpha_subpoints_nest():
    """`(١) (٢)` paragraphs (Arabic) with `(أ) (ب)` sub-points, should nest."""
    out = nest_enumerated_lines(
        ["(١) الفقرة الأولى", "(أ) فرع أول", "(ب) فرع ثاني", "(٢) الفقرة الثانية"]
    )
    top = [ln for ln in out if ln.startswith("POINT ")]
    nested = [ln for ln in out if ln.startswith("  POINT ")]
    assert top == ["POINT (١)", "POINT (٢)"]
    assert nested == ["  POINT (أ)", "  POINT (ب)"]


def test_uniform_paren_run_with_reset_opens_child():
    """When the same marker style runs forward then resets, treat the reset as a
    nested sub-list under the prior item, the dominant failure mode in old acts
    (uniform `(N)` for both outer and inner enumerations, no style cue to
    distinguish levels). Going from `(3)` back to `(1)` is invariably a child."""
    out = nest_enumerated_lines(
        ["(1) one", "(2) two", "(3) three", "(1) sub of three", "(2) more sub"]
    )
    top = [ln for ln in out if ln.startswith("POINT (")]
    nested = [ln for ln in out if ln.startswith("  POINT (")]
    assert top == ["POINT (1)", "POINT (2)", "POINT (3)"]
    assert nested == ["  POINT (1)", "  POINT (2)"]


def test_clean_ascending_run_stays_flat():
    """A monotonically ascending sequence must not false-nest. `(1)..(5)` stays
    at one level even though we never see a reset, POINT markers at depth 0."""
    out = nest_enumerated_lines([f"({i}) item {i}" for i in range(1, 6)])
    points_at_root = [ln for ln in out if ln.startswith("POINT (")]
    assert points_at_root == ["POINT (1)", "POINT (2)", "POINT (3)", "POINT (4)", "POINT (5)"]
    assert not any(ln.startswith("  POINT") for ln in out)


def test_letter_reset_opens_child():
    """Letter-style reset (a, b, a, b) nests the second run under the prior
    letter. Older acts often inline lettered sub-lists with the same shape
    as their parent enumeration."""
    out = nest_enumerated_lines(["(a) alpha", "(b) bravo", "(a) sub-of-b", "(b) more"])
    top = [ln for ln in out if ln.startswith("POINT (")]
    nested = [ln for ln in out if ln.startswith("  POINT (")]
    assert top == ["POINT (a)", "POINT (b)"]
    assert nested == ["  POINT (a)", "  POINT (b)"]


def test_alpha_run_stays_flat_through_the_roman_letters():
    """Seven letters double as roman numerals (c d i l m v x). A plain lettered
    list must stay flat across every one of them: `(a)..(z)` is one level, not a
    list that sprouts children at c, i, l, v and x. Anglophone acts routinely
    run paragraph lists well past (c)."""
    out = nest_enumerated_lines([f"({chr(c)}) item" for c in range(ord("a"), ord("z") + 1)])
    assert not any(ln.startswith("  POINT") for ln in out)
    assert sum(1 for ln in out if ln.startswith("POINT (")) == 26


def test_single_roman_letter_still_opens_a_child_when_it_breaks_the_alpha_run():
    """The other half of the same judgement: `(i)` straight after `(a)` cannot be
    continuing the letters (that would need (b)..(h) first), so it opens a roman
    child. This is what keeps genuine `(a)(i)(ii)` nesting working, and it is why
    the fix keys on alpha-run continuation rather than on the token alone."""
    out = nest_enumerated_lines(["(a) alpha", "(i) roman one", "(ii) roman two", "(b) bravo"])
    assert [ln for ln in out if ln.startswith("POINT (")] == ["POINT (a)", "POINT (b)"]
    assert [ln for ln in out if ln.startswith("  POINT (")] == ["  POINT (i)", "  POINT (ii)"]


def test_alpha_ordinals_survive_the_roman_letters():
    """`_token_order` used to read c/d/l/m as roman and return 0 for them, which
    silently zeroed an ordinary letter's ordinal and broke the reset check
    downstream. A reset after (d) must still open a child."""
    out = nest_enumerated_lines(["(a) a", "(b) b", "(c) c", "(d) d", "(a) reset", "(b) more"])
    assert [ln for ln in out if ln.startswith("POINT (")] == [
        "POINT (a)",
        "POINT (b)",
        "POINT (c)",
        "POINT (d)",
    ]
    assert [ln for ln in out if ln.startswith("  POINT (")] == ["  POINT (a)", "  POINT (b)"]


def test_ambiguous_letter_resumes_its_own_style_not_the_deepest_level():
    """The run an ambiguous letter belongs to is the one carrying its own marker
    style, which is not always the deepest open level. With `(a)(b)` at the root
    and a differently-styled `a)..d)` nested under it, the root `(c)` must resume
    the root run, not read the nested run's ordinal (4) and open a third level."""
    out = nest_enumerated_lines(
        ["(a) root", "(b) root", "a) sub", "b) sub", "c) sub", "d) sub", "(c) root", "(d) root"]
    )
    assert [ln for ln in out if ln.startswith("POINT (")] == [
        "POINT (a)",
        "POINT (b)",
        "POINT (c)",
        "POINT (d)",
    ]
    assert [ln for ln in out if ln.startswith("  POINT ")] == [
        "  POINT a)",
        "  POINT b)",
        "  POINT c)",
        "  POINT d)",
    ]
    assert not any(ln.startswith("    POINT") for ln in out)


def test_inline_lettered_run_splits_through_the_roman_letters():
    """`_split_inline` decides a glued run is a list by checking its ordinals
    ascend. While `_token_order` read c/d/l/m as roman it returned 0 for them, so
    a run reaching (c) never registered and the whole line stayed one blob. This
    is the inline half of the same ordinal fix."""
    out = nest_enumerated_lines(["(a) alpha (b) bravo (c) charlie (d) delta"])
    assert [ln for ln in out if ln.startswith("POINT (")] == [
        "POINT (a)",
        "POINT (b)",
        "POINT (c)",
        "POINT (d)",
    ]
    assert not any(ln.startswith("  POINT") for ln in out)


def test_parenthesised_acronyms_are_not_enumerators():
    """A definitions section introduces acronyms in parentheses, and the marker
    token class used to accept any one-to-four-letter run, so `(GIDA)` read as a
    list marker. That severed each definiendum from its definition, invented a
    point named after the acronym, and nested the next real item underneath it.
    Multi-letter parenthesised tokens are now roman numerals only."""
    out = nest_enumerated_lines(
        [
            "(c) Geographically Isolated and Disadvantaged Areas (GIDA) refer to remote areas;",
            "(d) Indigenous Peoples (IPs) refer to a group of people;",
        ]
    )
    assert [ln for ln in out if ln.startswith("POINT")] == ["POINT (c)", "POINT (d)"]
    # Each definition keeps its acronym inline rather than losing it to a new point.
    assert any("(GIDA) refer to remote areas" in ln for ln in out)
    assert any("(IPs) refer to a group of people" in ln for ln in out)


def test_acronyms_spelled_from_roman_letters_are_not_enumerators():
    """Narrowing the token class to a roman *character class* is not enough:
    plenty of acronyms are spelled entirely from i/v/x/l/c/d/m, so `(LLC)`,
    `(DCC)` and `(ID)` still matched. The class is an explicit list of the
    numerals `_token_order` actually knows, so anything else is prose."""
    # Two acronym-led lines must not become a list of their own.
    assert not [
        ln
        for ln in nest_enumerated_lines(
            ["(LLC) means a limited liability company;", "(DCC) means the coordination council;"]
        )
        if ln.startswith("POINT")
    ]
    # Nor may one attach itself to a genuine lettered run as a child.
    out = nest_enumerated_lines(["(a) first", "(b) second", "(ID) the identity document"])
    assert [ln for ln in out if "POINT" in ln] == ["POINT (a)", "POINT (b)"]


def test_roman_numerals_up_to_viii_still_parse_as_enumerators():
    """The counterpart to the above: narrowing the token class must not cost the
    multi-letter roman numerals it exists for."""
    out = nest_enumerated_lines(
        ["(a) alpha", "(i) one", "(ii) two", "(iii) three", "(viii) eight", "(b) bravo"]
    )
    assert [ln for ln in out if ln.startswith("POINT (")] == ["POINT (a)", "POINT (b)"]
    assert [ln for ln in out if ln.startswith("  POINT (")] == [
        "  POINT (i)",
        "  POINT (ii)",
        "  POINT (iii)",
        "  POINT (viii)",
    ]


def test_ascending_run_starting_above_one_stays_flat():
    """Sequences starting at 5 or some higher number (continuation of a larger
    document) must not false-nest. Only a backwards step in a non-trivial
    sequence (last>1) opens a child."""
    out = nest_enumerated_lines([f"({i}) item {i}" for i in range(5, 9)])
    assert not any(ln.startswith("  POINT") for ln in out)
    assert sum(1 for ln in out if ln.startswith("POINT (")) == 4


def test_arabic_letter_with_kashida_recognised():
    """`(هـ)` (Arabic ha + kashida) is the formal legal-text form of `(ه)`.
    Pre-fix `_MARK` only matched single-letter Arabic tokens [ء-ي] so the
    kashida-suffixed form silently failed, and the LLM-emitted (هـ) got
    swallowed into the preceding sibling's body. Now both forms are markers.
    This was the root cause of swallowed enumerators across a whole
    criminal code."""
    out = nest_enumerated_lines(
        [
            "The following acts are prohibited:",
            "(أ) caused harm (ب) littered the road (ج) caused damage (د) discarded refuse (هـ) threw on the road (و) used vulgar language",
        ]
    )
    top = [ln for ln in out if ln.startswith("POINT (")]
    assert "POINT (أ)" in top
    assert "POINT (ب)" in top
    assert "POINT (ج)" in top
    assert "POINT (د)" in top
    assert "POINT (هـ)" in top  # the previously-swallowed enumerator
    assert "POINT (و)" in top
    # 6 siblings, none nested under each other
    assert len(top) == 6


class TestInlineSplitGuards:
    """Cross-reference chains must not shred prose into phantom points."""

    def test_reference_chain_stays_one_line(self) -> None:
        # Competition 11/2025 art 10 shape: three reference nouns in a row.
        line = "يحظر مخالفة أحكام النقطتين (ي) من الفقرة (1) من المادة (5) من هذا القرار بقانون."
        assert nest_enumerated_lines([line]) == [line]

    def test_lead_in_with_stray_reference_intact(self) -> None:
        # AC 37/2018 lead-in shape: one reference marker mid-prose.
        line = "مع مراعاة أحكام الفقرة (2) من المادة السادسة يلتزم الموظف بما يلي:"
        assert nest_enumerated_lines([line]) == [line]

    def test_non_ascending_pair_stays_prose(self) -> None:
        line = "كما ورد في البند (4) وكذلك ما نصت عليه في (2) من ذات المادة."
        assert nest_enumerated_lines([line]) == [line]

    def test_genuine_glued_list_still_splits(self) -> None:
        line = "تختص اللجنة بما يلي: (1) دراسة الطلبات المقدمة. (2) إصدار التوصيات اللازمة."
        out = nest_enumerated_lines([line])
        assert any(ln.startswith("POINT (1)") or "POINT (1)" in ln for ln in out)
        assert any("POINT (2)" in ln for ln in out)

    def test_glued_arabic_letter_list_still_splits(self) -> None:
        line = "(أ) الاسم الكامل. (ب) مكان الإقامة. (ج) المهنة."
        out = nest_enumerated_lines([line])
        assert sum("POINT" in ln for ln in out) == 3


class TestInlineSplitTrades:
    """Deliberate behaviour boundaries of the ascending-chain requirement."""

    def test_abjad_run_splits_past_waw(self) -> None:
        # Abjad order (أ ب ج د ه و ز) is non-monotonic in codepoints at ز;
        # the chain check ranks through the abjad table, not Unicode.
        line = "(أ) الاسم. (ب) العنوان. (ج) المهنة. (د) العمر. (ه) الجنسية. (و) الحالة. (ز) الرقم."
        out = nest_enumerated_lines([line])
        assert sum("POINT" in ln for ln in out) == 7

    def test_degraded_list_with_misread_marker_stays_prose(self) -> None:
        # Known trade: a genuine glued list with one OCR-misread ordinal
        # ((1)..(7)..(3)) no longer splits; prose-preservation wins over
        # recovering a corrupted run.
        line = "(1) البند الأول. (7) البند الثاني. (3) البند الثالث."
        assert nest_enumerated_lines([line]) == [line]


class TestAmountAndCitationGuards:
    """Prose numerals with رقم precursors or currency/year followers
    are citations and amounts, never enumerators."""

    def test_law_number_citations_stay_whole(self) -> None:
        line = (
            "يستمر العمل بأحكام قانون رقم (6) لسنة 1999 بشأن العطاءات "
            "وقانون رقم (9) لسنة 1998 إلى حين إلغائهما."
        )
        assert nest_enumerated_lines([line]) == [line]

    def test_fine_amounts_stay_whole(self) -> None:
        line = "يعاقب بغرامة لا تقل عن (2000) دينار زرزوري ولا تزيد على (4000) دينار زرزوري."
        assert nest_enumerated_lines([line]) == [line]

    def test_genuine_list_still_splits(self) -> None:
        line = "تختص اللجنة بما يلي: (1) دراسة الطلبات المقدمة. (2) إصدار التوصيات اللازمة."
        out = nest_enumerated_lines([line])
        assert sum("POINT" in ln for ln in out) == 2


class TestAmountGuardEdges:
    def test_currency_with_attached_punctuation(self) -> None:
        line = "غرامة لا تقل عن (2000) دينار، ولا تزيد على (4000) دينار."
        assert nest_enumerated_lines([line]) == [line]

    def test_year_connector_without_raqm(self) -> None:
        line = "يلغى القانون (6) لسنة 1999 والقانون (9) لسنة 1998 المشار إليهما."
        assert nest_enumerated_lines([line]) == [line]

    def test_fine_range_with_gharama_precursor(self) -> None:
        line = "يعاقب بغرامة (1000) إلى (2000) وبالحبس مدة لا تزيد على سنة."
        assert nest_enumerated_lines([line]) == [line]


class TestDefinedTermLeadins:
    def test_definitions_after_nested_list_return_to_container_level(self) -> None:
        # A definitions article where one defined term carries a numbered
        # sub-list: the terms after it nested four levels deep under its last
        # point instead of returning to the container.
        lines = [
            "المسؤول المختص: تشمل عبارة المسؤول المختص:",
            "1. رئيس السلطة فيما يختص بإدارتها.",
            "2. مدير الميناء فيما يختص بمينائه.",
            "المشرف المختص: تشمل عبارة المشرف المختص:",
            "1. مشرف السجل فيما يختص بقيد الأدوات.",
            "2. مشرف الرسوم فيما يختص بتحصيلها.",
            "الشخص: الشخص الطبيعي أو الاعتباري.",
            "المورد: الشخص الذي يقوم بتوريد الأدوات.",
        ]
        out = nest_enumerated_lines(lines)
        assert "الشخص: الشخص الطبيعي أو الاعتباري." in out
        assert "المورد: الشخص الذي يقوم بتوريد الأدوات." in out

    def test_plain_continuation_stays_inside_point(self) -> None:
        lines = [
            "1. البند الأول.",
            "2. البند الثاني.",
            "نص متمم للبند الثاني دون رأس تعريف.",
        ]
        out = nest_enumerated_lines(lines)
        assert out[-1] == "  نص متمم للبند الثاني دون رأس تعريف."


class TestLeadinContinuations:
    def test_note_label_stays_inside_point(self) -> None:
        from codify.pipeline.enrich.points import nest_enumerated_lines

        lines = [
            "1. البند الأول.",
            "2. البند الثاني.",
            "ملاحظة: يستثنى من ذلك ما ورد أعلاه.",
        ]
        out = nest_enumerated_lines(lines)
        # The note stays indented under the open point, not dedented to
        # container level as a new defined term.
        assert out[-1].startswith(" ")


class TestFormFieldsAreNotProvisions:
    """A Lampiran form numbers its blanks, so the enumerator counts fields."""

    def test_a_numbered_blank_stays_plain_text(self) -> None:
        out = nest_enumerated_lines(
            [
                "1. ......................................",
                "2. ......................................",
            ]
        )
        assert not any(line.strip().startswith("POINT") for line in out), out

    def test_a_table_row_of_pipes_stays_plain_text(self) -> None:
        out = nest_enumerated_lines(["1. | | | | | | |", "2. | | | | | | |"])
        assert not any(line.strip().startswith("POINT") for line in out), out

    def test_a_labelled_blank_stays_plain_text(self) -> None:
        out = nest_enumerated_lines(
            [
                "1. Nama : ...............................................",
                "2. Jabatan : ............................................",
            ]
        )
        assert not any(line.strip().startswith("POINT") for line in out), out

    def test_a_genuine_ayat_run_still_nests(self) -> None:
        out = nest_enumerated_lines(
            ["(1) Pengusaha wajib membayar upah.", "(2) Upah dibayarkan setiap bulan."]
        )
        assert [line for line in out if line.strip().startswith("POINT")] == [
            "POINT (1)",
            "POINT (2)",
        ], out

    def test_an_ellipsis_inside_a_sentence_is_not_a_blank(self) -> None:
        """`dan seterusnya ...` ends a real provision; only a long run is a field."""
        out = nest_enumerated_lines(
            ["(1) Ketentuan ini berlaku ... dan seterusnya.", "(2) Ayat kedua."]
        )
        assert len([line for line in out if line.strip().startswith("POINT")]) == 2, out

    def test_a_bare_marker_takes_its_body_from_below(self) -> None:
        """An empty rest is a marker whose text follows, not a blank to fill."""
        out = nest_enumerated_lines(["(1)", "Pengusaha wajib membayar upah.", "(2)", "Ayat kedua."])
        assert len([line for line in out if line.strip().startswith("POINT")]) == 2, out

    def test_a_lone_punctuation_mark_is_not_a_blank(self) -> None:
        """`1. -` is a placeholder inside a genuine item, not a field to fill."""
        out = nest_enumerated_lines(["1. -", "2. ."])
        assert len([line for line in out if line.strip().startswith("POINT")]) == 2, out


def test_repeated_letter_definitions_remain_siblings():
    import string

    labels = list(string.ascii_lowercase)
    labels += [c * 2 for c in string.ascii_lowercase]
    labels += [c * 3 for c in string.ascii_lowercase[:15]]
    for left, right in [("", ")"), ("(", ")")]:
        out = nest_enumerated_lines([f"{left}{n}{right} Definition text." for n in labels])
        points = [line for line in out if line.lstrip().startswith("POINT ")]
        assert points == [f"POINT {left}{n}{right}" for n in labels]


def test_dotted_roman_items_nest_and_return_to_lettered_parent():
    numerals = "i ii iii iv v vi vii viii ix x".split()
    out = nest_enumerated_lines(["a) Parent.", *[f"{n}. Child." for n in numerals], "b) Next."])
    points = [line for line in out if line.lstrip().startswith("POINT ")]
    assert points == ["POINT a)", *[f"  POINT {n}." for n in numerals], "POINT b)"]
