"""The two consumers of the region classifier, on the shapes the cases produce."""

from __future__ import annotations

from lxml import etree

from codify.pipeline.enrich.conclusions import emit_conclusions
from codify.pipeline.enrich.notes import emit_authorial_notes
from codify.pipeline.enrich.regions import Region, RegionVocabulary

NS = {"akn": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"}
ARABIC_VOCAB = RegionVocabulary(closing_phrases=("صدر بمدينة",))


def _act(body: str) -> str:
    return (
        '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
        f'<act name="act"><meta/><body>{body}</body></act></akomaNtoso>'
    )


def _region(kind: str, text: str = "") -> dict[int, list[Region]]:
    """A note is corroborated against a region's own text, so it has to carry it."""
    return {
        1: [
            Region(
                kind=kind,
                page_number=1,
                block_index=0,
                block_type="references",
                text=text,
                top=0,
                bottom=1,
            )
        ]
    }


# The shape the arbitration regulation produces: attestation inside article 79.
ARTICLE_79 = _act(
    """<article eId="art_79"><num>79</num><content>
      <p eId="art_79__p_1">على جميع الجهات المختصة تنفيذ أحكام هذه اللائحة</p>
      <p eId="art_79__p_2">صدر بمدينة زرزورة بتاريخ : ١٢ / ٤ / ٢٠٠٤ ميلادية</p>
      <p eId="art_79__p_3">الموافق : ٢٢ / صفر / ١٤٢٥ هجرية</p>
      <p eId="art_79__p_4">سالم بن رشيد (أبو فارس)</p>
      <p eId="art_79__p_5">رئيس مجلس الوزراء</p>
    </content></article>"""
)


class TestConclusions:
    def test_the_final_article_is_left_with_only_its_own_paragraph(self) -> None:
        root = etree.fromstring(emit_conclusions(ARTICLE_79, vocab=ARABIC_VOCAB).encode())
        paragraphs = root.findall(".//akn:article/akn:content/akn:p", NS)
        assert [p.get("eId") for p in paragraphs] == ["art_79__p_1"]

    def test_the_attestation_becomes_conclusions_after_the_body(self) -> None:
        root = etree.fromstring(emit_conclusions(ARTICLE_79, vocab=ARABIC_VOCAB).encode())
        act = root.find("akn:act", NS)
        assert [etree.QName(el).localname for el in act] == [
            "meta",
            "body",
            "conclusions",
        ]

    def test_place_and_date_are_direct_paragraphs(self) -> None:
        """`<formula>` accepts only enactingFormula and promulgation as its name."""
        root = etree.fromstring(emit_conclusions(ARTICLE_79, vocab=ARABIC_VOCAB).encode())
        direct = root.findall(".//akn:conclusions/akn:p", NS)
        assert len(direct) == 2
        assert "صدر بمدينة" in "".join(direct[0].itertext())

    def test_the_signatory_is_a_block_container_not_a_signature_element(self) -> None:
        """`<signature>` is reserved for the inline name-block, per the EU lane."""
        out = emit_conclusions(ARTICLE_79, vocab=ARABIC_VOCAB)
        root = etree.fromstring(out.encode())
        block = root.find(".//akn:conclusions/akn:blockContainer", NS)
        assert block is not None and block.get("eId") == "sig_1"
        assert len(block.findall("akn:p", NS)) == 2
        assert "<signature" not in out

    def test_both_calendars_survive_the_move(self) -> None:
        text = "".join(
            etree.fromstring(emit_conclusions(ARTICLE_79, vocab=ARABIC_VOCAB).encode())
            .find(".//akn:conclusions", NS)
            .itertext()
        )
        assert "ميلادية" in text and "هجرية" in text

    def test_a_phrase_broken_across_a_line_still_lifts(self) -> None:
        split = ARTICLE_79.replace("صدر بمدينة", "صدر\nبمدينة")
        root = etree.fromstring(emit_conclusions(split, vocab=ARABIC_VOCAB).encode())
        paragraphs = root.findall(".//akn:article/akn:content/akn:p", NS)
        assert [p.get("eId") for p in paragraphs] == ["art_79__p_1"]

    def test_a_phrase_broken_by_a_blank_line_does_not_lift(self) -> None:
        split = ARTICLE_79.replace("صدر بمدينة", "صدر\n\nبمدينة")
        assert emit_conclusions(split, vocab=ARABIC_VOCAB) == split

    def test_a_document_with_no_closing_phrase_is_untouched(self) -> None:
        plain = _act('<article eId="art_1"><content><p>نص عادي</p></content></article>')
        assert emit_conclusions(plain, vocab=ARABIC_VOCAB) == plain

    def test_a_jurisdiction_declaring_no_phrase_is_untouched(self) -> None:
        assert emit_conclusions(ARTICLE_79, vocab=RegionVocabulary()) == ARTICLE_79

    def test_running_twice_does_not_produce_two_conclusions(self) -> None:
        once = emit_conclusions(ARTICLE_79, vocab=ARABIC_VOCAB)
        assert emit_conclusions(once, vocab=ARABIC_VOCAB) == once

    def test_a_closing_phrase_quoted_mid_document_is_left_as_prose(self) -> None:
        """Only the final container is considered; an earlier match is a quote."""
        quoted = _act(
            '<article eId="art_1"><content><p>ورد فيه صدر بمدينة زرزورة</p></content></article>'
            '<article eId="art_2"><content><p>نص ختامي</p></content></article>'
        )
        assert emit_conclusions(quoted, vocab=ARABIC_VOCAB) == quoted


_NOTE_ONE = "¹ عدلت بموجب المادة (٧) من القرار بقانون رقم (٧) لسنة ٢٠١٠"


class TestAuthorialNotes:
    # The shape the Illicit Gains Law produces: the footnote is a definition.
    ARTICLE_1 = _act(
        """<article eId="art_1"><num>1</num><intro>
          <p eId="art_1__intro__p_1">الكسب غير المشروع¹ هو كل مال حصل عليه</p>
          <p eId="art_1__intro__p_8">¹ عدلت بموجب المادة (٧) من القرار بقانون رقم (٧) لسنة ٢٠١٠</p>
        </intro></article>"""
    )

    def test_the_footnote_leaves_the_definitions(self) -> None:
        out = emit_authorial_notes(
            self.ARTICLE_1, vocab=RegionVocabulary(), regions=_region("footnote", _NOTE_ONE)
        )
        root = etree.fromstring(out.encode())
        assert [p.get("eId") for p in root.findall(".//akn:intro/akn:p", NS)] == [
            "art_1__intro__p_1"
        ]

    def test_it_binds_to_the_provision_carrying_the_same_marker(self) -> None:
        out = emit_authorial_notes(
            self.ARTICLE_1, vocab=RegionVocabulary(), regions=_region("footnote", _NOTE_ONE)
        )
        root = etree.fromstring(out.encode())
        note = root.find(".//akn:authorialNote", NS)
        assert note is not None
        assert note.getparent().get("eId") == "art_1__intro__p_1"
        assert note.get("placement") == "bottom"
        assert note.get("refersTo") is None

    def test_an_unlocatable_marker_binds_at_the_container_and_says_so(self) -> None:
        """Guessing a paragraph is worse than recording that it was inferred."""
        orphan = _act(
            """<article eId="art_2"><intro>
              <p eId="art_2__intro__p_1">نص المادة دون علامة</p>
              <p eId="art_2__intro__p_2">² أضيفت بموجب المادة (١٥)</p>
            </intro></article>"""
        )
        out = emit_authorial_notes(
            orphan,
            vocab=RegionVocabulary(),
            regions=_region("footnote", "² أضيفت بموجب المادة (١٥)"),
        )
        note = etree.fromstring(out.encode()).find(".//akn:authorialNote", NS)
        assert note is not None and note.get("refersTo") == "#inferred-binding"

    def test_the_older_parenthesised_marker_is_lifted_too(self) -> None:
        older = _act(
            """<article eId="art_3"><intro>
              <p eId="art_3__intro__p_1">نص المادة</p>
              <p eId="art_3__intro__p_2">(١) مفسوخ بموجب الأمر العسكري</p>
            </intro></article>"""
        )
        out = emit_authorial_notes(
            older,
            vocab=RegionVocabulary(),
            regions=_region("footnote", "(١) مفسوخ بموجب الأمر العسكري"),
        )
        assert etree.fromstring(out.encode()).find(".//akn:authorialNote", NS) is not None

    def test_without_a_region_a_mid_container_candidate_is_left_alone(self) -> None:
        """One signal is not a decision. A numbered list item is not a footnote."""
        listy = _act(
            """<article eId="art_4"><intro>
              <p eId="art_4__intro__p_1">(١) البند الأول</p>
              <p eId="art_4__intro__p_2">نص تال</p>
            </intro></article>"""
        )
        assert emit_authorial_notes(listy, vocab=RegionVocabulary(), regions=None) == listy

    def test_a_document_with_no_markers_is_untouched(self) -> None:
        plain = _act('<article eId="art_1"><content><p>نص عادي</p></content></article>')
        assert (
            emit_authorial_notes(plain, vocab=RegionVocabulary(), regions=_region("footnote", "x"))
            == plain
        )


_META = (
    '<meta><identification source="#codify"><FRBRWork>'
    '<FRBRthis value="/akn/xz/act/2004/41/main"/><FRBRuri value="/akn/xz/act/2004/41"/>'
    '<FRBRdate date="2004-04-12" name="Generation"/><FRBRauthor href="#codify"/>'
    '<FRBRcountry value="xz"/></FRBRWork><FRBRExpression>'
    '<FRBRthis value="/akn/xz/act/2004/41/ara@2004-04-12/main"/>'
    '<FRBRuri value="/akn/xz/act/2004/41/ara@2004-04-12"/>'
    '<FRBRdate date="2004-04-12" name="Generation"/><FRBRauthor href="#codify"/>'
    '<FRBRlanguage language="ara"/></FRBRExpression><FRBRManifestation>'
    '<FRBRthis value="/akn/xz/act/2004/41/ara@2004-04-12/main.xml"/>'
    '<FRBRuri value="/akn/xz/act/2004/41/ara@2004-04-12.xml"/>'
    '<FRBRdate date="2004-04-12" name="Generation"/><FRBRauthor href="#codify"/>'
    "</FRBRManifestation></identification></meta>"
)


def _valid_act(body: str) -> str:
    return (
        '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
        f'<act name="act">{_META}<body>{body}</body></act></akomaNtoso>'
    )


class TestTheSchemaGate:
    """A bare `<note>` fails the XSD; `validate_akn` raises rather than returns."""

    def test_conclusions_passes_strict_validation(self) -> None:
        from codify.akn._schema import validate_akn

        source = _valid_act(
            '<article eId="art_79"><num>79</num><content>'
            '<p eId="p1">tail of the article</p>'
            '<p eId="p2">صدر بمدينة زرزورة</p>'
            '<p eId="p3">second date line</p>'
            '<p eId="p4">signatory name</p>'
            '<p eId="p5">signatory role</p></content></article>'
        )
        out = emit_conclusions(source, vocab=ARABIC_VOCAB)
        # Assert the emission happened, or this only validates the fixture.
        assert "<conclusions" in out
        validate_akn(out, strict=True)

    def test_an_authorial_note_passes_strict_validation(self) -> None:
        from codify.akn._schema import validate_akn

        source = _valid_act(
            '<article eId="art_1"><num>1</num><intro>'
            '<p eId="q1">definition text¹ here</p>'
            '<p eId="q2">¹ amended by article 7</p></intro></article>'
        )
        out = emit_authorial_notes(
            source,
            vocab=RegionVocabulary(),
            regions=_region("footnote", "¹ amended by article 7"),
        )
        assert "<authorialNote" in out
        validate_akn(out, strict=True)


def test_a_marker_mid_sentence_is_not_a_footnote() -> None:
    """`notes.py` carries its own anchored pattern; unanchoring it would remove
    provision text from the document."""
    prose = _act(
        '<article eId="art_9"><content>'
        '<p eId="art_9__p_1">نص يذكر (١) داخل الجملة ولا يبدأ بها</p>'
        "</content></article>"
    )
    assert (
        emit_authorial_notes(prose, vocab=RegionVocabulary(), regions=_region("footnote")) == prose
    )


def test_an_article_emptied_by_the_lift_does_not_survive_as_a_bare_number() -> None:
    """An article whose whole content was attestation was never a provision. A
    numbered empty shell trips the export's empty-container gate."""
    only_attestation = _act(
        '<article eId="art_1"><num>1</num><content><p>نص عادي</p></content></article>'
        '<article eId="art_2"><num>2</num><content>'
        "<p>صدر بمدينة زرزورة</p></content></article>"
    )
    out = emit_conclusions(only_attestation, vocab=ARABIC_VOCAB)
    root = etree.fromstring(out.encode())
    assert [a.get("eId") for a in root.findall(".//akn:article", NS)] == ["art_1"]


class TestTextIsMovedNeverLost:
    """Every defect in this class is silent, so each gets its own test."""

    def test_an_ordinary_numbered_subsection_is_not_a_footnote(self) -> None:
        """`(1)` is the commonest opener in legislative prose. The marker alone
        would lift operative text out of the body."""
        act = _act(
            '<article eId="art_1"><num>1</num><content>'
            "<p>(1) The Minister shall issue regulations.</p>"
            "<p>(2) The Ministry shall publish them.</p>"
            "</content></article>"
        )
        regions = _region("footnote", "¹ عدلت بموجب المادة (٧) من القرار بقانون")
        assert emit_authorial_notes(act, vocab=RegionVocabulary(), regions=regions) == act

    def test_two_notes_in_one_container_both_survive(self) -> None:
        """The second was a pending candidate, so appending the first into it
        merged the two and destroyed one."""
        both = _act(
            '<article eId="art_1"><num>1</num><content>'
            "<p>Operative text of the article.</p>"
            "<p>¹ Amended by Law 5 of 1999.</p>"
            "<p>² Repealed by Law 9 of 2004.</p>"
            "</content></article>"
        )
        regions = {
            1: [
                Region(
                    kind="footnote",
                    page_number=1,
                    block_index=i,
                    block_type="references",
                    text=t,
                    top=900,
                    bottom=920,
                )
                for i, t in enumerate(
                    ["¹ Amended by Law 5 of 1999.", "² Repealed by Law 9 of 2004."]
                )
            ]
        }
        root = etree.fromstring(
            emit_authorial_notes(both, vocab=RegionVocabulary(), regions=regions).encode()
        )
        notes = root.findall(".//akn:authorialNote", NS)
        assert len(notes) == 2
        texts = ["".join(n.itertext()) for n in notes]
        assert "Amended by Law 5 of 1999." in texts[0]
        assert "Repealed by Law 9 of 2004." in texts[1]

    def test_a_citation_inside_a_note_survives_the_lift(self) -> None:
        """`emit_references` runs one pass earlier, and a footnote is where the
        amendment cross-references are."""
        cited = _act(
            '<article eId="art_1"><num>1</num><content>'
            "<p>Operative text.</p>"
            '<p>¹ Amended by <ref href="/akn/xz/act/2004/9">Law 9 of 2004</ref>.</p>'
            "</content></article>"
        )
        out = emit_authorial_notes(
            cited,
            vocab=RegionVocabulary(),
            regions=_region("footnote", "¹ Amended by Law 9 of 2004."),
        )
        note = etree.fromstring(out.encode()).find(".//akn:authorialNote", NS)
        assert note is not None
        assert note.find(".//akn:ref", NS) is not None

    def test_bare_tail_text_is_kept_when_the_wrapper_goes(self) -> None:
        """`remove()` discards the tail, which belongs to the container."""
        with_tail = (
            '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
            '<act name="act"><meta/><body>'
            '<article eId="art_1"><content><p>صدر بمدينة زرزورة</p></content>KEEP-ME</article>'
            "</body></act></akomaNtoso>"
        )
        assert "KEEP-ME" in emit_conclusions(with_tail, vocab=ARABIC_VOCAB)

    def test_a_provision_that_merely_quotes_the_phrase_keeps_its_text(self) -> None:
        """The final provision is exactly where the phrasing lives, so a match
        has to be followed only by attestation-shaped lines."""
        quoting = _act(
            '<article eId="art_40"><num>40</num><content>'
            "<p>The rule that a decree صدر بمدينة carries the force of law shall "
            "apply to every competent authority named in the preceding articles of "
            "this law and to any body that succeeds one of them.</p>"
            "</content></article>"
        )
        assert emit_conclusions(quoting, vocab=ARABIC_VOCAB) == quoting

    def test_a_scan_separator_rule_does_not_hide_the_marker(self) -> None:
        """The OCR prints the footnote rule as leading underscores, which an
        anchored marker match reads as ordinary text."""
        ruled = _act(
            '<article eId="art_1"><num>1</num><content>'
            "<p>Operative text with marker¹ here.</p>"
            "<p>____________________ ¹ Amended by Law 5 of 1999.</p>"
            "</content></article>"
        )
        out = emit_authorial_notes(
            ruled,
            vocab=RegionVocabulary(),
            regions=_region("footnote", "¹ Amended by Law 5 of 1999."),
        )
        note = etree.fromstring(out.encode()).find(".//akn:authorialNote", NS)
        assert note is not None
        assert "____" not in "".join(note.itertext())
        assert "Amended by Law 5 of 1999." in "".join(note.itertext())

    def test_a_note_with_nowhere_to_land_keeps_its_own_citation(self) -> None:
        """Building the note empties the paragraph, so it must not be built until
        a target is found; otherwise the skip path leaves a gutted paragraph."""
        alone = _act(
            '<article eId="art_1"><num>1</num><intro>'
            '<p>¹ Amended by <ref href="/akn/xz/act/1999/5">Law 5 of 1999</ref>.</p>'
            "</intro></article>"
        )
        out = emit_authorial_notes(
            alone,
            vocab=RegionVocabulary(),
            regions=_region("footnote", "¹ Amended by Law 5 of 1999."),
        )
        root = etree.fromstring(out.encode())
        assert root.find(".//akn:authorialNote", NS) is None
        assert root.find(".//akn:intro/akn:p/akn:ref", NS) is not None
        assert "Law 5 of 1999" in "".join(root.find(".//akn:intro", NS).itertext())

    def _signatory(self, body: str) -> list[str | None]:
        block = etree.fromstring(emit_conclusions(_act(body), vocab=ARABIC_VOCAB).encode()).find(
            ".//akn:blockContainer", NS
        )
        assert block is not None
        return [p.text for p in block]

    def test_a_second_calendar_line_does_not_displace_the_signatory(self) -> None:
        """An attestation can print both the Gregorian and the Hijri date, so the
        name and role are neither the first pair nor at a fixed offset."""
        assert self._signatory(
            '<article eId="art_79"><num>79</num><content>'
            "<p>صدر بمدينة Zerzura on 12 / 4 / 2004</p>"
            "<p>corresponding to 22 / Safar / 1425</p>"
            "<p>SIGNATORY-NAME</p><p>SIGNATORY-ROLE</p>"
            "</content></article>"
        ) == ["SIGNATORY-NAME", "SIGNATORY-ROLE"]

    def test_a_numbered_line_behind_the_signature_is_not_read_as_the_role(self) -> None:
        """A page number or gazette reference swept in behind the attestation
        would otherwise be grouped and tagged as the signatory's role."""
        assert self._signatory(
            '<article eId="art_79"><num>79</num><content>'
            "<p>صدر بمدينة Zerzura on 12 April 2004</p>"
            "<p>SIGNATORY-NAME</p><p>SIGNATORY-ROLE</p><p>issue 88 page 12</p>"
            "</content></article>"
        ) == ["SIGNATORY-NAME", "SIGNATORY-ROLE"]


class TestDownstreamKeepsNotesApart:
    """The lift is only worth doing if what reads the AKN afterwards agrees."""

    _ACT = (
        '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
        '<act name="act"><meta/><body>'
        '<article eId="art_1"><num>1</num><content>'
        '<p eId="art_1__p_1">Operative text.'
        '<authorialNote eId="fn_1" placement="bottom">'
        '<p>Amended by <ref href="/akn/xz/act/2010/7">Decree-Law 7 of 2010</ref>.</p>'
        "</authorialNote></p>"
        "</content></article>"
        "</body></act></akomaNtoso>"
    )

    def test_the_server_projection_gives_a_note_its_own_node(self) -> None:
        """Flattened into `InlineText`, the API-backed reader shows the footnote
        as a clause of the sentence it hangs off."""
        from codify.storage.documents import InlineNote, _walk_inline

        p = etree.fromstring(self._ACT.encode()).find(".//akn:p", NS)
        assert p is not None
        notes = [n for n in _walk_inline(p) if isinstance(n, InlineNote)]
        assert len(notes) == 1
        assert any(getattr(c, "href", None) == "/akn/xz/act/2010/7" for c in notes[0].children)

    def test_the_translator_is_not_handed_the_footnote(self) -> None:
        from codify.translate.anchors import provision_text

        p = etree.fromstring(self._ACT.encode()).find(".//akn:p", NS)
        assert p is not None
        assert provision_text(p).strip() == "Operative text."

    def test_writing_the_translated_line_does_not_delete_the_note(self) -> None:
        """`_set_p_text` clears children, and the note is one of them."""
        from codify.translate.write import _set_p_text

        p = etree.fromstring(self._ACT.encode()).find(".//akn:p", NS)
        assert p is not None
        _set_p_text(p, "Texte opératoire.")
        assert p.text == "Texte opératoire."
        note = p.find("akn:authorialNote", NS)
        assert note is not None
        assert note.find(".//akn:ref", NS) is not None
