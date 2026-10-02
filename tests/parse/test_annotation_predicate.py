"""One answer to whether an `<authorialNote>` is annotation or part of the sentence."""

from __future__ import annotations

from lxml import etree

from codify.akn import parse_akn
from codify.akn.vocabulary import is_lifted_note, provision_text

NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"


def _p(inner: str) -> etree._Element:
    return etree.fromstring(f'<p xmlns="{NS}">{inner}</p>'.encode())


class TestPredicate:
    def test_a_lifted_note_is_annotation(self) -> None:
        el = _p('<authorialNote placement="bottom"><p>Amended.</p></authorialNote>')[0]
        assert is_lifted_note(el)

    def test_a_note_without_placement_belongs_to_the_sentence(self) -> None:
        """The FORMEX converter emits non-FOOTNOTE notes with no placement, mid-prose."""
        el = _p("<authorialNote><p>see OJ L 1</p></authorialNote>")[0]
        assert not is_lifted_note(el)

    def test_the_tag_alone_is_not_the_discriminator(self) -> None:
        assert not is_lifted_note(_p("<ref>x</ref>")[0])


class TestProvisionText:
    def test_a_lifted_note_is_left_out_with_its_tail_kept(self) -> None:
        el = _p('Operative.<authorialNote placement="bottom"><p>Amended.</p></authorialNote> Tail.')
        assert provision_text(el) == "Operative. Tail."

    def test_an_inline_note_stays_in_its_sentence(self) -> None:
        el = _p("Recital (<authorialNote><p>see OJ L 1</p></authorialNote>) applies.")
        assert provision_text(el) == "Recital (see OJ L 1) applies."

    def test_exclusion_reaches_any_depth(self) -> None:
        """`_inner_text` recursed and `anchors.provision_text` did not, so the two
        disagreed about a note one level down."""
        el = _p('Head.<span><authorialNote placement="bottom"><p>N.</p></authorialNote></span>End.')
        assert provision_text(el) == "Head.End."


_META = """<meta><identification source="#src"><FRBRWork>
 <FRBRthis value="/akn/xz/act/2005/1"/><FRBRuri value="/akn/xz/act/2005/1"/>
 <FRBRdate date="2005-01-01" name="enacted"/></FRBRWork>
 <FRBRExpression><FRBRthis value="/akn/xz/act/2005/1/ara@2005-01-01"/>
 <FRBRuri value="/akn/xz/act/2005/1/ara@2005-01-01"/>
 <FRBRdate date="2005-01-01" name="validFrom"/><FRBRlanguage language="ara"/>
 </FRBRExpression></identification></meta>"""


def _doc(body: str) -> str:
    return f'<akomaNtoso xmlns="{NS}"><act name="act">{_META}<body>{body}</body></act></akomaNtoso>'


def _leaves(nodes: list) -> list:
    out = []
    for n in nodes:
        out.append(n)
        out.extend(_leaves(getattr(n, "children", None) or []))
    return out


class TestParser:
    """`provisions.text` feeds embeddings, the lens, the comparator and MCP, so a
    footnote folded in here is read as operative law by four consumers."""

    def _only(self, body: str):
        doc = parse_akn(_doc(body))
        found = [n for n in _leaves(doc.body) if (getattr(n, "text", "") or "").strip()]
        assert len(found) == 1, [n.akn_eid for n in found]
        return found[0]

    def test_a_lifted_note_does_not_reach_provision_text(self) -> None:
        el = self._only(
            '<article eId="art_1"><num>1</num><content>'
            '<p eId="art_1__p_1">Illicit gain means any asset.'
            '<authorialNote eId="fn_1" placement="bottom"><p>Amended by '
            '<ref href="/akn/xz/act/2010/7">Law 7 of 2010</ref>.</p></authorialNote>'
            "</p></content></article>"
        )
        assert "Amended by" not in el.text
        assert "Illicit gain means any asset." in el.text

    def test_a_reference_inside_a_lifted_note_is_not_the_provisions_own(self) -> None:
        """Offsets are computed against the merged string, so a citation in a
        footnote arrived as a cross-reference of the provision it annotates."""
        el = self._only(
            '<article eId="art_1"><num>1</num><content>'
            '<p eId="art_1__p_1">Operative.'
            '<authorialNote eId="fn_1" placement="bottom"><p>See '
            '<ref href="/akn/xz/act/2010/7">Law 7</ref>.</p></authorialNote>'
            "</p></content></article>"
        )
        assert list(el.references or []) == []

    def test_an_inline_note_is_still_provision_text(self) -> None:
        el = self._only(
            '<article eId="art_2"><num>2</num><content>'
            '<p eId="art_2__p_1">Recital (<authorialNote><p>see OJ L 1</p></authorialNote>)'
            " applies.</p></content></article>"
        )
        assert "see OJ L 1" in el.text

    def test_intro_and_wrap_up_get_the_same_treatment(self) -> None:
        """`_extract_plain` was a bare `itertext()`, so container prose folded the
        note in even where `content` did not."""
        doc = parse_akn(
            _doc(
                '<article eId="art_3"><num>3</num>'
                '<intro><p>Lead-in.<authorialNote placement="bottom"><p>NOTE-IN</p>'
                "</authorialNote></p></intro>"
                '<paragraph eId="art_3__para_1"><num>(1)</num><content><p>Body.</p>'
                "</content></paragraph>"
                '<wrapUp><p>Tail.<authorialNote placement="bottom"><p>NOTE-UP</p>'
                "</authorialNote></p></wrapUp>"
                "</article>"
            )
        )
        article = doc.body[0]
        assert "NOTE-IN" not in (article.intro or "")
        assert "NOTE-UP" not in (article.wrap_up or "")
        assert "Lead-in." in (article.intro or "")
