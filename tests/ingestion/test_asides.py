"""A marginal article-subject label is moved onto the article it labels as an
authorialNote, never dragged out of provision text, and clears strict XSD."""

from __future__ import annotations

from codify.pipeline.enrich.asides import emit_marginal_notes
from codify.pipeline.enrich.regions import Region

_META = (
    '<meta><identification source="#codify"><FRBRWork>'
    '<FRBRthis value="/akn/xz/act/1985/24/main"/><FRBRuri value="/akn/xz/act/1985/24"/>'
    '<FRBRdate date="1985-01-01" name="Generation"/><FRBRauthor href="#codify"/>'
    '<FRBRcountry value="xz"/></FRBRWork><FRBRExpression>'
    '<FRBRthis value="/akn/xz/act/1985/24/ara@1985-01-01/main"/>'
    '<FRBRuri value="/akn/xz/act/1985/24/ara@1985-01-01"/>'
    '<FRBRdate date="1985-01-01" name="Generation"/><FRBRauthor href="#codify"/>'
    '<FRBRlanguage language="ara"/></FRBRExpression><FRBRManifestation>'
    '<FRBRthis value="/akn/xz/act/1985/24/ara@1985-01-01/main.xml"/>'
    '<FRBRuri value="/akn/xz/act/1985/24/ara@1985-01-01.xml"/>'
    '<FRBRdate date="1985-01-01" name="Generation"/><FRBRauthor href="#codify"/>'
    "</FRBRManifestation></identification></meta>"
)


def _act(body: str) -> str:
    return (
        '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
        f'<act name="act">{_META}<body>{body}</body></act></akomaNtoso>'
    )


def _aside(text: str) -> dict[int, list[Region]]:
    return {
        1: [
            Region(
                kind="aside",
                page_number=1,
                block_index=0,
                block_type="aside_text",
                text=text,
                top=0,
                bottom=1,
            )
        ]
    }


def test_a_standalone_label_moves_to_the_article_as_a_note() -> None:
    source = _act(
        '<article eId="art_1"><num>1</num><content>'
        "<p>تعاريف</p>"
        "<p>يقصد بالكلمات التالية المعاني المبينة أدناه ما لم تدل القرينة على خلاف ذلك.</p>"
        "</content></article>"
    )
    out = emit_marginal_notes(source, regions=_aside("تعاريف"))
    # placement="bottom" is what is_lifted_note keys on, so the label is annotation
    # everywhere, not provision text.
    assert "<authorialNote" in out and 'placement="bottom"' in out
    # The label is retained inside the note, and the provision text is untouched.
    assert "تعاريف" in out
    assert "يقصد بالكلمات التالية" in out
    # It is no longer a standalone provision paragraph: exactly one <p> now opens
    # the content (the provision), the label having moved into the note.
    assert out.count("<p>يقصد") == 1


def test_a_provision_paragraph_opening_with_the_label_is_not_moved() -> None:
    # The label runs straight into the definition, so overlap is far below the
    # floor: never move a provision paragraph, only a bare standalone label.
    body = (
        '<article eId="art_1"><num>1</num><content>'
        "<p>تعاريف يقصد بالكلمات التالية المعاني المبينة أدناه ما لم تدل القرينة على خلاف ذلك.</p>"
        "</content></article>"
    )
    out = emit_marginal_notes(_act(body), regions=_aside("تعاريف"))
    assert "<authorialNote" not in out
    assert "تعاريف يقصد بالكلمات التالية" in out


def test_marginal_note_passes_strict_validation() -> None:
    from codify.akn._schema import validate_akn

    source = _act(
        '<article eId="art_1"><num>1</num><content>'
        "<p>تعاريف</p>"
        "<p>يقصد بالكلمات التالية المعاني المبينة أدناه.</p>"
        "</content></article>"
    )
    out = emit_marginal_notes(source, regions=_aside("تعاريف"))
    assert "<authorialNote" in out
    validate_akn(out, strict=True)


def test_the_moved_label_is_not_read_as_provision_text() -> None:
    # The lifted-note predicate excludes it from provision_text, so the label no
    # longer reads as the opening words of the article.
    from lxml import etree

    from codify.akn.vocabulary import provision_text

    source = _act(
        '<article eId="art_1"><num>1</num><content>'
        "<p>تعاريف</p><p>يقصد بالكلمات التالية المعاني المبينة أدناه.</p>"
        "</content></article>"
    )
    out = emit_marginal_notes(source, regions=_aside("تعاريف"))
    article = etree.fromstring(out.encode("utf-8")).find(
        ".//{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}article"
    )
    assert "تعاريف" not in provision_text(article)


def test_no_aside_regions_is_a_noop() -> None:
    source = _act('<article eId="art_1"><content><p>نص عادي</p></content></article>')
    assert emit_marginal_notes(source, regions={}) == source
