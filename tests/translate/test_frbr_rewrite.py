# ruff: noqa: E501  # AKN XML test fixtures: line wraps would change tested whitespace
"""FRBR expression/manifestation rewrite in the translation write path."""

from __future__ import annotations

from datetime import date

from lxml import etree

from codify.akn import AKN_NS
from codify.akn._schema import validate_akn
from codify.translate.write import apply_translation_to_akn

_NS = {"akn": AKN_NS}

_AKN = f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act name="act">
    <meta>
      <identification source="#codify">
        <FRBRWork>
          <FRBRthis value="/akn/xz/act/1999/7"/>
          <FRBRuri value="/akn/xz/act/1999/7"/>
          <FRBRdate date="1999-06-08" name="Generation"/>
          <FRBRauthor href="#codify"/>
          <FRBRcountry value="xz"/>
        </FRBRWork>
        <FRBRExpression>
          <FRBRthis value="/akn/xz/act/1999/7/ara@1999-06-08"/>
          <FRBRuri value="/akn/xz/act/1999/7/ara@1999-06-08"/>
          <FRBRdate date="1999-06-08" name="Generation"/>
          <FRBRauthor href="#codify"/>
          <FRBRlanguage language="ara"/>
        </FRBRExpression>
        <FRBRManifestation>
          <FRBRthis value="/akn/xz/act/1999/7/ara@1999-06-08.akn"/>
          <FRBRuri value="/akn/xz/act/1999/7/ara@1999-06-08.akn"/>
          <FRBRdate date="1999-06-08" name="Generation"/>
          <FRBRauthor href="#codify"/>
        </FRBRManifestation>
      </identification>
    </meta>
    <body><article eId="art_1"><num>1</num><content><p>نص</p></content></article></body>
  </act>
</akomaNtoso>
'''


def _attr(root: etree._Element, path: str, attr: str) -> str | None:
    el = root.find(path, _NS)
    return el.get(attr) if el is not None else None


def test_target_language_rewrites_expression_and_manifestation():
    out = apply_translation_to_akn(_AKN, [], [], target_language="English")
    root = etree.fromstring(out.encode("utf-8"))
    expr = "/akn/xz/act/1999/7/eng@1999-06-08"
    ident = ".//akn:identification/"
    assert _attr(root, f"{ident}akn:FRBRExpression/akn:FRBRthis", "value") == expr
    assert _attr(root, f"{ident}akn:FRBRExpression/akn:FRBRuri", "value") == expr
    assert _attr(root, f"{ident}akn:FRBRExpression/akn:FRBRlanguage", "language") == "eng"
    assert _attr(root, f"{ident}akn:FRBRManifestation/akn:FRBRthis", "value") == f"{expr}.akn"
    assert _attr(root, f"{ident}akn:FRBRManifestation/akn:FRBRuri", "value") == f"{expr}.akn"
    # Work identity untouched.
    assert _attr(root, f"{ident}akn:FRBRWork/akn:FRBRuri", "value") == "/akn/xz/act/1999/7"


def test_no_target_language_leaves_meta_untouched():
    out = apply_translation_to_akn(_AKN, [], [])
    root = etree.fromstring(out.encode("utf-8"))
    assert (
        _attr(root, ".//akn:identification/akn:FRBRExpression/akn:FRBRlanguage", "language")
        == "ara"
    )


def test_missing_identification_noops():
    bare = f'<akomaNtoso xmlns="{AKN_NS}"><act><body><article eId="art_1"><num>1</num><content><p>x</p></content></article></body></act></akomaNtoso>'
    out = apply_translation_to_akn(bare, [], [], target_language="English")
    assert "FRBRExpression" not in out


_WITH_ATTACHMENT = f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act>
    <meta>
      <identification source="#codify">
        <FRBRWork><FRBRthis value="/akn/xz/act/2016/18"/><FRBRuri value="/akn/xz/act/2016/18"/><FRBRdate date="2016-01-01" name="Generation"/><FRBRauthor href="#codify"/><FRBRcountry value="xz"/></FRBRWork>
        <FRBRExpression><FRBRthis value="/akn/xz/act/2016/18/ara@2016-01-01"/><FRBRuri value="/akn/xz/act/2016/18/ara@2016-01-01"/><FRBRdate date="2016-01-01" name="Generation"/><FRBRauthor href="#codify"/><FRBRlanguage language="ara"/></FRBRExpression>
        <FRBRManifestation><FRBRthis value="/akn/xz/act/2016/18/ara@2016-01-01.akn"/><FRBRuri value="/akn/xz/act/2016/18/ara@2016-01-01.akn"/><FRBRdate date="2016-01-01" name="Generation"/><FRBRauthor href="#codify"/></FRBRManifestation>
      </identification>
    </meta>
    <body><article eId="art_1"><num>1</num><content><p>نص.</p></content></article></body>
    <attachments>
      <attachment eId="att_1">
        <doc name="schedule">
          <meta>
            <identification source="#codify">
              <FRBRWork><FRBRthis value="/akn/xz/act/2016/18/!schedule_1"/><FRBRuri value="/akn/xz/act/2016/18"/><FRBRdate date="2016-01-01" name="Generation"/><FRBRauthor href="#codify"/><FRBRcountry value="xz"/></FRBRWork>
              <FRBRExpression><FRBRthis value="/akn/xz/act/2016/18/ara@2016-01-01/!schedule_1"/><FRBRuri value="/akn/xz/act/2016/18/ara@2016-01-01"/><FRBRdate date="2016-01-01" name="Generation"/><FRBRauthor href="#codify"/><FRBRlanguage language="ara"/></FRBRExpression>
              <FRBRManifestation><FRBRthis value="/akn/xz/act/2016/18/ara@2016-01-01.akn"/><FRBRuri value="/akn/xz/act/2016/18/ara@2016-01-01.akn"/><FRBRdate date="2016-01-01" name="Generation"/><FRBRauthor href="#codify"/></FRBRManifestation>
            </identification>
          </meta>
          <mainBody><p eId="att_1__p_1">جدول الرواتب.</p></mainBody>
        </doc>
      </attachment>
    </attachments>
  </act>
</akomaNtoso>
'''


class TestNestedAttachmentIdentification:
    """An annex carries its own FRBR block, which must agree with the
    language of the document it ships inside."""

    def test_no_source_language_survives_anywhere_including_attachments(self) -> None:
        out = apply_translation_to_akn(_WITH_ATTACHMENT, [], [], target_language="English")
        root = etree.fromstring(out.encode("utf-8"))
        langs = [el.get("language") for el in root.iter(f"{{{AKN_NS}}}FRBRlanguage")]
        assert langs == ["eng", "eng"], langs
        # `ara` now legitimately appears inside `<FRBRtranslation>`, which exists
        # to name the source. Drop those elements and the original whole-document
        # claim still holds, which keeps the manifestation blocks covered.
        for el in list(root.iter(f"{{{AKN_NS}}}FRBRtranslation")):
            el.getparent().remove(el)
        rest = etree.tostring(root, encoding="unicode")
        assert 'language="ara"' not in rest
        assert "/ara@" not in rest

    def test_attachment_keeps_its_own_work_identity(self) -> None:
        # Both blocks are rewritten from their own work URI, so the annex does
        # not inherit the parent's expression URI.
        out = apply_translation_to_akn(_WITH_ATTACHMENT, [], [], target_language="English")
        root = etree.fromstring(out.encode("utf-8"))
        this_values = [
            el.get("value")
            for el in root.iter(f"{{{AKN_NS}}}FRBRthis")
            if el.getparent() is not None and el.getparent().tag.endswith("FRBRExpression")
        ]
        assert this_values == [
            "/akn/xz/act/2016/18/eng@2016-01-01",
            # The annex keeps addressing its own component, rebased onto the
            # translated expression rather than collapsing to the whole work.
            "/akn/xz/act/2016/18/eng@2016-01-01/!schedule_1",
        ]


def _translations(root: etree._Element) -> list[etree._Element]:
    return list(root.iter(f"{{{AKN_NS}}}FRBRtranslation"))


class TestDeclaringTheTranslation:
    """A translated expression says so in the AKN, rather than the fact living
    only in a column this system knows to read."""

    def test_it_names_the_source_language_and_expression(self) -> None:
        out = apply_translation_to_akn(_AKN, [], [], target_language="English")
        (el,) = _translations(etree.fromstring(out.encode("utf-8")))
        assert el.get("fromLanguage") == "ara"
        assert el.get("href") == "/akn/xz/act/1999/7/ara@1999-06-08"
        assert el.get("by") == "#codify"
        assert el.get("authoritative") == "false"

    def test_it_sits_after_the_language(self) -> None:
        """`exprProperties` orders it last. The fixture puts `FRBRlanguage` at
        the end of the block, where appending happens to be right, so the source
        here carries a sibling after it: append then puts `FRBRtranslation` in
        the wrong place and the strict schema refuses the document."""
        with_sibling = _AKN.replace(
            '<FRBRlanguage language="ara"/>',
            '<FRBRlanguage language="ara"/>\n          '
            '<FRBRtranslation fromLanguage="heb" href="/akn/xz/act/1999/7/heb@1999-06-08"'
            ' by="#codify"/>',
        )
        out = apply_translation_to_akn(with_sibling, [], [], target_language="English")
        root = etree.fromstring(out.encode("utf-8"))
        expr = root.find(f".//{{{AKN_NS}}}FRBRExpression")
        assert expr is not None
        names = [etree.QName(c).localname for c in expr]
        assert names.index("FRBRtranslation") == names.index("FRBRlanguage") + 1
        validate_akn(out, strict=True)

    def test_retranslating_leaves_exactly_one(self) -> None:
        """The write path clones the source tree, so an inherited declaration
        would stack up, each naming a source further back than the last."""
        once = apply_translation_to_akn(_AKN, [], [], target_language="English")
        twice = apply_translation_to_akn(once, [], [], target_language="French")
        (el,) = _translations(etree.fromstring(twice.encode("utf-8")))
        assert el.get("fromLanguage") == "eng"

    def test_a_block_it_cannot_declare_keeps_the_one_it_had(self) -> None:
        """Removing the incumbent before knowing a replacement can be built is a
        net loss of provenance reported as a successful translation."""
        once = apply_translation_to_akn(_AKN, [], [], target_language="English")
        stripped = once.replace('<FRBRlanguage language="eng"/>', "")
        twice = apply_translation_to_akn(stripped, [], [], target_language="French")
        (el,) = _translations(etree.fromstring(twice.encode("utf-8")))
        assert el.get("fromLanguage") == "ara"

    def test_a_same_language_pass_declares_nothing(self) -> None:
        """It would have the expression cite its own previous URI as the thing
        it was translated from."""
        out = apply_translation_to_akn(_AKN, [], [], target_language="Arabic")
        assert _translations(etree.fromstring(out.encode("utf-8"))) == []

    def test_a_comment_in_the_block_is_not_mistaken_for_an_element(self) -> None:
        commented = _AKN.replace(
            '<FRBRlanguage language="ara"/>', '<!-- source --><FRBRlanguage language="ara"/>'
        )
        out = apply_translation_to_akn(commented, [], [], target_language="English")
        assert len(_translations(etree.fromstring(out.encode("utf-8")))) == 1

    def test_an_attachment_declares_its_own_component(self) -> None:
        out = apply_translation_to_akn(_WITH_ATTACHMENT, [], [], target_language="English")
        hrefs = [el.get("href") for el in _translations(etree.fromstring(out.encode("utf-8")))]
        assert hrefs == [
            "/akn/xz/act/2016/18/ara@2016-01-01",
            "/akn/xz/act/2016/18/ara@2016-01-01",
        ]


class TestTheTranslationIsDatedWhenItIsMade:
    """A translation comes into being when it is produced. Inheriting the
    source's date had every translated expression claiming to have existed since
    the law was enacted."""

    def test_the_expression_and_its_uri_take_the_creation_date(self) -> None:
        out = apply_translation_to_akn(
            _AKN, [], [], target_language="English", expression_date="2026-08-09"
        )
        root = etree.fromstring(out.encode("utf-8"))
        expr = "/akn/xz/act/1999/7/eng@2026-08-09"
        ident = ".//akn:identification/"
        assert _attr(root, f"{ident}akn:FRBRExpression/akn:FRBRuri", "value") == expr
        assert _attr(root, f"{ident}akn:FRBRExpression/akn:FRBRdate", "date") == "2026-08-09"
        # The manifestation URI names the expression, so it carries that date; its
        # own `FRBRdate` is when the file was made. Compared against today, not a
        # literal, which would pass only on the day it was written.
        assert _attr(root, f"{ident}akn:FRBRManifestation/akn:FRBRuri", "value") == f"{expr}.akn"
        assert (
            _attr(root, f"{ident}akn:FRBRManifestation/akn:FRBRdate", "date")
            == date.today().isoformat()
        )

    def test_the_manifestation_is_dated_when_the_file_was_made(self) -> None:
        """The expression date is deliberately stable across re-runs, so on a
        force retranslate days later it is not when this file was produced. The
        manifestation records that, and its URI still names the expression."""
        out = apply_translation_to_akn(
            _AKN, [], [], target_language="English", expression_date="1999-06-08"
        )
        root = etree.fromstring(out.encode("utf-8"))
        ident = ".//akn:identification/"
        manif = f"{ident}akn:FRBRManifestation/"
        assert _attr(root, f"{manif}akn:FRBRdate", "date") == date.today().isoformat()
        assert _attr(root, f"{manif}akn:FRBRuri", "value") == (
            "/akn/xz/act/1999/7/eng@1999-06-08.akn"
        )

    def test_the_work_keeps_its_own_date(self) -> None:
        out = apply_translation_to_akn(
            _AKN, [], [], target_language="English", expression_date="2026-08-09"
        )
        root = etree.fromstring(out.encode("utf-8"))
        assert (
            _attr(root, ".//akn:identification/akn:FRBRWork/akn:FRBRdate", "date") == "1999-06-08"
        )

    def test_the_source_it_names_is_still_the_source(self) -> None:
        """The declaration is read before the rewrite, so moving the date must
        not make the translation point at itself."""
        out = apply_translation_to_akn(
            _AKN, [], [], target_language="English", expression_date="2026-08-09"
        )
        (el,) = _translations(etree.fromstring(out.encode("utf-8")))
        assert el.get("href") == "/akn/xz/act/1999/7/ara@1999-06-08"
