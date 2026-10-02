"""A re-read may change a law's text and never its identity.

`reocr` re-runs metadata extraction, so without a pin the FRBR URI is rebuilt
from whatever the model reads off the page while `laws.frbr_work_uri` stays put,
and the document ends up asserting a different identity from its own columns."""

from __future__ import annotations

import inspect
import re

import pytest

from codify.frbr import is_citable_work_uri
from codify.pipeline import stages
from codify.pipeline.enrich.bluebell import parse_to_akn
from codify.pipeline.enrich.cobalt import _component_tail, enrich_akn, set_work_uri

_BLUEBELL = "PREFACE\n\nBODY\n\n  ARTICLE 1.\n\n    Text.\n"


def _parsed() -> str:
    """Real parser output: a hand-written fixture can be wrong in the same
    direction as the code reading it."""
    return parse_to_akn(_BLUEBELL, "xz", "act", "1999", "1", "ara")


def _stored(uri: str = "/akn/xz/act/1999/1") -> str:
    """What a stored version holds: parsed, then through enrich."""
    return enrich_akn(
        _parsed(), title="T", country="xz", doctype="act", year="1999", number="1", work_uri=uri
    )


def _block(xml: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", xml, re.S)
    return m.group(1) if m else None


def _expr(xml: str, tag: str) -> str | None:
    m = re.search(rf"<FRBRExpression>.*?<{tag} [^>]*value=\"([^\"]*)\"", xml, re.S)
    return m.group(1) if m else None


def _work(xml: str, tag: str) -> str | None:
    m = re.search(rf"<FRBRWork>.*?<{tag} [^>]*(?:value|date)=\"([^\"]*)\"", xml, re.S)
    return m.group(1) if m else None


class TestEnrichPin:
    def test_a_pinned_uri_wins_over_the_derived_one(self) -> None:
        # The case that matters: the number came from the title, not the page.
        out = enrich_akn(
            _parsed(),
            title="T",
            country="xz",
            doctype="act",
            year="2008",
            number="99",
            work_uri="/akn/xz/act/2008/8-11",
        )
        assert _work(out, "FRBRuri") == "/akn/xz/act/2008/8-11"
        assert _work(out, "FRBRnumber") == "8-11"

    def test_without_a_pin_the_derived_uri_still_applies(self) -> None:
        # First ingest and uploads carry no stored URI and must be unaffected.
        out = enrich_akn(
            _parsed(), title="T", country="xz", doctype="act", year="2008", number="99"
        )
        assert _work(out, "FRBRuri") == "/akn/xz/act/2008/99"

    def test_an_empty_pin_is_not_a_pin(self) -> None:
        out = enrich_akn(
            _parsed(), title="T", country="xz", doctype="act", year="2008", number="99", work_uri=""
        )
        assert _work(out, "FRBRuri") == "/akn/xz/act/2008/99"


class TestSetWorkUri:
    def test_it_restates_number_and_uri_from_the_work_uri(self) -> None:
        # Cobalt's setter is the whole repair: the components follow the URI.
        out = set_work_uri(_stored(), "/akn/xz/act/2009/259", "/akn/xz/act/2009/259/ara")
        assert out is not None
        assert _work(out, "FRBRuri") == "/akn/xz/act/2009/259"
        assert _work(out, "FRBRnumber") == "259"

    def test_an_already_correct_document_costs_no_write(self) -> None:
        assert (
            set_work_uri(_stored(), "/akn/xz/act/1999/1", _expr(_stored(), "FRBRuri") or "") is None
        )

    def test_an_enactment_date_survives_a_number_repair(self) -> None:
        # Cobalt writes the URI's bare year over FRBRWork/FRBRdate. Under the
        # same year that is a loss of precision, not a correction.
        doc = _stored().replace('<FRBRdate date="1999"', '<FRBRdate date="1999-05-04"', 1)
        assert _work(doc, "FRBRdate") == "1999-05-04"
        out = set_work_uri(doc, "/akn/xz/act/1999/8-11", "/akn/xz/act/1999/8-11/ara@2026-01-01")
        assert out is not None
        assert _work(out, "FRBRnumber") == "8-11"
        assert _work(out, "FRBRdate") == "1999-05-04"

    def test_a_date_under_a_different_year_is_not_carried_over(self) -> None:
        # It described a different work, so keeping it would assert a date the
        # repaired identity never had.
        doc = _stored().replace('<FRBRdate date="1999"', '<FRBRdate date="1994-05-04"', 1)
        out = set_work_uri(doc, "/akn/xz/act/2009/259", "/akn/xz/act/2009/259/ara@2026-01-01")
        assert out is not None
        assert _work(out, "FRBRdate") == "2009"


class TestTheSeamsThatCarryThePin:
    """The pin is only worth anything if it reaches `enrich_akn`. Each hop is a
    keyword that can be dropped without any test above noticing."""

    def test_the_enrich_sequence_hands_the_pin_to_cobalt(self) -> None:
        src = inspect.getsource(stages.run_enrich_passes)
        assert "work_uri: str | None = None" in src
        assert "work_uri=work_uri" in src


class TestSetWorkUriRefusals:
    def test_an_empty_document_is_refused_not_fabricated(self) -> None:
        """`Act("")` returns a synthetic empty act rather than raising, so
        without this guard an empty row is "repaired" into a stub carrying
        FRBRalias "Untitled" and no body."""
        with pytest.raises(ValueError, match="empty document"):
            set_work_uri("", "/akn/xz/act/2009/259", "/akn/xz/act/2009/259/ara")

    def test_the_expression_comes_from_the_row_not_the_document(self) -> None:
        """The document's own tail is not a safe source. Cobalt rebuilds its
        `@date` from the generation date, and one UA document says `ukr` where
        the row is the English expression, so carrying the document's tail
        writes a language mislabel onto a corrected work URI."""
        doc = _stored()
        out = set_work_uri(doc, "/akn/xz/act/2009/259", "/akn/xz/act/2009/259/eng@2015-06-01")
        assert out is not None
        assert _expr(out, "FRBRuri") == "/akn/xz/act/2009/259/eng@2015-06-01"
        assert _expr(out, "FRBRthis") == "/akn/xz/act/2009/259/eng@2015-06-01/!main"
        # Nothing of the document's own expression identity survives.
        assert "ara@" not in (_expr(out, "FRBRuri") or "")

    def test_an_expression_outside_its_work_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not under work"):
            set_work_uri(_stored(), "/akn/xz/act/2009/259", "/akn/xx/act/1900/9/ara")


class TestIsCitableWorkUri:
    @pytest.mark.parametrize(
        ("uri", "citable"),
        [
            ("/akn/xz/act/2008/8-11", True),
            ("/akn/eu/act/reg/2016/679", True),
            ("/akn/xz/act/0001/draft-3f9a2c1", False),
            ("/akn/xz/act/2008/draft-3f9a2c1", False),
            # The `_UNCITABLE_SEGMENT` shapes this module refuses elsewhere.
            ("/akn/xz/act/2008/", False),
            ("//", False),
            # A date URI: the selector rejects it, so the guard must agree.
            ("/akn/xz/act/2008-04-13/8", False),
            ("", False),
        ],
    )
    def test_the_shapes_that_may_be_written_back(self, uri: str, citable: bool) -> None:
        assert is_citable_work_uri(uri) is citable


class TestManifestationFormat:
    """The manifestation format is addressing the repair must not eat.

    It sits on `FRBRuri` with no `/!` marker, so splitting on the component
    alone drops it. Measured on demo that is 970 versions, 961 EU acquis."""

    @pytest.mark.parametrize(
        ("value", "tail"),
        [
            ("/akn/eu/act/reg/2016/679/eng@2016-05-04.xml", ".xml"),
            ("/akn/eu/act/reg/2016/679/eng@2016-05-04/!main.akn", "/!main.akn"),
            ("/akn/xz/act/2008/8-11/ara@2020-01-01/!main", "/!main"),
            ("/akn/xz/act/2008/8-11/ara@2020-01-01", ""),
            # A dotted work number is not a format.
            ("/akn/xx/act/2020/1.2/eng@2020-01-01", ""),
            (None, ""),
        ],
    )
    def test_the_tail_kept_from_each_shape(self, value: str | None, tail: str) -> None:
        assert _component_tail(value) == tail


def _with_attachment(doc: str) -> str:
    """A second `<identification>` inside a nested `<doc>`, as an attachment
    carries. 60 versions in one corpus hold one."""
    ident = re.search(r"<identification.*?</identification>", doc, re.S)
    assert ident is not None
    nested = ident.group(0).replace("/!main", "/!schedule1")
    # The `<attachments>` wrapper is load-bearing: Cobalt's `components()` xpath
    # is `./a:attachments/a:attachment/...`, so a bare `<attachment>` is never
    # visited. A fixture without it tests a shape the real path cannot produce
    # and passes on output the pipeline would never emit.
    attachment = (
        f'<attachments><attachment><doc name="schedule"><meta>{nested}</meta>'
        f"<mainBody><p>Schedule.</p></mainBody></doc></attachment></attachments>"
    )
    return doc.replace("</act>", f"{attachment}</act>")


def test_every_identification_block_moves_together() -> None:
    """Cobalt rewrites each component's blocks from the generation date, so
    correcting the root alone left 58 of 60 multi-component documents
    asserting two expressions of one work, at two dates, and `/!schedule1` no
    longer resolving under the root."""
    doc = _with_attachment(_stored())
    assert len(re.findall(r"<identification", doc)) == 2

    out = set_work_uri(doc, "/akn/xz/act/2009/259", "/akn/xz/act/2009/259/eng@2015-06-01")
    assert out is not None

    # Both blocks name the same work, which the bare-`<attachment>` shape does
    # not achieve because Cobalt never reaches it.
    works = re.findall(r'<FRBRWork>.*?<FRBRuri value="([^"]*)"', out, re.S)
    assert works == ["/akn/xz/act/2009/259", "/akn/xz/act/2009/259"]
    uris = re.findall(r'<FRBRExpression>.*?<FRBRuri value="([^"]*)"', out, re.S)
    assert len(uris) == 2
    assert set(uris) == {"/akn/xz/act/2009/259/eng@2015-06-01"}
    # The attachment keeps addressing its own component, not the whole work.
    this = re.findall(r'<FRBRExpression>.*?<FRBRthis value="([^"]*)"', out, re.S)
    assert this == [
        "/akn/xz/act/2009/259/eng@2015-06-01/!main",
        "/akn/xz/act/2009/259/eng@2015-06-01/!schedule1",
    ]
    # And no block is left declaring the document's own stale language.
    assert 'language="ara"' not in out


class TestExpressionDate:
    """The element, not only the URI.

    Cobalt rebuilds the expression `@date` from `FRBRExpression/FRBRdate`, so a
    repair that moved the URI and left the element stale made the document
    contradict itself and would be reverted by the next Cobalt-mediated write."""

    def test_both_date_elements_follow_the_row(self) -> None:
        out = set_work_uri(
            _stored(), "/akn/xz/act/2009/259", "/akn/xz/act/2009/259/eng@2015-06-01", "2015-06-01"
        )
        assert out is not None
        for block in ("FRBRExpression", "FRBRManifestation"):
            m = re.search(rf"<{block}>.*?<FRBRdate [^>]*date=\"([^\"]*)\"", out, re.S)
            assert m is not None and m.group(1) == "2015-06-01", block

    def test_without_an_expression_uri_the_row_is_not_consulted(self) -> None:
        """The opt-out path. Cobalt still rebases the expression onto the new
        work, which is its own behaviour and the pre-existing one; what this
        turns off is taking the expression from the row. The date therefore
        stays the document's, not the row's."""
        doc = _stored()
        out = set_work_uri(doc, "/akn/xz/act/2009/259")
        assert out is not None
        assert _work(out, "FRBRnumber") == "259"
        now = _expr(out, "FRBRuri")
        assert now is not None and now.startswith("/akn/xz/act/2009/259/ara@")
        was_date = re.search(r"<FRBRExpression>.*?<FRBRdate [^>]*date=\"([^\"]*)\"", doc, re.S)
        is_date = re.search(r"<FRBRExpression>.*?<FRBRdate [^>]*date=\"([^\"]*)\"", out, re.S)
        assert was_date is not None and is_date is not None
        assert is_date.group(1) == was_date.group(1)
