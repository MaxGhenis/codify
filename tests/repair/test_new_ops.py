"""The completed op vocabulary: annotate, split wiring, restore, terminal move."""

from __future__ import annotations

import hashlib

from lxml import etree

from codify.akn.vocabulary import provision_text
from codify.pipeline.enrich.bluebell import parse_to_akn
from codify.pipeline.enrich.validator import validate_akn
from codify.repair.edit_ops import SourceEvidence, apply_op, apply_plan
from codify.repair.ops import (
    Annotate,
    MoveToConclusions,
    RestoreFromSource,
    Split,
)

NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"


def _gap_fixture() -> tuple[str, dict[str, object]]:
    bb = (
        "BODY\n"
        "  ARTICLE 1\n    First body text.\n"
        "  ARTICLE 2\n    Second body text.\n"
        "  ARTICLE 5\n    Fifth body text.\n"
        "  ARTICLE 6\n    Sixth body text.\n"
    )
    xml = parse_to_akn(bb, country="gb", doctype="act", number="1", date="2020-01-01")
    finding = next(f for f in validate_akn(xml) if f["check"] == "number_gap")
    return xml, finding


class TestAnnotate:
    def test_gap_clears_and_substantive_text_is_untouched(self) -> None:
        xml, finding = _gap_fixture()
        assert finding.get("eid"), "number_gap must carry a host eid now"
        host = str(finding["eid"])
        before_root = etree.fromstring(xml.encode("utf-8"))
        before_texts = {
            el.get("eId"): provision_text(el) for el in before_root.iter() if el.get("eId")
        }

        res = apply_op(
            xml,
            Annotate(eid=host, note="Articles 3-4 are repealed; confirmed on page 2."),
            country="gb",
            target_finding=finding,
        )
        assert res.ok, res.error
        after_root = etree.fromstring(res.xml.encode("utf-8"))
        for el in after_root.iter():
            eid = el.get("eId")
            if eid in before_texts:
                # The lifted note is excluded from provision text at any depth.
                assert provision_text(el) == before_texts[eid], eid
        note = next(after_root.iter(f"{{{NS}}}authorialNote"))
        assert note.get("placement") == "bottom"
        assert note.get("marker") == "editorial-gap"

    def test_acknowledged_gap_stays_cleared_on_revalidation(self) -> None:
        xml, finding = _gap_fixture()
        res = apply_op(
            xml,
            Annotate(eid=str(finding["eid"]), note="Repealed."),
            country="gb",
            target_finding=finding,
        )
        assert res.ok, res.error
        gaps = [
            f
            for f in validate_akn(res.xml)
            if f["check"] == "number_gap" and f.get("eid") == finding["eid"]
        ]
        assert gaps == []

    def test_structural_note_is_rejected(self) -> None:
        xml, finding = _gap_fixture()
        res = apply_op(
            xml,
            Annotate(eid=str(finding["eid"]), note="ARTICLE 3\nInvented text."),
            country="gb",
            target_finding=finding,
        )
        assert not res.ok
        assert xml == res.xml


class TestSplitWiring:
    def test_matched_surface_reaches_the_split(self) -> None:
        """The validator's `matched` key carries the literal body surface; a
        split given the canonical form must still work when they coincide, and
        the tolerant surface when they do not."""
        bb = (
            "BODY\n  ARTICLE 1\n"
            "    POINT (أ)\n      First point text.\n"
            "    POINT (ب)\n      Second point وبعدها (ج) swallowed third text here.\n"
            "    POINT (د)\n      Fourth point text.\n"
        )
        xml = parse_to_akn(bb, country="xz", doctype="act", number="9", date="2020-01-01")
        finding = next(f for f in validate_akn(xml) if f["check"] == "swallowed_enumerator")
        assert "matched" in finding
        res = apply_op(
            xml,
            Split(eid=str(finding["eid"]), marker=str(finding["matched"])),
            country="xz",
            target_finding=finding,
        )
        assert res.ok, res.error
        assert not any(f["check"] == "swallowed_enumerator" for f in validate_akn(res.xml))

    def test_split_preserves_every_source_character(self) -> None:
        """Head + marker-number + tail all survive into the two siblings."""
        bb = (
            "BODY\n  ARTICLE 1\n"
            "    POINT (أ)\n      First point text.\n"
            "    POINT (ب)\n      Second point وبعدها (ج) swallowed third text here.\n"
            "    POINT (د)\n      Fourth point text.\n"
        )
        xml = parse_to_akn(bb, country="xz", doctype="act", number="9", date="2020-01-01")
        finding = next(f for f in validate_akn(xml) if f["check"] == "swallowed_enumerator")
        res = apply_op(
            xml,
            Split(eid=str(finding["eid"]), marker=str(finding["matched"])),
            country="xz",
            target_finding=finding,
        )
        assert res.ok, res.error

        def _prose(x: str) -> str:
            root = etree.fromstring(x.encode("utf-8"))
            body = root.find(f".//{{{NS}}}body")
            assert body is not None
            chars = "".join(
                t
                for el in body.iter()
                if etree.QName(el).localname not in {"num", "heading"}
                for t in (el.text, el.tail)
                if t
            )
            return "".join(chars.split())

        before = _prose(xml).replace("".join(str(finding["matched"]).split()), "")
        assert _prose(res.xml) == before  # only the marker left the prose

    def test_cross_family_marker_is_rejected(self) -> None:
        bb = (
            "BODY\n  ARTICLE 1\n"
            "    POINT (أ)\n      First point (2) not an abjad sibling.\n"
            "    POINT (ب)\n      Second.\n"
        )
        xml = parse_to_akn(bb, country="xz", doctype="act", number="9", date="2020-01-01")
        root = etree.fromstring(xml.encode("utf-8"))
        eid = next(str(el.get("eId")) for el in root.iter(f"{{{NS}}}point") if el.get("eId"))
        res = apply_op(xml, Split(eid=eid, marker="(2)"), country="xz")
        assert not res.ok
        assert "family" in res.error


class TestRestoreFromSource:
    SOURCE = (
        "Стаття 1. Перша\nПовний текст першої статті з достатнім обсягом.\n"
        "Стаття 2. Друга\nВідновлений текст другої статті.\nЩе один рядок тексту.\n"
    )

    def _fixture(self) -> tuple[str, dict[str, object]]:
        bb = "BODY\n  ARTICLE 1\n    Повний текст першої статті з достатнім обсягом.\n  ARTICLE 2\n"
        xml = parse_to_akn(bb, country="ua", doctype="act", number="1", date="2020-01-01")
        finding = next(f for f in validate_akn(xml) if f["check"] == "empty_article")
        return xml, finding

    def _evidence(self, text: str | None = None, sha: str | None = None) -> SourceEvidence:
        body = self.SOURCE if text is None else text
        return SourceEvidence(
            text=body,
            sha256=sha if sha is not None else hashlib.sha256(body.encode("utf-8")).hexdigest(),
        )

    def test_restores_verbatim_from_the_resolved_span(self) -> None:
        xml, finding = self._fixture()
        res = apply_op(
            xml,
            RestoreFromSource(eid=str(finding["eid"])),
            country="ua",
            target_finding=finding,
            evidence=self._evidence(),
        )
        assert res.ok, res.error
        assert "Відновлений текст другої статті." in res.xml
        assert "Ще один рядок тексту." in res.xml
        root = etree.fromstring(res.xml.encode("utf-8"))
        target = root.xpath(".//*[@eId=$e]", e=str(finding["eid"]))[0]
        body = " ".join(
            t
            for el in target.iter()
            if etree.QName(el).localname not in {"num", "heading"}
            for t in (el.text,)
            if t
        )
        # The anchor line is structure, not body: no duplicated num/heading prose.
        assert "Стаття 2" not in body

    def test_drifted_evidence_is_rejected(self) -> None:
        xml, finding = self._fixture()
        res = apply_op(
            xml,
            RestoreFromSource(eid=str(finding["eid"])),
            country="ua",
            target_finding=finding,
            evidence=self._evidence(sha="0" * 64),
        )
        assert not res.ok
        assert "drifted" in res.error

    def test_absent_evidence_is_rejected(self) -> None:
        xml, finding = self._fixture()
        res = apply_op(
            xml,
            RestoreFromSource(eid=str(finding["eid"])),
            country="ua",
            target_finding=finding,
        )
        assert not res.ok
        assert res.xml == xml


class TestMoveToConclusions:
    PHRASES = ["Issued in the city of"]

    def _fixture(self) -> str:
        bb = (
            "BODY\n  ARTICLE 1\n    Substantive obligation text that keeps going.\n"
            "  ARTICLE 2\n"
            "    Final substantive text.\n"
            "    Issued in the city of Zerzura on 1 March 2020.\n"
            "    The President\n"
        )
        return parse_to_akn(bb, country="xz", doctype="act", number="3", date="2020-01-01")

    def test_detector_finds_it_and_the_op_moves_it(self) -> None:
        xml = self._fixture()
        findings = validate_akn(xml, closing_phrases=self.PHRASES)
        finding = next(f for f in findings if f["check"] == "displaced_terminal_material")
        res = apply_plan(
            xml,
            [MoveToConclusions(eid=str(finding["eid"]))],
            country="xz",
            target_finding=finding,
            evidence=SourceEvidence(closing_phrases=self.PHRASES),
        )
        assert res.ok, res.error
        root = etree.fromstring(res.xml.encode("utf-8"))
        conclusions = root.find(f".//{{{NS}}}conclusions")
        assert conclusions is not None
        moved = " ".join(conclusions.itertext())
        assert "Issued in the city of" in moved
        body = root.find(f".//{{{NS}}}body")
        assert body is not None
        assert "Issued in the city of" not in " ".join(body.itertext())

    def test_without_vocabulary_the_op_abstains(self) -> None:
        xml = self._fixture()
        res = apply_plan(
            xml,
            [MoveToConclusions(eid="art_2")],
            country="xz",
            evidence=SourceEvidence(closing_phrases=[]),
        )
        assert not res.ok
        assert res.xml == xml
