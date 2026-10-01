"""A citation the body-fill broke into a subdivision is not a new provision.

`sebagaimana dimaksud dalam Pasal 41 ayat (3) huruf b` is split so the number
becomes its own subdivision: the body of (1) ends mid-sentence at `ayat`,
`POINT (3)` opens, and the rest of the sentence becomes its body. That
provision then collides with the article's own (3).
"""

from __future__ import annotations

from codify.jurisdictions import load_config
from codify.pipeline.enrich.bluebell import rejoin_citation_lines

# A synthetic jurisdiction of the same language and structuring shape: the
# mechanism under test is not specific to any real one.
_JURISDICTION = "xl"

# Bluebell's own shape: the keyword and number on one line, body beneath. The
# split falls inside a Bahasa cross-reference, whose grammar puts the
# article number and the paragraph number in separate numbered slots.
SPLIT_CITATION = (
    "        ARTICLE 64\n"
    "          POINT (1)\n"
    "            Pendaftaran Perkakas sebagaimana dimaksud dalam Pasal 41 ayat\n"
    "          POINT (3)\n"
    "            huruf b dilakukan dengan tahapan kegiatan:\n"
    "            POINT a.\n"
    "              pembuatan Daftar Perkakas Pelabuhan;\n"
    "          POINT (2)\n"
    "            Dalam rangka penatausahaan Kawasan Pelabuhan.\n"
    "          POINT (3)\n"
    "            Pendaftaran Perkakas dapat dilakukan secara elektronik.\n"
)


def test_a_broken_citation_is_rejoined() -> None:
    out, joined, _ = rejoin_citation_lines(SPLIT_CITATION, _JURISDICTION)
    assert joined == 1
    assert "dalam Pasal 41 ayat (3) huruf b dilakukan dengan tahapan kegiatan:" in out
    # Spliced, not copied: the marker's body moves into the sentence rather
    # than being left behind as well, which would duplicate the law.
    assert out.count("huruf b dilakukan dengan tahapan kegiatan:") == 1


def test_the_real_subdivisions_survive() -> None:
    """The article's own (2) and (3) must remain provisions: rejoining them
    would merge distinct law, which is worse than the defect being fixed."""
    out, _, _ = rejoin_citation_lines(SPLIT_CITATION, _JURISDICTION)
    assert "POINT (2)" in out
    assert out.count("POINT (3)") == 1  # the citation one is gone, the real one stays
    assert "Pendaftaran Perkakas dapat dilakukan" in out


def test_the_nested_list_reparents_onto_the_provision_that_owns_it() -> None:
    """`POINT a.` was nested under the spurious (3). Dropping that marker has to
    leave it a child of the provision whose sentence introduces it, not a
    sibling: textual order alone would still read correctly while the document
    said something different about what contains what."""
    from lxml import etree

    from codify.pipeline.enrich.bluebell import parse_to_akn

    out, _, _ = rejoin_citation_lines(SPLIT_CITATION, _JURISDICTION)
    xml = etree.fromstring(
        parse_to_akn(
            out, country=_JURISDICTION, doctype="act", date="2021-01-01", number="7"
        ).encode()
    )
    ns = {"a": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"}
    item = xml.xpath('//a:point[a:num[text()="a."]]', namespaces=ns)
    assert item, etree.tostring(xml, pretty_print=True).decode()[:800]
    ancestors = [
        e.findtext("{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}num")
        for e in item[0].iterancestors()
        if e.tag.endswith("}point")
    ]
    assert "(1)" in ancestors, ancestors


def test_a_subdivision_after_ordinary_prose_is_untouched() -> None:
    text = (
        "        ARTICLE 5\n"
        "          POINT (1)\n"
        "            Ketentuan ini berlaku.\n"
        "          POINT (2)\n"
        "            Ketentuan berikutnya berlaku.\n"
    )
    out, joined, _ = rejoin_citation_lines(text, _JURISDICTION)
    assert joined == 0 and out == text


def test_a_jurisdiction_declaring_no_nouns_is_unchanged() -> None:
    cfg = load_config("xz")
    assert cfg is not None and not cfg.structuring.reference_nouns
    out, joined, _ = rejoin_citation_lines(SPLIT_CITATION, "xz")
    assert joined == 0 and out == SPLIT_CITATION


def test_the_rejoin_is_idempotent() -> None:
    """`bluebell_to_akn` rejoins before parsing so the source map sees the same
    string the parser did, and `parse_to_akn` rejoins again inside."""
    once, first, _ = rejoin_citation_lines(SPLIT_CITATION, _JURISDICTION)
    twice, second, _ = rejoin_citation_lines(once, _JURISDICTION)
    assert first == 1 and second == 0 and twice == once


def test_the_source_map_keeps_the_callers_line_numbers() -> None:
    """The map drives the editor's linked scrolling. This join removes two
    lines, the marker and its body, so the prepared-to-original map has to
    account for both: an off-by-one would misalign every anchor after the first
    citation and show as scrolling to the wrong provision, not as an error.

    In SPLIT_CITATION the subdivisions that survive open on original lines 2, 6, 8, 10.
    """
    from codify.akn.bluebell import bluebell_to_akn

    akn, source_map, errors = bluebell_to_akn(
        SPLIT_CITATION, country=_JURISDICTION, doctype="act", date="2021-01-01", number="7"
    )
    assert not errors, errors
    points = [a for a in source_map if a.kind == "point"]
    assert [a.line for a in points] == [2, 6, 8, 10], [(a.eid, a.line) for a in points]
    # And every reported line really does open that provision in the caller's text.
    lines = SPLIT_CITATION.split("\n")
    for anchor in points:
        assert lines[anchor.line - 1].lstrip().startswith("POINT "), (
            anchor,
            lines[anchor.line - 1],
        )
