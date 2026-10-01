"""The harness is only useful if body prose is invisible to it and structure
is not, so both halves are tested against the same pair of documents."""

from __future__ import annotations

import pytest

from codify.akn.structure_diff import diff_structure, structure_of

NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"


def _doc(articles: str) -> str:
    return (
        f'<akomaNtoso xmlns="{NS}"><act><body>'
        f'<chapter eId="chp_1"><num>1</num><heading>General</heading>{articles}</chapter>'
        "</body></act></akomaNtoso>"
    )


def _article(eid: str, num: str, heading: str, body: str) -> str:
    return (
        f'<article eId="{eid}"><num>{num}</num><heading>{heading}</heading>'
        f'<content><p eId="{eid}__p_1">{body}</p></content></article>'
    )


ORIGINAL = _doc(_article("art_1", "1", "Definitions", "In this Law, the following terms apply."))


def test_reworded_body_is_not_a_structural_difference() -> None:
    """The model writes different prose every run; that must not read as drift."""
    reworded = _doc(_article("art_1", "1", "Definitions", "For the purposes of this Law:"))
    assert diff_structure(ORIGINAL, reworded) == []


def test_identical_input_reproduces_exactly() -> None:
    assert diff_structure(ORIGINAL, ORIGINAL) == []


def test_a_renumbered_eid_is_reported() -> None:
    renamed = _doc(_article("art_2", "1", "Definitions", "text"))
    differences = diff_structure(ORIGINAL, renamed)
    assert sorted(differences) == ["art_1: gone (was article)", "art_2: new (article)"]


def test_a_changed_num_is_reported() -> None:
    assert diff_structure(ORIGINAL, _doc(_article("art_1", "2", "Definitions", "text"))) == [
        "art_1: num '1' -> '2'"
    ]


def test_a_changed_heading_is_reported() -> None:
    assert diff_structure(ORIGINAL, _doc(_article("art_1", "1", "Scope", "text"))) == [
        "art_1: heading 'Definitions' -> 'Scope'"
    ]


def test_a_reparented_provision_is_reported() -> None:
    """An article escaping its chapter keeps its eId, so only nesting reveals it."""
    flattened = (
        f'<akomaNtoso xmlns="{NS}"><act><body>'
        '<chapter eId="chp_1"><num>1</num><heading>General</heading></chapter>'
        + _article("art_1", "1", "Definitions", "text")
        + "</body></act></akomaNtoso>"
    )
    assert diff_structure(ORIGINAL, flattened) == ["art_1: parent_eid 'chp_1' -> None"]


def test_reordered_provisions_are_reported() -> None:
    two = _doc(
        _article("art_1", "1", "Definitions", "a") + _article("art_2", "2", "Scope", "b"),
    )
    swapped = _doc(
        _article("art_2", "2", "Scope", "b") + _article("art_1", "1", "Definitions", "a"),
    )
    assert diff_structure(two, swapped) == ["document order changed"]


def test_a_dropped_provision_is_reported() -> None:
    two = _doc(
        _article("art_1", "1", "Definitions", "a") + _article("art_2", "2", "Scope", "b"),
    )
    assert diff_structure(two, ORIGINAL) == ["art_2: gone (was article)"]


def test_unparseable_input_is_not_reported_as_identical() -> None:
    assert diff_structure(ORIGINAL, "<not xml") is None
    assert structure_of("<not xml") is None


def test_skeleton_excludes_prose_and_keeps_containers() -> None:
    nodes = structure_of(ORIGINAL)
    assert nodes is not None
    assert [n.eid for n in nodes] == ["chp_1", "art_1"]


@pytest.mark.parametrize("missing", ["num", "heading"])
def test_absent_children_read_as_none_not_empty(missing: str) -> None:
    """A provision that never had a heading must not diff against one that lost it."""
    bare = (
        f'<akomaNtoso xmlns="{NS}"><act><body><article eId="art_1">'
        "<content><p>text</p></content></article></body></act></akomaNtoso>"
    )
    nodes = structure_of(bare)
    assert nodes is not None
    assert getattr(nodes[0], missing) is None


def test_tlc_metadata_is_not_structure() -> None:
    """<meta> names the organisations a document cites; adding one is not drift."""
    with_tlc = (
        f'<akomaNtoso xmlns="{NS}"><act><meta><references>'
        '<TLCOrganization eId="ministryOfJustice" href="/ontology/x" showAs="MoJ"/>'
        "</references></meta><body>"
        '<chapter eId="chp_1"><num>1</num><heading>General</heading>'
        + _article("art_1", "1", "Definitions", "In this Law, the following terms apply.")
        + "</chapter></body></act></akomaNtoso>"
    )
    assert diff_structure(ORIGINAL, with_tlc) == []


def _with_items(items: str) -> str:
    return _doc(
        '<article eId="art_1"><num>1</num><heading>Definitions</heading>'
        f"<content><p>In this Law:</p>{items}</content></article>"
    )


_LISTED = _with_items(
    '<blockList eId="art_1__list_1"><listIntroduction eId="art_1__intro">The following:'
    "</listIntroduction>"
    '<item eId="art_1__item_a"><num>a)</num><p>First offence.</p></item>'
    '<item eId="art_1__item_b"><num>b)</num><p>Second offence.</p></item>'
    "</blockList>"
)


def test_enumerated_items_are_provisions_not_prose() -> None:
    """codify.akn maps item to point and blockList to subparagraph, so both
    become provision rows; treating them as prose would hide a flattened list."""
    nodes = structure_of(_LISTED)
    assert nodes is not None
    assert [n.eid for n in nodes] == [
        "chp_1",
        "art_1",
        "art_1__list_1",
        "art_1__item_a",
        "art_1__item_b",
    ]


def test_a_list_flattened_into_prose_is_reported() -> None:
    flattened = _with_items("<p>a) First offence. b) Second offence.</p>")
    assert sorted(diff_structure(_LISTED, flattened)) == [
        "art_1__item_a: gone (was item)",
        "art_1__item_b: gone (was item)",
        "art_1__list_1: gone (was blockList)",
    ]


def test_a_quoted_amendment_is_not_this_document_structure() -> None:
    """A quotedStructure is a verbatim copy of the act being amended; counting
    it would let an amending act inherit the quoted law's skeleton."""
    amending = _doc(
        "<mod><quotedStructure>"
        '<article eId="art_99"><num>99</num></article>'
        "</quotedStructure></mod>" + _article("art_1", "1", "Definitions", "text")
    )
    nodes = structure_of(amending)
    assert nodes is not None
    assert "art_99" not in {n.eid for n in nodes}


def test_nested_points_are_compared_at_their_own_level() -> None:
    """Arabic acts carry more points than articles, so article-to-point loss is the
    realistic regression, not article-to-section."""
    nested = _doc(
        '<article eId="art_1"><num>1</num><content>'
        '<paragraph eId="art_1__para_1"><num>1</num>'
        '<point eId="art_1__para_1__point_a"><num>a</num><p>Text.</p></point>'
        "</paragraph></content></article>"
    )
    without_point = _doc(
        '<article eId="art_1"><num>1</num><content>'
        '<paragraph eId="art_1__para_1"><num>1</num><p>a Text.</p></paragraph>'
        "</content></article>"
    )
    assert diff_structure(nested, without_point) == ["art_1__para_1__point_a: gone (was point)"]


def test_an_attachment_schedule_is_part_of_the_skeleton() -> None:
    """Schedules carry fee and penalty tables, so losing one loses operative
    content even when every article survives."""
    with_schedule = _doc(_article("art_1", "1", "Definitions", "text")).replace(
        "</body>",
        '</body><attachment eId="att_1"><doc><mainBody>'
        '<section eId="att_1__sec_1"><num>1</num><content><p>Fees.</p></content></section>'
        "</mainBody></doc></attachment>",
    )
    differences = diff_structure(with_schedule, ORIGINAL)
    assert sorted(differences) == [
        "att_1: gone (was attachment)",
        "att_1__sec_1: gone (was section)",
    ]


def test_the_quoted_wrapper_itself_is_not_this_document_structure() -> None:
    """The amendment lifter gives <quotedStructure> its own eId, so excluding
    only its descendants still let another act's wrapper into the skeleton."""
    amending = _doc(
        '<mod><quotedStructure eId="art_1__mod_1__qstr_1">'
        '<article eId="art_99"><num>99</num></article>'
        "</quotedStructure></mod>" + _article("art_1", "1", "Definitions", "text")
    )
    nodes = structure_of(amending)
    assert nodes is not None
    assert [n.eid for n in nodes] == ["chp_1", "art_1"]
