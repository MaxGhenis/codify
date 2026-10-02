"""Counting provisions per kind, so two AKN documents can be compared for loss."""

from __future__ import annotations

from lxml import etree

from codify.akn import AKN_NS
from codify.akn.vocabulary import basic_unit_numbers, count_provisions_by_kind


def _root(body: str) -> etree._Element:
    return etree.fromstring(
        f'<akomaNtoso xmlns="{AKN_NS}"><act>'
        f'<meta><identification source="#codify"/></meta>'
        f"<body>{body}</body></act></akomaNtoso>".encode()
    )


def test_containers_and_articles_count_under_their_own_kind() -> None:
    root = _root(
        '<chapter eId="chp_1"><num>1</num>'
        '<article eId="chp_1__art_1"><num>1</num></article>'
        '<article eId="chp_1__art_2"><num>2</num></article>'
        "</chapter>"
    )
    assert count_provisions_by_kind(root) == {"chapter": 1, "article": 2}


def test_a_quoted_amendment_does_not_pad_the_count() -> None:
    """The quoted body belongs to the amended act. Counting it would let an
    amendment mask a loss in the act doing the amending."""
    root = _root(
        '<article eId="art_1"><num>1</num><content><p>'
        "<mod><quotedStructure>"
        '<article eId="q_art_9"><num>9</num></article>'
        '<article eId="q_art_10"><num>10</num></article>'
        "</quotedStructure></mod></p></content></article>"
    )
    assert count_provisions_by_kind(root) == {"article": 1}


def test_list_rendering_items_are_not_counted() -> None:
    """Bluebell derives `<item>` from a `<blockList>`, so its count tracks prose
    reflow. A re-read that recovers list punctuation must not look like a gain."""
    root = _root(
        '<article eId="art_1"><num>1</num>'
        "<blockList><item><p>a</p></item><item><p>b</p></item></blockList>"
        "</article>"
    )
    assert count_provisions_by_kind(root) == {"article": 1}


def test_a_numbered_point_is_counted() -> None:
    """A lettered point (Bahasa huruf) is a numbered unit of law, not a rendering artefact,
    so it stays in the comparison even though it shares a kind with `<item>`."""
    root = _root(
        '<article eId="art_1"><num>1</num>'
        '<point eId="art_1__point_a"><num>a</num></point>'
        "</article>"
    )
    assert count_provisions_by_kind(root)["point"] == 1


def test_a_named_hcontainer_is_law_and_is_counted() -> None:
    """A config may declare the bis article (`mukrrar`) as a named `<hcontainer>`. A tag
    test alone would drop it and hide the loss of every bis article."""
    named = _root('<hcontainer eId="art_1_bis" name="mukrrar"><num>1</num></hcontainer>')
    bare = _root('<hcontainer eId="hcontainer_1"><p>scaffolding</p></hcontainer>')
    assert count_provisions_by_kind(named) == {"hcontainer": 1}
    assert count_provisions_by_kind(bare) == {}


def test_basic_unit_numbers_are_the_article_numbers() -> None:
    root = _root(
        '<article eId="art_1"><num>1</num></article>'
        '<article eId="art_2"><num>2</num></article>'
        '<article eId="art_10"><num>10</num></article>'
    )
    assert basic_unit_numbers(root) == frozenset({"article:1", "article:2", "article:10"})


def test_digits_normalise_across_arabic_and_persian_blocks() -> None:
    """An old re-OCR renders the same article as ٩٤ one pass and ۹٤ the
    next; both must key to the same number so the set does not spuriously diverge."""
    arabic_indic = _root('<article eId="a"><num>٩٤</num></article>')
    persian = _root('<article eId="a"><num>۹٤</num></article>')
    assert (
        basic_unit_numbers(arabic_indic) == basic_unit_numbers(persian) == frozenset({"article:94"})
    )


def test_a_quoted_amendment_article_is_not_a_basic_unit() -> None:
    root = _root(
        '<article eId="art_1"><num>1</num><content><p>'
        '<mod><quotedStructure><article eId="q"><num>9</num></article>'
        "</quotedStructure></mod></p></content></article>"
    )
    assert basic_unit_numbers(root) == frozenset({"article:1"})


def test_the_structureless_fallback_section_is_not_a_basic_unit() -> None:
    """The no-anchor fallback wraps the whole body in one <section> numbered 1.
    A candidate that replaces it with real articles must not be charged for it."""
    fallback = _root(
        '<section eId="sec_1"><num>1</num><content><p>whole body</p></content></section>'
    )
    assert basic_unit_numbers(fallback) == frozenset()
    real = _root(
        '<article eId="art_1"><num>1</num></article><article eId="art_2"><num>2</num></article>'
    )
    assert not basic_unit_numbers(fallback) - basic_unit_numbers(real)  # nothing "lost"


def test_a_real_section_alongside_articles_is_a_basic_unit() -> None:
    root = _root(
        '<section eId="sec_1"><num>1</num></section><article eId="art_1"><num>1</num></article>'
    )
    assert basic_unit_numbers(root) == frozenset({"section:1", "article:1"})


def test_a_dropped_number_shows_even_when_the_count_rose() -> None:
    """act/1863's shape: the re-OCR gained articles overall but dropped specific
    numbers (296-301). A count rises and hides it; the set difference does not."""
    baseline = _root("".join(f'<article eId="a{n}"><num>{n}</num></article>' for n in (296, 297)))
    candidate = _root(
        "".join(f'<article eId="a{n}"><num>{n}</num></article>' for n in (400, 401, 402))
    )
    assert basic_unit_numbers(baseline) - basic_unit_numbers(candidate) == frozenset(
        {"article:296", "article:297"}
    )


def test_a_bis_article_is_distinct_from_its_base() -> None:
    """A `مكرر` (bis) article is its own unit of law. Keying by digits alone
    would collapse it onto the plain article of the same number, so a re-OCR that
    dropped the bis while keeping the base would slip the floor."""
    base = _root('<article eId="art_6"><num>6</num></article>')
    with_bis = _root(
        '<article eId="art_6"><num>6</num></article>'
        '<article eId="art_6_bis"><num>6 مكرر</num></article>'
    )
    assert basic_unit_numbers(with_bis) - basic_unit_numbers(base) == frozenset({"article:6مكرر"})


def test_a_named_bis_hcontainer_is_a_basic_unit() -> None:
    """A config may declare the bis article as <hcontainer name="mukrrar">; dropping it is
    dropping law, so the floor must see it. The bare Bluebell wrapper is not."""
    named = _root('<hcontainer eId="art_1_bis" name="mukrrar"><num>1</num></hcontainer>')
    bare = _root('<hcontainer eId="hcontainer_1"><p>scaffolding</p></hcontainer>')
    assert basic_unit_numbers(named) == frozenset({"mukrrar:1"})
    assert basic_unit_numbers(bare) == frozenset()
