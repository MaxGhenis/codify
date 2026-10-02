"""Projection tests, `_walk_block` + frontmatter extraction."""

from __future__ import annotations

from lxml import etree

from codify.storage.documents import (
    InlineText,
    _walk_block,
    _walk_text_block,
)


def _parse(xml: str) -> etree._Element:
    return etree.fromstring(xml)


def _inline_text(nodes: list) -> str:
    return "".join(n.text for n in nodes if isinstance(n, InlineText))


def _rendered(nodes: list) -> str:
    """Every node's text, mods included: what the reader puts on the page."""
    return "".join(
        n.text if isinstance(n, InlineText) else _rendered(getattr(n, "children", []))
        for n in nodes
    )


def test_walk_block_lifts_blocklist_items_under_content() -> None:
    """Bluebell shape: <article>/<content>/<p>+<blockList>/<item>."""
    el = _parse(
        """
        <article eId="art_3">
          <num>3</num>
          <heading>Principles</heading>
          <content>
            <p>The following principles apply:</p>
            <blockList>
              <item eId="art_3__item_1"><num>1)</num><p>first</p></item>
              <item eId="art_3__item_2"><num>2)</num><p>second</p></item>
              <item eId="art_3__item_3"><num>3)</num><p>third</p></item>
            </blockList>
          </content>
        </article>
        """.strip()
    )
    block = _walk_block(el)
    assert block.num == "3"
    assert block.heading == "Principles"
    assert _inline_text(block.intro).startswith("The following principles apply:")
    assert [b.num for b in block.blocks] == ["1)", "2)", "3)"]
    assert _inline_text(block.blocks[0].intro) == "first"


def test_walk_block_handles_nested_blocklist_inside_item() -> None:
    """An <item> may carry a sub-<blockList> for hierarchical lists."""
    el = _parse(
        """
        <item eId="i_a">
          <num>(a)</num>
          <p>outer</p>
          <blockList>
            <item eId="i_a__i_i"><num>(i)</num><p>inner</p></item>
            <item eId="i_a__i_ii"><num>(ii)</num><p>inner two</p></item>
          </blockList>
        </item>
        """.strip()
    )
    block = _walk_block(el)
    assert block.num == "(a)"
    assert _inline_text(block.intro) == "outer"
    assert [b.num for b in block.blocks] == ["(i)", "(ii)"]


def test_walk_block_splits_repeated_p_blocklist_into_sibling_groups() -> None:
    """The Bluebell pattern <p>1.</p><list><p>2.</p><list><p>3.</p><list>
    emits four sibling synthetic blocks so items render at a uniform
    depth, not crammed into the wrap_up of the first paragraph."""
    el = _parse(
        """
        <article eId="art_5">
          <content>
            <p>1. lead one</p>
            <blockList>
              <item><num>(a)</num><p>one-a</p></item>
              <item><num>(b)</num><p>one-b</p></item>
            </blockList>
            <p>2. lead two</p>
            <blockList>
              <item><num>(a)</num><p>two-a</p></item>
            </blockList>
            <p>3. lead three</p>
          </content>
        </article>
        """.strip()
    )
    block = _walk_block(el)
    # Sole groups fold into the outer block, but multi-group articles do
    # not; verify groups are nested as siblings with their own items.
    assert block.intro == []
    assert len(block.blocks) == 3
    leads = [_inline_text(g.intro) for g in block.blocks]
    assert leads == ["1. lead one", "2. lead two", "3. lead three"]
    assert [b.num for b in block.blocks[0].blocks] == ["(a)", "(b)"]
    assert [b.num for b in block.blocks[1].blocks] == ["(a)"]
    assert block.blocks[2].blocks == []


def test_walk_block_prunes_persisted_subparagraphs() -> None:
    """When subparagraphs are their own persisted Provision rows, the walker
    must skip them so the SPA doesn't double-render (once nested inside the
    paragraph's block tree, once at the section-provisions level).
    """
    el = _parse(
        """
        <paragraph eId="para_1">
          <num>1.</num>
          <intro><p>lead</p></intro>
          <subparagraph eId="para_1__sub_a">
            <num>(a)</num><content><p>one</p></content>
          </subparagraph>
          <subparagraph eId="para_1__sub_b">
            <num>(b)</num><content><p>two</p></content>
          </subparagraph>
        </paragraph>
        """.strip()
    )
    block = _walk_block(el, stop_at_eids=frozenset({"para_1__sub_a", "para_1__sub_b"}))
    assert _inline_text(block.intro) == "lead"
    # Both subparagraphs pruned; only the paragraph's own lead-in survives.
    assert block.blocks == []


def test_walk_block_canonical_paragraph_with_intro_and_wrapup() -> None:
    """The AKN canonical <paragraph>/<intro>/<subparagraph>/<wrapUp> path
    still works through the new unified walker."""
    el = _parse(
        """
        <paragraph eId="para_1">
          <num>1.</num>
          <intro><p>lead</p></intro>
          <subparagraph eId="para_1__sub_a">
            <num>(a)</num><content><p>one</p></content>
          </subparagraph>
          <subparagraph eId="para_1__sub_b">
            <num>(b)</num><content><p>two</p></content>
          </subparagraph>
          <wrapUp><p>tail</p></wrapUp>
        </paragraph>
        """.strip()
    )
    block = _walk_block(el)
    assert _inline_text(block.intro) == "lead"
    assert [b.num for b in block.blocks] == ["(a)", "(b)"]
    assert _inline_text(block.wrap_up) == "tail"


def test_walk_block_blocklist_intro_and_wrapup() -> None:
    """<listIntroduction> appends to intro; <listWrapUp> to wrap_up."""
    el = _parse(
        """
        <article eId="art_1">
          <content>
            <blockList>
              <listIntroduction><p>before</p></listIntroduction>
              <item eId="art_1__item_1"><num>1)</num><p>x</p></item>
              <listWrapUp><p>after</p></listWrapUp>
            </blockList>
          </content>
        </article>
        """.strip()
    )
    block = _walk_block(el)
    assert _inline_text(block.intro) == "before"
    assert [b.num for b in block.blocks] == ["1)"]
    assert _inline_text(block.wrap_up) == "after"


def test_walk_block_projects_a_table_with_spans() -> None:
    """<table> lands as a synthetic akn_type="table" block, not dropped, and
    carries colspan/rowspan through to DocumentTableCell."""
    el = _parse(
        """
        <article eId="art_4">
          <content>
            <p>Fees apply as follows:</p>
            <table eId="art_4__table_1">
              <tr>
                <th colspan="2"><p eId="art_4__table_1__p_1">Fee schedule</p></th>
              </tr>
              <tr>
                <th><p eId="art_4__table_1__p_2">Item</p></th>
                <th><p eId="art_4__table_1__p_3">Fee</p></th>
              </tr>
              <tr>
                <td rowspan="2"><p eId="art_4__table_1__p_4">Filing</p></td>
                <td><p eId="art_4__table_1__p_5">10</p></td>
              </tr>
              <tr>
                <td><p eId="art_4__table_1__p_6">20</p></td>
              </tr>
            </table>
          </content>
        </article>
        """.strip()
    )
    block = _walk_block(el)
    table_blocks = [b for b in block.blocks if b.akn_type == "table"]
    assert len(table_blocks) == 1
    table = table_blocks[0].table
    assert table is not None
    assert table.akn_eid == "art_4__table_1"
    assert len(table.rows) == 4
    caption = table.rows[0].cells[0]
    assert caption.colspan == 2 and caption.header
    assert caption.akn_eid == "art_4__table_1__p_1"
    assert _inline_text(caption.content) == "Fee schedule"
    filing = table.rows[2].cells[0]
    assert filing.akn_eid == "art_4__table_1__p_4"
    assert filing.rowspan == 2 and not filing.header


def test_walk_block_a_table_is_a_group_boundary() -> None:
    """<p>before</p><table/><p>after</p>: "after" must land in its own block
    after the table, not fold back into the outer intro that renders before
    `blocks` (which would put both prose runs above the table)."""
    el = _parse(
        """
        <article eId="art_5">
          <content>
            <p>before</p>
            <table eId="art_5__table_1">
              <tr><td><p eId="art_5__table_1__p_1">cell</p></td></tr>
            </table>
            <p eId="art_5__p_2">after</p>
          </content>
        </article>
        """.strip()
    )
    block = _walk_block(el)
    assert _inline_text(block.intro) == "before"
    assert [b.akn_type for b in block.blocks] == ["table", "p"]
    assert _inline_text(block.blocks[1].intro) == "after"
    # Parity with the client (akn-to-version-document.ts), which always keeps
    # a paragraph's own eid: a lone <p> after the table must keep its real
    # eid rather than the synthetic block's usual "".
    assert block.blocks[1].akn_eid == "art_5__p_2"


def test_walk_block_a_merged_multi_paragraph_group_has_no_single_eid() -> None:
    """Two <p>s merged into one group have no single citable identity, so
    the synthetic block still falls back to "" \u2014 the single-paragraph fix
    must not apply when there's more than one paragraph to name."""
    el = _parse(
        """
        <article eId="art_7">
          <content>
            <table eId="art_7__table_1">
              <tr><td><p eId="art_7__table_1__p_1">cell</p></td></tr>
            </table>
            <p eId="art_7__p_2">one</p>
            <p eId="art_7__p_3">two</p>
          </content>
        </article>
        """.strip()
    )
    block = _walk_block(el)
    merged = block.blocks[1]
    assert merged.akn_type == "p"
    assert merged.akn_eid == ""
    assert _inline_text(merged.intro) == "one\n\ntwo"


def test_walk_text_block_joins_paragraphs_with_blank_line() -> None:
    el = _parse(
        """
        <preamble>
          <p>The Parliament hereby adopts this Law:</p>
          <p>Second paragraph.</p>
        </preamble>
        """.strip()
    )
    nodes = _walk_text_block(el)
    text = _inline_text(nodes)
    assert "The Parliament hereby adopts this Law:" in text
    assert "Second paragraph." in text
    assert "\n\n" in text


def test_walk_text_block_recurses_into_recitals_and_formula() -> None:
    el = _parse(
        """
        <preamble>
          <formula><p>Be it enacted:</p></formula>
          <recitals>
            <recital><p>Whereas A;</p></recital>
            <recital><p>And whereas B;</p></recital>
          </recitals>
        </preamble>
        """.strip()
    )
    text = _inline_text(_walk_text_block(el))
    assert "Be it enacted:" in text
    assert "Whereas A;" in text
    assert "And whereas B;" in text


def test_a_javascript_href_never_reaches_the_reading_surface() -> None:
    """An href on an ingested `ref` is publisher-supplied, so it is untrusted in
    the same way its text is. The projection keeps the text and drops the
    target, which the reader then renders as plain emphasis rather than a live
    link."""
    from codify.storage.documents import InlineCitation, InlineRef, _push_inline

    out: list = []
    _push_inline(
        _parse(
            '<ref xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0" '
            "href=\"javascript:fetch('/api/proxy/admin/glossary')\">see s.4</ref>"
        ),
        out,
    )
    assert isinstance(out[0], InlineRef)
    assert out[0].text == "see s.4"
    assert out[0].href is None

    out = []
    _push_inline(
        _parse(
            '<rref xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0" '
            'href="data:text/html;base64,PHNjcmlwdD4=">annex</rref>'
        ),
        out,
    )
    assert isinstance(out[0], InlineCitation)
    assert out[0].href is None
    # The citation pill still shows what the document declared.
    assert out[0].uri.startswith("data:")


def test_ordinary_hrefs_survive() -> None:
    from codify.storage.documents import InlineRef, _push_inline

    ns = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
    for href in ("#art_5", "/akn/xz/act/2014/4", "https://example.test/x", "mailto:a@b.test"):
        out: list = []
        _push_inline(_parse(f'<ref xmlns="{ns}" href="{href}">x</ref>'), out)
        assert isinstance(out[0], InlineRef)
        assert out[0].href == href, href


def test_a_quoted_phrase_stays_in_the_sentence_that_introduces_it() -> None:
    """The amending clause is one sentence: the words that introduce an
    insertion, the words inserted, and the punctuation that closes it."""
    el = _parse(
        """
        <subparagraph eId="sec_2__subpara_a">
          <num>(a)</num>
          <content>
            <p>in the heading, after "Money laundering" insert</p>
            <mod eId="sec_2__subpara_a__mod_1">
              <quotedStructure eId="q1">
                <p eId="q1__p_1">and terrorist financing</p>
              </quotedStructure>
            </mod>
            <p>; and</p>
          </content>
        </subparagraph>
        """.strip()
    )
    block = _walk_block(el)

    assert not block.blocks, "a phrase is not a structure and gets no block of its own"
    kinds = [n.kind for n in block.intro]
    assert kinds == ["text", "mod", "text"], kinds
    assert "\n" not in _inline_text(block.intro), "the closing punctuation finishes the sentence"
    # The space between the introducing words and the insertion: a block
    # boundary carries none, so the projection has to put it back.
    assert block.intro[0].text.endswith(" "), "insert[the words] is not a sentence"


def test_a_quoted_structure_keeps_its_own_hierarchy() -> None:
    el = _parse(
        """
        <paragraph eId="sec_2__para_2">
          <num>(2)</num>
          <content>
            <p>In regulation 4, substitute—</p>
            <mod eId="sec_2__para_2__mod_1">
              <quotedStructure eId="q2">
                <paragraph eId="q2__para_1">
                  <num>(1)</num><content><p>A record must be kept.</p></content>
                </paragraph>
                <paragraph eId="q2__para_2">
                  <num>(2)</num><content><p>For five years.</p></content>
                </paragraph>
              </quotedStructure>
            </mod>
          </content>
        </paragraph>
        """.strip()
    )
    block = _walk_block(el)

    assert [b.akn_type for b in block.blocks] == ["quote"]
    quote = block.blocks[0]
    # Keyed on the mod: that is the element carrying an eId a citation can name.
    assert quote.akn_eid == "sec_2__para_2__mod_1"
    assert [(b.num, _inline_text(b.intro)) for b in quote.blocks] == [
        ("(1)", "A record must be kept."),
        ("(2)", "For five years."),
    ]


def test_an_amendment_is_never_silently_dropped() -> None:
    """The defect this projection had: `<mod>` matched no branch of the walker,
    so an amending provision rendered with its amendment missing."""
    el = _parse(
        """
        <subparagraph eId="s__a">
          <num>(a)</num>
          <content>
            <p>substitute</p>
            <mod eId="s__a__mod_1">
              <quotedStructure eId="q3"><p eId="q3__p_1">Treasury</p></quotedStructure>
            </mod>
          </content>
        </subparagraph>
        """.strip()
    )
    block = _walk_block(el)

    assert _rendered(block.intro) == "substitute Treasury"


def test_a_pretty_printed_quote_does_not_break_the_line() -> None:
    """The source's own indentation is not part of the law: the reader renders
    newlines, so a quote indented in the XML reads as three broken lines."""
    el = _parse(
        """
        <subparagraph eId="s">
          <num>(a)</num>
          <content>
            <p>insert</p>
            <mod eId="m">
              <quotedStructure eId="q">
                <p eId="q__p_1">and terrorist financing</p>
              </quotedStructure>
            </mod>
            <p>; and</p>
          </content>
        </subparagraph>
        """.strip()
    )
    block = _walk_block(el)

    assert _rendered(block.intro) == "insert and terrorist financing; and"


def test_a_quoted_text_amendment_survives() -> None:
    """A `<mod>` can hold `<quotedText>` instead of a structure. Nothing else
    carries those words, so dropping the mod drops the amendment."""
    el = _parse(
        """
        <subparagraph eId="s2">
          <num>(b)</num>
          <content>
            <p>substitute</p>
            <mod eId="m2"><quotedText>alpha</quotedText></mod>
          </content>
        </subparagraph>
        """.strip()
    )
    block = _walk_block(el)

    assert _rendered(block.intro) == "substitute alpha"
    assert [n.kind for n in block.intro] == ["text", "mod"]


def test_a_heading_makes_a_quote_a_structure() -> None:
    """One paragraph beside a heading is not a phrase: inlining it would drop
    the heading, which is the part naming what the amendment replaces."""
    el = _parse(
        """
        <subparagraph eId="s4">
          <num>(d)</num>
          <content>
            <p>substitute</p>
            <mod eId="m4">
              <quotedStructure eId="q5">
                <heading>Records</heading>
                <p eId="q5__p_1">Keep them.</p>
              </quotedStructure>
            </mod>
          </content>
        </subparagraph>
        """.strip()
    )
    block = _walk_block(el)

    assert [b.akn_type for b in block.blocks] == ["quote"]
    assert block.blocks[0].heading == "Records"


def test_the_spaces_between_marked_up_words_survive_the_quote() -> None:
    """Whitespace between inline elements is its own text node. Normalising
    each node alone deletes it, and the words run together."""
    el = _parse(
        """
        <subparagraph eId="s5">
          <num>(e)</num>
          <content>
            <p>insert</p>
            <mod eId="m5">
              <quotedStructure eId="q6">
                <p eId="q6__p_1">the <b>new</b> wording</p>
              </quotedStructure>
            </mod>
          </content>
        </subparagraph>
        """.strip()
    )
    block = _walk_block(el)

    assert _rendered(block.intro) == "insert the new wording"


def test_a_quoted_section_reaches_the_reader() -> None:
    """The lifter's canonical shape: a whole section substituted for another.
    The quote walker is its only reader, so a container it cannot model is
    a bordered strip with the new law missing from it."""
    el = _parse(
        """
        <section eId="sec_3">
          <num>3</num>
          <content>
            <p>substitute—</p>
            <mod eId="sec_3__mod_1">
              <quotedStructure eId="q">
                <section eId="q__sec_14">
                  <num>14.</num>
                  <heading>Records</heading>
                  <subsection eId="q__sec_14__subsec_1">
                    <num>(1)</num><content><p>Keep them.</p></content>
                  </subsection>
                </section>
              </quotedStructure>
            </mod>
          </content>
        </section>
        """.strip()
    )
    block = _walk_block(el)

    quote = block.blocks[0]
    assert quote.akn_type == "quote"
    assert quote.blocks, "the substituted section is the whole of the amendment"
    inner = quote.blocks[0]
    assert (inner.num, inner.heading) == ("14.", "Records")
    assert _rendered(inner.blocks[0].intro) == "Keep them."


def test_a_quote_the_walker_cannot_model_still_shows_its_words() -> None:
    el = _parse(
        """
        <subparagraph eId="s6">
          <num>(f)</num>
          <content>
            <p>substitute—</p>
            <mod eId="m6">
              <quotedStructure eId="q7">
                <foreign>words nobody modelled</foreign>
                <foreign>and more of them</foreign>
              </quotedStructure>
            </mod>
          </content>
        </subparagraph>
        """.strip()
    )
    block = _walk_block(el)

    assert _rendered(block.blocks[0].intro) == "words nobody modelled and more of them"


def test_an_inline_mod_marks_only_the_words_it_amended() -> None:
    """The published shape: the whole instruction sits in the `<mod>`, and only
    its quoted child was inserted. Marking all of it brackets the instruction."""
    el = _parse(
        """
        <level eId="lvl_b">
          <num>(b)</num>
          <content>
            <p><mod>in paragraph (f), after "(r)" insert
              <quotedText>"to (t)"</quotedText>.</mod></p>
          </content>
        </level>
        """.strip()
    )
    block = _walk_block(el)

    kinds = [n.kind for n in block.intro]
    assert kinds == ["text", "mod", "text"], kinds
    assert block.intro[0].text.startswith("in paragraph (f)")
    assert "insert" in block.intro[0].text
    assert _rendered(block.intro[1].children) == '"to (t)"'
    assert block.intro[2].text == "."


def test_a_quote_keeps_words_the_walker_cannot_model() -> None:
    """Nothing else carries a quote's words, so an element with no branch here
    cannot be dropped merely because a sibling had one."""
    el = _parse(
        """
        <subparagraph eId="s7">
          <num>(g)</num>
          <content>
            <p>substitute—</p>
            <mod eId="m7">
              <quotedStructure eId="q8">
                <p eId="q8__p_1">known</p><foreign>suffix</foreign>
              </quotedStructure>
            </mod>
          </content>
        </subparagraph>
        """.strip()
    )
    quote = _walk_block(el).blocks[0]

    assert _rendered(quote.intro) == "known suffix"


def test_a_heading_only_quote_is_not_printed_twice() -> None:
    el = _parse(
        """
        <subparagraph eId="s8">
          <num>(h)</num>
          <content>
            <p>substitute—</p>
            <mod eId="m8">
              <quotedStructure eId="q9"><heading>Records</heading></quotedStructure>
            </mod>
          </content>
        </subparagraph>
        """.strip()
    )
    quote = _walk_block(el).blocks[0]

    assert quote.heading == "Records"
    assert _rendered(quote.intro) == ""


def test_prose_after_a_quoted_section_stays_below_it() -> None:
    """A quote's intro renders before its blocks, so a trailing paragraph that
    folds into the intro is printed above the section it follows."""
    el = _parse(
        """
        <subparagraph eId="s9">
          <num>(i)</num>
          <content>
            <p>substitute—</p>
            <mod eId="m9">
              <quotedStructure eId="q10">
                <section eId="q10__sec_1">
                  <num>1.</num><heading>Records</heading>
                  <content><p>Keep them.</p></content>
                </section>
                <p eId="q10__p_tail">tail</p>
              </quotedStructure>
            </mod>
          </content>
        </subparagraph>
        """.strip()
    )
    quote = _walk_block(el).blocks[0]

    assert _rendered(quote.intro) == ""
    assert [b.num for b in quote.blocks] == ["1.", None]
    assert _rendered(quote.blocks[1].intro) == "tail"


def test_a_hierarchy_quoted_mid_sentence_becomes_a_block() -> None:
    """The published shape: the instruction, the provision being inserted, and
    the punctuation after it, all inside one `<mod>` inside one `<p>`. Inlining
    the structure prints the inserted provision's number as prose."""
    el = _parse(
        """
        <level eId="lvl_c">
          <num>(c)</num>
          <content>
            <p><mod>after paragraph (d), insert—<quotedStructure eId="q">
              <level eId="q__lvl_1"><num>(da)</num>
                <content><p>the Chartered Institute of Legal Executives;</p></content>
              </level>
            </quotedStructure><inline name="appendText">;</inline></mod></p>
          </content>
        </level>
        """.strip()
    )
    block = _walk_block(el)

    assert _rendered(block.intro) == "after paragraph (d), insert—"
    quote = block.blocks[0]
    assert quote.akn_type == "quote"
    assert [(b.num, _rendered(b.intro)) for b in quote.blocks] == [
        ("(da)", "the Chartered Institute of Legal Executives;")
    ]
    # The punctuation closing the instruction follows the quote, not the intro.
    assert _rendered(block.blocks[1].intro) == ";"


def test_a_phrase_quoted_mid_sentence_is_still_a_phrase() -> None:
    el = _parse(
        """
        <level eId="lvl_d">
          <num>(d)</num>
          <content>
            <p><mod>after "(r)" insert <quotedText>"to (t)"</quotedText>.</mod></p>
          </content>
        </level>
        """.strip()
    )
    block = _walk_block(el)

    assert not block.blocks, "a phrase does not become a block"
    assert [n.kind for n in block.intro] == ["text", "mod", "text"]


def test_quoted_text_beside_a_structure_stays_in_the_sentence() -> None:
    """One mod can carry both. Only the structure is promoted; the words next
    to it are still words and keep their markers."""
    el = _parse(
        """
        <level eId="lvl_e">
          <num>(e)</num>
          <content>
            <p><mod>for <quotedText>"old"</quotedText> substitute—<quotedStructure eId="q">
              <level eId="q__lvl_1"><num>(a)</num><content><p>new text;</p></content></level>
            </quotedStructure></mod></p>
          </content>
        </level>
        """.strip()
    )
    block = _walk_block(el)

    assert [n.kind for n in block.intro] == ["text", "mod", "text"]
    assert _rendered(block.intro[1].children) == '"old"'
    assert block.blocks[0].akn_type == "quote"


def test_a_quote_ending_the_paragraph_leaves_no_empty_row() -> None:
    """Nothing follows the insertion, so there is no sentence to resume. An
    opened group with nothing in it renders as a blank row under the quote."""
    el = _parse(
        """
        <level eId="lvl_f">
          <num>(f)</num>
          <content>
            <p><mod>insert—<quotedStructure eId="q">
              <level eId="q__lvl_1"><num>(da)</num><content><p>new;</p></content></level>
            </quotedStructure></mod></p>
          </content>
        </level>
        """.strip()
    )
    block = _walk_block(el)

    assert _rendered(block.intro) == "insert—"
    assert [b.akn_type for b in block.blocks] == ["quote"]


def test_walk_block_keeps_the_row_eid_a_citation_anchors_on() -> None:
    """A row search cites `(version, table, row)` by the `<tr>` eId, and the
    reader scrolls to whatever carries that id. Dropping it here leaves the
    citation with nothing to land on."""
    el = _parse(
        """
        <article eId="art_5">
          <content>
            <table eId="art_5__table_1">
              <tr eId="art_5__table_1__tr_1">
                <th><p>Old</p></th><th><p>New</p></th>
              </tr>
              <tr eId="art_5__table_1__tr_2">
                <td><p>Article 12</p></td><td><p>Article 30</p></td>
              </tr>
            </table>
          </content>
        </article>
        """.strip()
    )
    table = next(b.table for b in _walk_block(el).blocks if b.akn_type == "table")
    assert table is not None
    assert [r.akn_eid for r in table.rows] == [
        "art_5__table_1__tr_1",
        "art_5__table_1__tr_2",
    ]


def test_a_minted_row_eid_matches_the_one_the_row_resolver_cites() -> None:
    """Search cites the resolver's id. If the reader mints a different one, the
    citation opens the law and lands nowhere, silently."""
    from codify.tables.resolve import resolve_tables

    xml = """
        <akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0"><act>
        <body><section eId="sec_1"><content>
          <table eId="t_1">
            <tr><th><p>Old</p></th><th><p>New</p></th></tr>
            <tr><td><p>Article 12</p></td><td><p>Article 30</p></td></tr>
            <tr><td><p>Article 13</p></td><td><p>Article 31</p></td></tr>
          </table>
        </content></section></body></act></akomaNtoso>
    """.strip()

    cited = {r.akn_eid for r in resolve_tables(xml)}
    el = _parse(
        """
        <article eId="art_9"><content>
          <table eId="t_1">
            <tr><th><p>Old</p></th><th><p>New</p></th></tr>
            <tr><td><p>Article 12</p></td><td><p>Article 30</p></td></tr>
            <tr><td><p>Article 13</p></td><td><p>Article 31</p></td></tr>
          </table>
        </content></article>
        """.strip()
    )
    table = next(b.table for b in _walk_block(el).blocks if b.akn_type == "table")
    assert table is not None
    rendered = {r.akn_eid for r in table.rows}

    assert cited, "the resolver produced no rows to cite"
    assert cited <= rendered, f"cited ids the reader never renders: {cited - rendered}"
