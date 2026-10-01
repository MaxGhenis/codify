# ruff: noqa: E501  # AKN XML test fixtures: line wraps would change tested whitespace
"""Tests for lift_amendment_markup.

The lift is pure XML transformation: every <block name="quote"> with an
<embeddedStructure> containing a structural AKN child becomes a <mod>.
No heading-regex gate, classification is the amendment_parser agent's job.
"""

from __future__ import annotations

import pytest
from lxml import etree

from codify.akn import AKN_NS, NS
from codify.pipeline.enrich.amendments import lift_amendment_markup


def _act(body_inner: str) -> str:
    return f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act>
    <body>{body_inner}</body>
  </act>
</akomaNtoso>
'''


def _quote_block(inner: str) -> str:
    return f"""<block name="quote">
      <embeddedStructure>{inner}</embeddedStructure>
    </block>"""


class TestHeadingIndependence:
    """Drafting convention is not a gate. Any quote block with structural
    content is lifted regardless of the section heading wording."""

    @pytest.mark.parametrize(
        "heading",
        [
            # UK/OPC
            "Amendment of section 5 of the principal Act",
            "Substitution of section 5",
            # Commonwealth revised-edition style (Cap. numbers)
            "Repeal and replacement of section 14 of Cap. 167",
            "Insertion of new section 79A",
            # Targets other than sections
            "Repeal and replacement of PART XI of Cap. 167",
            "Amendment of Schedule 1",
            "Repeal and replacement of long title",
            # Civil law
            "L'article 5 est ainsi modifié",
            "§5 wird wie folgt gefasst",
            # Pathological: no heading at all
            "",
            # Pathological: heading that says nothing about amendments
            "Short title",
        ],
    )
    def test_heading_wording_does_not_gate_lift(self, heading: str) -> None:
        body = f"""
        <section eId="sec_3">
          <num>3.</num>
          <heading>{heading}</heading>
          <content>
            <p>The principal Act is amended…</p>
            {_quote_block(f'<section xmlns="{AKN_NS}" eId="sec_5"><num>5.</num><content><p>Replacement.</p></content></section>')}
          </content>
        </section>
        """
        lifted = lift_amendment_markup(_act(body))
        assert "<akn:mod" in lifted or "<mod" in lifted
        assert 'eId="sec_3__mod_1"' in lifted
        assert "quotedStructure" in lifted


class TestEmbeddedStructureGate:
    """The only gate is: does the quote block contain an <embeddedStructure>?
    Content shape (nested provisions vs bare prose) is the agent's problem."""

    @pytest.mark.parametrize(
        "child",
        [
            # Nested provision, drafter re-encoded replacement as AKN structure
            '<section xmlns="{ns}" eId="sec_5"><num>5.</num><content><p>X</p></content></section>',
            '<subsection xmlns="{ns}" eId="sec_5__subsec_2"><num>(2)</num><content><p>X</p></content></subsection>',
            # Bare prose, the Police (Amendment) Act 2025 case, where the LLM
            # transcribed quoted text as loose paragraphs rather than nested AKN.
            '<p xmlns="{ns}">Replacement sentence one.</p><p xmlns="{ns}">Replacement sentence two.</p>',
            '<p xmlns="{ns}">&#8220;An Act to provide for the Atlantis Police Service.&#8221;</p>',
        ],
    )
    def test_quote_block_with_embedded_structure_is_lifted(self, child: str) -> None:
        body = f"""
        <section eId="sec_3">
          <content>
            {_quote_block(child.format(ns=AKN_NS))}
          </content>
        </section>
        """
        lifted = lift_amendment_markup(_act(body))
        assert "quotedStructure" in lifted
        assert 'eId="sec_3__mod_1"' in lifted

    def test_quote_block_without_embedded_structure_is_ignored(self) -> None:
        # Bluebell didn't wrap this quote in an embeddedStructure, not our
        # concern, leave it alone.
        body = """
        <section eId="sec_3">
          <content>
            <block name="quote"><p>just a plain quote</p></block>
          </content>
        </section>
        """
        lifted = lift_amendment_markup(_act(body))
        assert "<akn:mod" not in lifted and "<mod " not in lifted


class TestEidScoping:
    def test_sequential_eids_within_section(self) -> None:
        body = f"""
        <section eId="sec_8">
          <content>
            <p>Section 14 is amended as follows.</p>
            {_quote_block(f'<section xmlns="{AKN_NS}" eId="sec_14"><content><p>A</p></content></section>')}
            <p>And further.</p>
            {_quote_block(f'<section xmlns="{AKN_NS}" eId="sec_14"><content><p>B</p></content></section>')}
          </content>
        </section>
        """
        lifted = lift_amendment_markup(_act(body))
        assert 'eId="sec_8__mod_1"' in lifted
        assert 'eId="sec_8__mod_2"' in lifted

    def test_sibling_sections_do_not_collide(self) -> None:
        body = f"""
        <section eId="sec_8">
          <content>
            {_quote_block(f'<section xmlns="{AKN_NS}" eId="sec_14"><content><p>A</p></content></section>')}
          </content>
        </section>
        <section eId="sec_9">
          <content>
            {_quote_block(f'<section xmlns="{AKN_NS}" eId="sec_15"><content><p>B</p></content></section>')}
          </content>
        </section>
        """
        lifted = lift_amendment_markup(_act(body))
        assert 'eId="sec_8__mod_1"' in lifted
        assert 'eId="sec_9__mod_1"' in lifted


class TestReprefix:
    def test_child_eids_rebased_under_quoted_structure(self) -> None:
        body = f"""
        <section eId="sec_3">
          <content>
            {
            _quote_block(f'''<section xmlns="{AKN_NS}" eId="sec_14">
              <subsection eId="sec_14__subsec_1"><num>(1)</num><content><p>X</p></content></subsection>
            </section>''')
        }
          </content>
        </section>
        """
        lifted = lift_amendment_markup(_act(body))
        root = etree.fromstring(lifted.encode("utf-8"))
        # Descendant subsection should be rebased under the new quotedStructure eId.
        subsec = root.findall(".//akn:subsection", NS)
        assert len(subsec) == 1
        eid = subsec[0].get("eId", "")
        assert eid.startswith("sec_3__mod_1__qstr_1"), f"got {eid!r}"
