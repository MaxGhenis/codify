"""Tests for hcontainer post-processing."""

from __future__ import annotations

import pytest
from lxml import etree

from codify.akn import AKN_NS, NS
from codify.jurisdictions import JurisdictionConfigError
from codify.pipeline.enrich.hcontainers import postprocess_hcontainers


def _act_with_subsection(heading: str) -> str:
    return f'''<?xml version="1.0"?>
<akomaNtoso xmlns="{AKN_NS}">
  <act>
    <body>
      <section eId="sec_1">
        <num>1.</num>
        <heading>Main provision</heading>
        <subsection eId="sec_1__subsec_1">
          <heading>{heading}</heading>
          <content><p>Some text.</p></content>
        </subsection>
      </section>
    </body>
  </act>
</akomaNtoso>
'''


def _act_class(*containers: dict) -> dict:
    return {"document_classes": {"act": {"label": "Act", "hcontainers": list(containers)}}}


_EXPLANATION = {"local_term": "Explanation", "name": "explanation", "requires_postprocessing": True}
_TABSAREH = {"local_term": "Tabsareh (Note/Proviso)", "name": "tabsareh"}


@pytest.fixture(autouse=True)
def _declared_hcontainers(monkeypatch, tmp_path):
    """Synthetic configs, so the renaming is tested without any shipped profile
    having to declare these containers."""
    from codify.jurisdictions import try_load_config
    from tests.config_fixtures import isolated_configs

    configs = {
        "in": _act_class(_EXPLANATION),
        "pk": _act_class(_EXPLANATION),
        "ir": _act_class(_TABSAREH),
    }
    with isolated_configs(monkeypatch, tmp_path / "data" / "jurisdictions", configs):
        yield
    try_load_config.cache_clear()


class TestPostprocessHcontainers:
    def test_india_explanation_renamed(self):
        out = postprocess_hcontainers(
            _act_with_subsection("Explanation"),
            country="in",
            doctype="act",
        )
        root = etree.fromstring(out.encode("utf-8"))
        hcs = root.findall(".//akn:hcontainer", NS)
        assert len(hcs) == 1
        assert hcs[0].get("name") == "explanation"

    def test_pakistan_explanation(self):
        out = postprocess_hcontainers(
            _act_with_subsection("Explanation"),
            country="pk",
            doctype="act",
        )
        root = etree.fromstring(out.encode("utf-8"))
        hcs = root.findall(".//akn:hcontainer", NS)
        assert len(hcs) == 1

    def test_iran_tabsareh(self):
        out = postprocess_hcontainers(
            _act_with_subsection("Tabsareh (Note/Proviso)"),
            country="ir",
            doctype="act",
        )
        root = etree.fromstring(out.encode("utf-8"))
        hcs = root.findall(".//akn:hcontainer", NS)
        assert len(hcs) == 1
        assert hcs[0].get("name") == "tabsareh"

    def test_no_match_leaves_element_alone(self):
        out = postprocess_hcontainers(
            _act_with_subsection("Definitions"),
            country="in",
            doctype="act",
        )
        root = etree.fromstring(out.encode("utf-8"))
        # Should still be a subsection, not renamed
        subs = root.findall(".//akn:subsection", NS)
        assert len(subs) == 1
        hcs = root.findall(".//akn:hcontainer", NS)
        assert len(hcs) == 0

    def test_an_absent_config_raises(self):
        """Distinct from the configured no-op below, which is a jurisdiction
        that has a config and declares no hcontainers."""
        with pytest.raises(JurisdictionConfigError):
            postprocess_hcontainers(
                _act_with_subsection("Explanation"), country="zz", doctype="act"
            )

    def test_country_without_hcontainers_noop(self):
        out = postprocess_hcontainers(
            _act_with_subsection("Explanation"),
            country="gb",  # gb declares no hcontainers
            doctype="act",
        )
        root = etree.fromstring(out.encode("utf-8"))
        assert root.findall(".//akn:hcontainer", NS) == []

    def test_case_insensitive_match(self):
        out = postprocess_hcontainers(
            _act_with_subsection("explanation"),  # lowercase
            country="in",
            doctype="act",
        )
        root = etree.fromstring(out.encode("utf-8"))
        assert len(root.findall(".//akn:hcontainer", NS)) == 1

    def test_idempotent(self):
        once = postprocess_hcontainers(
            _act_with_subsection("Explanation"),
            country="in",
            doctype="act",
        )
        twice = postprocess_hcontainers(once, country="in", doctype="act")
        root = etree.fromstring(twice.encode("utf-8"))
        assert len(root.findall(".//akn:hcontainer", NS)) == 1
