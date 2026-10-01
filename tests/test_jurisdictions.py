from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from codify.jurisdictions import (
    JURISDICTIONS_DIR,
    AknSource,
    JurisdictionConfig,
    JurisdictionConfigError,
    SourceAdapter,
    canonicalise_body,
    load_config,
    load_registry,
    try_load_config,
)
from codify.open_wheel import ships_in_open_wheel

# Relative to this file, not the cwd: the monorepo path only resolves when the
# suite is run from the repository root, and does not exist in the open tree.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _discover_codes() -> list[str]:
    return sorted(p.parent.name for p in JURISDICTIONS_DIR.glob("*/config.json"))


_CODES = _discover_codes()


def test_corpus_is_present() -> None:
    assert _CODES, f"No jurisdictions found under {JURISDICTIONS_DIR}"
    assert {"fi", "gb", "ie", "it", "nz"} <= set(_CODES)
    assert all(ships_in_open_wheel(JURISDICTIONS_DIR / code / "config.json") for code in _CODES)


@pytest.mark.parametrize("code", _CODES, ids=_CODES)
def test_config_parses(code: str) -> None:
    try_load_config.cache_clear()
    config = load_config(code)
    assert isinstance(config, JurisdictionConfig)
    assert config.code == code


def test_load_config_caches() -> None:
    try_load_config.cache_clear()
    first = load_config(_CODES[0])
    second = load_config(_CODES[0])
    assert first is second


def test_try_load_config_unknown_returns_none() -> None:
    """The probing form. `load_config` raises for the same input, which is what
    makes it the safe default."""
    assert try_load_config("zz-not-a-code") is None
    with pytest.raises(JurisdictionConfigError):
        load_config("zz-not-a-code")


def test_canonicalise_known_alias_resolves() -> None:
    # Supranational alias map is loaded from data/jurisdictions/supranational_bodies.json.
    # Pick a body that's present and a documented alias for it.
    bodies_path = JURISDICTIONS_DIR / "supranational_bodies.json"
    bodies = json.loads(bodies_path.read_text())["bodies"]
    aliased = next((b for b in bodies if b.get("aliases")), None)
    if aliased is None:
        pytest.skip("supranational_bodies.json has no aliased entries")
    alias = aliased["aliases"][0]
    assert canonicalise_body(alias) == aliased["code"]
    assert canonicalise_body(alias.upper()) == aliased["code"]


def test_canonicalise_unknown_passes_through() -> None:
    assert canonicalise_body("totally-not-a-real-body") == "totally-not-a-real-body"


def test_aknsource_defaults_for_legacy_config() -> None:
    """A bare AknSource (just name + url_template) must validate, so that
    legacy configs with no format/protocol/coverage fields still load.
    Defaults reflect the lowest-common-denominator: PDF over HTTP."""
    src = AknSource.model_validate(
        {"name": "legacy", "url_template": "https://x.example/{frbr_uri}"}
    )
    assert src.format == "pdf"
    assert src.protocol == "http"
    assert src.coverage == "unknown"
    assert src.schema_id is None
    assert src.license is None


def test_aknsource_rejects_unknown_format() -> None:
    """Strict validation guards against typos like 'akn-xml' (hyphen)
    or 'XML' (case), the Literal must catch these."""
    with pytest.raises(ValueError):
        AknSource.model_validate({"name": "x", "url_template": "y", "format": "akn-xml"})


def test_effective_tier_walks_sources() -> None:
    """effective_tier picks the best (lowest) tier across all sources,
    so a config with both AKN XML and PDF fallbacks reports tier 1."""
    cfg = JurisdictionConfig.model_validate(
        {
            "code": "test",
            "name": "Test",
            "tradition": ["common_law"],
            "languages": ["eng"],
            "akn_sources": [
                {"name": "pdf-only", "url_template": "x", "format": "pdf"},
                {"name": "akn", "url_template": "y", "format": "akn_xml"},
                {"name": "html", "url_template": "z", "format": "html"},
            ],
        }
    )
    assert cfg.effective_tier == 1


def test_effective_tier_falls_back_to_declared_tier_when_no_sources() -> None:
    cfg = JurisdictionConfig.model_validate(
        {
            "code": "test",
            "name": "Test",
            "tradition": ["common_law"],
            "languages": ["eng"],
            "tier": 2,
        }
    )
    assert cfg.effective_tier == 2


def test_effective_tier_returns_none_when_nothing_known() -> None:
    cfg = JurisdictionConfig.model_validate(
        {
            "code": "test",
            "name": "Test",
            "tradition": ["common_law"],
            "languages": ["eng"],
        }
    )
    assert cfg.effective_tier is None


def test_effective_tier_token_gating_demotes_when_token_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Laws.Africa-pattern jurisdiction: AKN behind a token, HTML public.
    Without the token, effective_tier reflects the public fallback."""
    monkeypatch.delenv("LAWS_AFRICA_API_TOKEN", raising=False)
    cfg = JurisdictionConfig.model_validate(
        {
            "code": "gh",
            "name": "Ghana",
            "tradition": ["common_law"],
            "languages": ["eng"],
            "akn_sources": [
                {
                    "name": "GhaLII (Laws.Africa)",
                    "url_template": "https://ghalii.org/...",
                    "format": "structured_html",
                },
                {
                    "name": "Laws.Africa Content API",
                    "url_template": "https://api.laws.africa/...",
                    "format": "akn_xml",
                    "requires_token": True,
                    "auth_env_var": "LAWS_AFRICA_API_TOKEN",
                    "fallback_format": "structured_html",
                },
            ],
        }
    )
    assert cfg.effective_tier == 2  # structured_html (gated source falls back)

    monkeypatch.setenv("LAWS_AFRICA_API_TOKEN", "abc")
    cfg2 = JurisdictionConfig.model_validate(cfg.model_dump())
    assert cfg2.effective_tier == 1  # akn_xml unlocked


def test_effective_tier_skips_gated_source_with_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Token-gated source without fallback_format is dropped entirely
    when the env var isn't set, the next-best source wins."""
    monkeypatch.delenv("SOME_TOKEN", raising=False)
    cfg = JurisdictionConfig.model_validate(
        {
            "code": "test",
            "name": "Test",
            "tradition": ["common_law"],
            "languages": ["eng"],
            "akn_sources": [
                {
                    "name": "gated",
                    "url_template": "x",
                    "format": "akn_xml",
                    "requires_token": True,
                    "auth_env_var": "SOME_TOKEN",
                    # no fallback_format
                },
                {"name": "public", "url_template": "y", "format": "html"},
            ],
        }
    )
    assert cfg.effective_tier == 3  # only the html source is usable


def test_tiers_on_the_five_tier_ladder() -> None:
    """Tiers follow the 2026-07 audit ladder (1 AKN native .. 5 no online
    statute base). Out-of-scope jurisdictions declare coverage_status
    ('subsumed'/'no_corpus'), not a tier."""
    for code in _CODES:
        try_load_config.cache_clear()
        cfg = load_config(code)
        assert cfg is not None, code
        assert cfg.tier in (1, 2, 3, 4, 5, None), f"{code} has tier={cfg.tier}"


def test_subsumed_by_references_existing_jurisdiction() -> None:
    """Every subsumed_by must point to a jurisdiction whose config exists.
    Catches typos like subsumed_by='france' instead of 'fr'."""
    valid_codes = set(_CODES)
    for code in _CODES:
        try_load_config.cache_clear()
        cfg = load_config(code)
        assert cfg is not None, code
        if cfg.subsumed_by is not None:
            assert cfg.subsumed_by in valid_codes, (
                f"{code} declares subsumed_by={cfg.subsumed_by!r} but no such jurisdiction exists"
            )


def test_subsumed_by_requires_subsumed_status() -> None:
    """subsumed_by is only legal when coverage_status='subsumed'."""
    with pytest.raises(ValueError, match="subsumed_by is only meaningful"):
        JurisdictionConfig.model_validate(
            {
                "code": "x",
                "name": "X",
                "tradition": ["common_law"],
                "languages": ["eng"],
                "coverage_status": "in_scope",
                "subsumed_by": "fr",
            }
        )


def test_subsumed_status_requires_subsumed_by() -> None:
    """coverage_status='subsumed' without subsumed_by is rejected."""
    with pytest.raises(ValueError, match="no subsumed_by code"):
        JurisdictionConfig.model_validate(
            {
                "code": "x",
                "name": "X",
                "tradition": ["common_law"],
                "languages": ["eng"],
                "coverage_status": "subsumed",
            }
        )


def test_coverage_status_default_is_in_scope() -> None:
    """Legacy configs without coverage_status set must default to in_scope."""
    cfg = JurisdictionConfig.model_validate(
        {
            "code": "x",
            "name": "X",
            "tradition": ["common_law"],
            "languages": ["eng"],
        }
    )
    assert cfg.coverage_status == "in_scope"
    assert cfg.subsumed_by is None


def test_aknsource_full_shape() -> None:
    src = AknSource.model_validate(
        {
            "name": "BCN Ley Chile",
            "url_template": "https://nuevo.leychile.cl/...",
            "format": "akn_xml_v2",
            "protocol": "rest",
            "schema_id": "AKN-2.0-BCN",
            "schema_url": "http://datos.bcn.cl/XMLSchema/2013/akomantoso20_BCN.xsd",
            "coverage": "complete",
            "coverage_note": "All consolidated normas",
            "license": "CC-BY-3.0",
        }
    )
    assert src.format == "akn_xml_v2"
    assert src.protocol == "rest"
    assert src.coverage == "complete"


@pytest.mark.parametrize(
    "scheme",
    [
        "none",
        "token",
        "bearer",
        "api_key",
        "operator_code",
        "basic",
        "soap_session",
        "email_registration",
    ],
)
def test_auth_scheme_accepts_extended_set(scheme: str) -> None:
    """Each auth_scheme value used by audited configs (gh, kr, ro, fk, etc.)
    must validate, including `api_key`, `operator_code` and
    `email_registration`."""
    src = AknSource.model_validate(
        {
            "name": "test",
            "url_template": "https://example.com/{frbr_uri}",
            "auth_scheme": scheme,
        }
    )
    assert src.auth_scheme == scheme


def test_registry_matches_jurisdiction_directories() -> None:
    registry = load_registry()
    registry_codes = {entry["code"] for entry in registry if "code" in entry}
    # Synthetic (fictional) jurisdictions are test fixtures, deliberately absent
    # from the registry; exclude them from the on-disk expectation.
    dir_codes = {c for c in _CODES if not ((cfg := load_config(c)) and cfg.synthetic)}
    missing_in_registry = dir_codes - registry_codes
    missing_on_disk = registry_codes - dir_codes
    assert not missing_in_registry, (
        f"Codes on disk but not in registry: {sorted(missing_in_registry)}"
    )
    assert not missing_on_disk, f"Codes in registry but no config dir: {sorted(missing_on_disk)}"


# --- N1: SourceAdapter schema --------------------------------------------------


def test_source_adapter_kind_is_typed() -> None:
    sa = SourceAdapter(kind="akn_native", name="x", url_template="https://x{frbr_uri}")
    assert sa.kind == "akn_native"
    with pytest.raises(Exception):
        SourceAdapter(kind="not-a-real-kind", name="x")  # type: ignore[arg-type]


def test_source_adapter_rejects_unknown_fields() -> None:
    with pytest.raises(Exception):
        SourceAdapter(kind="akn_native", name="x", magic_token="leak")  # type: ignore[call-arg]


def test_auth_env_var_value_never_serialised() -> None:
    cfg = JurisdictionConfig(
        code="xx",
        name="X",
        tradition=["civil_law"],
        languages=["eng"],
        source_adapters=[
            SourceAdapter(kind="akn_native", name="x", auth_env_var="MY_SECRET_TOKEN")
        ],
    )
    dumped = cfg.model_dump()
    assert dumped["source_adapters"][0]["auth_env_var"] == "MY_SECRET_TOKEN"
    # Belt-and-braces: the env var name itself appears, but never a value.
    # Ensure no actual secret-shaped string ("sk-...", "Bearer ...") leaks.
    serialised = str(dumped)
    assert "Bearer " not in serialised
    assert "sk-" not in serialised


@pytest.mark.parametrize(
    "code", ["eu", "gb", "za", "african-union", "caricom", "uemoa"], ids=lambda c: c
)
def test_seeded_configs_have_explicit_source_adapters(code: str) -> None:
    try_load_config.cache_clear()
    cfg = load_config(code)
    assert cfg is not None
    assert cfg.source_adapters, f"{code}: source_adapters empty after migration"


def test_source_adapter_requires_auth_env_var_when_secured() -> None:
    """auth_scheme != none ⇒ auth_env_var non-empty."""
    SourceAdapter(kind="akn_native", name="ok", auth_scheme="token", auth_env_var="MY_TOKEN")
    with pytest.raises(Exception):
        SourceAdapter(kind="akn_native", name="bad", auth_scheme="token")


def test_act_short_label_and_code_from_config(tmp_path, monkeypatch) -> None:
    try_load_config.cache_clear()
    from codify.jurisdictions import act_short_code, act_short_label
    from tests.config_fixtures import isolated_configs

    configs = {
        "xq": {
            "document_classes": {
                "order": {"label": "Order", "short_label": "ORD"},
                "act": {"label": "Act", "short_label": "Act"},
            }
        }
    }
    with isolated_configs(monkeypatch, tmp_path / "configs", configs):
        assert act_short_label("xq", "order") == "ORD"
        assert act_short_code("/akn/xq/act/order/2021/285", "order") == "ORD 285/2021"
        assert act_short_code("/akn/xq/act/2020/162", "act") == "Act 162/2020"
        assert act_short_label("zz", "foo") == "Foo"
        assert act_short_code("/akn/xq", "order") == "ORD"


def test_authoritative_language() -> None:
    try_load_config.cache_clear()
    from codify.jurisdictions import authoritative_language

    assert authoritative_language("al") == "sqi"
    with pytest.raises(JurisdictionConfigError, match="no jurisdiction config"):
        authoritative_language("zz-not-a-code")


def test_discovery_queries_must_be_doctypes() -> None:
    raw = {
        "code": "xq",
        "name": "Example",
        "tradition": ["civil_law"],
        "languages": ["eng"],
        "document_classes": {"act": {"label": "Act"}},
        "source_adapters": [
            {
                "kind": "akn_native",
                "name": "Example source",
                "discovery": {"queries": {"act": "title:example"}},
            }
        ],
    }
    assert JurisdictionConfig.model_validate(raw)
    raw["source_adapters"][0]["discovery"]["queries"]["not_a_doctype"] = "title:x"
    with pytest.raises(ValueError, match="not in document_classes"):
        JurisdictionConfig.model_validate(raw)


def test_al_discovery_config_present() -> None:
    try_load_config.cache_clear()
    cfg = load_config("al")
    assert cfg is not None
    disc = next((a.discovery for a in cfg.source_adapters if a.discovery), None)
    assert disc is not None
    assert set(disc.queries) == {"ligj", "vendim"}


# Bahasa instrument titles are formulaic, so the class is decidable from the
# title alone. The ordering hazards are that "Peraturan Pemerintah Pengganti
# Undang-Undang" must beat "Peraturan Pemerintah", and both Perda tiers must
# beat the bare Undang-Undang rule.
XL_TITLE_CLASSES = [
    (
        "UNDANG-UNDANG REPUBLIK LANGKASUKA NOMOR 13 TAHUN 2003 TENTANG KETENAGAKERJAAN",
        "act",
    ),
    ("PERATURAN PEMERINTAH PENGGANTI UNDANG-UNDANG NOMOR 2 TAHUN 2022", "perppu"),
    ("PERATURAN PEMERINTAH NOMOR 5 TAHUN 2021", "pp"),
    ("PERATURAN PRESIDEN NOMOR 68 TAHUN 2021", "perpres"),
    ("PERATURAN MENTERI KETENAGAKERJAAN NOMOR 14 TAHUN 2023", "permen"),
    ("PERATURAN DAERAH PROVINSI TANJUNG SERI NOMOR 3 TAHUN 2022", "perda_provinsi"),
    ("PERATURAN DAERAH KABUPATEN PULAU SERI NOMOR 2 TAHUN 2012", "perda_kabkota"),
    ("PERATURAN GUBERNUR TANJUNG SERI NOMOR 10 TAHUN 2020", "perkada"),
    ("PERATURAN KOMISI INFORMASI NOMOR 11 TAHUN 2020", "peraturan_lembaga"),
    ("UNDANG-UNDANG DASAR REPUBLIK LANGKASUKA TAHUN 1957", "constitution"),
    # The ikhtisar's title contains the judgment's, so it must outrank it.
    ("PUTUSAN Nomor 12/PUL-III/2029", "putusan"),
    ("Ikhtisar Putusan Mahkamah Konstitusi Nomor 12/PUL-III/2029", "ikhtisar"),
]


@pytest.mark.parametrize("title,expected", XL_TITLE_CLASSES, ids=lambda v: v if len(v) < 24 else "")
def test_classification_rules_resolve_instrument_type(title: str, expected: str) -> None:
    try_load_config.cache_clear()
    cfg = load_config("xl")
    assert cfg is not None
    assert cfg.classify_document_class(title=title) == expected


def test_classification_targets_all_exist() -> None:
    """A rule pointing at a class that isn't declared classifies into nothing."""
    try_load_config.cache_clear()
    cfg = load_config("xl")
    assert cfg is not None
    assert cfg.structuring is not None
    for rule in cfg.structuring.classification_rules:
        assert rule.target_document_class in cfg.document_classes


def test_ruling_and_its_summary_do_not_share_a_work(tmp_path, monkeypatch) -> None:
    """Different document classes can share a number without sharing a work."""
    from codify.frbr import build_frbr_work_uri
    from tests.config_fixtures import isolated_configs

    configs = {
        "xq": {
            "frbr": {
                "country_code": "xq",
                "uri_patterns": {
                    "judgment": "/akn/xq/judgment/{year}/{number}",
                    "summary": "/akn/xq/doc/summary/{year}/{number}",
                },
            }
        }
    }
    with isolated_configs(monkeypatch, tmp_path / "configs", configs):
        uris = {build_frbr_work_uri("xq", dt, 2099, "17") for dt in ("judgment", "summary")}
        assert len(uris) == 2
        assert all("/akn/xq/act/" not in u for u in uris)


def test_atlas_module_imports_cycle_free_in_both_orders() -> None:
    """The atlas split defers `JURISDICTIONS_DIR` into a loader to avoid an
    import cycle. Existing tests import `codify.jurisdictions` first, so exercise
    the untested atlas-first order (and the reverse) in fresh subprocesses."""
    import subprocess
    import sys

    atlas_first = (
        "import codify.jurisdictions_atlas as a; "
        "assert a.canonicalise_body('EU'); "
        "a.SupranationalMembership(body='EU'); "
        "import codify.jurisdictions as j; "
        "assert j.canonicalise_body is a.canonicalise_body"
    )
    juris_first = (
        "import codify.jurisdictions as j; "
        "assert j.SuspensionPeriod and j.MembershipStatus; "
        "assert j.load_config('gb') is not None"
    )
    for snippet in (atlas_first, juris_first):
        proc = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


def test_public_reference_defaults_to_excluded() -> None:
    """The allowlist only protects anything if a jurisdiction has to be added to
    it deliberately. Excluded by default, not included and simplified: a
    simplified config still says which jurisdictions were studied."""
    from codify.jurisdictions import JurisdictionConfig

    cfg = JurisdictionConfig.model_validate(
        {
            "code": "zz",
            "name": "Nowhere",
            "tradition": ["civil_law"],
            "languages": ["eng"],
        }
    )
    assert cfg.public_reference is False


def test_the_flagged_set_is_small_and_deliberate() -> None:
    """A flag that spreads quietly is the failure this design guards against, so
    the count is asserted rather than left to drift."""
    import json

    from codify.jurisdictions import JURISDICTIONS_DIR

    registry = json.loads((JURISDICTIONS_DIR / "registry.json").read_text())
    flagged = sorted(j["code"] for j in registry["jurisdictions"] if j.get("public_reference"))
    assert flagged == ["ee", "fi", "gb", "ie", "it", "nz"]


def test_the_export_carries_a_filtered_registry(tmp_path, monkeypatch) -> None:
    """An unselected source entry must not travel with the public profiles."""
    import sys

    from tests.config_fixtures import isolated_configs

    sys.path.insert(0, str(_SCRIPTS))
    import export_public_configs as exporter

    with isolated_configs(monkeypatch, tmp_path / "source", {"xq": {}}) as source:
        registry_path = source / "registry.json"
        registry = json.loads(registry_path.read_text())
        registry["jurisdictions"].append({"code": "xq", "name": "Unselected example"})
        registry_path.write_text(json.dumps(registry))
        with monkeypatch.context() as patch:
            patch.setattr(exporter, "JURISDICTIONS_DIR", source)
            dest = tmp_path / "export"
            assert exporter.export(dest) == 0
            out = dest / "data" / "jurisdictions"
            exported = sorted(p.name for p in out.iterdir() if p.is_dir())
            assert exported == exporter.public_codes()
            listed = {
                j["code"] for j in json.loads((out / "registry.json").read_text())["jurisdictions"]
            }
            everything = {j["code"] for j in registry["jurisdictions"]}
            assert listed == set(exported) & everything
            assert listed < everything
            assert "xq" not in listed


def test_every_exported_config_loads_from_the_exported_tree(tmp_path) -> None:
    """The export is a tree the core must run against, not a file copy. A real
    config's membership validator reads `supranational_bodies.json`."""
    import subprocess
    import sys

    sys.path.insert(0, str(_SCRIPTS))
    from export_public_configs import export, public_codes

    dest = tmp_path / "export"
    assert export(dest) == 0
    probe = (
        "from codify.jurisdictions import load_config\n"
        f"[load_config(c) for c in {public_codes()!r}]\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=dest,
        env={"PYTHONPATH": str(dest / "packages" / "codify"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr


def test_the_export_is_a_tree_that_resolves_to_itself(tmp_path) -> None:
    """`JURISDICTIONS_DIR` is package-relative, so configs without the package
    beside them resolve back to the full set and the export proves nothing."""
    import sys

    sys.path.insert(0, str(_SCRIPTS))
    from export_public_configs import export

    dest = tmp_path / "export"
    assert export(dest) == 0
    jurisdictions = dest / "packages" / "codify" / "codify" / "jurisdictions.py"
    assert jurisdictions.is_file()
    # The depth the module computes from: parents[3] of the source file must be
    # the export root, or it resolves past the export into the real tree.
    assert jurisdictions.parents[3] == dest


def test_the_export_refuses_a_non_empty_destination(tmp_path) -> None:
    """Merging would leave behind a jurisdiction since removed from the allowlist,
    which is the one thing this script exists to prevent."""
    import sys

    sys.path.insert(0, str(_SCRIPTS))
    from export_public_configs import export

    dest = tmp_path / "export"
    assert export(dest) == 0
    stale = dest / "data" / "jurisdictions" / "zz"
    stale.mkdir(parents=True)
    assert export(dest) == 1
    assert stale.exists(), "a refused export must not delete what it found"


# Additional distribution data is explicitly inventoried.
_WORKING_FILE_DIRS = {
    "gb": {"gold_basket.json", "short_titles.json"},
}


def test_only_known_directories_carry_working_files() -> None:
    """Reject unreviewed additional files in the packaged configurations."""
    found: dict[str, set[str]] = {}
    for cfg in JURISDICTIONS_DIR.glob("*/config.json"):
        # Every immediate entry, directories included: a `notes/` folder is working
        # material as surely as a loose file, and filtering to files let one through.
        extra = {
            f.name for f in cfg.parent.iterdir() if f.name not in {"config.json", "profile.md"}
        }
        if extra:
            found[cfg.parent.name] = extra
    assert found == _WORKING_FILE_DIRS, (
        "jurisdiction working files changed; scrub the addition before allowlisting it: "
        f"{ {k: sorted(v) for k, v in sorted(found.items())} }"
    )


class TestConfigAbsenceIsAFault:
    """`load_config` answering None was read as "no rules apply", and every site
    that reached it produced a plausible, thinner result. `load_config` exists
    so a caller naming a jurisdiction cannot silently get defaults."""

    def test_a_jurisdiction_with_no_config_raises(self) -> None:
        with pytest.raises(JurisdictionConfigError, match="zz"):
            load_config("zz")

    def test_the_message_names_the_directory_it_searched(self) -> None:
        """A reader needs to know whether the jurisdiction is unwritten or the
        tree is absent, because the two have different fixes."""
        with pytest.raises(JurisdictionConfigError) as excinfo:
            load_config("zz")
        assert str(JURISDICTIONS_DIR) in str(excinfo.value)

    def test_absent_data_says_so_rather_than_blaming_the_jurisdiction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Installed as a wheel the package carries no `data/`, so every lookup
        would answer with defaults. Naming the jurisdiction there would send a
        reader to write a config that already exists."""
        monkeypatch.setattr(
            "codify.jurisdictions.JURISDICTIONS_DIR", Path("/nonexistent/jurisdictions")
        )
        try_load_config.cache_clear()
        with pytest.raises(JurisdictionConfigError, match="installed"):
            load_config("gb")
        try_load_config.cache_clear()

    def test_a_declared_jurisdiction_still_loads(self) -> None:
        assert load_config("gb").code == "gb"

    def test_a_named_country_with_no_config_cannot_be_scanned(self) -> None:
        """The anchor regex is where absence did the most damage: without a config
        the alternation is English keywords, and a non-Latin document then scans
        to almost no anchors and reads as one with almost no structure."""
        from codify.pipeline.enrich.anchors import cached_regex

        cached_regex.cache_clear()
        with pytest.raises(JurisdictionConfigError, match="zz"):
            cached_regex("zz", "act")
        cached_regex.cache_clear()

    def test_an_empty_country_still_scans_with_the_builtin_alternation(self) -> None:
        """Distinct from a named country with no config: an empty code is the
        caller saying it has no jurisdiction, which the corpus census does."""
        from codify.pipeline.enrich.anchors import cached_regex

        assert cached_regex("", "act") is not None

    def test_an_undeclared_doctype_resolves_through_the_default_class(self) -> None:
        """Why the builtin-alias fallback is unreachable from a loaded config:
        `get_document_class` falls back to the default class, and every shipped
        config declares one."""
        for code in _CODES:
            cfg = load_config(code)
            assert cfg.get_document_class("no-such-doctype") is not None, code

    def test_scanning_a_country_with_no_config_raises_from_the_scan_itself(
        self,
    ) -> None:
        """Four helpers in the anchor module fell back on None, two of them
        swallowing the error. Asserted through `scan_anchors` rather than the
        helpers, so a later caller cannot route around them."""
        from codify.pipeline.enrich.anchors import build_anchor_regex, scan_anchors

        regex = build_anchor_regex(load_config("gb"), "act")
        with pytest.raises(JurisdictionConfigError, match="zz"):
            scan_anchors("SECTION 1 - Short title\n\nText.\n", regex, country="zz", doctype="act")


class TestTheWheelCarriesTheFlaggedConfigs:
    """The build hook reads the flags rather than a list, which would drift from
    the tree without saying so."""

    def test_the_selection_is_exactly_the_flagged_configs(self) -> None:
        expected = {
            code
            for code in _CODES
            if (cfg := load_config(code)) and (cfg.synthetic or cfg.public_reference)
        }
        shipped = {
            p.parent.name for p in JURISDICTIONS_DIR.glob("*/config.json") if ships_in_open_wheel(p)
        }
        assert shipped == expected
        assert expected, "no config is flagged; the wheel would carry no data"

    def test_a_config_that_does_not_parse_is_not_shipped(self, tmp_path: Path) -> None:
        """A wheel is the wrong place to discover a broken config. The suite
        validates every one of them, so a real breakage fails there and by name,
        rather than here and as a quietly thinner wheel."""
        broken = tmp_path / "config.json"
        broken.write_text("{not json", encoding="utf-8")
        assert ships_in_open_wheel(broken) is False

    def test_an_unflagged_config_is_excluded(self, tmp_path: Path) -> None:
        config = tmp_path / "config.json"
        config.write_text(
            json.dumps(
                {
                    "code": "xq",
                    "name": "Example",
                    "tradition": ["civil_law"],
                    "languages": ["eng"],
                }
            )
        )
        assert not ships_in_open_wheel(config)
        raw = json.loads(config.read_text())
        raw["public_reference"] = True
        config.write_text(json.dumps(raw))
        assert ships_in_open_wheel(config)

    def test_the_export_script_selects_the_same_set(self) -> None:
        """Both sides read the configs; the export used to read the registry,
        which has no entry for the synthetic jurisdictions."""
        spec = importlib.util.spec_from_file_location(
            "codify_export_public_configs",
            Path(__file__).resolve().parents[1] / "scripts" / "export_public_configs.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        shipped = {
            p.parent.name for p in JURISDICTIONS_DIR.glob("*/config.json") if ships_in_open_wheel(p)
        }
        assert set(module.public_codes()) == shipped


def test_an_absent_registry_raises() -> None:
    """It answered `[]`, which reads as a deployment holding no jurisdictions."""
    from codify import jurisdictions as j

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(j, "JURISDICTIONS_DIR", Path("/nonexistent-jurisdictions"))
        with pytest.raises(JurisdictionConfigError, match="registry"):
            j.load_registry()


def test_absent_supranational_bodies_raise() -> None:
    """It answered `{}`, so every supranational alias stopped resolving."""
    from codify import jurisdictions as j
    from codify import jurisdictions_atlas as atlas

    atlas._load_supranational_bodies.cache_clear()
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(j, "JURISDICTIONS_DIR", Path("/nonexistent-jurisdictions"))
            with pytest.raises(JurisdictionConfigError, match="supranational"):
                atlas._load_supranational_bodies()
    finally:
        atlas._load_supranational_bodies.cache_clear()


def test_the_resolver_prefers_the_project_root_to_an_ancestor(tmp_path) -> None:
    """An sdist unpacked inside a checkout must read what it ships, not the 257
    the checkout carries."""
    from codify.data_paths import data_dir

    inner = tmp_path / "dist" / "codify-0.1.0"
    (inner / "codify").mkdir(parents=True)
    for root in (tmp_path, inner):
        (root / "data" / "jurisdictions").mkdir(parents=True)
    assert data_dir(str(inner / "codify" / "jurisdictions.py"), "jurisdictions") == (
        inner / "data" / "jurisdictions"
    )


def test_the_resolver_survives_a_shallow_tree() -> None:
    """`parents[n]` raises rather than answering nothing, and this runs at import,
    so a tree unpacked near the filesystem root took the package down with it."""
    from codify.data_paths import data_dir

    assert data_dir("/codify/jurisdictions.py", "jurisdictions").name == "jurisdictions"


def test_the_resolver_falls_back_to_the_wheel_copy(tmp_path) -> None:
    """Whatever the install directory is called: `pip --target` names it freely."""
    from codify.data_paths import data_dir

    pkg = tmp_path / "vendor" / "codify"
    (pkg / "data" / "frameworks").mkdir(parents=True)
    # What another distribution leaving a generic `data/` there would look like.
    (pkg.parent / "data" / "frameworks").mkdir(parents=True)
    assert data_dir(str(pkg / "frameworks" / "acquis.py"), "frameworks") == (
        pkg / "data" / "frameworks"
    )


def test_a_partial_wheel_does_not_read_past_itself(tmp_path) -> None:
    """A build shipping one leaf and not the other must report the missing one
    absent, not resolve it from whatever sits beside the install directory."""
    from codify.data_paths import data_dir

    pkg = tmp_path / "vendor" / "codify"
    (pkg / "data" / "frameworks").mkdir(parents=True)
    (pkg.parent / "data" / "jurisdictions").mkdir(parents=True)
    got = data_dir(str(pkg / "jurisdictions.py"), "jurisdictions")
    assert got == pkg / "data" / "jurisdictions"
    assert not got.exists(), "absent, so the caller raises rather than reads on"


# ── document class `extends` ─────────────────────────────────────────────────

_SECTION = {
    "local_term": "Section",
    "akn_element": "section",
    "level": "basic",
    "bluebell_keyword": "SECTION",
    "numbering": "arabic",
}


def _config_with(classes: dict) -> JurisdictionConfig:
    return JurisdictionConfig(
        code="xq",
        name="Nowhere",
        tradition=["civil_law"],
        languages=["en"],
        document_classes=classes,
    )


def test_extends_lays_own_fields_over_the_parent() -> None:
    config = _config_with(
        {
            "act": {"label": "Act", "basic_unit": "section", "hierarchy": [_SECTION]},
            "token": {"extends": "act", "label": "Token Act", "search_rank": 3},
        }
    )
    act, token = config.document_classes["act"], config.document_classes["token"]
    assert token.hierarchy == act.hierarchy
    assert token.basic_unit == act.basic_unit
    assert token.label == "Token Act"
    assert token.search_rank == 3
    assert token.extends is None  # resolved, not left as a pointer
    assert act.label == "Act" and act.search_rank is None  # parent untouched


def test_extends_refuses_a_missing_parent() -> None:
    with pytest.raises(ValueError, match="missing"):
        _config_with({"act": {"label": "Act"}, "token": {"extends": "nope", "label": "T"}})


def test_extends_refuses_a_chain() -> None:
    with pytest.raises(ValueError, match="itself extends"):
        _config_with(
            {
                "act": {"label": "Act"},
                "mid": {"extends": "act", "label": "Mid"},
                "leaf": {"extends": "mid", "label": "Leaf"},
            }
        )
