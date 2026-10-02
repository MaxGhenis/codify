"""Manifest validation + deliverable packaging."""

import io
import zipfile

import pytest

from codify.deliverable import DeliverableManifest, LawArtifacts, build_deliverable


def test_manifest_parses_and_strips_trailing_slash() -> None:
    m = DeliverableManifest.from_yaml(
        "jurisdiction_code: xz\nsource_language: ara\ntarget_languages: [en]\n"
        "laws:\n  - {filename: a.pdf, frbr_work_uri: /akn/xz/act/2005/1/}\n"
    )
    assert m.jurisdiction_code == "xz"
    assert m.laws[0].frbr_work_uri == "/akn/xz/act/2005/1"  # slash stripped
    assert m.include_blocking is False  # off by default


def test_manifest_include_blocking_opt_in() -> None:
    m = DeliverableManifest.from_yaml(
        "jurisdiction_code: xz\nsource_language: ara\ntarget_languages: [en]\n"
        "include_blocking: true\n"
        "laws:\n  - {filename: a.pdf, frbr_work_uri: /akn/xz/act/2005/1}\n"
    )
    assert m.include_blocking is True


@pytest.mark.parametrize(
    "bad",
    [
        "jurisdiction_code: xz\nsource_language: ara\nlaws: []\n",  # empty
        # duplicate filename
        "jurisdiction_code: xz\nsource_language: ara\nlaws:\n"
        "  - {filename: a.pdf, frbr_work_uri: /akn/xz/1}\n"
        "  - {filename: a.pdf, frbr_work_uri: /akn/xz/2}\n",
        # not an /akn/ uri
        "jurisdiction_code: xz\nsource_language: ara\nlaws:\n"
        "  - {filename: a.pdf, frbr_work_uri: xz/act/1}\n",
        # unknown field (extra=forbid)
        "jurisdiction_code: xz\nsource_language: ara\nbogus: 1\nlaws:\n"
        "  - {filename: a.pdf, frbr_work_uri: /akn/xz/1}\n",
        # target language equals source (no-op translation)
        "jurisdiction_code: xz\nsource_language: ara\ntarget_languages: [ara]\nlaws:\n"
        "  - {filename: a.pdf, frbr_work_uri: /akn/xz/1}\n",
        # duplicate target languages
        "jurisdiction_code: xz\nsource_language: ara\ntarget_languages: [en, en]\nlaws:\n"
        "  - {filename: a.pdf, frbr_work_uri: /akn/xz/1}\n",
    ],
)
def test_manifest_rejects_bad_input(bad: str) -> None:
    with pytest.raises(ValueError):
        DeliverableManifest.from_yaml(bad)


def test_packager_txt_decodes_entities() -> None:
    # regex tag-stripping would leave &amp; encoded; real text extraction decodes it
    laws = [
        LawArtifacts(
            "/akn/xz/act/2005/1", "clean", akn_by_lang={"ara": "<akn><p>A &amp; B</p></akn>"}
        )
    ]
    zip_bytes, _ = build_deliverable(
        jurisdiction_code="xz",
        source_language="ara",
        target_languages=[],
        laws=laws,
        codify_version="test",
    )
    txt = zipfile.ZipFile(io.BytesIO(zip_bytes)).read("txt/xz-act-2005-1.ara.txt").decode("utf-8")
    assert txt == "A & B"


def test_packager_excludes_ungraded_via_allowlist() -> None:
    # a grade outside the shippable allow-list must not reach the body even with content
    laws = [LawArtifacts("/akn/xz/act/2005/1", "ungraded", akn_by_lang={"ara": "<akn/>"})]
    zip_bytes, report = build_deliverable(
        jurisdiction_code="xz",
        source_language="ara",
        target_languages=[],
        laws=laws,
        codify_version="test",
    )
    names = set(zipfile.ZipFile(io.BytesIO(zip_bytes)).namelist())
    assert not any("2005-1" in n for n in names), "ungraded must be excluded from body"
    assert "/akn/xz/act/2005/1" in report  # but still named


def _names(zip_bytes: bytes) -> set[str]:
    return set(zipfile.ZipFile(io.BytesIO(zip_bytes)).namelist())


def test_manifest_carries_provenance() -> None:
    m = DeliverableManifest.from_yaml(
        "jurisdiction_code: xz\nsource_language: ara\nlaws:\n"
        "  - {filename: a.pdf, frbr_work_uri: /akn/xz/act/2005/1, "
        "source_id: C06-089, source_doctype: qanun}\n"
    )
    assert m.laws[0].source_id == "C06-089"
    assert m.laws[0].source_doctype == "qanun"


def test_packager_assembles_body_and_reports() -> None:
    laws = [
        LawArtifacts(
            "/akn/xz/act/decree-law/2031/12",
            "clean",
            source_id="C21-198",
            source_doctype="qarar_bi_qanun",
            akn_by_lang={"ara": "<akn><p>نص</p></akn>", "en": "<akn><p>text</p></akn>"},
            pdf_by_lang={"ara": b"%PDF-1.4 a"},
        ),
    ]
    zip_bytes, report = build_deliverable(
        jurisdiction_code="xz",
        source_language="ara",
        target_languages=["en"],
        laws=laws,
        codify_version="test",
    )
    names = _names(zip_bytes)
    assert "akn/xz-act-decree-law-2031-12.ara.xml" in names
    assert "akn/xz-act-decree-law-2031-12.en.xml" in names
    assert "txt/xz-act-decree-law-2031-12.ara.txt" in names
    assert "pdf/xz-act-decree-law-2031-12.ara.pdf" in names
    assert {"README.txt", "quality-report.txt", "manifest.txt"} <= names
    assert report.startswith("---")  # YAML frontmatter for machine parsing
    assert "clean: 1" in report
    assert "C21-198" in report and "qarar_bi_qanun" in report  # provenance surfaced
    # txt is the tag-stripped plain text, not the raw AKN
    txt = zipfile.ZipFile(io.BytesIO(zip_bytes)).read("txt/xz-act-decree-law-2031-12.ara.txt")
    assert txt.decode("utf-8").strip() == "نص"


def test_packager_reports_translation_grade_and_review_findings() -> None:
    laws = [
        LawArtifacts(
            "/akn/xz/act/2024/22",
            "clean",
            translation_grade="C",
            review_findings=["missing monetary amounts under ['art_26']"],
            akn_by_lang={"ara": "<akn/>", "en": "<akn/>"},
        ),
    ]
    _, report = build_deliverable(
        jurisdiction_code="xz",
        source_language="ara",
        target_languages=["en"],
        laws=laws,
        codify_version="test",
    )
    assert "| translation | review |" in report  # columns present
    assert "translation_grades:" in report and "C: 1" in report  # histogram
    assert "review_required: 1" in report
    assert "missing monetary amounts under ['art_26']" in report  # named per law


def test_review_required_counts_grade_c_and_blocking_without_delivery_rows() -> None:
    # A C-grade law whose C came from the composite rubric (no review-tier
    # delivery row) and a blocking-structure law both count as review-required.
    laws = [
        LawArtifacts(
            "/akn/xz/act/2024/1", "clean", translation_grade="C", akn_by_lang={"ara": "<akn/>"}
        ),
        LawArtifacts("/akn/xz/act/2024/2", "blocking", akn_by_lang={"ara": "<akn/>"}),
        LawArtifacts(
            "/akn/xz/act/2024/3", "clean", translation_grade="A", akn_by_lang={"ara": "<akn/>"}
        ),
    ]
    _, report = build_deliverable(
        jurisdiction_code="xz",
        source_language="ara",
        target_languages=["en"],
        laws=laws,
        codify_version="test",
    )
    assert "review_required: 2" in report  # the C law + the blocking law, not the A


def test_packager_include_blocking_ships_blocking_law_body() -> None:
    # With the opt-in, a blocking-structural law that was translated must reach
    # the body (not just the report), else the toggle burns compute for nothing.
    laws = [
        LawArtifacts("/akn/xz/act/2027/9", "blocking", akn_by_lang={"ara": "<akn><p>x</p></akn>"}),
    ]
    zip_bytes, report = build_deliverable(
        jurisdiction_code="xz",
        source_language="ara",
        target_languages=[],
        laws=laws,
        codify_version="test",
        include_blocking=True,
    )
    names = _names(zip_bytes)
    assert any("2027-9" in n for n in names), "include_blocking must ship the blocking law body"
    # default (off) still excludes it, the existing gate
    zip_off, _ = build_deliverable(
        jurisdiction_code="xz",
        source_language="ara",
        target_languages=[],
        laws=laws,
        codify_version="test",
    )
    assert not any("2027-9" in n for n in _names(zip_off))


def test_packager_readme_notes_pdf_omitted_when_render_off() -> None:
    laws = [LawArtifacts("/akn/xz/act/2024/9", "clean", akn_by_lang={"ara": "<akn/>"})]
    zip_bytes, _ = build_deliverable(
        jurisdiction_code="xz",
        source_language="ara",
        target_languages=[],
        laws=laws,
        codify_version="test",
        render_pdf=False,
    )
    readme = zipfile.ZipFile(io.BytesIO(zip_bytes)).read("README.txt").decode("utf-8")
    assert "pdf/ omitted" in readme
    assert "rendered PDF, one file per law" not in readme


def test_packager_excludes_blocking_from_body_but_names_it() -> None:
    laws = [
        LawArtifacts("/akn/xz/act/2031/12", "clean", akn_by_lang={"ara": "<akn/>"}),
        LawArtifacts("/akn/xz/act/2027/9", "blocking", akn_by_lang={"ara": "<akn/>"}),
        LawArtifacts("/akn/xz/act/2005/1", "failed", akn_by_lang={"ara": "<akn/>"}),
    ]
    zip_bytes, report = build_deliverable(
        jurisdiction_code="xz",
        source_language="ara",
        target_languages=[],
        laws=laws,
        codify_version="test",
    )
    names = _names(zip_bytes)
    assert any("2031-12" in n for n in names), "clean law must be in the body"
    assert not any("2027-9" in n for n in names), "blocking law excluded from body"
    assert not any("2005-1" in n for n in names), "failed law excluded from body"
    # still named in the report + manifest
    assert "/akn/xz/act/2027/9" in report
    assert "blocking: 1" in report and "failed: 1" in report
