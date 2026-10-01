"""A drop must not silently take body text out of the body with it.

Every anchor pass that removes a marker leaves the text that marker governed
behind, and that text falls to whatever precedes the drop. For the first anchor
in a document that is the preamble, which is how an act can lose most of its
provisions while keeping nearly all of its characters.
"""

from __future__ import annotations

from codify.pipeline.enrich.anchors import cached_regex, scan_anchors_with_ambiguity
from codify.pipeline.enrich.validator import _check_orphaned_drops

# The shape that caused the loss: an OCR-damaged source where the `Pasal 1`
# heading never made it through, so the citation anchor in `Mengingat` is the
# only marker holding the definitions that follow it.
LOAD_BEARING_CITATION = (
    "Menimbang: bahwa penyelenggaraan pemerintahan daerah diarahkan;\n"
    "Mengingat:\n"
    "Pasal 5 ayat (1), Pasal 20, Pasal 22D Undang-Undang Dasar 1957;\n"
    "MEMUTUSKAN:\n"
    "Menetapkan: UNDANG-UNDANG TENTANG PEMERINTAHAN DAERAH.\n"
    # No `Pasal 1` line: these definitions hang off the citation anchor above.
    + "Pemerintah Pusat adalah Presiden Republik Langkasuka yang memegang kekuasaan "
    "pemerintahan negara Republik Langkasuka yang dibantu oleh Wakil Presiden dan menteri "
    "sebagaimana dimaksud dalam Undang-Undang Dasar Republik Langkasuka Tahun 1957.\n"
    "Pemerintah Daerah adalah kepala daerah sebagai unsur penyelenggara Pemerintahan "
    "Daerah yang memimpin pelaksanaan urusan pemerintahan yang menjadi kewenangan daerah "
    "otonom dan tugas pembantuan yang diberikan kepada daerah otonom tersebut.\n"
    "Urusan Pemerintahan adalah kekuasaan pemerintahan yang menjadi kewenangan Presiden "
    "yang pelaksanaannya dilakukan oleh kementerian negara dan penyelenggara Pemerintahan "
    "Daerah untuk melindungi, melayani dan memberdayakan masyarakat demi kesejahteraan.\n"
    "BAB I\n"
    "KETENTUAN UMUM\n"
    "Pasal 2\n"
    "Republik Langkasuka dibagi atas daerah provinsi.\n"
    "Pasal 3\n"
    "Daerah provinsi dibagi atas daerah kabupaten dan kota.\n"
    "Pasal 4\n"
    "Daerah kabupaten dibagi atas kecamatan.\n"
)

# The same convention, undamaged: the citation is followed straight away by the
# body's first container, so the drop takes nothing with it.
CLEAN_CITATION = (
    "Mengingat:\n"
    "Pasal 5 ayat (1), Pasal 20 Undang-Undang Dasar 1957;\n"
    "MEMUTUSKAN:\n"
    "BAB I\n"
    "KETENTUAN UMUM\n"
    "Pasal 1\n"
    "Dalam Peraturan ini yang dimaksud dengan ruang adalah wadah.\n"
    "Pasal 2\n"
    "Penataan ruang diselenggarakan berdasarkan asas keterpaduan.\n"
    "Pasal 3\n"
    "Penyelenggaraan penataan ruang bertujuan mewujudkan ruang wilayah.\n"
)


def _spans(text: str, country: str = "xl"):
    scan = scan_anchors_with_ambiguity(
        text, cached_regex(country, "act"), country=country, doctype="act"
    )
    return [s for s in scan.ambiguity if s.emitted_by == "declare_orphaned_drops"]


def test_a_load_bearing_drop_is_measured() -> None:
    declared = _spans(LOAD_BEARING_CITATION)
    assert len(declared) == 1
    detail = declared[0].detail
    assert detail["dropped_by"] == "drop_preamble_citation_articles"
    # The three definitions, which is what the drop actually took out of the body.
    assert detail["orphaned_chars"] > 400
    assert detail["ratio"] >= 10


def test_a_drop_that_takes_nothing_is_silent() -> None:
    assert _spans(CLEAN_CITATION) == []


def test_the_measure_never_blocks_the_ingest() -> None:
    """A refused document is worse than one with a fat preamble, so this must
    not reach `AnchorInvariantError` at the structure seam."""
    declared = _spans(LOAD_BEARING_CITATION)
    assert declared and not any(s.blocking for s in declared)


def test_the_span_covers_the_orphaned_run_not_the_marker() -> None:
    """The drop's own span ends at its matched text. This one runs to the next
    anchor, which is the whole point: it measures what was left behind."""
    span = _spans(LOAD_BEARING_CITATION)[0]
    assert span.end - span.start == span.detail["orphaned_chars"]
    assert span.end - span.start > 400


def test_another_jurisdiction_is_unaffected() -> None:
    """The drop is config-gated, so nothing is dropped here and
    nothing is measured."""
    assert _spans(LOAD_BEARING_CITATION, "al") == []


def test_the_validator_turns_the_measurement_into_a_warning() -> None:
    findings = _check_orphaned_drops(
        [{"dropped_by": "drop_toc_duplicates", "orphaned_chars": 1045, "ratio": 41.8}]
    )
    assert len(findings) == 1
    assert findings[0]["check"] == "orphaned_drops"
    # Warning, not error: every character survives, so the document is readable.
    assert findings[0]["severity"] == "warning"
    assert "1,045" in findings[0]["message"]


def test_the_validator_reports_the_worst_drop_of_several() -> None:
    findings = _check_orphaned_drops(
        [
            {"dropped_by": "drop_toc_duplicates", "orphaned_chars": 500, "ratio": 12.0},
            {
                "dropped_by": "drop_preamble_citation_articles",
                "orphaned_chars": 8402,
                "ratio": 236.7,
            },
        ]
    )
    assert "8,402" in findings[0]["message"]
    assert "drop_preamble_citation_articles" in findings[0]["message"]
    assert findings[0]["count"] == 2


def test_no_measurements_means_no_finding() -> None:
    assert _check_orphaned_drops([]) == []


def test_a_drop_with_no_comparable_gap_still_reads_as_a_sentence() -> None:
    """The scanner reports no ratio when the drops left no undisturbed gap, which
    is the case where they took nearly everything. The message must not say
    `Nonex this document's median anchor gap`."""
    findings = _check_orphaned_drops(
        [{"dropped_by": "drop_toc_duplicates", "orphaned_chars": 9000, "ratio": None}]
    )
    assert "None" not in findings[0]["message"]
    assert "no undisturbed anchor gap" in findings[0]["message"]


def test_a_drop_that_leaves_too_few_survivors_is_still_measured() -> None:
    """The more a drop takes, the fewer anchors survive it. Taking the median
    over the survivors would let the largest loss escape for want of a
    denominator, so the median comes from the pre-drop list."""
    body = "Ketentuan umum yang berlaku bagi seluruh penyelenggaraan urusan pemerintahan. " * 12
    almost_everything = (
        "Mengingat:\nPasal 5 ayat (1) Undang-Undang Dasar 1957;\nMEMUTUSKAN:\n"
        f"{body}\n"
        "BAB I\nKETENTUAN UMUM\n"
        "Pasal 2\nRepublik Langkasuka dibagi atas daerah provinsi.\n"
    )
    declared = _spans(almost_everything)
    assert len(declared) == 1
    assert declared[0].detail["orphaned_chars"] > 900


def test_a_large_loss_is_not_forgiven_for_being_proportionate() -> None:
    """In a document whose provisions are long, a 10x ratio is thousands of
    characters. The absolute ceiling declares it whatever the ratio says."""
    long_provision = "Ketentuan ini mengatur penyelenggaraan urusan pemerintahan daerah. " * 90
    spaced = (
        "Mengingat:\nPasal 5 ayat (1) Undang-Undang Dasar 1957;\nMEMUTUSKAN:\n"
        f"{long_provision}\n"
        "BAB I\nKETENTUAN UMUM\n"
        f"Pasal 2\n{long_provision}\n"
        f"Pasal 3\n{long_provision}\n"
        f"Pasal 4\n{long_provision}\n"
    )
    span = _spans(spaced)[0]
    # Proportionate to this document's own gaps, and still several pages of law.
    assert span.detail["ratio"] < 10
    assert span.detail["orphaned_chars"] > 4000
