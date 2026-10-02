"""Tests for codify.acquisition.pdf_metadata.extract_pdf_metadata.

Exercises three classes:
1. Date-string parsing, PDF 1.7 'D:' format with various omissions / TZs.
2. Extraction contrast, an OCR'd-style scan carries a sparse info dict (no
   /Producer, no /Title) but still yields a page count, while a digital-source
   PDF populates /Producer and /CreationDate. Built synthetically with pypdf
   (the same reader the code uses) so the open tree ships no real PDFs.
3. Defensive behaviour, corrupted / empty bytes don't raise.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from io import BytesIO

from pypdf import PdfWriter

from codify.acquisition.pdf_metadata import _parse_pdf_date, extract_pdf_metadata


def _scanned_style_pdf(pages: int = 3) -> bytes:
    """An OCR'd-scan stand-in: image-era pages, no info-dict metadata."""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=595, height=842)
    # pypdf stamps its own /Producer; blank it so the info dict is sparse like an
    # OCR'd scan's.
    writer.add_metadata({"/Producer": "", "/Title": ""})
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _digital_source_pdf() -> bytes:
    """A born-digital stand-in: an info dict with /Producer and /CreationDate."""
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    writer.add_metadata(
        {
            "/Producer": "Codify Synthetic Composer",
            "/Title": "Synthetic Digital Act",
            "/CreationDate": "D:20240115093045Z",
        }
    )
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_parse_pdf_date_full_with_tz():
    out = _parse_pdf_date("D:20240115093045+02'00'")
    assert out == datetime(2024, 1, 15, 9, 30, 45, tzinfo=timezone(timedelta(hours=2)))


def test_parse_pdf_date_no_tz_defaults_utc():
    out = _parse_pdf_date("D:20240115093045")
    assert out == datetime(2024, 1, 15, 9, 30, 45, tzinfo=UTC)


def test_parse_pdf_date_year_only():
    out = _parse_pdf_date("D:2024")
    assert out == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)


def test_parse_pdf_date_z_terminator():
    out = _parse_pdf_date("D:20240115093045Z")
    assert out == datetime(2024, 1, 15, 9, 30, 45, tzinfo=UTC)


def test_parse_pdf_date_with_parens():
    out = _parse_pdf_date("(D:20240115093045+02'00')")
    assert out == datetime(2024, 1, 15, 9, 30, 45, tzinfo=timezone(timedelta(hours=2)))


def test_parse_pdf_date_malformed_returns_none():
    assert _parse_pdf_date("not a date") is None
    assert _parse_pdf_date("") is None
    assert _parse_pdf_date(None) is None
    # Out-of-range fields
    assert _parse_pdf_date("D:20241345993099") is None


def test_extract_metadata_handles_corrupt_bytes():
    # Garbage bytes → empty metadata, never raises
    out = extract_pdf_metadata(b"not a pdf at all")
    assert out == {}


def test_extract_metadata_empty_bytes():
    out = extract_pdf_metadata(b"")
    assert out == {}


def test_scanned_style_pdf_has_page_count_but_sparse_info_dict():
    """OCR'd scans yield a page count even though the info dict is sparse, the
    extractor must not depend on /Title or /Producer being present."""
    out = extract_pdf_metadata(_scanned_style_pdf(pages=3))
    assert out.get("page_count") == 3
    # Sparse: no producer or title. (`extract_pdf_metadata` drops empty values.)
    assert not out.get("pdf_producer")
    assert not out.get("pdf_title")


def test_digital_source_pdf_populates_producer_and_creation_date():
    """A born-digital PDF carries the info-dict fields the scan lacks, the
    OCR-vs-digital contrast scanned fixtures used to exercise."""
    out = extract_pdf_metadata(_digital_source_pdf())
    assert out.get("page_count") == 1
    assert out.get("pdf_producer") == "Codify Synthetic Composer"
    assert out.get("pdf_creation_date") == datetime(2024, 1, 15, 9, 30, 45, tzinfo=UTC)
