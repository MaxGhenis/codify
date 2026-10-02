"""The constitutional citation in a Bahasa enacting formula is not a provision.

A `Mengingat` recital citing the constitution by Pasal can build a phantom
article: anchored on `Pasal 5`, it lands at <body> root before BAB I and
swallows `MEMUTUSKAN`.
"""

from __future__ import annotations

from codify.pipeline.enrich.anchors import cached_regex, scan_anchors_with_ambiguity

# A statute's opening as page text reads: the citation opens its
# own line, which is why `sameline_precursors` cannot catch it.
PREAMBLE = (
    "Menimbang: bahwa ruang wilayah Republik Langkasuka;\n"
    "Mengingat:\n"
    "Pasal 5 ayat (1), Pasal 20, Pasal 25A, dan Pasal 33 ayat (3)\n"
    "Undang-Undang Dasar Republik Langkasuka Tahun 1957;\n"
    "MEMUTUSKAN:\n"
    "Menetapkan: UNDANG-UNDANG TENTANG PENATAAN RUANG.\n"
    "BAB I\n"
    "KETENTUAN UMUM\n"
    "Pasal 1\n"
    "Dalam Undang-Undang ini yang dimaksud dengan ruang adalah wadah.\n"
    "BAB II\n"
    "ASAS DAN TUJUAN\n"
    "Pasal 2\n"
    "Penataan ruang diselenggarakan berdasarkan asas keterpaduan.\n"
)


def _scan(text: str, country: str):
    return scan_anchors_with_ambiguity(
        text, cached_regex(country, "act"), country=country, doctype="act"
    )


def test_the_citation_before_the_first_chapter_is_not_anchored() -> None:
    scan = _scan(PREAMBLE, "xl")
    articles = [a for a in scan.anchors if a.kind == "article"]
    assert [a.number for a in articles] == ["1", "2"]
    # The enacting formula stays ahead of every anchor, so the preamble split
    # that follows can see it.
    assert min(a.char_offset for a in scan.anchors) > PREAMBLE.index("MEMUTUSKAN")


def test_the_drop_is_recorded_as_resolved() -> None:
    scan = _scan(PREAMBLE, "xl")
    dropped = [s for s in scan.ambiguity if s.emitted_by == "drop_preamble_citation_articles"]
    assert len(dropped) == 1
    assert dropped[0].resolved is True
    assert dropped[0].detail["reads_as"] == "preamble_citation"
    assert dropped[0].blocking is False


def test_an_amending_act_keeps_its_roman_pasal() -> None:
    """An amending act's `Pasal I` carries the amendments and legitimately precedes a
    quoted container, so the pass declines for the whole document."""
    amending = (
        "Mengingat:\n"
        "Pasal 20, Pasal 21 Undang-Undang Dasar 1957;\n"
        "MEMUTUSKAN:\n"
        "Pasal I\n"
        "Beberapa ketentuan dalam Undang-Undang Nomor 8 Tahun 1991 diubah:\n"
        "BAB IX\n"
        "KETENTUAN PIDANA\n"
        "Pasal 40\n"
        "Setiap orang dilarang melakukan kegiatan yang mengakibatkan kerusakan.\n"
    )
    numbers = [a.number for a in _scan(amending, "xl").anchors if a.kind == "article"]
    assert "I" in numbers
    assert "20" in numbers  # untouched: the pass declines document-wide


def test_a_document_with_no_container_keeps_its_root_articles() -> None:
    flat = (
        "MEMUTUSKAN:\n"
        "Pasal 1\n"
        "Dalam Peraturan ini yang dimaksud dengan Menteri adalah menteri.\n"
        "Pasal 2\n"
        "Peraturan ini mulai berlaku pada tanggal diundangkan.\n"
    )
    numbers = [a.number for a in _scan(flat, "xl").anchors if a.kind == "article"]
    assert numbers == ["1", "2"]


def test_a_flat_body_survives_a_structured_annex() -> None:
    """A LAMPIRAN carries its own hierarchy. If the annex opens with BAB I and
    the body is flat, the annex's chapter must not be read as the body's first
    container, which would drop every real article ahead of it."""
    flat_body_structured_annex = (
        "Mengingat:\n"
        "Pasal 5 ayat (1) Undang-Undang Dasar 1957;\n"
        "MEMUTUSKAN:\n"
        "Pasal 1\n"
        "Dalam Peraturan Pemerintah ini yang dimaksud dengan Menteri adalah menteri.\n"
        "Pasal 2\n"
        "Peraturan Pemerintah ini mulai berlaku pada tanggal diundangkan.\n"
        "LAMPIRAN\n"
        "PEDOMAN TEKNIS\n"
        "BAB I\n"
        "UMUM\n"
        "Pasal 1\n"
        "Pedoman teknis ini menjadi acuan pelaksanaan.\n"
    )
    numbers = [
        a.number for a in _scan(flat_body_structured_annex, "xl").anchors if a.kind == "article"
    ]
    assert "2" in numbers  # the flat body's own articles are untouched
    assert numbers.count("1") >= 1


def test_another_jurisdiction_is_unaffected() -> None:
    """The pass is config-gated, so a jurisdiction that has not declared the
    convention keeps whatever it anchored before."""
    scan = _scan(PREAMBLE, "al")
    assert not [s for s in scan.ambiguity if s.emitted_by == "drop_preamble_citation_articles"]
