"""Ingest EU directives via AKN4EU passthrough or FORMEX transformation.

The mapper follows AKN4EU v3.0, LegalDocML AKN-Core v1.0 and FORMEX 5.55.
Cellar table-of-contents wrappers are rejected because they contain no act text.
"""

from __future__ import annotations

import asyncio
import contextvars
import re
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from lxml import etree

from codify.acquisition.adapters.eu.cellar import celex_to_frbr as _celex_to_frbr_tuple
from codify.akn import Document
from codify.akn._schema import parse_xml
from codify.akn.io import parse_akn
from codify.akn.vocabulary import table_row_eid
from codify.core.tracing import attach_trace_attribution, langfuse_trace_context
from codify.frbr import build_frbr_work_uri
from codify.pipeline.events import Complete, Failed, IngestionEvent, Parsed
from codify.pipeline.formats.eu_html import html_to_akn4eu, is_eurlex_html

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"


class FormexNotAnActError(ValueError):
    """Raised when FORMEX input is not a directive/regulation/decision body."""


class FormexDateMissingError(ValueError):
    """Raised when no act-level date can be extracted from FORMEX."""


# Roots that indicate metadata/index wrappers, not the act itself.
_NON_ACT_ROOTS = {"DOC", "PUBLICATION", "ROOT.CONS"}

# AKN abbreviations for hierarchical eId construction (chp_1__art_2__para_1).
# Per OASIS Naming Convention v1.0 §5, these are normative.
_EID_ABBREV = {
    "title": "tit",
    "part": "part",
    "chapter": "chp",
    "section": "sec",
    "subsection": "subsec",
    "article": "art",
    "paragraph": "para",
    "subparagraph": "subpara",
    "point": "pt",
    "list": "list",
    "item": "item",
}

# FORMEX HT/@TYPE → AKN inline tag. Non-mapped types (UC, STROKE, etc.)
# are handled in _serialize_inline's HT branch, UC text-upper-cases,
# STROKE preserves the legal-semantic via <inline name="strike">.
_HT_TYPE_TO_TAG: dict[str, str] = {
    "ITALIC": "i",
    "BOLD": "b",
    "SUP": "sup",
    "SUB": "sub",
    "EXPANDED": "b",
}

# Compact-date pattern used by _formex_date and _publication_from_bib.
_FORMEX_COMPACT_DATE = re.compile(r"^\d{8}$")

# DIVISION/@TYPE → AKN container element.
_DIV_TYPE_TO_AKN = {
    "PT": "part",
    "CHA": "chapter",
    "SEC": "section",
    "SUB": "subsection",
}


# Its own pool, not the loop's default executor: that one is shared with the
# readiness probe, the PDF export and the OCR crops.
DEFAULT_CPU_THREADS = 4
_CPU_POOL = ThreadPoolExecutor(max_workers=DEFAULT_CPU_THREADS, thread_name_prefix="codify-cpu")


def configure_cpu_pool(max_workers: int) -> None:
    """Resize the pipeline's CPU pool. The API calls this at startup from its own
    settings, so the value follows `.env` and `make run` rather than the process
    environment this package would otherwise have to read at import.
    """
    global _CPU_POOL
    if max_workers < 1:
        raise ValueError(f"pipeline CPU threads must be at least 1, got {max_workers}")
    old, _CPU_POOL = (
        _CPU_POOL,
        ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="codify-cpu"),
    )
    old.shutdown(wait=False)


def _validate_native(xml: str) -> list[dict[str, Any]]:
    """The validator for publisher-structured AKN, whose dates and numbering are the
    publisher's own."""
    from codify.pipeline.enrich.validator import validate_akn

    return validate_akn(xml, provenance="native")


async def _on_the_cpu_pool(fn: Callable[..., Any], *args: Any) -> Any:
    """Run a synchronous pipeline pass on the CPU pool, carrying the caller's context.

    `run_in_executor` does not copy contextvars the way `to_thread` does, so without
    this the structlog bindings and the enclosing langfuse span are absent in the
    worker and the pass logs unattributed.
    """
    ctx = contextvars.copy_context()
    return await asyncio.get_running_loop().run_in_executor(_CPU_POOL, lambda: ctx.run(fn, *args))


async def on_the_cpu_pool(fn: Callable[..., Any], *args: Any) -> Any:
    """Public name for the pool, for callers outside this module.

    The API's routes need the same bounded pool the pipeline uses, so that one
    interactive request and a batch cannot each claim their own threads. Exposed
    by name rather than by handing out the module.
    """
    return await _on_the_cpu_pool(fn, *args)


def _detect_format(xml: str | bytes) -> Literal["akn4eu", "formex"]:
    """Inspect the root element to decide format."""
    try:
        root = parse_xml(xml)
    except etree.XMLSyntaxError as exc:
        raise ValueError(f"input is not valid XML: {exc}") from exc
    tag = etree.QName(root).localname
    if tag == "akomaNtoso":
        return "akn4eu"
    if tag in {
        "DOCUMENT",
        "FORMEX",
        "REG",
        "DIR",
        "DEC",
        "CONS_TEXT",
        "CONS_ACT",
        "ACT",
        "DOC",
        "GENERAL",
    }:
        return "formex"
    raise ValueError(f"unknown EU XML root element: {root.tag!r}")


_SUBTYPE_TO_DOCTYPE = {"dir": "directive", "reg": "regulation", "dec": "decision"}


def celex_to_frbr(celex: str) -> str:
    """Map a CELEX to its /akn/eu/act/{subtype}/{year}/{number} FRBR work URI.

    Follows OASIS Naming Convention §3, matching Indigo and legislation.gov.uk.
    Cellar publishes no AKN4EU manifestations, so we adopt the convention.
    """
    uri, _doctype = _celex_to_frbr_tuple(celex)
    return uri


def _strip_foreign_namespaces(root: etree._Element) -> None:
    """Drop elements outside the AKN namespace.

    EUR-Lex AKN4EU wraps EU-specific identifiers (eli:, eu:, fmx:) around the
    canonical body. None are load-bearing for ingestion.
    """
    for el in list(root.iter()):
        ns = etree.QName(el).namespace
        if ns and ns != AKN_NS:
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)


def parse_akn4eu(xml: str | bytes) -> Document:
    """Strip EU-specific namespaces, then parse via codify.akn.io.parse_akn."""
    root = parse_xml(xml)
    _strip_foreign_namespaces(root)
    cleaned = etree.tostring(root, xml_declaration=True, encoding="UTF-8")
    return parse_akn(cleaned)


# FORMEX carries no doctype field, so the title is the only signal in the
# document itself. A CELEX, where the caller has one, is definitive instead.
_TITLE_DOCTYPE = re.compile(r"\b(directive|regulation|decision)\b", re.IGNORECASE)


def _doctype_from_title(src: etree._Element) -> str | None:
    """The act's own type, read from its title."""
    title = src.find(".//TITLE")
    if title is None:
        return None
    text = " ".join(title.itertext())
    found = _TITLE_DOCTYPE.search(text)
    return found.group(1).lower() if found else None


def _frbr_from_formex(xml: str | bytes) -> str:
    """Build a FRBR work URI from FORMEX BIB.DOC fields. Falls back to placeholder.

    Emits the /akn/eu/act/{subtype}/{year}/{number} shape (see celex_to_frbr).
    """
    src = parse_xml(xml)
    doctype = _doctype_from_title(src) or "directive"
    no_doc = src.find(".//BIB.DOC/NO.DOC")
    if no_doc is None:
        no_doc = src.find(".//NO.DOC")
    if no_doc is not None:
        year_el = no_doc.find("YEAR")
        number_el = no_doc.find("NO.CURRENT")
        if year_el is not None and number_el is not None:
            year = (year_el.text or "").strip()
            number = (number_el.text or "").strip().lstrip("0") or "0"
            return build_frbr_work_uri("eu", doctype, year, number)
    return build_frbr_work_uri("eu", doctype, "0000", "0")


def _unwrap_general(src: etree._Element) -> None:
    """Lift a GENERAL document's CONTENTS children to the root, the shape of an ACT.

    Older FORMEX publishes some acts as GENERAL: BIB.INSTANCE, TITLE, then
    CONTENTS holding PREAMBLE, DIVISION/ARTICLE and FINAL.
    """
    if etree.QName(src).localname != "GENERAL":
        return
    contents = src.find("CONTENTS")
    if contents is None:
        return
    for child in list(contents):
        contents.addprevious(child)
    src.remove(contents)


def _require_act_with_articles(src: etree._Element) -> None:
    """Reject TOC-stub inputs (DOC/PUBLICATION roots, ACT without ARTICLE)."""
    root_tag = etree.QName(src).localname
    if root_tag in _NON_ACT_ROOTS:
        raise FormexNotAnActError(f"root <{root_tag}> is an OJ index wrapper, not a directive body")
    if not src.xpath(".//ARTICLE"):
        raise FormexNotAnActError(
            f"<{root_tag}> contains zero <ARTICLE> elements — likely a TOC stub"
        )


def _formex_date(src: etree._Element) -> str | None:
    """Extract the act-level date from FORMEX, or None if missing.

    Search order, most authoritative first:
      1. BIB.INSTANCE/DATE, canonical in FORMEX 5.x
      2. DATE.OF.DOCUMENT / DATE.OF.SIGNATURE / DATE.OF.PUBLICATION, older dialect
      3. PREAMBLE.FINAL//DATE, signature date in the preamble closer
      4. FINAL//DATE / SIGNATURE//DATE, signature in conclusions
      5. TITLE//DATE, inline date within the long title
    """
    candidates: list[etree._Element] = []
    candidates.extend(src.findall(".//BIB.INSTANCE/DATE"))
    for tag in ("DATE.OF.DOCUMENT", "DATE.OF.SIGNATURE", "DATE.OF.PUBLICATION"):
        candidates.extend(src.findall(f".//{tag}"))
    candidates.extend(src.findall(".//PREAMBLE.FINAL//DATE"))
    candidates.extend(src.findall(".//FINAL//DATE"))
    candidates.extend(src.findall(".//SIGNATURE//DATE"))
    candidates.extend(src.findall(".//TITLE//DATE"))
    for el in candidates:
        iso = el.get("ISO") or el.get("DATE") or (el.text or "").strip()
        if iso and _FORMEX_COMPACT_DATE.match(iso):
            iso = f"{iso[:4]}-{iso[4:6]}-{iso[6:8]}"
        if iso and re.match(r"^\d{4}-\d{2}-\d{2}$", iso):
            return iso
    return None


def _doctype_from_frbr(frbr_work_uri: str) -> str:
    """Pull the doctype slug from a FRBR work URI; falls back to 'directive'.

    The {subtype} segment is `dir`/`reg`/`dec`, mapped to the readable form for
    <act @name>.
    """
    parts = frbr_work_uri.strip("/").split("/")
    if len(parts) >= 4 and parts[2] == "act":
        return _SUBTYPE_TO_DOCTYPE.get(parts[3], "directive")
    return "directive"


def _publication_from_bib(
    src: etree._Element, provenance: dict[str, object] | None = None
) -> dict[str, str]:
    """Extract the OJ publication reference from BIB.INSTANCE/DOCUMENT.REF.

    Always non-None, so the section is emitted even when partly populated. Missing
    fields land in `provenance['publication_missing']`, which the ingest generator
    yields as a `ValidationIssued` event against the AKN4EU profile.
    """
    ref = src.find(".//BIB.INSTANCE/DOCUMENT.REF")
    coll = (ref.findtext("COLL") or "").strip() if ref is not None else ""
    no_oj = (ref.findtext("NO.OJ") or "").strip() if ref is not None else ""
    year = (ref.findtext("YEAR") or "").strip() if ref is not None else ""
    iso = ""
    date_el = src.find(".//BIB.INSTANCE/DATE")
    if date_el is not None:
        raw = date_el.get("ISO") or (date_el.text or "").strip()
        if _FORMEX_COMPACT_DATE.match(raw):
            iso = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
        elif re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
            iso = raw

    missing = [
        f for f, v in (("coll", coll), ("no_oj", no_oj), ("year", year), ("date", iso)) if not v
    ]
    if missing and provenance is not None:
        provenance["publication_missing"] = missing

    show_as = (
        f"OJ {coll} {no_oj}, {iso}"
        if (coll and no_oj and iso)
        else f"OJ {coll or '?'} {no_oj or '?'}"
    )
    return {
        "name": "Official Journal of the European Union",
        "showAs": show_as,
        "number": no_oj or "unknown",
        "date": iso,
    }


def _stub_meta(
    *,
    frbr_work_uri: str,
    language: str = "eng",
    expression_date: str,
    src: etree._Element | None = None,
    acquis_chapter: int | None = None,
    provenance: dict[str, object] | None = None,
) -> etree._Element:
    """Build the AKN <meta> block: identification + publication + lifecycle
    + classification + references container.
    """
    doctype = _doctype_from_frbr(frbr_work_uri)
    meta = etree.Element("{%s}meta" % AKN_NS)

    ident = _akn(meta, "identification", source="#codify")
    work = _akn(ident, "FRBRWork")
    _akn(work, "FRBRthis", value=f"{frbr_work_uri}/main")
    _akn(work, "FRBRuri", value=frbr_work_uri)
    _akn(work, "FRBRdate", date=expression_date, name="generation")
    _akn(work, "FRBRauthor", href="#eu")
    _akn(work, "FRBRcountry", value="eu")
    _akn(work, "FRBRsubtype", value=doctype)
    expr = _akn(ident, "FRBRExpression")
    _akn(expr, "FRBRthis", value=f"{frbr_work_uri}/{language}@{expression_date}/main")
    _akn(expr, "FRBRuri", value=f"{frbr_work_uri}/{language}@{expression_date}")
    _akn(expr, "FRBRdate", date=expression_date, name="generation")
    _akn(expr, "FRBRauthor", href="#eu")
    _akn(expr, "FRBRlanguage", language=language)
    manif = _akn(ident, "FRBRManifestation")
    _akn(manif, "FRBRthis", value=f"{frbr_work_uri}/{language}@{expression_date}/main.xml")
    _akn(manif, "FRBRuri", value=f"{frbr_work_uri}/{language}@{expression_date}.xml")
    _akn(manif, "FRBRdate", date=expression_date, name="generation")
    _akn(manif, "FRBRauthor", href="#codify")
    _akn(manif, "FRBRformat", value="xml")

    pub: dict[str, str] = {}
    if src is not None:
        pub = _publication_from_bib(src, provenance=provenance)
        # Always emit <publication>, partial is better than absent for
        # AKN4EU profile compliance. Missing fields surface via provenance.
        _akn(meta, "publication", **pub)

    # Lifecycle: emit the publication event from BIB.INSTANCE/DATE if we
    # have one; else fall back to the expression date. AKN convention is
    # that <lifecycle> records real legal events, not authorial actions.
    lifecycle = _akn(meta, "lifecycle", source="#codify")
    pub_date = pub.get("date") or expression_date
    _akn(
        lifecycle,
        "eventRef",
        eId="evt_publication",
        date=pub_date,
        type="generation",
        source="#publication",
    )

    if acquis_chapter is not None:
        classification = _akn(meta, "classification", source="#codify")
        _akn(
            classification,
            "keyword",
            value=f"acquis-chapter-{acquis_chapter:02d}",
            showAs=f"Acquis chapter {acquis_chapter}",
            dictionary="#dgnear",
        )

    refs = _akn(meta, "references", source="#codify")
    _akn(
        refs,
        "TLCOrganization",
        eId="eu",
        href="/ontology/organization/eu",
        showAs="European Union",
    )
    _akn(
        refs,
        "TLCOrganization",
        eId="codify",
        href="/ontology/organization/codify",
        showAs="Codify",
    )
    if acquis_chapter is not None:
        _akn(
            refs,
            "TLCConcept",
            eId=f"acquis_{acquis_chapter:02d}",
            href=f"/ontology/concept/acquis/chapter-{acquis_chapter:02d}",
            showAs=f"Acquis chapter {acquis_chapter}",
        )
    return meta


def _akn(parent: etree._Element, tag: str, **attrs: str) -> etree._Element:
    """Append an AKN-namespaced child with attributes, skipping None values."""
    el = etree.SubElement(parent, "{%s}%s" % (AKN_NS, tag))
    for k, v in attrs.items():
        if v is not None:
            el.set(k, v)
    return el


def _localname(el: etree._Element) -> str:
    return etree.QName(el).localname if isinstance(el.tag, str) else ""


def _eid(path: tuple[str, ...], leaf: str) -> str:
    return "__".join((*path, leaf)) if path else leaf


def _ref_doc_oj_href(ref: etree._Element) -> str | None:
    """Build a FRBR href from REF.DOC.OJ subelements."""
    coll = (ref.findtext("COLL") or "").strip()
    no_oj = (ref.findtext("NO.OJ") or "").strip()
    year = (ref.findtext("YEAR") or "").strip()
    if coll and no_oj and year:
        return f"/akn/eu/officialgazette/{year}/{coll}/{no_oj}"
    return None


def _serialize_inline(src: etree._Element, akn_parent: etree._Element) -> None:
    """Copy mixed text and recognised inline children into `akn_parent`.

      HT             <i>/<b>/<sup>/<sub>/<inline name="uc">
      DATE           <date date="ISO">text</date>
      NOTE           <authorialNote>
      FT             <noteRef> when it references a note, plain text otherwise
      REF.DOC.OJ     <ref href="...">
      QUOT.START/END the curly mark as text; AKN4EU reserves <quotedText> for
                     amendment contexts, so intervening markup stays as siblings

    Unknown inline elements have their text flattened rather than dropped.
    """
    if src.text:
        akn_parent.text = (akn_parent.text or "") + src.text
    last: etree._Element | None = None

    def _add_text(text: str) -> None:
        nonlocal last
        if not text:
            return
        if last is not None:
            last.tail = (last.tail or "") + text
        else:
            akn_parent.text = (akn_parent.text or "") + text

    children = list(src)
    i = 0
    while i < len(children):
        child = children[i]
        if not isinstance(child.tag, str):
            if child.tail:
                _add_text(child.tail)
            i += 1
            continue
        local = _localname(child)
        if local == "HT":
            type_attr = (child.get("TYPE") or "ITALIC").upper()
            mapping = _HT_TYPE_TO_TAG.get(type_attr)
            if mapping is None:
                if type_attr == "STROKE":
                    # Strikethrough has legal semantics in amendment text, so it is
                    # preserved as <inline name="strike"> for round-trippability.
                    new = _akn(akn_parent, "inline", name="strike")
                    _serialize_inline(child, new)
                    last = new
                elif type_attr == "UC":
                    # UC is presentational; AKN has no semantic equivalent.
                    # Apply the upper-case to the text content and inline it.
                    _add_text("".join(child.itertext()).upper())
                else:
                    # Unrecognised decoration, preserve text, drop the tag.
                    _add_text("".join(child.itertext()))
            else:
                new = _akn(akn_parent, mapping)
                _serialize_inline(child, new)
                last = new
        elif local == "DATE":
            iso = child.get("ISO") or ""
            iso_norm = (
                f"{iso[:4]}-{iso[4:6]}-{iso[6:8]}" if _FORMEX_COMPACT_DATE.match(iso) else iso
            )
            new = _akn(akn_parent, "date", date=iso_norm or "")
            new.text = "".join(child.itertext()).strip() or " "
            last = new
        elif local in {"QUOT.START", "QUOT.END"}:
            _add_text("“" if local == "QUOT.START" else "”")
        elif local == "NOTE":
            # FORMEX NOTE to AKN <authorialNote eId="fn_<NOTE.ID>">. The marker
            # is the visible footnote number, not the NUMBERING style enum.
            note = _akn(akn_parent, "authorialNote")
            note_type = (child.get("TYPE") or "").upper()
            if note_type == "FOOTNOTE":
                note.set("placement", "bottom")
            note_id = child.get("NOTE.ID") or ""
            if note_id:
                note.set("eId", f"fn_{note_id}")
            inner = _akn(note, "p")
            _serialize_inline(child, inner)
            last = note
        elif local == "FT":
            # FT by @TYPE: FOOTNOTE or @REF.NOTE emits <noteRef>; NUMBER and
            # anything else emits plain text, AKN having no semantic for them.
            ft_type = (child.get("TYPE") or "").upper()
            ref_note = child.get("REF.NOTE") or ""
            text = "".join(child.itertext())
            if ft_type == "FOOTNOTE" or ref_note:
                ref_el = _akn(
                    akn_parent,
                    "noteRef",
                    href=f"#fn_{ref_note}" if ref_note else "#unresolved",
                    marker=text or " ",
                )
                last = ref_el
            else:
                _add_text(text)
        elif local == "REF.DOC.OJ":
            ref_el = _akn(akn_parent, "ref", href=_ref_doc_oj_href(child) or "#unresolved")
            ref_el.text = "".join(child.itertext()).strip() or " "
            last = ref_el
        else:
            _add_text("".join(child.itertext()))
        if child.tail:
            _add_text(child.tail)
        i += 1


# --- Block-level emitters ----------------------------------------------------


def _wrapped_find(formex: etree._Element, tag: str) -> etree._Element | None:
    """A division's own child, or the one inside its TITLE wrapper."""
    found = formex.find(tag)
    if found is None:
        title_wrap = formex.find("TITLE")
        if title_wrap is not None:
            found = title_wrap.find(tag)
    return found


# What a title splits across, measured on the corpus: a TI holds NP or P, and a
# number and its text sit in sibling NO.P and TXT. Everything else inside a title
# is inline markup whose tail continues the sentence.
_TITLE_BLOCK_PARTS = {"NP", "P", "TXT", "NO.P", "TI", "STI"}


def _title_text(element: etree._Element) -> str:
    """Flatten a title. Block parts are separated, inline content stays adjacent:
    concatenating throughout welds a number to its text, and separating throughout
    parts an inline element from the punctuation after it."""
    chunks: list[str] = []
    buf: list[str] = []

    def walk(node: etree._Element, inline: bool) -> None:
        buf.append(node.text or "")
        for child in node:
            if not isinstance(child.tag, str):
                buf.append(child.tail or "")
                continue
            # Inside an inline element everything is inline, whatever it is named:
            # a NOTE wraps its body in a P, and splitting there strands the
            # punctuation that follows the note.
            if not inline and _localname(child) in _TITLE_BLOCK_PARTS:
                chunks.append("".join(buf))
                buf.clear()
                walk(child, False)
                chunks.append("".join(buf))
                buf.clear()
            else:
                walk(child, True)
            buf.append(child.tail or "")

    walk(element, False)
    chunks.append("".join(buf))
    return " ".join(" ".join(c.split()) for c in chunks if c.strip())


def _emit_num_heading(formex: etree._Element, akn_parent: etree._Element) -> None:
    """Pull <NO>/<NO.SEQ>, <TI> and <STI> into <num>/<heading>/<subheading>."""
    no = formex.find("NO")
    if no is None:
        no = formex.find("NO.SEQ")
    if no is not None and (no.text or "").strip():
        _akn(akn_parent, "num").text = no.text.strip()
    for tag, akn_name in (("TI", "heading"), ("STI", "subheading")):
        found = _wrapped_find(formex, tag)
        if found is None:
            continue
        text = _title_text(found)
        if text:
            _akn(akn_parent, akn_name).text = text


def _flat(element: etree._Element) -> str:
    """An element's text, whitespace collapsed. Not for a title: `_title_text`
    separates a number from its text, which this welds. The quoted-prose fallback
    renders this into a `<p>`, so it can weld there too."""
    return " ".join("".join(element.itertext()).split())


def _emit_article_num_heading(formex: etree._Element, akn_parent: etree._Element) -> None:
    """ARTICLE: <TI.ART><STI.ART>num</STI.ART><TI><P>heading</P></TI></TI.ART>."""
    ti_art = formex.find("TI.ART")
    if ti_art is None:
        return
    sti = ti_art.find("STI.ART")
    num = _title_text(sti) if sti is not None else ""
    heading = ""
    ti = ti_art.find("TI")
    if ti is not None:
        heading = _title_text(ti)
    if not num:
        # Plain Formex 4 puts the number in TI.ART itself and the subtitle in a
        # sibling STI.ART, inverting the nesting the branch above reads.
        num = _title_text(ti_art)
        sibling = formex.find("STI.ART")
        if not heading and sibling is not None:
            heading = _title_text(sibling)
    if num:
        _akn(akn_parent, "num").text = num
    if heading:
        _akn(akn_parent, "heading").text = heading


_HEADING_TYPE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(BOOK|LIVRE|TITLE)\b", re.IGNORECASE), "title"),
    (re.compile(r"^(PART|PARTIE)\b", re.IGNORECASE), "part"),
    (re.compile(r"^(CHAPTER|CHAPITRE)\b", re.IGNORECASE), "chapter"),
    (re.compile(r"^(SUBSECTION|SOUS-SECTION)\b", re.IGNORECASE), "subsection"),
    (re.compile(r"^(SECTION)\b", re.IGNORECASE), "section"),
)


def _infer_division_type(formex: etree._Element) -> str:
    """Infer the AKN container type when DIVISION/@TYPE is absent.

    FORMEX directives commonly nest TITLE→CHAPTER→SECTION as three untyped
    <DIVISION> elements distinguished only by their <TI> heading keyword, so the
    keyword picks the container type; no match falls back to <chapter>.
    """
    ti = formex.find("TI")
    if ti is None:
        title_wrap = formex.find("TITLE")
        if title_wrap is not None:
            ti = title_wrap.find("TI")
    if ti is not None:
        text = _title_text(ti)
        for pattern, akn_type in _HEADING_TYPE_PATTERNS:
            if pattern.match(text):
                return akn_type
    return "chapter"


# Children the body emitters read for a heading, or drop on purpose: counting
# them as losses would report a clean conversion as lossy.
# Read by another pass from the document root, so their level does not matter.
# What a pass emits only sometimes, or reads only as a direct child, is passed in
# instead, so an unread one still counts as a loss.
_HANDLED_ELSEWHERE = {
    "BIB.INSTANCE",
    "DATE.OF.DOCUMENT",
    "DATE.OF.SIGNATURE",
    "DATE.OF.PUBLICATION",
}
# What `_emit_paragraph` reads; a PARAG child outside this set is dropped.
_PARAG_EMITTED = {"NO.PARAG", "ALINEA", "LIST", "TXT", "P"}


def _heading_sources(src: etree._Element) -> set[etree._Element]:
    """The elements `_emit_num_heading` actually reads, by making the same finds.

    Naming the tags instead once claimed a `TITLE`-nested `NO` was read when
    nothing reads it, which silenced the loss rather than counting it.
    """
    read: set[etree._Element] = set()
    no = src.find("NO")
    if no is None:
        no = src.find("NO.SEQ")
    if no is not None and (no.text or "").strip():
        read.add(no)
    for tag in ("TI", "STI"):
        found = _wrapped_find(src, tag)
        if found is not None:
            read.add(found)
    return read


def _article_heading_sources(src: etree._Element) -> set[etree._Element]:
    """The same for `_emit_article_num_heading`. It branches on the number having
    text, not on the element being there, and reads the sibling subtitle only
    when the wrapper gave it no heading."""
    read: set[etree._Element] = set()
    ti_art = src.find("TI.ART")
    if ti_art is None:
        return read
    nested = ti_art.find("STI.ART")
    ti = ti_art.find("TI")
    if nested is not None and _flat(nested):
        read.add(nested)
        if ti is not None:
            read.add(ti)
        return read
    # No number in the nested subtitle, so the whole wrapper is flattened into it.
    read.add(ti_art)
    if ti is None or not _flat(ti):
        sibling = src.find("STI.ART")
        if sibling is not None:
            read.add(sibling)
    return read


def _bears_text(element: etree._Element) -> bool:
    """Comments and processing instructions are nodes, not elements, and itertext
    raises on them."""
    if not isinstance(element.tag, str):
        return False
    return bool("".join(element.itertext()).strip())


def _count_discarded_body_text(
    body_source: etree._Element, emitted_apart: set[etree._Element] | None = None
) -> int:
    """Source text the body walk has no branch for, counted rather than emitted.

    A bare paragraph is not valid under <body>, so a shape with nowhere to go is
    reported instead of guessed at. Mirrors the walk, so it drifts if that changes.
    """
    lost = 0

    def loose(parent: etree._Element) -> int:
        n = 1 if (parent.text or "").strip() else 0
        return n + sum(1 for c in parent if (c.tail or "").strip())

    def carries_something(child: etree._Element) -> bool:
        # Attributes count: `itertext` never sees a file reference, so an element
        # holding only one is dropped without a word of text going missing. A
        # comment is not content, and `len` counts one.
        elements = any(isinstance(g.tag, str) for g in child)
        return _bears_text(child) or elements or bool(child.attrib)

    apart = emitted_apart or set()

    def paragraph(src: etree._Element) -> int:
        """One level into a PARAG. It stops there, so the total is a floor."""
        return loose(src) + sum(
            1
            for c in src
            if isinstance(c.tag, str)
            and c not in apart
            and _localname(c) not in _PARAG_EMITTED
            and carries_something(c)
        )

    def article(src: etree._Element) -> int:
        kids = [c for c in src if isinstance(c.tag, str) and c not in apart]
        parags = [c for c in kids if _localname(c) in {"PARAG", "PAR"}]
        alineas = [c for c in kids if _localname(c) == "ALINEA"]
        read = _article_heading_sources(src)
        if not parags and not alineas:
            # The text fallback keeps every word, so only what has no text is lost.
            return sum(
                1 for c in kids if c not in read and not _bears_text(c) and carries_something(c)
            )
        kept = {"PARAG", "PAR"} if parags else {"ALINEA"}
        skipped = sum(paragraph(c) for c in parags)
        for c in kids:
            if _localname(c) in kept or c in read:
                continue
            if _localname(c) == "TI.ART":
                # Only its number and title are read; anything else it holds is not.
                skipped += sum(1 for g in c if g not in read and carries_something(g))
            elif carries_something(c):
                skipped += 1
        return skipped + loose(src)

    def division(src: etree._Element) -> int:
        # No cutoff: the walk it mirrors has none, so one here would count zero
        # for a document it converts fine.
        n = loose(src)
        read = _heading_sources(src)
        for c in src:
            if not isinstance(c.tag, str) or c in apart:
                continue
            local = _localname(c)
            if local in {"DIVISION", "CHAP", "SECTION"}:
                n += division(c)
            elif local == "ARTICLE":
                n += article(c)
            elif c in read:
                continue
            elif local == "TITLE":
                # Only the wrapper's TI is read; a NO or STI beside it is not.
                n += sum(1 for g in c if g not in read and carries_something(g))
            elif carries_something(c):
                n += 1
        return n

    lost += loose(body_source)
    for child in body_source:
        if not isinstance(child.tag, str):
            continue
        if child in apart:
            continue
        local = _localname(child)
        if local in {"DIVISION", "CHAP", "SECTION"}:
            lost += division(child)
        elif local == "ARTICLE":
            lost += article(child)
        elif local not in _HANDLED_ELSEWHERE and carries_something(child):
            lost += 1
    return lost


def _emit_division(
    formex: etree._Element,
    parent: etree._Element,
    eid_path: tuple[str, ...],
    counts: dict[str, int],
) -> None:
    """DIVISION → title/part/chapter/section/subsection.

    Resolution order: explicit @TYPE → heading-keyword inference → default.
    """
    type_attr = (formex.get("TYPE") or "").upper()
    if type_attr in _DIV_TYPE_TO_AKN:
        akn_type = _DIV_TYPE_TO_AKN[type_attr]
    elif _localname(formex) == "SECTION":
        akn_type = "section"
    elif _localname(formex) == "CHAP":
        akn_type = "chapter"
    else:
        akn_type = _infer_division_type(formex)
    counts[akn_type] = counts.get(akn_type, 0) + 1
    leaf = f"{_EID_ABBREV[akn_type]}_{counts[akn_type]}"
    div = _akn(parent, akn_type, eId=_eid(eid_path, leaf))
    _emit_num_heading(formex, div)
    new_path = (*eid_path, leaf)
    sub_counts: dict[str, int] = {}
    for child in formex:
        local = _localname(child)
        if local in {"DIVISION", "CHAP", "SECTION"}:
            _emit_division(child, div, new_path, sub_counts)
        elif local == "ARTICLE":
            _emit_article(child, div, new_path, sub_counts)


def _emit_article(
    formex: etree._Element,
    parent: etree._Element,
    eid_path: tuple[str, ...],
    counts: dict[str, int],
) -> None:
    """ARTICLE → <article eId="art_<N>">; identifier from @IDENTIFIER if present."""
    identifier = (formex.get("IDENTIFIER") or "").lstrip("0") or ""
    if not identifier:
        counts["article"] = counts.get("article", 0) + 1
        identifier = str(counts["article"])
    leaf = f"art_{identifier}"
    art = _akn(parent, "article", eId=_eid(eid_path, leaf))
    _emit_article_num_heading(formex, art)
    new_path = (*eid_path, leaf)

    parags = [c for c in formex if _localname(c) in {"PARAG", "PAR"}]
    if parags:
        for index, parag in enumerate(parags, start=1):
            _emit_paragraph(parag, art, new_path, index)
        return

    alineas = [c for c in formex if _localname(c) == "ALINEA"]
    if alineas:
        para = _akn(art, "paragraph", eId=_eid(new_path, "para_1"))
        content = _akn(para, "content")
        for alinea in alineas:
            _alinea_into_content(alinea, content, (*new_path, "para_1"))
        return

    text = " ".join("".join(formex.itertext()).split())
    if text:
        para = _akn(art, "paragraph", eId=_eid(new_path, "para_1"))
        content = _akn(para, "content")
        _akn(content, "p").text = text


def _emit_paragraph(
    formex: etree._Element,
    parent: etree._Element,
    eid_path: tuple[str, ...],
    index: int,
) -> None:
    """PARAG → <paragraph>; NO.PARAG → <num>; ALINEA + LIST → <content>."""
    no = formex.find("NO.PARAG")
    if no is not None and (no.text or "").strip():
        ident = re.sub(r"[^\d]", "", no.text) or str(index)
    else:
        ident = str(index)
    leaf = f"para_{ident}"
    para = _akn(parent, "paragraph", eId=_eid(eid_path, leaf))
    if no is not None and (no.text or "").strip():
        _akn(para, "num").text = no.text.strip()
    content = _akn(para, "content")
    new_path = (*eid_path, leaf)

    for child in formex:
        local = _localname(child)
        if local == "ALINEA":
            _alinea_into_content(child, content, new_path)
        elif local == "LIST":
            _emit_list(child, content, new_path)
        elif local in {"TXT", "P"}:
            # Direct PARAG > TXT|P children, emit each as its own <p>.
            p = _akn(content, "p")
            _serialize_inline(child, p)


def _prose_text(element: etree._Element, emitted_apart: list[etree._Element]) -> str:
    """The element's text, leaving out the children emitted as their own nodes."""
    parts = [element.text or ""]
    for child in element:
        parts.append("" if child in emitted_apart else _prose_text(child, emitted_apart))
        parts.append(child.tail or "")
    return " ".join("".join(parts).split())


def _alinea_into_content(
    formex: etree._Element,
    content: etree._Element,
    eid_path: tuple[str, ...],
) -> None:
    """ALINEA → <p> (from TXT) + optional <blockList> (from LIST), then any
    quoted structure, which follows the instruction introducing it."""
    quotes = _quoted_blocks(formex)
    lists = formex.findall("LIST")
    emitted_apart = [*quotes, *lists]
    txt = formex.find("TXT")
    if txt is not None:
        p = _akn(content, "p")
        _serialize_inline(txt, p)
    else:
        ps = [c for c in formex if _localname(c) == "P" and c not in quotes]
        if ps:
            for src_p in ps:
                text = _prose_text(src_p, emitted_apart)
                if not text:
                    continue
                if any(child in quotes for child in src_p):
                    _akn(content, "p").text = text
                else:
                    _serialize_inline(src_p, _akn(content, "p"))
        else:
            text = _prose_text(formex, emitted_apart)
            if text:
                _akn(content, "p").text = text
    for quoted in quotes:
        _emit_quoted(quoted, content, eid_path)
    for list_el in lists:
        _emit_list(list_el, content, eid_path)


def _quoted_blocks(formex: etree._Element) -> list[etree._Element]:
    """The `QUOT.S` blocks a Formex element carries, directly or inside a `<P>`.

    Formex hangs the text an amendment introduces beside the lead-in that
    announces it, not inside it: `<ITEM><NP><TXT>…following:</TXT></NP><P><QUOT.S>`.
    """
    out: list[etree._Element] = []
    for child in formex:
        local = _localname(child)
        if local == "QUOT.S":
            out.append(child)
        elif local == "P":
            out.extend(c for c in child if _localname(c) == "QUOT.S")
    return out


def _quoted_prose(element: etree._Element, parent: etree._Element) -> None:
    """Quoted structure this walker does not model, kept as prose."""
    children = [c for c in element if isinstance(c.tag, str)]
    # Pure element content is structure, so its children each get their own <p>
    # rather than fusing; text around a child makes that child inline.
    mixed = bool((element.text or "").strip()) or any((c.tail or "").strip() for c in children)
    if children and not mixed:
        for child in children:
            _quoted_prose(child, parent)
        return
    p = _akn(parent, "p")
    _serialize_inline(element, p)
    if not _flat(p):
        parent.remove(p)


def _emit_quoted(
    quot_s: etree._Element,
    parent: etree._Element,
    eid_path: tuple[str, ...],
) -> None:
    """QUOT.S → `<mod><quotedStructure>`, the shape the reader renders as a quote.

    Walked with the same emitters as the body, so a quoted article keeps its own
    number and heading rather than arriving as a paragraph of prose.
    """
    # `mod` is inline, so it rides a block of its own; numbered off the parent,
    # because several branches feed one item.
    leaf = f"mod_{len(parent.findall('.//{%s}mod' % AKN_NS)) + 1}"
    block = _akn(parent, "p")
    mod = _akn(block, "mod", eId=_eid(eid_path, leaf))
    quoted = _akn(mod, "quotedStructure", eId=_eid((*eid_path, leaf), "qstr_1"))
    quoted_path = (*eid_path, leaf, "qstr_1")
    counts: dict[str, int] = {}
    for index, child in enumerate(quot_s, start=1):
        local = _localname(child)
        if local == "ARTICLE":
            _emit_article(child, quoted, quoted_path, counts)
        elif local == "DIVISION":
            _emit_division(child, quoted, quoted_path, counts)
        elif local in {"PARAG", "PAR"}:
            _emit_paragraph(child, quoted, quoted_path, index)
        elif local == "LIST":
            _emit_list(child, quoted, quoted_path)
        elif local == "ALINEA":
            _alinea_into_content(child, quoted, quoted_path)
        elif local in {"TXT", "P", "NP"}:
            _serialize_inline(child, _akn(quoted, "p"))
        elif local == "TBL":
            _emit_table(child, quoted, "__".join(quoted_path), counts)
        else:
            _quoted_prose(child, quoted)
    if len(quoted) == 0:
        text = _flat(quot_s)
        if text:
            _akn(quoted, "p").text = text


def _emit_list(
    formex: etree._Element,
    parent: etree._Element,
    eid_path: tuple[str, ...],
) -> None:
    """LIST → <blockList>; ITEM → <item> with NO.ITEM marker → <num>."""
    existing = sum(
        1 for sib in parent if isinstance(sib.tag, str) and _localname(sib) == "blockList"
    )
    leaf = f"list_{existing + 1}"
    blocklist = _akn(parent, "blockList", eId=_eid(eid_path, leaf))
    new_path = (*eid_path, leaf)
    for index, item in enumerate(formex.findall("ITEM"), start=1):
        _emit_item(item, blocklist, new_path, index)


def _emit_item(
    formex: etree._Element,
    parent: etree._Element,
    eid_path: tuple[str, ...],
    index: int,
) -> None:
    """ITEM → <item>; marker from NO.ITEM or NP/NO.P, content from NP/TXT, ITEM.CONT,
    ALINEA or direct TXT.

    Canonical FORMEX 5.x is `<ITEM><NP><NO.P>(1)</NO.P><TXT>…</TXT></NP></ITEM>`, but
    both marker shapes are common; the more specific NO.P wins when both are present.
    """
    no_item = formex.find("NO.ITEM")
    np = formex.find("NP")
    no_p = np.find("NO.P") if np is not None else None
    marker = ""
    if no_item is not None and (no_item.text or "").strip():
        marker = no_item.text.strip()
    elif no_p is not None and (no_p.text or "").strip():
        marker = no_p.text.strip()
    ident = re.sub(r"[^\w]", "", marker).lower() if marker else str(index)
    if not ident:
        ident = str(index)
    item = _akn(parent, "item", eId=_eid(eid_path, f"item_{ident}"))
    if marker:
        _akn(item, "num").text = marker

    # Content extraction, try each known shape in turn.
    new_path = (*eid_path, f"item_{ident}")
    if np is not None:
        # NP wraps the item content. TXT and LIST may be siblings of NO.P.
        np_txt = np.find("TXT")
        if np_txt is not None:
            p = _akn(item, "p")
            _serialize_inline(np_txt, p)
        for nested in np.findall("LIST"):
            _emit_list(nested, item, new_path)
        # Fallback: NP with no TXT but direct text content.
        if np_txt is None and not np.findall("LIST"):
            text = _prose_text(np, _quoted_blocks(np))
            # Strip the marker we already extracted.
            if marker and text.startswith(marker):
                text = text[len(marker) :].strip()
            if text:
                _akn(item, "p").text = text

    # `NP` first: canonical Formex nests the quote beside the lead-in inside it,
    # `<NP><NO.P/><TXT/><P><QUOT.S/></P></NP>`, and only some dialects hang it
    # directly under the item.
    quotes = _quoted_blocks(np) if np is not None else []
    quotes += _quoted_blocks(formex)
    for quoted in quotes:
        _emit_quoted(quoted, item, new_path)

    cont = formex.find("ITEM.CONT")
    if cont is not None:
        cont_quotes = _quoted_blocks(cont)
        cont_lists = cont.findall("LIST")
        alineas = [c for c in cont if _localname(c) == "ALINEA"]
        if alineas:
            for alinea in alineas:
                _alinea_into_content(alinea, item, new_path)
        else:
            cont_txt = cont.find("TXT")
            if cont_txt is not None:
                p = _akn(item, "p")
                _serialize_inline(cont_txt, p)
            else:
                text = _prose_text(cont, [*cont_quotes, *cont_lists])
                if text:
                    _akn(item, "p").text = text
        for quoted in cont_quotes:
            _emit_quoted(quoted, item, new_path)
        for nested in cont_lists:
            _emit_list(nested, item, new_path)

    # Direct ALINEA siblings (some FORMEX dialects skip ITEM.CONT/NP wrappers).
    # Guarded against double-emission: only fires when neither NP nor ITEM.CONT supplied content.
    if np is None and cont is None:
        for alinea in formex.findall("ALINEA"):
            _alinea_into_content(alinea, item, new_path)

    # Direct TXT child (rare but seen).
    direct_txt = formex.find("TXT")
    if direct_txt is not None and np is None and cont is None:
        p = _akn(item, "p")
        _serialize_inline(direct_txt, p)

    # AKN <item> requires content beyond bare <num>; emit minimal <p> only if
    # we found nothing, this guards a degenerate ITEM with only a marker.
    has_content = any(isinstance(c.tag, str) and _localname(c) != "num" for c in item)
    if not has_content:
        _akn(item, "p").text = " "


def _build_preface(src: etree._Element) -> etree._Element | None:
    """Map FORMEX `<TITLE>` → AKN `<preface>` with semantic inline markup.

    AKN4EU Vol II requires the act title to carry `<docType>`, `<docNumber>` and
    `<docDate>` as inline children of `<longTitle>/<p>`. The first paragraph block
    typically holds the formal designation ("Directive (EU) 2016/943 of …") and gets
    that markup; the rest stay plain `<p>` blocks.
    """
    title = src.find("TITLE")
    toc_src = src.find(".//TOC")
    ti = title.find("TI") if title is not None else None
    if ti is None and toc_src is None:
        return None

    preface = etree.Element("{%s}preface" % AKN_NS)

    if ti is not None:
        long_title = _akn(preface, "longTitle")
        paragraphs = [c for c in ti if _localname(c) == "P"]
        if paragraphs:
            for src_p in paragraphs:
                p = _akn(long_title, "p")
                _serialize_inline(src_p, p)
        else:
            text = _title_text(ti)
            if text:
                _akn(long_title, "p").text = text

    if toc_src is not None:
        _emit_toc(toc_src, preface)

    return preface if len(preface) else None


def _emit_toc(formex_toc: etree._Element, parent: etree._Element) -> None:
    """FORMEX TOC/TOC.BLK/TOC.ITEM → AKN <toc>/<tocItem>.

    Nested TOC.BLK structure becomes nested tocItem with `level` attribute
    derived from depth. NO.ITEM → text prefix; ITEM.CONT → text suffix.
    """
    toc = _akn(parent, "toc")

    def _walk(src_node: etree._Element, akn_parent: etree._Element, level: int) -> None:
        for child in src_node:
            local = _localname(child)
            if local == "TOC.BLK":
                _walk(child, akn_parent, level + 1)
            elif local == "TOC.ITEM":
                no = (child.findtext("NO.ITEM") or "").strip()
                cont = (child.findtext("ITEM.CONT") or "").strip()
                text = (
                    f"{no} {cont}".strip()
                    if no or cont
                    else (" ".join("".join(child.itertext()).split()))
                )
                if text:
                    item = _akn(akn_parent, "tocItem", level=str(level), href="#unresolved")
                    item.text = text

    _walk(formex_toc, toc, level=1)


def _build_preamble(src: etree._Element) -> etree._Element | None:
    """Map FORMEX `<PREAMBLE>` → AKN `<preamble>` per AKN4EU.

    PREAMBLE.INIT/FINAL → `<formula name="enactingFormula">` with one `<p>`
    per source line. VISA → `<citation>`. CONSID → `<recital eId="rec_N">`
    with `<num>` from NO.P and inline-rich `<p>` from TXT.
    """
    preamble_src = src.find("PREAMBLE")
    if preamble_src is None:
        return None
    preamble = etree.Element("{%s}preamble" % AKN_NS)

    init = preamble_src.find("PREAMBLE.INIT")
    if init is not None:
        _emit_formula_block(init, preamble, name="enactingFormula")

    visas = preamble_src.findall(".//VISA")
    if visas:
        citations = _akn(preamble, "citations")
        for index, visa in enumerate(visas, start=1):
            citation = _akn(citations, "citation", eId=f"citations__cit_{index}")
            p = _akn(citation, "p")
            _serialize_inline(visa, p)

    seen_recital_eids: set[str] = set()
    consids = preamble_src.findall(".//CONSID")
    if consids:
        recitals = _akn(preamble, "recitals", eId="recitals")
        for index, consid in enumerate(consids, start=1):
            np_el = consid.find("NP")
            num_text: str | None = None
            txt_el: etree._Element | None = None
            if np_el is not None:
                no_p = np_el.find("NO.P")
                if no_p is not None and (no_p.text or "").strip():
                    num_text = no_p.text.strip()
                txt_el = np_el.find("TXT")
            if txt_el is None:
                txt_el = consid.find("TXT")
            ident = re.sub(r"[^\d]", "", num_text or "") or str(index)
            # A preamble can number two recitals alike. AKN requires unique
            # eIds, and a positional fallback can collide with a number, so the
            # repeat is suffixed instead.
            base = ident
            repeat = 1
            while f"recitals__rec_{ident}" in seen_recital_eids:
                repeat += 1
                ident = f"{base}-{repeat}"
            seen_recital_eids.add(f"recitals__rec_{ident}")
            recital = _akn(recitals, "recital", eId=f"recitals__rec_{ident}")
            if num_text:
                _akn(recital, "num").text = num_text
            p = _akn(recital, "p")
            if txt_el is not None:
                _serialize_inline(txt_el, p)
            else:
                p.text = " ".join("".join(consid.itertext()).split()) or " "

    final = preamble_src.find("PREAMBLE.FINAL")
    if final is not None:
        _emit_formula_block(final, preamble, name="enactingFormula")

    return preamble if len(preamble) else None


def _emit_formula_block(formex: etree._Element, parent: etree._Element, *, name: str) -> None:
    """Emit `<formula>` with one `<p>` per source `<P>` (or one if flat)."""
    formula = _akn(parent, "formula", name=name)
    ps = [c for c in formex if _localname(c) == "P"]
    if ps:
        for src_p in ps:
            p = _akn(formula, "p")
            _serialize_inline(src_p, p)
    else:
        text = " ".join("".join(formex.itertext()).split())
        if text:
            _akn(formula, "p").text = text


def _build_conclusions(src: etree._Element) -> etree._Element | None:
    """Map FORMEX `<FINAL>`/`<SIGNATURE>` → AKN `<conclusions>`.

    PL.DATE (place + date) → `<formula name="conclusions">`.
    Each SIGNATORY → `<blockContainer>` with one `<p>` per signature line.
    AKN's `<signature>` element is reserved for the inline name-block; the
    block-level grouping uses `<blockContainer>` per AKN4EU Vol II.
    """
    final = src.find("FINAL")
    signature = final.find("SIGNATURE") if final is not None else src.find("SIGNATURE")
    if signature is None:
        return None
    conclusions = etree.Element("{%s}conclusions" % AKN_NS)

    # AKN <conclusions> takes (formula | blockContainer | p)*; only valid
    # @name values for <formula> are 'enactingFormula' and 'promulgation'.
    # PL.DATE is just place+date prose, so emit as direct <p>s.
    pl_date = signature.find("PL.DATE")
    if pl_date is not None:
        for src_p in pl_date.findall("P"):
            p = _akn(conclusions, "p")
            _serialize_inline(src_p, p)

    # Each SIGNATORY → <blockContainer> with one <p> per line. AKN's
    # <signature> element is reserved for inline name-blocks within a
    # signature line (refersTo a <person> in references).
    for index, signatory in enumerate(signature.findall("SIGNATORY"), start=1):
        block = _akn(conclusions, "blockContainer", eId=f"sig_{index}")
        for src_p in signatory.findall("P"):
            p = _akn(block, "p")
            _serialize_inline(src_p, p)

    return conclusions if len(conclusions) else None


# Above this a table is kept whole but flagged: the document stays complete and
# the run says which ones will strain a reader or an embedding.
_LARGE_TABLE_ROWS = 1000


def _emit_title(title: etree._Element | None, out: etree._Element, *, block: bool = False) -> None:
    """FORMEX splits a title: TI carries the label, STI the subject.

    An annex alone on its TI is labelled ("ANNEX I"), so that is a num. A table
    alone on its TI is usually named rather than numbered, so that is a heading.
    """
    if title is None:
        return
    texts = [_title_text(el) for el in (title.find("TI"), title.find("STI")) if el is not None]
    texts = [t for t in texts if t]
    if not texts:
        return
    names = ("num", "heading") if len(texts) > 1 else ("num" if block else "heading",)
    for text, name in zip(texts, names, strict=False):
        if block:
            _akn(out, "block", name=name).text = text
        else:
            _akn(out, name).text = text


# Lost or untranscribed text moves the grade; counts and a missing OJ reference are advisory.
_SIGNAL_SEVERITY = {
    "unmapped_blocks": "warning",
    "annex_images": "warning",
    "annex_included_docs": "warning",
    "annex_unmapped_elements": "warning",
    "body_discarded_text": "warning",
    "html_images": "warning",
}


def signal_finding(signal: str, value: object) -> dict[str, Any]:
    """A provenance signal in the validator's finding shape, the raw key kept beside it."""
    return {
        signal: value,
        "check": signal,
        "severity": _SIGNAL_SEVERITY.get(signal, "info"),
        "message": f"{signal.replace('_', ' ')}: {value}",
    }


def _emit_table(
    tbl: etree._Element,
    parent: etree._Element,
    prefix: str,
    counts: dict[str, int],
    provenance: dict[str, object] | None = None,
) -> None:
    """Map FORMEX `<TBL>` → AKN, in the AKN4EU shape: the table inside an
    `hcontainer name="TAB"` whose num and heading carry its number and subject.
    `table` and `tr` carry eIds, cells do not."""
    counts["table"] = counts.get("table", 0) + 1
    eid = f"{prefix}__table_{counts['table']}"
    wrapper = _akn(parent, "hcontainer", name="TAB", eId=eid)
    _emit_title(tbl.find("TITLE"), wrapper)
    # AKN4EU: a hierarchical element's <content> holds one <p>, one <table> or
    # one <foreign>, and nothing else.
    table_eid = f"{eid}__tbl"
    table = _akn(_akn(wrapper, "content"), "table", eId=table_eid)
    rows = tbl.findall(".//ROW")
    for r_index, row in enumerate(rows, start=1):
        header_row = row.get("TYPE") == "HEADER"
        tr = _akn(table, "tr", eId=table_row_eid(table_eid, r_index))
        expected = 1
        for cell in row:
            if _localname(cell) != "CELL":
                continue
            # FORMEX declares each cell's column. A ragged row that omits one
            # would otherwise shift left and pair with the wrong header.
            declared = cell.get("COL")
            if declared and declared.isdigit():
                for _ in range(max(0, int(declared) - expected)):
                    _akn(_akn(tr, "td"), "p")
                    expected += 1
            is_header = header_row or cell.get("TYPE") == "HEADER"
            td = _akn(
                tr,
                "th" if is_header else "td",
                colspan=cell.get("COLSPAN"),
                rowspan=cell.get("ROWSPAN"),
            )
            _serialize_inline(cell, _akn(td, "p"))
            span = cell.get("COLSPAN")
            expected += int(span) if span and span.isdigit() else 1
    if provenance is not None and len(rows) >= _LARGE_TABLE_ROWS:
        cur = provenance.get("large_tables", 0)
        provenance["large_tables"] = (int(cur) if isinstance(cur, int) else 0) + 1


# Grouping wrappers with no legal meaning of their own; the emitter reads
# through them to the blocks inside.
_TRANSPARENT = {"CONTENTS", "GR.SEQ", "GR.TBL", "GR.ANNOTATION", "GR.NOTES"}
# Carries the annex's own bibliographic record, never body text.
_ANNEX_METADATA = {"BIB.INSTANCE", "TITLE", "TOC"}
# Blocks that serialise straight to a paragraph.
_ANNEX_PROSE = {"P", "TXT", "NP", "LIST", "DLIST", "QUOT.S", "ADDR.S", "ANNOTATION"}
# Numbering labels for a group, carried by the group's own heading.
_ANNEX_IGNORED = {"NO.GR.SEQ", "NO.P", "NO.SEQ"}


def _scope_note_ids(attachment: etree._Element, eid: str) -> None:
    """Prefix an attachment's footnote eIds with its own: each annex numbers
    from E0001, so an act and its annexes collide on `fn_E0001`."""
    for note in attachment.iter("{%s}authorialNote" % AKN_NS):
        note_id = note.get("eId")
        if note_id:
            note.set("eId", f"{eid}__{note_id}")
    for ref in attachment.iter("{%s}noteRef" % AKN_NS):
        href = ref.get("href") or ""
        if href.startswith("#fn_"):
            ref.set("href", f"#{eid}__{href[1:]}")


def _attachment_meta(
    *, frbr_work_uri: str, language: str, expression_date: str, eid: str
) -> etree._Element:
    """`<meta>` for an attachment's `<doc>`, mandatory under `openStructure`.
    FRBRthis names the component (`/!att_1`), which translation rebasing keeps."""
    meta = etree.Element("{%s}meta" % AKN_NS)
    ident = _akn(meta, "identification", source="#codify")
    expression_uri = f"{frbr_work_uri}/{language}@{expression_date}"
    work = _akn(ident, "FRBRWork")
    _akn(work, "FRBRthis", value=f"{frbr_work_uri}/!{eid}")
    _akn(work, "FRBRuri", value=frbr_work_uri)
    _akn(work, "FRBRdate", date=expression_date, name="generation")
    _akn(work, "FRBRauthor", href="#eu")
    _akn(work, "FRBRcountry", value="eu")
    # AKN4EU: an annex of an act carries norm. A Work property, after country.
    _akn(work, "FRBRprescriptive", value="true")
    expr = _akn(ident, "FRBRExpression")
    _akn(expr, "FRBRthis", value=f"{expression_uri}/!{eid}")
    _akn(expr, "FRBRuri", value=expression_uri)
    _akn(expr, "FRBRdate", date=expression_date, name="generation")
    _akn(expr, "FRBRauthor", href="#eu")
    _akn(expr, "FRBRlanguage", language=language)
    manif = _akn(ident, "FRBRManifestation")
    _akn(manif, "FRBRthis", value=f"{expression_uri}/!{eid}.xml")
    _akn(manif, "FRBRuri", value=f"{expression_uri}.xml")
    _akn(manif, "FRBRdate", date=expression_date, name="generation")
    _akn(manif, "FRBRauthor", href="#codify")
    _akn(manif, "FRBRformat", value="xml")
    return meta


def _annex_block(
    parent: etree._Element, name: str, prefix: str, counts: dict[str, int]
) -> etree._Element:
    """An annex block container, numbered within its annex.

    Every element the mapper mints needs an eId: `akn_wid` falls back to it,
    and a provision with neither cannot be stored or cited.
    """
    counts[name] = counts.get(name, 0) + 1
    return _akn(parent, "hcontainer", name=name, eId=f"{prefix}__{name.lower()}_{counts[name]}")


def _emit_included_marker(
    included: etree._Element, out: etree._Element, prefix: str, counts: dict[str, int]
) -> None:
    """A page scan or a nested document, named in place rather than dropped."""
    kind = included.get("TYPE", "") or "source"
    hc = _annex_block(out, "includedSource", prefix, counts)
    _akn(_akn(hc, "content"), "p").text = f"[{kind} not transcribed: {included.get('FILEREF', '')}]"


def _count_included_sources(annex: etree._Element, provenance: dict[str, object] | None) -> None:
    """Every page scan and nested document the annex references, wherever it
    sits. A run that reports none of these reads as complete when it is not."""
    if provenance is None:
        return
    for el in annex.iter("INCL.ELEMENT"):
        key = "annex_images" if el.get("TYPE") == "TIFF" else "annex_included_docs"
        cur = provenance.get(key, 0)
        provenance[key] = (int(cur) if isinstance(cur, int) else 0) + 1


def _emit_annex_children(
    parent_src: etree._Element,
    out: etree._Element,
    prefix: str,
    counts: dict[str, int],
    provenance: dict[str, object] | None,
) -> None:
    """Walk one annex level, reading through grouping wrappers."""
    _emit_annex_text(parent_src.text, out, prefix, counts, provenance)
    for child in parent_src:
        # Comments and processing instructions are nodes, not elements, and
        # answer nothing an element does.
        if isinstance(child.tag, str):
            _emit_annex_child(child, out, prefix, counts, provenance)
        # A block's tail is body text wherever the walk went: emitted here so no
        # branch above can drop it by returning early.
        _emit_annex_text(child.tail, out, prefix, counts, provenance)


def _group_heading(wrapper: etree._Element, tag: str) -> str:
    """The wrapper's TI or STI, out of its TITLE. Joined across text nodes rather
    than concatenated: every title in the corpus splits its number from its text
    into sibling elements, which run together without the space."""
    title = wrapper.find("TITLE")
    found = title.find(tag) if title is not None else None
    return _title_text(found) if found is not None else ""


def _emit_titled_group(
    wrapper: etree._Element,
    out: etree._Element,
    prefix: str,
    counts: dict[str, int],
    provenance: dict[str, object] | None,
) -> None:
    """A transparent wrapper with a title becomes a container of its own; without
    one it stays transparent, because a container with no heading says nothing and
    would move its contents' eIds for no reader's benefit."""
    # A wrapper under an annex DIVISION is not reached: `_emit_division` walks
    # structural children only. Two of 2,500 documents, so it is recorded, not built.
    heading = _group_heading(wrapper, "TI")
    if not heading:
        _emit_annex_children(wrapper, out, prefix, counts, provenance)
        return
    group = _annex_block(out, "group", prefix, counts)
    number = wrapper.find("NO.GR.SEQ")
    if number is not None and (number.text or "").strip():
        _akn(group, "num").text = number.text.strip()
    _akn(group, "heading").text = heading
    subheading = _group_heading(wrapper, "STI")
    if subheading:
        _akn(group, "subheading").text = subheading
    # The annex's counts, not its own: a number the group consumed must never be
    # handed to a later sibling, or an old eId survives carrying different text.
    _emit_annex_children(wrapper, group, str(group.get("eId")), counts, provenance)


def _emit_annex_text(
    text: str | None,
    out: etree._Element,
    prefix: str,
    counts: dict[str, int],
    provenance: dict[str, object] | None,
) -> None:
    """Loose text sitting between an annex's blocks, as its own paragraph."""
    if text is None or not text.strip():
        return
    hc = _annex_block(out, "paragraph", prefix, counts)
    _akn(_akn(hc, "content"), "p").text = text.strip()
    if provenance is not None:
        cur = provenance.get("annex_loose_text", 0)
        provenance["annex_loose_text"] = (int(cur) if isinstance(cur, int) else 0) + 1


def _emit_annex_child(
    child: etree._Element,
    out: etree._Element,
    prefix: str,
    counts: dict[str, int],
    provenance: dict[str, object] | None,
) -> None:
    """One annex child, by shape. Its own tail is the caller's to emit."""
    local = _localname(child)
    if local in _ANNEX_METADATA or local in _ANNEX_IGNORED:
        return
    if local in _TRANSPARENT:
        _emit_titled_group(child, out, prefix, counts, provenance)
    elif local in {"DIVISION", "CHAP", "SECTION", "ARTICLE"}:
        if local == "ARTICLE":
            _emit_article(child, out, (prefix,), counts)
        else:
            _emit_division(child, out, (prefix,), counts)
        # Those emitters carry no table branch, so a table inside an annex
        # article or division would flatten into paragraph text.
        for nested in child.iter("TBL"):
            _emit_table(nested, out, prefix, counts, provenance)
    elif local in {"TBL", "TABLE"}:
        _emit_table(child, out, prefix, counts, provenance)
    elif local == "INCL.ELEMENT":
        _emit_included_marker(child, out, prefix, counts)
    elif local in {"FORMULA", "FIGURE"}:
        placeholder = _annex_block(out, f"omitted{local.title()}", prefix, counts)
        _akn(_akn(placeholder, "content"), "p").text = f"[{local} omitted: see source]"
        if provenance is not None:
            cur = provenance.get("unmapped_blocks", 0)
            provenance["unmapped_blocks"] = (int(cur) if isinstance(cur, int) else 0) + 1
    elif child.find(".//TBL") is not None:
        # Tables sit inside lists, quoted amendments and numbered
        # paragraphs. Flattening the wrapper would turn rows into a
        # run-on sentence, so descend whatever the wrapper is.
        _emit_annex_children(child, out, prefix, counts, provenance)
    elif local in _ANNEX_PROSE:
        hc = _annex_block(out, "paragraph", prefix, counts)
        _serialize_inline(child, _akn(_akn(hc, "content"), "p"))
        # The inline serialiser drops an include, which leaves an empty
        # paragraph where a page of law should be.
        for included in child.iter("INCL.ELEMENT"):
            _emit_included_marker(included, out, prefix, counts)
    elif "".join(child.itertext()).strip():
        # An element this walk does not know, carrying text. Kept, and
        # counted separately from the shapes we have decided to drop: an
        # unrecognised one is the case nobody has looked at yet.
        hc = _annex_block(out, "paragraph", prefix, counts)
        _serialize_inline(child, _akn(_akn(hc, "content"), "p"))
        if provenance is not None:
            cur = provenance.get("annex_unmapped_elements", 0)
            provenance["annex_unmapped_elements"] = (int(cur) if isinstance(cur, int) else 0) + 1


def _attachment_sources(src: etree._Element) -> list[etree._Element]:
    """The elements `_build_attachments` maps, so nothing names the tags twice."""
    found: list[etree._Element] = src.findall(".//ANNEX")
    found.extend(src.findall(".//APPENDIX"))
    return found


def _build_attachments(
    src: etree._Element,
    provenance: dict[str, object] | None = None,
    *,
    frbr_work_uri: str = "/akn/eu/act/dir/0000/0",
    language: str = "eng",
    expression_date: str = "",
) -> etree._Element | None:
    """Map FORMEX `<ANNEX>`/`<APPENDIX>` → AKN `<attachments>`.

    Each ANNEX becomes an `<attachment>` holding a `<doc name="annex">` with its
    own `<meta>`, a `<preface>` from TITLE/TI, and a `<mainBody>` walked by
    `_emit_annex_children`.
    """
    annexes = _attachment_sources(src)
    if not annexes:
        return None
    attachments = etree.Element("{%s}attachments" % AKN_NS)
    for index, annex in enumerate(annexes, start=1):
        att = _akn(attachments, "attachment", eId=f"att_{index}")
        # AKN4EU names the document type from the resource-type authority table.
        doc = _akn(att, "doc", name="ANNEX")
        doc.append(
            _attachment_meta(
                frbr_work_uri=frbr_work_uri,
                language=language,
                expression_date=expression_date,
                eid=f"att_{index}",
            )
        )
        # AKN4EU: an annex takes a headerOfAnnex container, not a longTitle.
        ann_title = annex.find("TITLE")
        if ann_title is not None:
            header = _akn(_akn(doc, "preface"), "container", name="headerOfAnnex")
            _emit_title(ann_title, header, block=True)
        # Inner mainBody, recurse over structural children.
        inner_body = _akn(doc, "mainBody")
        sub_counts: dict[str, int] = {}
        _emit_annex_children(annex, inner_body, f"att_{index}", sub_counts, provenance)
        _count_included_sources(annex, provenance)
        _scope_note_ids(att, f"att_{index}")
        # Annex with no recognised children, wrap raw text so it isn't lost.
        if not len(inner_body):
            text = " ".join("".join(annex.itertext()).split())
            if text:
                hc = _annex_block(inner_body, "paragraph", f"att_{index}", {})
                _akn(_akn(hc, "content"), "p").text = text
    return attachments


def formex_to_akn4eu(
    xml: str | bytes,
    *,
    frbr_work_uri: str = "/akn/eu/act/dir/0000/0",
    language: str = "eng",
    acquis_chapter: int | None = None,
    provenance: dict[str, object] | None = None,
) -> str:
    """Deterministic FORMEX → AKN4EU transformation.

    Maps the FORMEX structural skeleton (DIVISION/ARTICLE/PARAG/ALINEA/LIST/ITEM)
    and inline markup (HT, DATE, QUOT, NOTE, FT, REF.DOC.OJ) to AKN4EU.
    """
    src = parse_xml(xml)
    _unwrap_general(src)
    _require_act_with_articles(src)

    expr_date = _formex_date(src)
    if expr_date is None:
        raise FormexDateMissingError(
            "no act-level date in BIB.INSTANCE, PREAMBLE.FINAL, or FINAL/SIGNATURE"
        )

    root = etree.Element("{%s}akomaNtoso" % AKN_NS, nsmap={None: AKN_NS})
    act_name = _doctype_from_frbr(frbr_work_uri)
    act = etree.SubElement(root, "{%s}act" % AKN_NS, contains="originalVersion", name=act_name)
    act.append(
        _stub_meta(
            frbr_work_uri=frbr_work_uri,
            language=language,
            expression_date=expr_date,
            src=src,
            acquis_chapter=acquis_chapter,
            provenance=provenance,
        )
    )
    preface = _build_preface(src)
    if preface is not None:
        act.append(preface)
    preamble = _build_preamble(src)
    if preamble is not None:
        act.append(preamble)
    body = etree.SubElement(act, "{%s}body" % AKN_NS)

    candidates = src.xpath(".//ENACTING.TERMS | .//MAIN.BODY | .//BODY")
    body_source = candidates[0] if candidates else src
    top_counts: dict[str, int] = {}
    for child in body_source:
        local = _localname(child)
        if local in {"DIVISION", "CHAP", "SECTION"}:
            _emit_division(child, body, (), top_counts)
        elif local == "ARTICLE":
            _emit_article(child, body, (), top_counts)

    conclusions = _build_conclusions(src)
    if conclusions is not None:
        act.append(conclusions)
    attachments = _build_attachments(
        src,
        provenance=provenance,
        frbr_work_uri=frbr_work_uri,
        language=language,
        expression_date=expr_date,
    )
    if attachments is not None:
        act.append(attachments)

    # Post-walk counts, rather than plumbed through every emitter: text the body
    # walk had no branch for, and anchors we could not resolve.
    if provenance is not None:
        emitted_apart = {
            node
            for built, tags in ((preamble, ("PREAMBLE",)), (conclusions, ("FINAL", "SIGNATURE")))
            if built is not None
            for node in (src.find(tag) for tag in tags)
            if node is not None
        }
        if attachments is not None:
            emitted_apart.update(_attachment_sources(src))
        if preface is not None:
            # It reads TITLE as a direct child, so one deeper is not consumed.
            emitted_apart.update(
                node for node in (src.find("TITLE"), src.find(".//TOC")) if node is not None
            )
        discarded = _count_discarded_body_text(body_source, emitted_apart)
        if discarded:
            provenance["body_discarded_text"] = discarded
        unresolved = root.xpath(
            'count(.//*[@href="#unresolved"])',
            namespaces={"akn": AKN_NS},
        )
        if unresolved:
            provenance["unresolved_refs"] = int(unresolved)

    serialised: bytes = etree.tostring(root, xml_declaration=True, encoding="UTF-8")
    return serialised.decode("utf-8")


async def ingest(
    source: Path | str,
    jurisdiction_code: str,
    *,
    frbr_work_uri: str | None = None,
    language: str = "eng",
) -> AsyncIterator[IngestionEvent]:
    """Read EU directive XML, run FORMEX→AKN, then the deterministic enrich passes.

    Enrichment composes hcontainers, references, inline_markup and validator. Cobalt
    is skipped deliberately: it would overwrite the AKN4EU FRBR URI shape with the
    generic `/akn/{country}/act/…` pattern. Enacting and amendments are skipped too,
    the formula already being in the preamble and FORMEX carrying no Bluebell quote
    markers.
    """
    import structlog
    from langfuse import get_client

    from codify.jurisdictions import placeholder_statuses_for_code
    from codify.pipeline.enrich.hcontainers import postprocess_hcontainers
    from codify.pipeline.enrich.inline_markup import emit_inline_markup
    from codify.pipeline.enrich.placeholder_status import mark_placeholder_status
    from codify.pipeline.enrich.references import emit_references
    from codify.pipeline.events import Enriched, ValidationIssued

    logger = structlog.get_logger()
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="span",
        name="ingest_document.eu_directive",
        trace_context=langfuse_trace_context(),
        input={"jurisdiction": jurisdiction_code, "source": str(source)},
        metadata={"jurisdiction": jurisdiction_code, "pipeline": "ingest"},
    ) as root:
        attach_trace_attribution(
            langfuse, tags=[f"jurisdiction:{jurisdiction_code}", "pipeline:ingest"]
        )
        try:
            provenance: dict[str, object] = {}

            def formex_conversion(path: Path | None, text: str | None) -> str:
                # Reading, sniffing the format and deriving the URI each touch the
                # whole document, so all of them belong on this side of the handoff.
                if text is not None:
                    src = text
                elif path is not None:
                    src = path.read_text(encoding="utf-8")
                else:
                    raise ValueError("formex_conversion was given neither a path nor the text")
                if is_eurlex_html(src):
                    # Pre-FORMEX acts: Cellar's HTML rendering of the OJ text.
                    if not frbr_work_uri:
                        raise ValueError("an HTML act needs the acquirer's FRBR work URI")
                    return html_to_akn4eu(
                        src, frbr_work_uri=frbr_work_uri, language=language, provenance=provenance
                    )
                if _detect_format(src) != "formex":
                    return src
                return formex_to_akn4eu(
                    src,
                    frbr_work_uri=frbr_work_uri or _frbr_from_formex(src),
                    language=language,
                    provenance=provenance,
                )

            if isinstance(source, str) and source.startswith(("http://", "https://")):
                from codify.pipeline.fetchers.eurlex import fetch_url

                fetched = await fetch_url(source)
                xml = await _on_the_cpu_pool(formex_conversion, None, fetched)
            else:
                xml = await _on_the_cpu_pool(formex_conversion, Path(source), None)
            yield Parsed(akn_xml_len=len(xml))

            # Surface FORMEX→AKN provenance signals as ValidationIssued before
            # enrichment. An annex holding an untranscribed scan is incomplete,
            # so the run says so rather than completing quietly.
            for signal in (
                "unmapped_blocks",
                "unresolved_refs",
                "publication_missing",
                "annex_images",
                "annex_included_docs",
                "annex_unmapped_elements",
                "body_discarded_text",
                "large_tables",
                "html_source",
                "html_images",
            ):
                if provenance.get(signal):
                    yield ValidationIssued(issue=signal_finding(signal, provenance[signal]))

            doctype = "directive"

            EnrichName = Literal["placeholders", "hcontainers", "references"]
            sync_passes: list[tuple[EnrichName, Callable[[str], str]]] = [
                # This path emits the untranscribed-image marker itself; the marker
                # carries its status from the start.
                (
                    "placeholders",
                    lambda x: mark_placeholder_status(
                        x, placeholder_statuses_for_code(jurisdiction_code)
                    ),
                ),
                ("hcontainers", lambda x: postprocess_hcontainers(x, jurisdiction_code, doctype)),
                ("references", lambda x: emit_references(x, jurisdiction_code)),
            ]
            for name, fn in sync_passes:
                try:
                    xml = fn(xml)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("enrich_pass_failed", pass_name=name, error=str(exc))
                    # Failed (not ValidationIssued): a pass crash is an execution failure.
                    yield Failed(stage=name, error=f"{type(exc).__name__}: {exc}")
                    continue
                yield Enriched(pass_name=name)

            try:
                xml = await emit_inline_markup(xml, jurisdiction_code, doctype, None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("enrich_pass_failed", pass_name="inline_markup", error=str(exc))  # noqa: S106
                yield Failed(stage="inline_markup", error=f"{type(exc).__name__}: {exc}")
            else:
                yield Enriched(pass_name="inline_markup")  # noqa: S106

            # Fatal: Complete is the downstream success signal, so emitting it
            # over un-validated AKN corrupts the corpus silently.
            try:
                # Off the loop for the same reason, and it is the heavier of the
                # two: about four times the conversion on the same document.
                for issue in await _on_the_cpu_pool(_validate_native, xml):
                    yield ValidationIssued(issue=issue)
                yield Enriched(pass_name="validator")  # noqa: S106
            except Exception as exc:  # noqa: BLE001
                logger.warning("validator_failed", error=str(exc))
                yield Failed(stage="validator", error=f"{type(exc).__name__}: {exc}")
                return

            document = parse_akn4eu(xml)
            root.update(output={"frbr": document.frbr_work_uri, "akn_xml_len": len(xml)})
            yield Complete(document=document, akn_xml=xml)
        except Exception as exc:  # noqa: BLE001
            root.update(level="ERROR", status_message=str(exc))
            yield Failed(stage="eu_directive", error=f"{type(exc).__name__}: {exc}")


__all__ = [
    "FormexDateMissingError",
    "FormexNotAnActError",
    "celex_to_frbr",
    "formex_to_akn4eu",
    "ingest",
    "parse_akn4eu",
]
