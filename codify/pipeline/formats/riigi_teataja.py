"""Ingest Estonian legislation from Riigi Teataja XML into Akoma Ntoso 3.0.

Deterministic transformation, no LLM required. Maps the official Riigi Teataja
`<oigusakt>` schema to OASIS LegalDocML AKN 3.0 matching `data/jurisdictions/ee/config.json`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import AsyncIterator
from html import unescape
from pathlib import Path
from typing import Any, cast

import structlog
from lxml import etree

from codify.akn._schema import AKN_NS, parse_xml, validate_akn
from codify.akn.document import Document
from codify.akn.eids import ensure_unique_eids
from codify.akn.io import parse_akn
from codify.frbr import build_frbr_work_uri
from codify.pipeline.events import (
    Complete,
    Failed,
    IngestionEvent,
    MetadataExtracted,
    Parsed,
    ValidationIssued,
)

logger = structlog.get_logger()

NSMAP = {None: AKN_NS}


def _q(tag: str) -> str:
    return f"{{{AKN_NS}}}{tag}"


_SUPERSCRIPT_MAP = {
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁰": "0",
}

_LEADING_PUNCT = {".", ",", ";", ":", ")", "]", "}", "»", "“", "”"}


def _extract_number(el: etree._Element | None, default: str = "1") -> str:
    """Extract provision number accounting for Estonian ylaIndeks attributes and superscripts."""
    if el is None:
        return default
    text = (el.text or "").strip()
    yla = el.get("ylaIndeks")
    if yla:
        text = f"{text}{yla}"
    for sup, digit in _SUPERSCRIPT_MAP.items():
        text = text.replace(sup, digit)
    cleaned = re.sub(r"[^\w-]", "", text)
    return cleaned or default


def _clean_number_str(raw: str | None, default: str = "1") -> str:
    if not raw:
        return default
    text = raw.strip()
    for sup, digit in _SUPERSCRIPT_MAP.items():
        text = text.replace(sup, digit)
    cleaned = re.sub(r"[^\w-]", "", text)
    return cleaned or default


def _romanise(text: str) -> str:
    """Romanise Estonian text for FRBR URI slugs per ee/config.json."""
    trans = {
        "ä": "a",
        "ö": "o",
        "ü": "u",
        "õ": "o",
        "š": "s",
        "ž": "z",
        "Ä": "a",
        "Ö": "o",
        "Ü": "u",
        "Õ": "o",
        "Š": "s",
        "Ž": "z",
    }
    s = text.lower()
    for k, v in trans.items():
        s = s.replace(k, v)
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "seadus"


def _rewrite_viide_uri(raw_uri: str) -> str:
    """Rewrite Riigi Teataja internal links to canonical public URLs."""
    m = re.search(r"id=(\d+)(?:!pr(\d+)(?:lg(\d+))?)?", raw_uri)
    if m:
        akt_id = m.group(1)
        pr = m.group(2)
        lg = m.group(3)
        frag = f"#pr{pr}" if pr else ""
        if pr and lg:
            frag = f"#pr{pr}lg{lg}"
        return f"https://www.riigiteataja.ee/akt/{akt_id}{frag}"
    return raw_uri


def is_riigi_teataja(source: Path | str | bytes | etree._Element) -> bool:
    """Check if source is an Estonian Riigi Teataja XML document."""
    try:
        if isinstance(source, etree._Element):
            root = source
        elif isinstance(source, bytes):
            prefix = source[:65536]
        elif isinstance(source, Path):
            with source.open("rb") as stream:
                prefix = stream.read(65536)
        elif isinstance(source, str):
            prefix_text = source[:65536].lstrip()
            if prefix_text.startswith("<"):
                prefix = prefix_text.encode()
            else:
                if len(source) > 4096 or "\n" in source or "\0" in source:
                    return False
                p = Path(source)
                if p.exists():
                    with p.open("rb") as stream:
                        prefix = stream.read(65536)
                else:
                    return False
        else:
            return False
        if not isinstance(source, etree._Element):
            if b"<!doctype" in prefix.lower():
                return False
            parser = etree.XMLPullParser(
                events=("start",), resolve_entities=False, load_dtd=False, no_network=True
            )
            parser.feed(prefix)
            _event, root = next(parser.read_events())
        name = etree.QName(root)
        return bool(name.localname == "oigusakt" and name.namespace == "Juurakt")
    except Exception:
        return False


def _clean_text(el: etree._Element) -> str:
    parts = [text for text in el.itertext() if text]
    return " ".join("".join(parts).split())


def _append_text(target: etree._Element, chunk: str) -> None:
    if not chunk:
        return
    if len(target) == 0:
        target.text = (target.text or "") + chunk
    else:
        target[-1].tail = (target[-1].tail or "") + chunk


def _clean_tail(raw_tail: str | None) -> str:
    if not raw_tail:
        return ""
    if not raw_tail.strip():
        return ""
    return re.sub(r"\s+", " ", raw_tail)


def _serialize_inline(src_node: etree._Element, parent_akn: etree._Element) -> None:
    """Serialize inline elements and text from sisuTekst in document order."""
    if src_node.text:
        txt = _clean_tail(src_node.text)
        if txt:
            _append_text(parent_akn, txt)

    for child in src_node:
        tag = etree.QName(child).localname
        if tag == "tavatekst":
            _serialize_inline(child, parent_akn)
            if child.tail:
                _append_text(parent_akn, _clean_tail(child.tail))
        elif tag == "viide":
            v_text = child.findtext("{*}kuvatavTekst") or "".join(child.itertext())
            ref = etree.SubElement(parent_akn, _q("ref"))
            ref.text = v_text
            v_uri = child.findtext(".//{*}viideURI")
            if v_uri:
                ref.set("href", _rewrite_viide_uri(v_uri))
            if child.tail:
                _append_text(parent_akn, _clean_tail(child.tail))
        elif tag in {"b", "i", "sup", "sub"}:
            inline = etree.SubElement(parent_akn, _q(tag))
            _serialize_inline(child, inline)
            if child.tail:
                _append_text(parent_akn, _clean_tail(child.tail))
        elif tag == "reavahetus":
            etree.SubElement(parent_akn, _q("eol"))
            if child.tail:
                _append_text(parent_akn, _clean_tail(child.tail))
        elif tag == "HTMLKonteiner":
            raw_html = child.text or ""
            unescaped = unescape(raw_html)
            clean = " ".join(re.sub(r"<[^>]+>", " ", unescaped).split())
            if clean:
                _append_text(parent_akn, clean)
            if child.tail:
                _append_text(parent_akn, _clean_tail(child.tail))


def _convert_paragrahv(p_src: etree._Element, parent_akn: etree._Element, parent_eid: str) -> None:
    p_nr = _extract_number(p_src.find("{*}paragrahvNr"))
    p_eid = f"{parent_eid}__art_{p_nr}" if parent_eid else f"art_{p_nr}"
    art = etree.SubElement(parent_akn, _q("article"), eId=p_eid, wId=p_eid)

    knr = p_src.findtext("{*}kuvatavNr")
    if knr and knr.strip():
        num = etree.SubElement(art, _q("num"))
        num.text = knr.strip()

    h = p_src.findtext("{*}paragrahvPealkiri")
    if h and h.strip():
        heading = etree.SubElement(art, _q("heading"))
        heading.text = h.strip()

    loiked = p_src.findall("{*}loige")
    if not loiked:
        c = etree.SubElement(art, _q("content"))
        p = etree.SubElement(c, _q("p"), eId=f"{p_eid}__p_1")
        st = p_src.find("{*}sisuTekst")
        mm = p_src.find("{*}muutmismarge")
        hk = p_src.find("{*}HTMLKonteiner")
        if st is not None:
            _serialize_inline(st, p)
            if not p.text and len(p) == 0:
                p.text = _clean_text(st)
        elif hk is not None:
            raw_html = hk.text or ""
            p.text = " ".join(re.sub(r"<[^>]+>", " ", unescape(raw_html)).split())
        elif mm is not None:
            p.text = _clean_text(mm)
        else:
            p.text = ""
        return

    # Single unnumbered loige: collapse only if it has NO sub-items (alampunkt)
    single_unnumbered = len(loiked) == 1 and not (loiked[0].findtext("{*}loigeNr") or "").strip()
    single_has_points = single_unnumbered and bool(loiked[0].findall("{*}alampunkt"))

    if single_unnumbered and not single_has_points:
        loige_el = loiked[0]
        c = etree.SubElement(art, _q("content"))
        p = etree.SubElement(c, _q("p"), eId=f"{p_eid}__p_1")
        st = loige_el.find("{*}sisuTekst")
        if st is not None:
            _serialize_inline(st, p)
            if not p.text and len(p) == 0:
                p.text = _clean_text(st)
        return

    for l_idx, loige_el in enumerate(loiked, start=1):
        raw_l_nr = loige_el.find("{*}loigeNr")
        l_nr = _extract_number(raw_l_nr, default=str(l_idx)) if not single_unnumbered else "1"
        l_eid = f"{p_eid}__para_{l_nr}"
        para = etree.SubElement(art, _q("paragraph"), eId=l_eid, wId=l_eid)

        l_knr = loige_el.findtext("{*}kuvatavNr")
        if l_knr and l_knr.strip():
            num = etree.SubElement(para, _q("num"))
            num.text = l_knr.strip()

        alampunktid = loige_el.findall("{*}alampunkt")
        st = loige_el.find("{*}sisuTekst")

        if alampunktid:
            intro = etree.SubElement(para, _q("intro"))
            p = etree.SubElement(intro, _q("p"), eId=f"{l_eid}__intro__p_1")
            if st is not None:
                _serialize_inline(st, p)
                if not p.text and len(p) == 0:
                    p.text = _clean_text(st)
            for ap_idx, ap in enumerate(alampunktid, start=1):
                raw_ap_nr = ap.find("{*}alampunktNr")
                ap_nr = _extract_number(raw_ap_nr, default=str(ap_idx))
                ap_eid = f"{l_eid}__point_{ap_nr}"
                pt = etree.SubElement(para, _q("point"), eId=ap_eid, wId=ap_eid)
                ap_knr = ap.findtext("{*}kuvatavNr")
                if ap_knr and ap_knr.strip():
                    pt_num = etree.SubElement(pt, _q("num"))
                    pt_num.text = ap_knr.strip()
                pt_c = etree.SubElement(pt, _q("content"))
                pt_p = etree.SubElement(pt_c, _q("p"), eId=f"{ap_eid}__p_1")
                ap_st = ap.find("{*}sisuTekst")
                if ap_st is not None:
                    _serialize_inline(ap_st, pt_p)
                    if not pt_p.text and len(pt_p) == 0:
                        pt_p.text = _clean_text(ap_st)
        else:
            c = etree.SubElement(para, _q("content"))
            p = etree.SubElement(c, _q("p"), eId=f"{l_eid}__p_1")
            if st is not None:
                _serialize_inline(st, p)
                if not p.text and len(p) == 0:
                    p.text = _clean_text(st)


def _convert_jaotis(jt_src: etree._Element, parent_akn: etree._Element, parent_eid: str) -> None:
    raw_nr = jt_src.find("{*}jaotisNr")
    if raw_nr is None:
        raw_nr = jt_src.find("{*}alljaotisNr")
    jt_nr = _extract_number(raw_nr)
    jt_eid = f"{parent_eid}__subsec_{jt_nr}" if parent_eid else f"subsec_{jt_nr}"
    subsec = etree.SubElement(parent_akn, _q("subsection"), eId=jt_eid, wId=jt_eid)

    knr = jt_src.findtext("{*}kuvatavNr")
    if knr and knr.strip():
        num = etree.SubElement(subsec, _q("num"))
        num.text = knr.strip()

    h = jt_src.findtext("{*}jaotisPealkiri") or jt_src.findtext("{*}alljaotisPealkiri")
    if h and h.strip():
        heading = etree.SubElement(subsec, _q("heading"))
        heading.text = h.strip()

    for child in jt_src:
        ctag = etree.QName(child).localname
        if ctag == "paragrahv":
            _convert_paragrahv(child, subsec, jt_eid)
        elif ctag in {"jaotis", "alljaotis"}:
            _convert_jaotis(child, subsec, jt_eid)


def _convert_jagu(j_src: etree._Element, parent_akn: etree._Element, parent_eid: str) -> None:
    raw_nr = j_src.find("{*}jaguNr")
    j_nr = _extract_number(raw_nr)
    j_eid = f"{parent_eid}__sec_{j_nr}" if parent_eid else f"sec_{j_nr}"
    sec = etree.SubElement(parent_akn, _q("section"), eId=j_eid, wId=j_eid)

    knr = j_src.findtext("{*}kuvatavNr")
    if knr and knr.strip():
        num = etree.SubElement(sec, _q("num"))
        num.text = knr.strip()

    h = j_src.findtext("{*}jaguPealkiri")
    if h and h.strip():
        heading = etree.SubElement(sec, _q("heading"))
        heading.text = h.strip()

    for child in j_src:
        ctag = etree.QName(child).localname
        if ctag == "paragrahv":
            _convert_paragrahv(child, sec, j_eid)
        elif ctag in {"jaotis", "alljaotis"}:
            _convert_jaotis(child, sec, j_eid)


def _convert_peatykk(
    p_src: etree._Element, parent_akn: etree._Element, parent_eid: str
) -> tuple[etree._Element, str]:
    prefix = "jagu" if etree.QName(p_src).localname == "jagu" else "peatykk"
    raw_nr = p_src.find(f"{{*}}{prefix}Nr")
    c_nr = _extract_number(raw_nr)
    c_eid = f"{parent_eid}__chp_{c_nr}" if parent_eid else f"chp_{c_nr}"
    chp = etree.SubElement(parent_akn, _q("chapter"), eId=c_eid, wId=c_eid)

    knr = p_src.findtext("{*}kuvatavNr")
    if knr and knr.strip():
        num = etree.SubElement(chp, _q("num"))
        num.text = knr.strip()

    h = p_src.findtext(f"{{*}}{prefix}Pealkiri")
    if h and h.strip():
        heading = etree.SubElement(chp, _q("heading"))
        heading.text = h.strip()

    for child in p_src:
        ctag = etree.QName(child).localname
        if ctag == "jagu":
            _convert_jagu(child, chp, c_eid)
        elif ctag == "paragrahv":
            _convert_paragrahv(child, chp, c_eid)
        elif ctag in {"jaotis", "alljaotis"}:
            _convert_jaotis(child, chp, c_eid)

    return chp, c_eid


def _convert_osa(o_src: etree._Element, parent_akn: etree._Element) -> tuple[etree._Element, str]:
    raw_nr = o_src.find("{*}osaNr")
    o_nr = _extract_number(raw_nr)
    o_eid = f"part_{o_nr}"
    part = etree.SubElement(parent_akn, _q("part"), eId=o_eid, wId=o_eid)

    knr = o_src.findtext("{*}kuvatavNr")
    if knr and knr.strip():
        num = etree.SubElement(part, _q("num"))
        num.text = knr.strip()

    h = o_src.findtext("{*}osaPealkiri")
    if h and h.strip():
        heading = etree.SubElement(part, _q("heading"))
        heading.text = h.strip()

    for child in o_src:
        ctag = etree.QName(child).localname
        if ctag == "peatykk":
            _convert_peatykk(child, part, o_eid)
        elif ctag == "jagu":
            _convert_jagu(child, part, o_eid)
        elif ctag == "paragrahv":
            _convert_paragrahv(child, part, o_eid)

    return part, o_eid


def riigi_teataja_to_akn(
    source_xml: str | bytes,
    *,
    frbr_work_uri: str | None = None,
    language: str = "est",
) -> tuple[str, dict[str, Any]]:
    """Transform Riigi Teataja XML to canonical Akoma Ntoso 3.0.

    Returns (akn_xml, metadata).
    """
    root = parse_xml(source_xml, huge_tree=True)
    meta = root.find("{*}metaandmed")

    title = root.findtext(".//{*}pealkiri") or root.findtext(".//{*}aktinimi") or "Seadus"
    title = " ".join(title.split())

    d_type_raw = (meta.findtext("{*}dokumentLiik") if meta is not None else "") or "seadus"
    d_type_raw = d_type_raw.lower().strip()

    short_title = (meta.findtext("{*}lyhend") if meta is not None else "") or title
    slug = _romanise(short_title)

    is_constitution = (
        d_type_raw == "põhiseadus"
        or "põhiseadus" in title.lower()
        or (meta is not None and (meta.findtext("{*}lyhend") or "").strip().lower() == "ps")
    )
    if is_constitution:
        doctype = "constitution"
    elif d_type_raw == "määrus":
        doctype = "maarrus"
    elif d_type_raw in {"otsus", "korraldus"}:
        doctype = "act"
    else:
        doctype = "act"

    enactment_date = ""
    publication_date = ""
    number = ""

    if meta is not None:
        m_vastu = meta.find("{*}vastuvoetud")
        if m_vastu is not None:
            raw_enact = m_vastu.findtext("{*}aktikuupaev")
            if raw_enact:
                enactment_date = raw_enact.split("+")[0].strip()
            number = (m_vastu.findtext("{*}aktiNr") or "").strip()

        raw_pub = meta.findtext(".//{*}avaldamineKuupaev")
        if raw_pub:
            publication_date = raw_pub.split("+")[0].strip()

        if not number:
            number = (meta.findtext(".//{*}RTartikkel") or "").strip()

    primary_date = enactment_date or publication_date
    if not primary_date:
        raise ValueError(
            f"Riigi Teataja XML document {title!r} lacks both enactment and publication dates"
        )

    year = primary_date.split("-")[0]

    if frbr_work_uri:
        work_uri = frbr_work_uri
    elif doctype == "constitution":
        work_uri = "/akn/ee/act/1992/pohiseadus"
    else:
        work_uri = build_frbr_work_uri("ee", doctype, year, slug)

    exp_date = (
        meta.findtext(".//{*}kehtivuseAlgus") if meta is not None else primary_date
    ) or primary_date
    if "+" in exp_date:
        exp_date = exp_date.split("+")[0].strip()

    exp_uri = f"{work_uri}/{language}@{exp_date}"
    man_uri = f"{exp_uri}.akn"

    # Determine author TLC based on issuing body
    valjaandja = (meta.findtext("{*}valjaandja") if meta is not None else "") or ""
    valjaandja_lower = valjaandja.lower()
    if "valitsus" in valjaandja_lower:
        author_id = "valitsus"
        author_role_type = "TLCOrganization"
        author_href = "/ontology/org/ee/vabariigi-valitsus"
        author_name = "Vabariigi Valitsus"
    elif "minister" in valjaandja_lower:
        author_id = "minister"
        author_role_type = "TLCRole"
        author_href = "/ontology/role/ee/minister"
        author_name = valjaandja or "Minister"
    else:
        author_id = "riigikogu"
        author_role_type = "TLCOrganization"
        author_href = "/ontology/org/ee/riigikogu"
        author_name = "Riigikogu"

    akn_root = etree.Element(_q("akomaNtoso"), nsmap=NSMAP)
    act = etree.SubElement(akn_root, _q("act"), name=doctype)

    # Meta
    meta_el = etree.SubElement(act, _q("meta"))
    ident = etree.SubElement(meta_el, _q("identification"), source="#codify")

    # FRBRWork
    f_work = etree.SubElement(ident, _q("FRBRWork"))
    etree.SubElement(f_work, _q("FRBRthis"), value=work_uri)
    etree.SubElement(f_work, _q("FRBRuri"), value=work_uri)
    etree.SubElement(f_work, _q("FRBRdate"), date=enactment_date or primary_date, name="enacted")
    etree.SubElement(f_work, _q("FRBRauthor"), href=f"#{author_id}")
    etree.SubElement(f_work, _q("FRBRcountry"), value="ee")

    # FRBRExpression
    f_exp = etree.SubElement(ident, _q("FRBRExpression"))
    etree.SubElement(f_exp, _q("FRBRthis"), value=exp_uri)
    etree.SubElement(f_exp, _q("FRBRuri"), value=exp_uri)
    etree.SubElement(f_exp, _q("FRBRdate"), date=exp_date, name="expression")
    etree.SubElement(f_exp, _q("FRBRauthor"), href=f"#{author_id}")
    etree.SubElement(f_exp, _q("FRBRlanguage"), language=language)

    # FRBRManifestation
    f_man = etree.SubElement(ident, _q("FRBRManifestation"))
    etree.SubElement(f_man, _q("FRBRthis"), value=man_uri)
    etree.SubElement(f_man, _q("FRBRuri"), value=man_uri)
    etree.SubElement(f_man, _q("FRBRdate"), date=exp_date, name="generation")
    etree.SubElement(f_man, _q("FRBRauthor"), href="#codify")

    # Publication
    rt_art = meta.findtext(".//{*}RTartikkel") if meta is not None else ""
    rt_osa = meta.findtext(".//{*}RTosa") if meta is not None else "RT I"
    pub_date = publication_date or primary_date
    pub_show = (
        f"{rt_osa}, {pub_date}, {rt_art}"
        if rt_art
        else f"{rt_osa}, {pub_date}"
        if pub_date
        else rt_osa
    )
    etree.SubElement(
        meta_el,
        _q("publication"),
        date=pub_date,
        name=rt_osa or "RT I",
        showAs=pub_show,
    )

    # References
    refs = etree.SubElement(meta_el, _q("references"), source="#codify")
    etree.SubElement(
        refs,
        _q("TLCOrganization"),
        eId="codify",
        href="/ontology/org/codify",
        showAs="Codify",
    )
    etree.SubElement(
        refs,
        _q(author_role_type),
        eId=author_id,
        href=author_href,
        showAs=author_name,
    )

    # Preface
    preface = etree.SubElement(act, _q("preface"))
    p_title = etree.SubElement(preface, _q("p"), eId="preface__p_1")
    p_title.text = title

    # Preamble
    sisu = root.find("{*}sisu")
    if sisu is not None:
        preambul = sisu.find("{*}preambul")
        if preambul is not None:
            preamble = etree.SubElement(act, _q("preamble"))
            formula = etree.SubElement(preamble, _q("formula"), name="enactingFormula")
            p_f = etree.SubElement(formula, _q("p"), eId="preamble__formula__p_1")
            _serialize_inline(preambul, p_f)
            if not p_f.text and len(p_f) == 0:
                p_f.text = _clean_text(preambul)

    # Body
    body = etree.SubElement(act, _q("body"))
    if sisu is not None:
        direct_st_counts: dict[str, int] = defaultdict(int)
        current_osa_akn: etree._Element | None = None
        current_osa_eid: str = ""
        current_peatykk_akn: etree._Element | None = None
        current_peatykk_eid: str = ""

        for child in sisu:
            ctag = etree.QName(child).localname
            if ctag == "osa":
                current_osa_akn, current_osa_eid = _convert_osa(child, body)
                current_peatykk_akn = None
                current_peatykk_eid = ""
            elif ctag == "peatykk":
                current_osa_akn = None
                current_osa_eid = ""
                current_peatykk_akn, current_peatykk_eid = _convert_peatykk(child, body, "")
            elif ctag == "jagu":
                target_parent = (
                    current_peatykk_akn
                    if current_peatykk_akn is not None
                    else (current_osa_akn if current_osa_akn is not None else body)
                )
                target_eid = (
                    current_peatykk_eid
                    if current_peatykk_akn is not None
                    else (current_osa_eid if current_osa_akn is not None else "")
                )
                if doctype == "constitution":
                    _convert_peatykk(child, target_parent, target_eid)
                else:
                    _convert_jagu(child, target_parent, target_eid)
            elif ctag == "paragrahv":
                target_parent = (
                    current_peatykk_akn
                    if current_peatykk_akn is not None
                    else (current_osa_akn if current_osa_akn is not None else body)
                )
                target_eid = (
                    current_peatykk_eid
                    if current_peatykk_akn is not None
                    else (current_osa_eid if current_osa_akn is not None else "")
                )
                _convert_paragrahv(child, target_parent, target_eid)
            elif ctag == "punkt":
                target_parent = (
                    current_peatykk_akn
                    if current_peatykk_akn is not None
                    else (current_osa_akn if current_osa_akn is not None else body)
                )
                target_eid = (
                    current_peatykk_eid
                    if current_peatykk_akn is not None
                    else (current_osa_eid if current_osa_akn is not None else "")
                )
                raw_nr = child.find("{*}punktNr")
                p_nr = _extract_number(raw_nr)
                p_eid = f"{target_eid}__point_{p_nr}" if target_eid else f"point_{p_nr}"
                pt = etree.SubElement(target_parent, _q("point"), eId=p_eid, wId=p_eid)
                knr = child.findtext("{*}kuvatavNr")
                if knr and knr.strip():
                    pt_num = etree.SubElement(pt, _q("num"))
                    pt_num.text = knr.strip()
                pt_c = etree.SubElement(pt, _q("content"))
                pt_p = etree.SubElement(pt_c, _q("p"), eId=f"{p_eid}__p_1")
                st = child.find("{*}sisuTekst")
                if st is not None:
                    _serialize_inline(st, pt_p)
                    if not pt_p.text and len(pt_p) == 0:
                        pt_p.text = _clean_text(st)
            elif ctag in {"sisuTekst", "HTMLKonteiner"}:
                target_parent = (
                    current_peatykk_akn
                    if current_peatykk_akn is not None
                    else (current_osa_akn if current_osa_akn is not None else body)
                )
                target_eid = (
                    current_peatykk_eid
                    if current_peatykk_akn is not None
                    else (current_osa_eid if current_osa_akn is not None else "")
                )
                direct_st_counts[target_eid] += 1
                st_count = direct_st_counts[target_eid]
                p_eid = f"{target_eid}__para_{st_count}" if target_eid else f"para_{st_count}"
                para = etree.SubElement(target_parent, _q("paragraph"), eId=p_eid, wId=p_eid)
                c_el = etree.SubElement(para, _q("content"))
                p_el = etree.SubElement(c_el, _q("p"), eId=f"{p_eid}__p_1")
                if ctag == "sisuTekst":
                    _serialize_inline(child, p_el)
                    if not p_el.text and len(p_el) == 0:
                        p_el.text = _clean_text(child)
                else:
                    raw_html = child.text or ""
                    p_el.text = " ".join(re.sub(r"<[^>]+>", " ", unescape(raw_html)).split())

    # Attachments
    lisad = root.findall(".//{*}lisaViide") + root.findall(".//{*}lisa")
    if lisad:
        attachments = etree.SubElement(act, _q("attachments"))
        akt_viide = meta.findtext(".//{*}aktViide") if meta is not None else None
        for l_idx, lisa in enumerate(lisad, start=1):
            att_eid = f"att_{l_idx}"
            att = etree.SubElement(attachments, _q("attachment"), eId=att_eid)

            lp_el = lisa.find(".//{*}lisaPealkiri")
            lisa_title = _clean_text(lp_el) if lp_el is not None else ""
            if not lisa_title:
                fn_el = lisa.find(".//{*}fail")
                lisa_title = (fn_el.get("failNimi") if fn_el is not None else "") or f"Lisa {l_idx}"
            heading_el = etree.SubElement(att, _q("heading"))
            heading_el.text = lisa_title

            doc_att = etree.SubElement(att, _q("doc"), name=f"lisa_{l_idx}")

            doc_meta = etree.SubElement(doc_att, _q("meta"))
            doc_ident = etree.SubElement(doc_meta, _q("identification"), source="#codify")
            f_w = etree.SubElement(doc_ident, _q("FRBRWork"))
            etree.SubElement(f_w, _q("FRBRthis"), value=f"{work_uri}/!{att_eid}")
            etree.SubElement(f_w, _q("FRBRuri"), value=work_uri)
            etree.SubElement(
                f_w, _q("FRBRdate"), date=enactment_date or primary_date, name="enacted"
            )
            etree.SubElement(f_w, _q("FRBRauthor"), href=f"#{author_id}")
            etree.SubElement(f_w, _q("FRBRcountry"), value="ee")

            f_e = etree.SubElement(doc_ident, _q("FRBRExpression"))
            etree.SubElement(f_e, _q("FRBRthis"), value=f"{exp_uri}/!{att_eid}")
            etree.SubElement(f_e, _q("FRBRuri"), value=exp_uri)
            etree.SubElement(f_e, _q("FRBRdate"), date=exp_date, name="expression")
            etree.SubElement(f_e, _q("FRBRauthor"), href=f"#{author_id}")
            etree.SubElement(f_e, _q("FRBRlanguage"), language=language)

            f_m = etree.SubElement(doc_ident, _q("FRBRManifestation"))
            etree.SubElement(f_m, _q("FRBRthis"), value=f"{man_uri}/!{att_eid}")
            etree.SubElement(f_m, _q("FRBRuri"), value=man_uri)
            etree.SubElement(
                f_m,
                _q("FRBRdate"),
                date=exp_date,
                name="generation",
            )
            etree.SubElement(f_m, _q("FRBRauthor"), href="#codify")

            doc_body = etree.SubElement(doc_att, _q("mainBody"))
            lisa_p = etree.SubElement(doc_body, _q("p"), eId=f"{att_eid}__p_1")

            fail_el = lisa.find(".//{*}fail")
            fail_nimi = fail_el.get("failNimi") if fail_el is not None else None
            if fail_nimi and akt_viide:
                ref_link = f"https://www.riigiteataja.ee/aktilisa/{akt_viide}/{fail_nimi}"
                ref = etree.SubElement(lisa_p, _q("ref"), href=ref_link)
                ref.text = f"{lisa_title} ({fail_nimi})"
            else:
                lisa_p.text = lisa_title

    raw_xml_out = cast(
        str,
        etree.tostring(akn_root, pretty_print=True, xml_declaration=True, encoding="utf-8").decode(
            "utf-8"
        ),
    )
    unique_xml, _ = ensure_unique_eids(raw_xml_out, huge_tree=True)

    metadata: dict[str, Any] = {
        "title": title,
        "number": number or "1",
        "year": int(year),
        "date": enactment_date or primary_date,
        "language": language,
        "doctype": doctype,
        "frbr_work_uri": work_uri,
        "frbr_expression_uri": exp_uri,
        "gazette": {
            "name": rt_osa or "RT I",
            "date": publication_date or primary_date,
            "number": rt_art or number,
            "year": int(publication_date.split("-")[0]) if publication_date else int(year),
        },
    }

    return unique_xml, metadata


async def ingest(
    source: Path | str,
    jurisdiction_code: str = "ee",
    *,
    frbr_work_uri: str | None = None,
    language: str = "est",
) -> AsyncIterator[IngestionEvent]:
    """Ingest Riigi Teataja XML document deterministically."""
    from codify.pipeline.enrich.validator import validate_akn as run_validator
    from codify.pipeline.formats.eu_directive import _on_the_cpu_pool

    try:
        path = source if isinstance(source, Path) else Path(source)
        raw_xml = path.read_bytes()

        def convert() -> tuple[str, dict[str, Any]]:
            return riigi_teataja_to_akn(
                raw_xml,
                frbr_work_uri=frbr_work_uri,
                language=language,
            )

        akn_xml, metadata = await _on_the_cpu_pool(convert)

        yield MetadataExtracted(metadata=metadata)
        yield Parsed(akn_xml_len=len(akn_xml))

        # Schema validation (offloaded to CPU pool)
        def _validate_huge(xml: str) -> None:
            validate_akn(xml, huge_tree=True)

        try:
            await _on_the_cpu_pool(_validate_huge, akn_xml)
        except Exception as exc:
            logger.warning("akn_schema_validation_failed", error=str(exc))
            yield Failed(stage="schema_validation", error=f"{type(exc).__name__}: {exc}")
            return

        # Pipeline validator findings
        def _run_validator_huge(xml: str) -> list[dict[str, Any]]:
            return run_validator(xml, huge_tree=True, provenance="native")

        try:
            for issue in await _on_the_cpu_pool(_run_validator_huge, akn_xml):
                yield ValidationIssued(issue=issue)
        except Exception as exc:
            logger.warning("validator_failed", error=str(exc))
            yield Failed(stage="validator", error=f"{type(exc).__name__}: {exc}")
            return

        doc: Document = parse_akn(akn_xml, huge_tree=True)

        logger.info(
            "riigi_teataja_ingested",
            jurisdiction=jurisdiction_code,
            frbr=doc.frbr_work_uri,
            provisions_len=len(doc.body),
        )

        yield Complete(document=doc, akn_xml=akn_xml)

    except Exception as exc:
        yield Failed(stage="riigi_teataja", error=f"{type(exc).__name__}: {exc}")


__all__ = [
    "ingest",
    "is_riigi_teataja",
    "riigi_teataja_to_akn",
]
