"""Inject semantic `<ref>`, `<def>` and `<term>` elements into AKN text.

Deterministic passes always run; optional LLM-assisted definition and external
reference passes are skipped when no client is supplied.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import structlog
from lxml import etree

from codify.agents.tools.frbr import construct_frbr_uri
from codify.akn import AKN_NS, NS
from codify.akn._schema import parse_xml
from codify.akn.eid import eid_abbrev
from codify.akn.references import DERIVED_REF_CLASS
from codify.frbr import default_token, law_number_token, token_country, token_family
from codify.jurisdictions import load_config, load_short_titles
from codify.lang import normalise_digits
from codify.pipeline.enrich.eid import build_term_map, resolve_informal_reference

if TYPE_CHECKING:
    from codify.core.llm import LLMClient
    from codify.jurisdictions import JurisdictionConfig

logger = structlog.get_logger()

_DEFINITION_HEADING_RE = re.compile(
    r"(?i)^(interpretation|definitions?|meaning of terms|glossary)",
)
_QUOTED_TERM_RE = re.compile(r'["\u201c]([^"\u201d]{1,80})["\u201d]')

MAX_CONCURRENT = 10

_OTHER_ACT_NAME = re.compile(
    r"(?:Act|Regulations|Order)\s+(?:19|20)\d{2}\b|\bthe\s+(?:19|20)\d{2}\s+(?:Act|Regulations)\b"
)
# "Section 12 (heading) is amended as follows", an amendment instruction ABOUT
# the unit (the amended act is named in an earlier paragraph), not a self-ref.
_AMENDED_PREDICATE = re.compile(
    r"^\s*(?:\([^)]{0,120}\)\s*)?(?:is|are)\s+(?:amended|repealed|revoked|omitted|substituted)"
)
_TAIL_PREP = re.compile(r"^\s*(?:to\s+\d+[A-Za-z]?\s*)?,?\s*(?:of|under)\s+")
_TAIL_EXTERNAL = re.compile(
    r"^(?:that|those|the\s+said)\b"
    r"|(?:Act|Regulations|Order)\s+(?:19|20)\d{2}\b"
    r"|\bthe\s+(?:19|20)\d{2}\s+(?:Act|Regulations)\b"
)
_TAIL_SELF = re.compile(r"\b(?:this|these)\b")


# Marks a `<ref>` this pass created by reading prose, as against one the
# publisher wrote into the source. The resolver weighs the two differently: an
# authored link is evidence, a match on text is a reading, and the string alone
# cannot tell them apart because publisher AKN carries relative `/akn/...`
# hrefs of exactly the shape this pass mints.
# Carried in `class`, which AKN 3.0 allows on `<ref>` and this pass already
# uses for the amendment operation. A bespoke attribute fails schema validation,
# which is the one thing the output must never do.


def _derived_ref(href: str) -> etree._Element:
    """A `<ref>` minted from prose, marked as such."""
    return etree.Element(f"{{{AKN_NS}}}ref", attrib={"href": href, "class": DERIVED_REF_CLASS})


def _tail_redirects_elsewhere(tail: str) -> bool:
    """True when the text after a unit match hands the reference to another instrument
    ("of the 1998 Act"). Internal container paths ("of Schedule 2 to this Act") stay
    self-links. The tail is a single text node, so a qualifier split across inline
    elements is unseen, which fails toward the old self-link behaviour.
    """
    m = _TAIL_PREP.match(tail)
    if m is None:
        return _AMENDED_PREDICATE.match(tail) is not None
    rest = tail[m.end() :]
    ext = _TAIL_EXTERNAL.search(rest)
    if ext is None:
        return False
    return _TAIL_SELF.search(rest[: ext.start()]) is None


def _tail_is_self_qualified(tail: str) -> bool:
    """ "of this Act" / "under these Regulations", explicit self-reference."""
    m = _TAIL_PREP.match(tail)
    return m is not None and _TAIL_SELF.match(tail[m.end() :]) is not None


# --- public API --------------------------------------------------------------


async def emit_inline_markup(
    akn_xml: str,
    country: str,
    doctype: str = "act",
    client: "LLMClient | None" = None,
    skip_external_refs: bool = False,
) -> str:
    """Inject <ref>, <def>, <term> into AKN body text. Idempotent. ``skip_external_refs``
    turns off the external-citation passes for documents whose references arrived
    pre-resolved from a structured parser; internal references still run.
    """
    cfg = load_config(country)
    root = parse_xml(akn_xml)
    body = root.find(".//akn:body", NS)
    if body is None:
        return akn_xml

    # Pass 1: internal cross-references
    ref_count = _mark_internal_refs(root, body, country, doctype, cfg)
    logger.debug("inline_pass_1_refs", country=country, count=ref_count)

    # Pass 2: definition extraction
    terms, def_section_eids = await _mark_definitions(root, body, client)
    logger.debug(
        "inline_pass_2_defs",
        country=country,
        terms=list(terms.keys())[:20],
        def_sections=list(def_section_eids),
    )

    # Pass 3: term occurrences (skip definition sections)
    term_count = _mark_term_occurrences(body, terms, def_section_eids) if terms else 0
    logger.debug("inline_pass_3_terms", country=country, count=term_count)

    ext_count = 0
    if not skip_external_refs:
        # Pass 4a: deterministic external references, Arabic title-year + Slavic
        # № N від DATE patterns. Cheap, content-triggered, runs before LLM Pass 4
        # so the model sees fewer redundant candidates.
        ext_count_det = _mark_external_refs_deterministic(root, body, country, doctype)
        ext_count_det += _mark_external_refs_cyrillic(body, country, doctype)
        ext_count_det += _mark_external_refs_numbered(body, country)
        logger.debug("inline_pass_4a_ext_refs_det", country=country, count=ext_count_det)

        # Pass 4a2: named-act references via the jurisdiction's short-title index
        # ("the Freedom of Information Act 2000" carries no number; the index does).
        title_count = _mark_title_refs(body, country)
        logger.debug("inline_pass_4a2_title_refs", country=country, count=title_count)

        # Pass 4b: LLM-assisted external references (catches whatever 4a missed).
        ext_count = ext_count_det
        if client is not None:
            ext_count += await _mark_external_refs(root, body, country, doctype, client)
            logger.debug("inline_pass_4_ext_refs", country=country, count=ext_count)

    # Pass 5: type refs in amendment context (class="amendment-<op>" -> mod_* edges).
    amend_count = _classify_amendment_refs(body, country)
    logger.debug("inline_pass_5_amendments", country=country, count=amend_count)

    # Pass 6: amendment-schedule scope, "The X Act is amended as follows"
    # binds following op-paragraphs to that act.
    scope_count = _mark_scoped_amendments(body)
    logger.debug("inline_pass_6_scoped_amendments", country=country, count=scope_count)

    # Inject TLCTerm entries into <references>
    if terms:
        _inject_tlc_terms(root, terms)

    if ref_count + len(terms) + term_count + ext_count > 0:
        logger.info(
            "inline_markup_complete",
            refs=ref_count,
            defs=len(terms),
            terms=term_count,
            ext_refs=ext_count,
        )

    return cast(str, etree.tostring(root, pretty_print=True, encoding="unicode"))


# --- Pass 1: internal refs ---------------------------------------------------


def _mark_internal_refs(
    root: etree._Element,
    body: etree._Element,
    country: str,
    doctype: str,
    cfg: JurisdictionConfig,
) -> int:
    term_map, _ = build_term_map(cfg, doctype)
    if not term_map:
        return 0

    # Build unanchored regex from jurisdiction's term vocabulary.
    units = sorted(term_map.keys(), key=len, reverse=True)
    unit_alt = "|".join(re.escape(u) for u in units)
    pattern = re.compile(
        rf"\b(?P<unit>{unit_alt})\.?\s*(?P<num>\d+[A-Za-z]?(?:[-/]\d+[A-Za-z]?)?)(?P<subs>(?:\s*\(\s*[0-9A-Za-z]+\s*\))*)",
        re.IGNORECASE,
    )

    # The scaffolder emits parent-prefixed eIds (e.g. `chp_1__art_5__para_2`);
    # resolve_informal_reference returns the bare suffix. Map back to the
    # concrete eId, preferring matches under the calling element's ancestor.
    eid_index: list[str] = [el.get("eId", "") for el in root.iter() if el.get("eId")]

    eid_set = set(eid_index)

    # Publisher-convention bridge: native AKN uses hyphenated element-name
    # eIds ("section-55-2") where ours are abbrev paths ("sec_55__subsec_2").
    # Key both by (unit element, numeric path) so either convention resolves.
    doc_class = cfg.get_document_class(doctype)
    abbrev_to_element = {
        (e.eid_abbrev or eid_abbrev(e.akn_element)): e.akn_name or e.akn_element
        for e in (doc_class.hierarchy if doc_class else [])
        if e.akn_element
    }
    publisher_key: dict[tuple[str, tuple[str, ...]], str] = {}
    for eid in eid_index:
        m = re.match(r"^([a-z]+)-([0-9a-z-]+)$", eid)
        if m:
            key = (m.group(1), tuple(m.group(2).split("-")))
            publisher_key.setdefault(key, eid)

    def _publisher_lookup(candidate: str) -> str | None:
        parts = candidate.split("__")
        m = re.match(r"^([a-z]+)_(.+)$", parts[0])
        if m is None:
            return None
        unit = abbrev_to_element.get(m.group(1), m.group(1))
        nums = [m.group(2)]
        for part in parts[1:]:
            pm = re.match(r"^[a-z]+_(.+)$", part)
            if pm is None:
                return None
            nums.append(pm.group(1))
        return publisher_key.get((unit, tuple(n.lower() for n in nums)))

    def _resolve_in_document(candidate: str, p: etree._Element) -> str | None:
        """Map a candidate eId to one the document contains, or None: a dangling `<ref>` is
        worse than plain text. Tries exact, suffix match under the calling element's
        ancestors, global suffix, then walks the candidate's own path up
        (``sec_5__subsec_2`` to ``sec_5``) so a sub-reference still links to the nearest
        real unit.
        """
        if candidate in eid_set:
            return candidate
        published = _publisher_lookup(candidate)
        if published is not None:
            return published
        ancestor_eids: list[str] = []
        node = p.getparent()
        while node is not None:
            eid = node.get("eId") if hasattr(node, "get") else None
            if eid:
                ancestor_eids.append(eid)
            node = node.getparent()
        suffix = f"__{candidate}"
        for ancestor_eid in ancestor_eids:
            prefix = f"{ancestor_eid}__"
            target = ancestor_eid + suffix
            for eid in eid_index:
                if eid.startswith(prefix) and (eid == target or eid.endswith(suffix)):
                    return eid
        for eid in eid_index:
            if eid.endswith(suffix):
                return eid
        parts = candidate.split("__")
        for cut in range(len(parts) - 1, 0, -1):
            ancestor = "__".join(parts[:cut])
            if ancestor in eid_set:
                return ancestor
            anc_suffix = f"__{ancestor}"
            for eid in eid_index:
                if eid.endswith(anc_suffix):
                    return eid
        return None

    count = 0
    for p in body.iter(f"{{{AKN_NS}}}p"):
        ptext = "".join(p.itertext())
        # Amendment instructions to another act make bare "section N" refer to
        # the amended act, not this one, never self-link there.
        other_act_context = bool(_AMEND_CONTEXT.search(ptext) and _OTHER_ACT_NAME.search(ptext))

        def make_ref(m: re.Match[str]) -> etree._Element | None:
            tail = m.string[m.end() : m.end() + 160]
            if other_act_context and not _tail_is_self_qualified(tail):
                return None
            if _tail_redirects_elsewhere(tail):
                return None
            full = m.group(0).strip()
            candidate = resolve_informal_reference(full, country, doctype)
            if candidate is None:
                return None
            concrete = _resolve_in_document(candidate, p)
            if concrete is None:
                return None  # unresolved target, leave as plain text
            return _derived_ref(f"#{concrete}")

        count += _wrap_text_matches(p, pattern, make_ref)

    return count


# --- Pass 2: definitions -----------------------------------------------------


async def _mark_definitions(
    root: etree._Element,
    body: etree._Element,
    client: "LLMClient | None",
) -> tuple[dict[str, str], set[str]]:
    """Find defined terms. Returns ({normalised_term: slug}, {def_section_eids})."""
    terms: dict[str, str] = {}
    def_section_eids: set[str] = set()

    for section in body.iter(f"{{{AKN_NS}}}section"):
        heading = section.find("akn:heading", NS)
        if heading is None or not heading.text:
            continue
        if not _DEFINITION_HEADING_RE.search(heading.text):
            continue

        # Tier 1: quoted terms
        for p in section.iter(f"{{{AKN_NS}}}p"):
            text = _collect_text(p)
            for m in _QUOTED_TERM_RE.finditer(text):
                term = m.group(1).strip()
                if len(term) < 2:
                    continue
                slug = _slugify(term)
                terms[term.lower()] = slug

            # Wrap the quoted terms in <def>
            def make_def(m: re.Match[str]) -> etree._Element | None:
                inner = m.group(1).strip()
                if len(inner) < 2:
                    return None
                slug = _slugify(inner)
                el = etree.Element(
                    f"{{{AKN_NS}}}def",
                    attrib={"refersTo": f"#term-{slug}"},
                )
                return el

            # Match the full quoted form including quotes for wrapping
            _wrap_text_matches(p, _QUOTED_TERM_RE, make_def)

        # Tier 2: LLM extraction for non-quoted terms
        if client is not None:
            section_text = _collect_text(section)
            if len(section_text) > 50:
                llm_terms = await _extract_definitions_llm(section_text, client)
                for t, s in llm_terms.items():
                    terms.setdefault(t, s)

        eid = section.get("eId", "")
        if eid:
            def_section_eids.add(eid)

    return terms, def_section_eids


async def _extract_definitions_llm(text: str, client: "LLMClient") -> dict[str, str]:
    """LLM-assisted definition extraction for civil-law and non-quoted terms."""
    prompt = (
        "Extract all explicitly defined terms from this legislative text.\n"
        "Include only terms with 'means', 'includes', 'refers to', or equivalent.\n"
        'Return JSON: {"terms": [{"term": "company", "slug": "company"}]}\n'
        'If no terms found, return {"terms": []}\n\n'
        f"Text:\n{text}"
    )
    try:
        result = await client.chat_json(prompt=prompt)
        out: dict[str, str] = {}
        for entry in result.get("terms") or []:
            term = str(entry.get("term", "")).strip()
            slug = str(entry.get("slug", "")).strip() or _slugify(term)
            if term and len(term) >= 2:
                out[term.lower()] = slug
        return out
    except Exception:
        logger.warning("definition_extraction_llm_failed", exc_info=True)
        return {}


# --- Pass 3: term occurrences ------------------------------------------------


def _mark_term_occurrences(
    body: etree._Element,
    terms: dict[str, str],
    def_section_eids: set[str],
) -> int:
    if not terms:
        return 0

    sorted_terms = sorted(terms.keys(), key=len, reverse=True)
    alt = "|".join(re.escape(t) for t in sorted_terms)
    pattern = re.compile(rf"\b({alt})\b", re.IGNORECASE)
    slug_map = terms

    count = 0
    for section in body.iter(f"{{{AKN_NS}}}section"):
        if section.get("eId", "") in def_section_eids:
            continue
        for p in section.iter(f"{{{AKN_NS}}}p"):

            def make_term(m: re.Match[str]) -> etree._Element | None:
                matched = m.group(0)
                slug = slug_map.get(matched.lower())
                if slug is None:
                    return None
                return etree.Element(
                    f"{{{AKN_NS}}}term",
                    attrib={"refersTo": f"#term-{slug}"},
                )

            count += _wrap_text_matches(p, pattern, make_term)

    return count


# --- Pass 4a: deterministic external refs (Arabic) ---------------------------

# Arabic legal citation, anchored on the `لسنة YYYY` suffix: everything between the
# kind word (قانون / مرسوم / أمر / نظام / قرار) and it is optional title text or a
# `رقم N` reference. Year is 4 digits, ASCII or Arabic-Indic. Title words exclude
# digits, so a leading `رقم N` lands in `num` rather than the first title word.
_ARABIC_KINDS = "قانون|مرسوم|أمر|نظام|قرار"
_ARABIC_CITATION_RE = re.compile(
    rf"(?P<kind>{_ARABIC_KINDS})"
    r"(?:\s+رقم\s+(?P<num>[\d٠-٩]+))?"  # optional `رقم N` right after kind
    r"(?P<title>(?:\s+[^\d٠-٩\s][^\s]*){0,8}?)"  # 0-8 title words; no leading digit
    r"(?:\s+رقم\s+(?P<num2>[\d٠-٩]+))?"  # optional `رقم N` after title (rarer form)
    r"\s+لسنة\s+(?P<year>[\d٠-٩]{4})"  # mandatory year marker
)


def _mark_external_refs_deterministic(
    root: etree._Element,
    body: etree._Element,
    country: str,
    doctype: str,
) -> int:
    """Detect external Arabic act citations by regex, before the LLM pass so the model sees
    fewer redundant candidates.

    Triggered by Arabic letters in the body rather than by jurisdiction code, so an
    act drafted in Arabic gets the same treatment whatever its jurisdiction, and a
    bilingual document is not skipped for having a primary code absent from a
    hand-maintained allowlist.
    """
    if not _has_arabic_content(body):
        return 0

    count = 0
    for p in body.iter(f"{{{AKN_NS}}}p"):

        def make_ref(m: re.Match[str]) -> etree._Element | None:
            year = normalise_digits(m.group("year"))
            num_raw = m.group("num") or m.group("num2")
            number = normalise_digits(num_raw) if num_raw else ""
            if not number:
                # No explicit number, fall back to a slug of the kind+title so
                # the FRBR URI stays unique-per-cited-act rather than collapsing
                # every untitled citation for a given year into the same target.
                kind = m.group("kind") or ""
                title = (m.group("title") or "").strip()
                number = _unicode_slug(f"{kind} {title}")
            try:
                frbr = construct_frbr_uri(country, doctype, year, number)
            except Exception:  # noqa: BLE001, defensive
                return None
            return _derived_ref(frbr)

        count += _wrap_text_matches(p, _ARABIC_CITATION_RE, make_ref)
    return count


_ARABIC_LETTER_RE = re.compile(r"[ء-ي٠-٩۰-۹]")
_CYRILLIC_LETTER_RE = re.compile(r"[Ѐ-ӿ]")


def _has_arabic_content(body: etree._Element, sample_chars: int = 4000) -> bool:
    """Return True if the body's body-text carries Arabic letters. Samples up
    to ~4 KB of text to keep the check fast on long documents, Arabic content
    cluster early in any document that has it."""
    return _has_script_content(body, _ARABIC_LETTER_RE, sample_chars)


def _has_cyrillic_content(body: etree._Element, sample_chars: int = 4000) -> bool:
    """Return True if the body's body-text carries Cyrillic letters."""
    return _has_script_content(body, _CYRILLIC_LETTER_RE, sample_chars)


def _has_script_content(body: etree._Element, pat: re.Pattern[str], sample_chars: int) -> bool:
    sampled = 0
    for p in body.iter(f"{{{AKN_NS}}}p"):
        for chunk in p.itertext():
            sampled += len(chunk)
            if pat.search(chunk):
                return True
            if sampled >= sample_chars:
                return False
    return False


# Ukrainian (also Russian, Belarusian) law citation. The "-IX" suffix is the
# Verkhovna Rada convocation marker, which with the date pins the law identity
# unambiguously. Reverse word order (`від DATE № N`) is equally common, hence the
# second pattern.
_CYR_PREP_OF = "від|от|ад|од"  # ukr / rus / bel / sr/mk
_CYR_LAW_CITATION_FWD_RE = re.compile(
    r"№\s*(?P<num>\d+(?:-[A-ZА-Я0-9IVXLCDM]+)?)"
    rf"\s+(?:{_CYR_PREP_OF})\s+"
    r"(?P<date>\d{1,2}\.\d{1,2}\.\d{4})"
)
_CYR_LAW_CITATION_REV_RE = re.compile(
    rf"(?:{_CYR_PREP_OF})\s+(?P<date>\d{{1,2}}\.\d{{1,2}}\.\d{{4}})\s+"
    r"№\s*(?P<num>\d+(?:-[A-ZА-Я0-9IVXLCDM]+)?)"
)


def _mark_external_refs_cyrillic(body: etree._Element, country: str, doctype: str) -> int:
    """Detect canonical Slavic-language law citations. Triggered by Cyrillic
    content in the body so a multilingual doc with a single Cyrillic block
    still benefits."""
    if not _has_cyrillic_content(body):
        return 0
    count = 0
    for p in body.iter(f"{{{AKN_NS}}}p"):
        count += _wrap_text_matches(p, _CYR_LAW_CITATION_FWD_RE, _cyr_make_ref(country, doctype))
        count += _wrap_text_matches(p, _CYR_LAW_CITATION_REV_RE, _cyr_make_ref(country, doctype))
    return count


def _cyr_make_ref(country: str, doctype: str) -> Callable[[re.Match[str]], etree._Element | None]:
    def builder(m: re.Match[str]) -> etree._Element | None:
        num = m.group("num")
        date = m.group("date")  # DD.MM.YYYY
        try:
            _day, _month, year = date.split(".")
            int(year)
        except ValueError:
            return None
        try:
            # Year slot only (naming-convention house style); the number's
            # convocation suffix disambiguates within the year.
            frbr = construct_frbr_uri(country, doctype, year, num)
        except Exception:  # noqa: BLE001
            return None
        return _derived_ref(frbr)

    return builder


# --- Pass 4a3: numbered-instrument citations --------------------------------

# Citations by a yearless series number. The lookahead refuses a list: splitting
# OCR-damaged lists ("580,1577 and5") would invent confident targets.
_NUMBERED_TAIL = (
    r"\s+(?:No\.?|Nos\.?|Numbered|Blg\.?)?\s*(?P<num>\d+)\b"
    r"(?!\s*(?:,|and|&)\s*(?:Nos?\.?\s*)?\d)"
)

# Kind → FRBR doctype. Full names precede abbreviations so the specific citation
# wraps first and the abbreviation pass leaves the wrapped span alone. Executive
# Orders are excluded: their FRBR pattern needs a president slug that a citation
# never carries. Plain "Act No. N" is excluded too: it has no distinct doctype
# and would collide with Republic Act URIs.
_NUMBERED_CITATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(rf"\bRepublic\s+Acts?{_NUMBERED_TAIL}", re.I), "act"),
    (re.compile(rf"\bPresidential\s+Decrees?{_NUMBERED_TAIL}", re.I), "pd"),
    (re.compile(rf"\bBatas\s+Pambansa{_NUMBERED_TAIL}", re.I), "bp"),
    (re.compile(rf"\bCommonwealth\s+Acts?{_NUMBERED_TAIL}", re.I), "ca"),
    (re.compile(rf"\bR\.?\s?A\.?{_NUMBERED_TAIL}", re.I), "act"),
    (re.compile(rf"\bP\.?\s?D\.?{_NUMBERED_TAIL}", re.I), "pd"),
    (re.compile(rf"\bB\.?\s?P\.?{_NUMBERED_TAIL}", re.I), "bp"),
    (re.compile(rf"\bC\.?\s?A\.?{_NUMBERED_TAIL}", re.I), "ca"),
)


def _mark_external_refs_numbered(body: etree._Element, country: str) -> int:
    """Detect numbered-instrument citations and build FRBR work URIs straight
    from the instrument number.

    The series never reuses a number, so the year slot takes the unknown-year
    placeholder. Gated on jurisdiction: Latin script gives no content trigger.

    Logs per-doctype hit counts so ingest can report hit rates.
    """
    if country != "ph":
        return 0

    hits: dict[str, int] = {}

    def builder(doctype: str) -> Callable[[re.Match[str]], etree._Element | None]:
        def make(m: re.Match[str]) -> etree._Element | None:
            try:
                frbr = construct_frbr_uri(country, doctype, "", m.group("num"))
            except Exception:  # noqa: BLE001, defensive
                return None
            hits[doctype] = hits.get(doctype, 0) + 1
            return _derived_ref(frbr)

        return make

    count = 0
    for p in body.iter(f"{{{AKN_NS}}}p"):
        for pattern, doctype in _NUMBERED_CITATIONS:
            count += _wrap_text_matches(p, pattern, builder(doctype))

    if count:
        logger.info("numbered_citation_pass", hits=hits, total=count)
    return count


# --- Pass 4b: LLM-assisted external refs -------------------------------------


async def _mark_external_refs(
    root: etree._Element,
    body: etree._Element,
    country: str,
    doctype: str,
    client: "LLMClient",
) -> int:
    """LLM-assisted external act reference detection."""
    # Size-bounded chunks that partition the WHOLE body, the old
    # section-only iteration silently skipped schedules, where UK amendment
    # references live. Small siblings pack into one call.
    chunks = _external_chunks(body)
    if not chunks:
        return 0
    batches: list[list[tuple[etree._Element, str]]] = []
    cur: list[tuple[etree._Element, str]] = []
    size = 0
    for el, text in chunks:
        if cur and size + len(text) > _EXT_CHUNK_CHARS:
            batches.append(cur)
            cur, size = [], 0
        cur.append((el, text))
        size += len(text)
    if cur:
        batches.append(cur)

    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def extract_one(batch: list[tuple[etree._Element, str]]) -> list[dict[str, Any]]:
        joined = "\n\n".join(t for _, t in batch)
        async with sem:
            return await _extract_external_refs_llm(joined, country, client)

    all_refs = await asyncio.gather(*[extract_one(b) for b in batches])

    count = 0
    for batch, refs in zip(batches, all_refs):
        for ref_info in refs:
            ref_text = ref_info.get("text", "")
            frbr = ref_info.get("frbr_uri", "")
            if not ref_text or not frbr:
                continue
            escaped = re.escape(ref_text)
            pat = re.compile(rf"({escaped})", re.IGNORECASE)
            for el, _ in batch:
                for p in el.iter(f"{{{AKN_NS}}}p"):

                    def make_ext_ref(m: re.Match[str], _href: str = frbr) -> etree._Element:
                        return _derived_ref(_href)

                    count += _wrap_text_matches(p, pat, make_ext_ref)

    return count


_EXT_CHUNK_CHARS = 6000


def _external_chunks(body: etree._Element) -> list[tuple[etree._Element, str]]:
    """Partition the body into elements small enough to prompt with; large
    containers recurse into their children so nothing is skipped."""
    out: list[tuple[etree._Element, str]] = []

    _INLINE = {"ref", "term", "def", "b", "i", "u", "sup", "sub", "span", "inline", "marker"}

    def emit(el: etree._Element) -> None:
        text = _collect_text(el)
        if len(text) <= 30:
            return
        children = [
            k for k in el if isinstance(k.tag, str) and etree.QName(k).localname not in _INLINE
        ]
        if len(text) <= _EXT_CHUNK_CHARS or not children:
            out.append((el, text))
            return
        for k in children:
            emit(k)

    emit(body)
    return out


# LLM `kind` → (jurisdiction override, FRBR doctype). Default: host-country act.
_REF_KIND_TARGETS: dict[str, tuple[str | None, str]] = {
    "eu-regulation": ("eu", "regulation"),
    "eu-directive": ("eu", "directive"),
    "eu-decision": ("eu", "decision"),
    "secondary": (None, "si"),
}


def _digit_token(value: object) -> str:
    """LLM-extracted year field folded to a digit string, or "".
    JSON null arrives as None; str(None) is the literal "None" that once
    minted /None/ FRBR segments, so anything non-digit is treated absent."""
    token = normalise_digits(str(value if value is not None else "")).strip()
    if token and not token.isdigit():
        logger.debug("ref_year_discarded", value=token)
        return ""
    return token


async def _extract_external_refs_llm(
    text: str,
    country: str,
    client: "LLMClient",
) -> list[dict[str, Any]]:
    prompt = (
        "Extract references to external legislation: other Acts, Codes,\n"
        "Ordinances, Decrees, Regulations, Directives, statutory instruments,\n"
        "named conventions.\n"
        "Do NOT include internal references (article 5, статті 7, المادة ٥).\n"
        "Do NOT include short citations already covered by canonical patterns:\n"
        "  - `№ NNNN-IX від DD.MM.YYYY` (Ukrainian)\n"
        "  - `قانون ... لسنة YYYY` / `قانون رقم N لسنة YYYY` (Arabic)\n"
        "DO include bare numeric citations of whole instruments, e.g.\n"
        "`Regulation (EU) 2016/679`, `Directive 95/46/EC`, `S.I. 2019/419`,\n"
        "`the Data Protection Act 1998`.\n"
        '\nReturn JSON: {"refs": [{"text": "the exact text as it appears",\n'
        ' "title": "short canonical title", "year": "YYYY", "number": "NNN",\n'
        ' "kind": "act|secondary|eu-regulation|eu-directive|eu-decision",\n'
        ' "series": "publisher series token when the citation names one"}]}\n'
        "kind: `act` = primary legislation of this jurisdiction (default);\n"
        "`secondary` = statutory instrument / subsidiary legislation;\n"
        "series: the legislation.gov.uk type token a devolved citation carries\n"
        "(`2020 asp 8` → asp, `S.S.I. 2020/1` → ssi, `W.S.I.` → wsi,\n"
        "`S.R.` → nisr, `anaw`, `asc`, `nia`); omit for the default series;\n"
        "`eu-*` = EU instruments (year is the 4-digit year: Directive 95/46/EC\n"
        "→ year 1995, number 46; Regulation (EU) 2016/679 → 2016, 679).\n"
        'If none found, return {"refs": []}\n\n'
        f"Text:\n{text}"
    )
    try:
        result = await client.chat_json(prompt=prompt)
        out: list[dict[str, Any]] = []
        for r in result.get("refs") or []:
            year = _digit_token(r.get("year"))
            number = law_number_token(r.get("number"))
            ref_text = str(r.get("text", ""))
            # A work URI without a number segment is a dangling trailing-slash
            # href; year alone is not enough to cite anything.
            if ref_text and number:
                juris, doctype = _REF_KIND_TARGETS.get(
                    str(r.get("kind", "") or "").lower(), (None, "act")
                )
                target_country = juris or country
                # A series of the same family wins, under the country that publishes it.
                series = str(r.get("series") or "").strip().lower()
                series_country = token_country(series)
                if juris is None and series_country and token_family(series) == doctype:
                    frbr = construct_frbr_uri(series_country, series, year, number)
                else:
                    frbr = construct_frbr_uri(
                        target_country, default_token(target_country, doctype), year, number
                    )
                out.append({"text": ref_text, "frbr_uri": frbr})
        return out
    except Exception:
        logger.warning("external_ref_extraction_failed", exc_info=True)
        return []


# --- Pass 4a2: short-title index -----------------------------------------------

# Literal SI citations: each prefix names a series; "(N.I. n)" marks an Order
# in Council.
_SI_REF = re.compile(r"\b(S\.S\.I\.|W\.S\.I\.|S\.I\.)\s*(\d{4})/(\d+)\b(\s*\(N\.I\.\s*\d+\))?")
_SI_SERIES = {"S.S.I.": "ssi", "W.S.I.": "wsi"}

# Cyrillic statute citations quote the title: Закон України «Про …».
_TITLE_REF_CYR = re.compile(r"[Зз]акон(?:у|ом|і)?\s+України\s*[«\"]([^»\"]{3,160})[»\"]")

_TITLE_WORD = r"(?:\(?[A-Z][\w'&()\u2019.-]*\)?|of|and|on|etc\.?)"
# Trailing lookahead: inside "the X Act 2000 (...) Order 2001" the inner Act
# span must not link, the citation names the Order.
_TITLE_REF = re.compile(
    rf"\b[Tt]he\s+({_TITLE_WORD}(?:,?\s+{_TITLE_WORD}){{0,14}}?"
    r"\s+(?:Act|Regulations|Order)\s+(?:19|20)\d{2})\b"
    r"(?!\s*(?:\([^)]*\)\s*)*(?:Act|Regulations|Order|Rules)\s+(?:19|20)\d{2})"
)


def _mark_title_refs(body: etree._Element, country: str) -> int:
    # An empty index misses every title; the literal SI matcher still runs.
    index = load_short_titles(country)
    count = 0
    for p in body.iter(f"{{{AKN_NS}}}p"):

        def make_ref(m: re.Match[str]) -> etree._Element | None:
            work = index.get(re.sub(r"\s+", " ", m.group(1)).strip().lower())
            if work is None:
                return None
            return _derived_ref(work)

        count += _wrap_text_matches(p, _TITLE_REF, make_ref)
        count += _wrap_text_matches(p, _TITLE_REF_CYR, make_ref)

        def make_si_ref(m: re.Match[str]) -> etree._Element:
            # A recognised series names the country that publishes it.
            token = "nisi" if m.group(4) else _SI_SERIES.get(m.group(1))
            if token is None:
                token = default_token(country, "si")
            # Through the one constructor, like every other site: this was the
            # last place minting an unmarked ref, so a series citation read out
            # of prose landed as publisher-authored and skipped the gate.
            return _derived_ref(
                construct_frbr_uri(token_country(token) or country, token, m.group(2), m.group(3))
            )

        count += _wrap_text_matches(p, _SI_REF, make_si_ref)
    return count


# --- Pass 5: amendment classification ----------------------------------------

# Op cues by priority, shared by passes 5 and 6 so they cannot classify one
# instruction differently. Defaults still differ: pass 5 falls back to "replace" on
# an amendment paragraph with no cue, pass 6 emits nothing.
_OP_CUES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsubstitut|\breplac", re.I), "replace"),
    (
        re.compile(r"\bomit|\brepeal|\brevok|\bdelet|\bcease[sd]?\s+to\s+have\s+effect", re.I),
        "delete",
    ),
    (
        re.compile(r"\binsert|\badd(?:ed)?\b|\bAfter\s+(?:section|paragraph|subsection)", re.I),
        "insert",
    ),
    (re.compile(r"\brenumber", re.I), "renumber"),
)
_AMEND_CONTEXT = re.compile(r"\bamend|\brepeal|\brevok|\bsubstitut|\bomit(?:ted)?\b|\binsert", re.I)


def _classify_amendment_refs(body: etree._Element, country: str) -> int:
    """Set class="amendment-<op>" on refs whose paragraph reads as an
    amendment instruction; jurisdiction phrasing (config) adds a cue."""
    cfg = load_config(country)
    phrasing = getattr(getattr(cfg, "amendments", None), "phrasing", None)
    phrase_re = re.compile(re.escape(phrasing), re.I) if phrasing else None
    count = 0
    for p in body.iter(f"{{{AKN_NS}}}p"):
        text = "".join(p.itertext())
        if not (_AMEND_CONTEXT.search(text) or (phrase_re and phrase_re.search(text))):
            continue
        op = "replace"
        for cue, candidate in _OP_CUES:
            if cue.search(text):
                op = candidate
                break
        for ref in p.iter(f"{{{AKN_NS}}}ref"):
            if ref.get("href", "").startswith("#"):
                continue  # internal refs aren't the amended work
            if "amendment" in (ref.get("class") or ""):
                continue
            # Appended, not replaced: this ref may already carry the marker
            # saying this pass minted it, and overwriting it filed a minted
            # amendment reference as publisher-authored.
            existing = (ref.get("class") or "").split()
            ref.set("class", " ".join([*existing, f"amendment-{op}"]))
            count += 1
    return count


# --- Pass 6: amendment-schedule scope -----------------------------------------

_SCOPE_CONTAINERS = {"part", "chapter", "schedule", "hcontainer", "division"}


def _scope_container(p: etree._Element) -> etree._Element | None:
    """Nearest structural ancestor, the span an amendment opener governs.
    Openers live in `<paragraph><content><p>`; sibling `<paragraph>`s carry
    the ops, so the scope must anchor above the numbered unit."""
    node = p.getparent()
    while node is not None:
        if isinstance(node.tag, str) and etree.QName(node).localname in _SCOPE_CONTAINERS:
            return node
        node = node.getparent()
    return None


_SCOPE_OPENER = re.compile(r"\b(?:is|are)\s+amended\s+(?:as\s+follows|in\s+accordance)", re.I)


def _mark_scoped_amendments(body: etree._Element) -> int:
    """Bind bare amendment instructions to the act a scope opener named.

    UK amendment schedules name the amended act once ("The X Act is amended as follows")
    and follow with op-paragraphs that never repeat it. The opener's <ref> sets the scope
    for following paragraphs under the same container, and each gains a typed ref to the
    scoped act. Paragraphs already carrying an amendment-classed ref are left alone.
    """
    count = 0
    scope_href: str | None = None
    scope_parent: etree._Element | None = None
    for p in body.iter(f"{{{AKN_NS}}}p"):
        text = "".join(p.itertext())
        if _SCOPE_OPENER.search(text):
            refs = [
                r for r in p.iter(f"{{{AKN_NS}}}ref") if not (r.get("href") or "").startswith("#")
            ]
            if refs:
                scope_href = refs[0].get("href")
                scope_parent = _scope_container(p)
                for r in refs[:1]:
                    tokens = (r.get("class") or "").split()
                    if not any(t.startswith("amendment-") for t in tokens):
                        # Composed, not replaced: the provenance marker may
                        # already be here and overwriting it filed a minted
                        # reference as publisher-authored.
                        r.set("class", " ".join([*tokens, "amendment-replace"]))
            else:
                scope_href = None
                scope_parent = None
            continue
        if scope_href is None or scope_parent is None:
            continue
        node = p
        inside = False
        while node is not None:
            if node is scope_parent:
                inside = True
                break
            node = node.getparent()
        if not inside:
            scope_href = None
            continue
        if any("amendment" in (r.get("class") or "") for r in p.iter(f"{{{AKN_NS}}}ref")):
            continue
        # Multi-op paragraphs ("omit X ... insert Y") emit one ref per op.
        for cue, op in _OP_CUES:
            if not cue.search(text):
                continue
            # Minted from prose like every other ref this pass makes, so it
            # carries the provenance marker beside the operation. Without it
            # these landed as publisher-authored and skipped the gate.
            ref = etree.Element(
                f"{{{AKN_NS}}}ref",
                attrib={
                    "href": scope_href,
                    "class": f"{DERIVED_REF_CLASS} amendment-{op}",
                },
            )
            ref.text = ""
            if p.text is not None:
                ref.tail = p.text
                p.text = None
            p.insert(0, ref)
            count += 1
    return count


# --- TLCTerm injection -------------------------------------------------------


def _inject_tlc_terms(root: etree._Element, terms: dict[str, str]) -> None:
    """Add <TLCTerm> entries to <references> for each defined term."""
    meta = root.find(".//akn:meta", NS)
    if meta is None:
        return
    refs = meta.find("akn:references", NS)
    if refs is None:
        refs = etree.SubElement(meta, f"{{{AKN_NS}}}references", attrib={"source": "#codify"})

    existing = {el.get("eId") for el in refs}
    for term, slug in terms.items():
        eid = f"term-{slug}"
        if eid in existing:
            continue
        etree.SubElement(
            refs,
            f"{{{AKN_NS}}}TLCTerm",
            attrib={
                "eId": eid,
                "href": f"/ontology/term/{slug}",
                "showAs": term,
            },
        )


# --- XML text manipulation helpers -------------------------------------------


def _wrap_text_matches(
    element: etree._Element,
    pattern: re.Pattern[str],
    make_wrapper: Callable[[re.Match[str]], etree._Element | None],
) -> int:
    """Wrap regex matches in element's text nodes with inline AKN elements.

    Processes right-to-left so character offsets remain valid. Handles both
    element.text and child .tail attributes. Returns count of wraps applied.
    """
    count = 0
    if element.tag == f"{{{AKN_NS}}}ref":
        return 0  # never mark up inside an existing reference

    # Process element.text
    if element.text:
        count += _wrap_in_text_attr(element, "text", pattern, make_wrapper)

    # Process .tail of each direct child (including tails after semantic
    # elements, "section 5" might appear in text trailing a <ref>).
    for child in list(element):
        if child.tail:
            count += _wrap_in_tail(element, child, pattern, make_wrapper)

    return count


def _wrap_in_text_attr(
    element: etree._Element,
    attr: str,
    pattern: re.Pattern[str],
    make_wrapper: Callable[[re.Match[str]], etree._Element | None],
) -> int:
    text = getattr(element, attr)
    if not text:
        return 0

    matches = list(pattern.finditer(text))
    if not matches:
        return 0

    count = 0
    # Right-to-left: always insert at position 0 so document order is preserved.
    for m in reversed(matches):
        wrapper = make_wrapper(m)
        if wrapper is None:
            continue

        before = text[: m.start()]
        matched = text[m.start() : m.end()]
        after = text[m.end() :]

        wrapper.text = matched
        wrapper.tail = after

        setattr(element, attr, before)
        element.insert(0, wrapper)
        text = before
        count += 1

    return count


def _wrap_in_tail(
    parent: etree._Element,
    child: etree._Element,
    pattern: re.Pattern[str],
    make_wrapper: Callable[[re.Match[str]], etree._Element | None],
) -> int:
    text = child.tail
    if not text:
        return 0

    matches = list(pattern.finditer(text))
    if not matches:
        return 0

    count = 0
    child_idx = list(parent).index(child)

    # Right-to-left: each wrapper goes at child_idx + 1, pushing previous
    # wrappers rightward, preserving document order.
    for m in reversed(matches):
        wrapper = make_wrapper(m)
        if wrapper is None:
            continue

        before = text[: m.start()]
        matched = text[m.start() : m.end()]
        after = text[m.end() :]

        wrapper.text = matched
        wrapper.tail = after
        child.tail = before

        parent.insert(child_idx + 1, wrapper)
        text = before
        count += 1

    return count


def _collect_text(element: etree._Element) -> str:
    return " ".join(t.strip() for t in element.itertext() if t and t.strip())


def _slugify(term: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")
    return slug or "unknown"


_SLUG_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)
_SLUG_HYPHEN_RUN_RE = re.compile(r"-+")


def _unicode_slug(text: str) -> str:
    """Slugify a Unicode title, keeps letters from any script, drops
    punctuation. Used for FRBR URIs when an Arabic/Cyrillic citation gives
    a title but no canonical number; the slug encodes the cited act's
    identity so distinct citations don't collapse to the same target."""
    text = _SLUG_NON_WORD_RE.sub("-", text)
    text = _SLUG_HYPHEN_RUN_RE.sub("-", text).strip("-").lower()
    return text or "unknown"
