"""EUR-Lex legacy HTML → AKN4EU.

Acts published before FORMEX exist at Cellar only as an HTML 4.01 rendering of
the Official Journal text: one `<p>` per block inside `div#TexteOnly`, with the
title first, the preamble up to the enacting formula, `Article N` on a line of
its own before each article, and the signature block from `Done at`. The
structure is regular enough to segment deterministically, so these acts take
the same enrichment and validation as a FORMEX act rather than the LLM lane,
which reads every cross-reference to an article as an article.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree, html

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"

# A heading is `Article N` or `ARTICLE N`, alone or with a short capitalised
# title; a sentence starting "Article 1 shall apply" and a lowercase "article 2" are prose.
_ARTICLE = re.compile(
    r"^(?:Article|ARTICLE)[\s\u2013\u2014-]+(\d+)([A-Za-z]{0,2})(?:\s*[-:\u2013]?\s+([A-Z][^.;:]{0,80}))?$"
)
_ANNEX = re.compile(r"^ANNEX(?:\s+([IVXLC]+|\d+|[A-Z]))?$")
_ENACTING = re.compile(
    r"^(HAS|HAVE)\s+(ADOPTED|DECIDED|MADE)\b.*:$|^HAS DECIDED AS FOLLOWS:$", re.IGNORECASE
)
_DONE_AT = re.compile(r"^Done at\b", re.IGNORECASE)
_NUMBERED_PARA = re.compile(r"^(\d{1,3})\.\s+(.*)$", re.DOTALL)
_MONTHS = {
    m: i
    for i, m in enumerate(
        (
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december",
        ),
        start=1,
    )
}  # fmt: skip
_LONG_DATE = re.compile(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b")
_OJ_SOURCE = re.compile(
    r"Official Journal\s+(?:L\s*)?(\w+)\s*,\s*(\d{2})/(\d{2})/(\d{4})\s+P\.\s*(\d+)"
)


class EurlexHtmlError(ValueError):
    """The HTML is not an act this converter can segment."""


@dataclass
class _Act:
    title: str
    preamble: list[str] = field(default_factory=list)
    formula: str | None = None
    postamble: list[str] = field(default_factory=list)
    articles: list[tuple[str, str | None, list[str]]] = field(default_factory=list)
    conclusions: list[str] = field(default_factory=list)
    annexes: list[tuple[str, list[str]]] = field(default_factory=list)
    oj: dict[str, str] = field(default_factory=dict)
    duplicate_paragraphs: int = 0


# Older renderings drop the `oj-` prefix from these classes.
_HEADER_CLASS = re.compile(r"^(?:oj-)?hd-")
_XML_DECLARATION = re.compile(r"^\ufeff?\s*<\?xml[^>]*\?>", re.IGNORECASE)


def is_eurlex_html(text: str) -> bool:
    """HTML 4.01 or XHTML, with or without an XML declaration in front."""
    head = _XML_DECLARATION.sub("", text[:300]).lstrip("\ufeff \t\r\n").lower()
    return head.startswith("<!doctype html") or head.startswith("<html")


def _blocks(text: str) -> tuple[list[str], dict[str, str], dict[str, int]]:
    """The act's text blocks in order, and the OJ reference from the page's metadata."""
    # Already decoded: lxml refuses a str that still declares its encoding.
    root = html.fromstring(_XML_DECLARATION.sub("", text, count=1))
    oj: dict[str, str] = {}
    for meta in root.iter("meta"):
        if meta.get("name") == "DC.source":
            m = _OJ_SOURCE.search(meta.get("content") or "")
            if m:
                number, day, month, year, page = m.groups()
                oj = {"number": number, "date": f"{year}-{month}-{day}", "page": page}
    container = root.get_element_by_id("TexteOnly", None)
    if container is None:
        container = root.find("body")
        if container is None:
            raise EurlexHtmlError("no body in the HTML")
    # The XHTML rendering's running header (date, OJ page) is a table, not the title.
    for table in list(container.iter("table")):
        if any(_HEADER_CLASS.match(p.get("class") or "") for p in table.iter("p")):
            table.drop_tree()
    # Cellar's older pages open `<p><TXT_TE>` and never close the `<p>`, so the parser
    # nests the whole act inside it; a paragraph holding paragraphs is unwrapped.
    for outer in [p for p in container.iter("p") if next(p.iterdescendants("p"), None) is not None]:
        outer.drop_tag()
    # A line break is a space, so `Effective<br>immediately` keeps its two words.
    for br in container.iter("br"):
        br.tail = " " + (br.tail or "")
    # An image is content the page holds only as pixels: named in place, and counted.
    images = 0
    for img in list(container.iter("img")):
        images += 1
        src = img.get("src") or ""
        # An inline image's src is the whole picture in base64; its alt names it.
        name = (img.get("alt") or "") if src.startswith("data:") else (src or img.get("alt") or "")
        marker = f"[image not transcribed: {name}]"
        if _inside_block(img, container):
            img.tail = f" {marker} " + (img.tail or "")
        else:
            p = img.makeelement("p")
            p.text = marker
            img.addprevious(p)
    blocks: list[str] = []
    previous_title = False
    for el in container.iter(*_BLOCK_TAGS):
        if _inside_block(el, container):
            continue  # a nested block's text is already in its outer block
        # The XHTML rendering splits the title over several lines; it is one block.
        is_title = el.get("class") in ("oj-doc-ti", "doc-ti")
        if is_title and previous_title:
            blocks[-1] += " " + " ".join(el.text_content().split())
            continue
        if el.tag == "tr":
            # A table row is one block, its cells separated; the table's shape is not kept.
            cells = [" ".join(c.text_content().split()) for c in el if c.tag in ("td", "th")]
            joined = " | ".join(c for c in cells if c)
        else:
            joined = " ".join(el.text_content().split())
        if joined:
            blocks.append(joined)
        previous_title = is_title and bool(joined)
    tables = sum(1 for _ in container.iter("table"))
    return (
        blocks,
        oj,
        {
            "tables": tables,
            "images": images,
            "discarded_chars": _uncaptured_chars(container),
        },
    )


_BLOCK_TAGS = ("p", "h1", "h2", "h3", "li", "tr")


def _inside_block(el: etree._Element, container: etree._Element) -> bool:
    parent = el.getparent()
    while parent is not None and parent is not container:
        if parent.tag in _BLOCK_TAGS:
            return True
        parent = parent.getparent()
    return False


def _uncaptured_chars(container: etree._Element) -> int:
    """Text no block element wraps, so a page with content elsewhere is not silently thin."""
    total = 0
    for el in container.iter():
        if not isinstance(el.tag, str) or el.tag in ("script", "style"):
            continue
        if el.tag not in _BLOCK_TAGS and not _inside_block(el, container):
            total += len("".join((el.text or "").split()))
        parent = el.getparent()
        if (
            parent is not None
            and parent.tag not in _BLOCK_TAGS
            and not _inside_block(parent, container)
        ):
            total += len("".join((el.tail or "").split()))
    return total


def _segment(blocks: list[str], oj: dict[str, str]) -> _Act:
    if not blocks:
        raise EurlexHtmlError("no text blocks in the HTML")
    act = _Act(title=blocks[0], oj=oj)
    i = 1
    # The OJ rendering repeats the title as the first body line.
    if i < len(blocks) and blocks[i] == act.title:
        i += 1
    # Preamble runs to the enacting formula; an act with none starts at Article 1.
    while i < len(blocks) and not _ARTICLE.match(blocks[i]) and not _ANNEX.match(blocks[i]):
        if _ENACTING.match(blocks[i]):
            act.formula = blocks[i]
            i += 1
            break
        act.preamble.append(blocks[i])
        i += 1
    current: list[str] | None = None
    section: str = "body"
    while i < len(blocks):
        block = blocks[i]
        i += 1
        if _DONE_AT.match(block) and not act.conclusions:
            section = "conclusions"
            act.conclusions.append(block)
            continue
        if _ANNEX.match(block):
            section = "annex"
            current = []
            act.annexes.append((block, current))
            continue
        if section == "conclusions":
            act.conclusions.append(block)
            continue
        m = _ARTICLE.match(block)
        if m and section == "body" and (m.group(3) is None or _is_title(m.group(3))):
            current = []
            act.articles.append((m.group(1) + m.group(2), m.group(3), current))
            continue
        if current is None:
            # Text between the formula and the first article: preamble, after the formula.
            act.postamble.append(block)
            continue
        current.append(block)
    if not act.articles:
        raise EurlexHtmlError("no `Article N` blocks: not an act, or not the OJ rendering")
    return act


_TITLE_WORDS = 10


def _is_title(text: str) -> bool:
    """Short, capitalised, unpunctuated: a heading such as "Scope", not a sentence."""
    return len(text.split()) <= _TITLE_WORDS and not text.rstrip().endswith((",", ";"))


def _date_in(text: str) -> str | None:
    for day, month, year in _LONG_DATE.findall(text):
        num = _MONTHS.get(month.lower())
        if num:
            return f"{year}-{num:02d}-{int(day):02d}"
    return None


def expression_date(act: _Act) -> str | None:
    """The act's own date from its title, else from the signature, else the OJ date."""
    return (
        _date_in(act.title)
        or next((d for d in (_date_in(b) for b in act.conclusions) if d), None)
        or act.oj.get("date")
    )


def _akn(parent: etree._Element, tag: str, **attrs: str) -> etree._Element:
    el = etree.SubElement(parent, f"{{{AKN_NS}}}{tag}")
    for k, v in attrs.items():
        if v is not None:
            el.set(k, v)
    return el


# A chapeau and its first item run together in one block: `... shall apply: 1. to ...`.
_INLINE_FIRST_ITEM = re.compile(r"^(.*?:)\s+(1\.\s+.*)$", re.DOTALL)
_POINT = re.compile(r"^\(([a-z]{1,2}|[ivxl]+)\)\s+(.*)$", re.DOTALL)


def _split_inline_first_item(blocks: list[str]) -> list[str]:
    out: list[str] = []
    for block in blocks:
        m = None if _NUMBERED_PARA.match(block) else _INLINE_FIRST_ITEM.match(block)
        out.extend([m.group(1), m.group(2)] if m else [block])
    return out


def _fill_paragraph(para: etree._Element, eid: str, blocks: list[str]) -> None:
    """Prose into `content`; lettered points make it the `intro` and follow as `point`s."""
    holder: etree._Element | None = None
    point_content: etree._Element | None = None
    seen: dict[str, int] = {}
    for block in blocks:
        m = _POINT.match(block)
        if m:
            if holder is not None:
                holder.tag = f"{{{AKN_NS}}}intro"
            letter = m.group(1)
            seen[letter] = seen.get(letter, 0) + 1
            leaf = f"point_{letter}" if seen[letter] == 1 else f"point_{letter}_dup{seen[letter]}"
            point = _akn(para, "point", eId=f"{eid}__{leaf}")
            _akn(point, "num").text = f"({letter})"
            point_content = _akn(point, "content")
            _akn(point_content, "p").text = m.group(2)
        elif point_content is not None:
            _akn(point_content, "p").text = block  # a continuation stays with its point
        else:
            if holder is None:
                holder = _akn(para, "content")
            _akn(holder, "p").text = block


def _emit_paragraphs(article: etree._Element, eid: str, blocks: list[str]) -> int:
    """Numbered blocks (`1. text`) become numbered paragraphs, any prose before the
    first of them the article's intro; an article with no numbering shares one."""
    blocks = _split_inline_first_item(blocks)
    matches = [_NUMBERED_PARA.match(b) for b in blocks]
    first = next((i for i, m in enumerate(matches) if m), None)
    if first is None:
        _fill_paragraph(_akn(article, "paragraph", eId=f"{eid}__para_1"), f"{eid}__para_1", blocks)
        return 0
    if first:
        intro = _akn(article, "intro")
        for block in blocks[:first]:
            _akn(intro, "p").text = block
    seen: dict[str, int] = {}
    groups: list[tuple[str, list[str]]] = []
    for block, m in zip(blocks[first:], matches[first:], strict=True):
        if m is not None:
            groups.append((m.group(1), [m.group(2)]))
        else:
            groups[-1][1].append(block)
    for num, body in groups:
        seen[num] = seen.get(num, 0) + 1
        # A repeated number keeps a unique eId, as a repeated article does.
        leaf = f"para_{num}" if seen[num] == 1 else f"para_{num}_dup{seen[num]}"
        para = _akn(article, "paragraph", eId=f"{eid}__{leaf}")
        _akn(para, "num").text = f"{num}."
        _fill_paragraph(para, f"{eid}__{leaf}", body)
    return sum(n - 1 for n in seen.values())


_PREAMBLE_KINDS = (
    ("having regard", "citations", "citation", "cit"),
    ("whereas", "recitals", "recital", "rec"),
)


def _preamble_kind(block: str) -> tuple[str, str, str] | None:
    lower = block.lower()
    return next((k[1:] for k in _PREAMBLE_KINDS if lower.startswith(k[0])), None)


def _emit_preamble(preamble: etree._Element, blocks: list[str]) -> None:
    """Blocks in source order; a run of citations or recitals shares one container."""
    container: etree._Element | None = None
    counts: dict[str, int] = {}
    for block in blocks:
        kind = _preamble_kind(block)
        if kind is None:
            _akn(preamble, "p").text = block
            container = None
            continue
        wrapper, leaf, abbr = kind
        if container is None or container.tag != f"{{{AKN_NS}}}{wrapper}":
            container = _akn(preamble, wrapper)
        counts[abbr] = counts.get(abbr, 0) + 1
        _akn(_akn(container, leaf, eId=f"{wrapper}__{abbr}_{counts[abbr]}"), "p").text = block


def build_body(
    act: _Act,
    root_act: etree._Element,
    *,
    frbr_work_uri: str = "/akn/eu/act/dir/0000/0",
    language: str = "eng",
    expression_date: str = "0000-01-01",
) -> None:
    """Preface, preamble, body, conclusions and annexes onto the `<act>`."""
    from codify.pipeline.formats.eu_directive import _attachment_meta

    preface = _akn(root_act, "preface")
    _akn(_akn(preface, "longTitle"), "p").text = act.title
    if act.preamble or act.formula or act.postamble:
        preamble = _akn(root_act, "preamble")
        _emit_preamble(preamble, act.preamble)
        if act.formula:
            _akn(_akn(preamble, "formula", name="enactingFormula"), "p").text = act.formula
        for block in act.postamble:
            _akn(preamble, "p").text = block
    body = _akn(root_act, "body")
    seen: dict[str, int] = {}
    duplicate_paragraphs = 0
    for shown, heading, blocks in act.articles:
        number = shown.lower()  # the eId lower-cases an inserted article's letter; num keeps it
        seen[number] = seen.get(number, 0) + 1
        # A repeated number keeps a unique eId; the count reaches the run as a finding.
        eid = f"art_{number}" if seen[number] == 1 else f"art_{number}_dup{seen[number]}"
        article = _akn(body, "article", eId=eid)
        _akn(article, "num").text = f"Article {shown}"
        if heading:
            _akn(article, "heading").text = heading
        duplicate_paragraphs += _emit_paragraphs(article, eid, blocks)
    act.duplicate_paragraphs = duplicate_paragraphs
    if act.conclusions:
        conclusions = _akn(root_act, "conclusions")
        for block in act.conclusions:
            _akn(conclusions, "p").text = block
    if act.annexes:
        attachments = _akn(root_act, "attachments")
        for index, (heading, blocks) in enumerate(act.annexes, start=1):
            attachment = _akn(attachments, "attachment", eId=f"att_{index}")
            _akn(attachment, "heading").text = heading
            doc = _akn(attachment, "doc", name="ANNEX")
            doc.append(
                _attachment_meta(
                    frbr_work_uri=frbr_work_uri,
                    language=language,
                    expression_date=expression_date,
                    eid=f"att_{index}",
                )
            )
            main = _akn(doc, "mainBody")
            for block in blocks:
                _akn(main, "p").text = block


def html_to_akn4eu(
    text: str,
    *,
    frbr_work_uri: str,
    language: str = "eng",
    provenance: dict[str, object] | None = None,
) -> str:
    """Deterministic EUR-Lex HTML → AKN4EU, the FORMEX converter's meta and shape."""
    # Local: eu_directive imports this module for its sniff.
    from codify.pipeline.formats.eu_directive import (
        FormexDateMissingError,
        _doctype_from_frbr,
        _stub_meta,
    )

    blocks, oj, walk = _blocks(text)
    act = _segment(blocks, oj)
    date = expression_date(act)
    if date is None:
        raise FormexDateMissingError("no date in the title, the signature or the OJ reference")
    root = etree.Element(f"{{{AKN_NS}}}akomaNtoso", nsmap={None: AKN_NS})
    root_act = etree.SubElement(
        root, f"{{{AKN_NS}}}act", contains="originalVersion", name=_doctype_from_frbr(frbr_work_uri)
    )
    meta = _stub_meta(frbr_work_uri=frbr_work_uri, language=language, expression_date=date)
    if oj:
        publication = _akn(
            meta,
            "publication",
            date=oj["date"],
            name="OJ",
            showAs=f"Official Journal {oj['number']}",
            number=oj["number"],
        )
        meta.insert(1, publication)
        # The lifecycle's publication event records the same date as the publication.
        for event in meta.iterfind(f".//{{{AKN_NS}}}eventRef[@eId='evt_publication']"):
            event.set("date", oj["date"])
    elif provenance is not None:
        provenance["publication_missing"] = True
    root_act.append(meta)
    build_body(act, root_act, frbr_work_uri=frbr_work_uri, language=language, expression_date=date)
    if provenance is not None:
        if walk["discarded_chars"]:
            provenance["body_discarded_text"] = walk["discarded_chars"]
        if walk["images"]:
            provenance["html_images"] = walk["images"]
        numbers = [n.lower() for n, _, _ in act.articles]
        provenance["html_source"] = {
            "articles": len(act.articles),
            "duplicate_articles": len(numbers) - len(set(numbers)),
            "duplicate_paragraphs": act.duplicate_paragraphs,
            "tables": walk["tables"],
            "images": walk["images"],
            "recitals": sum(1 for b in act.preamble if b.lower().startswith("whereas")),
            "annexes": len(act.annexes),
        }
    serialised: bytes = etree.tostring(root, xml_declaration=True, encoding="UTF-8")
    return serialised.decode("utf-8")


__all__ = ["EurlexHtmlError", "expression_date", "html_to_akn4eu", "is_eurlex_html"]
