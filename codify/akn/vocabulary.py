"""Classify AKN elements as provisions, prose or borrowed text."""

from __future__ import annotations

import re

from lxml import etree

# AKN element name to provision kind. Unlisted eIds are not provisions.
TAG_TO_KIND: dict[str, str] = {
    "title": "title",
    "subtitle": "title",
    "book": "title",
    "tome": "title",
    "part": "title",
    "chapter": "chapter",
    "subchapter": "chapter",
    "division": "chapter",
    "subdivision": "chapter",
    "subpart": "chapter",
    "section": "section",
    "subsection": "section",
    "rule": "section",
    "subrule": "section",
    "article": "article",
    "paragraph": "paragraph",
    "alinea": "paragraph",
    "subparagraph": "subparagraph",
    "clause": "subparagraph",
    "subclause": "subparagraph",
    "proviso": "subparagraph",
    "point": "point",
    "indent": "point",
    "item": "point",
    "hcontainer": "subparagraph",
    "blockContainer": "subparagraph",
    "blockList": "subparagraph",
    "level": "paragraph",
}

PROVISION_ELEMENTS = frozenset(TAG_TO_KIND)

# Model-written text. Compared for presence, never for content.
PROSE_ELEMENTS = frozenset({"p", "content", "listIntroduction"})

# Quoted amendments are verbatim text from the amended act.
QUOTED_AMENDMENT_ANCESTORS = frozenset({"mod", "quotedStructure", "embeddedStructure"})

# ``<meta>`` describes the referenced world, not the act body.
BORROWED_ANCESTORS = QUOTED_AMENDMENT_ANCESTORS | {"meta"}

# Wrappers without inherent numbering; Bluebell derives grouping per parse.
GENERIC_CONTAINER_TAGS = frozenset({"hcontainer", "blockContainer", "blockList"})

# Provisions carrying their own identity; count differences indicate loss.
IDENTIFIED_PROVISIONS = PROVISION_ELEMENTS - GENERIC_CONTAINER_TAGS


# Named containers that hold scaffolding, not a unit of law: a table or figure
# wrapper, and the group a titled annex wrapper becomes. A restructure renders
# the contents without the wrapper, so counting one as a basic unit makes the
# preservation gate reject a candidate that dropped nothing.
PRESENTATION_CONTAINER_NAMES = frozenset({"TAB", "FGR", "SFR", "IMG", "group"})


def is_generic_container(tag: str, name: str | None) -> bool:
    """Whether an element is scaffolding rather than a provision. Not settled by the tag:
    `<hcontainer>` also carries named jurisdiction units (a declared bis article), which
    are law. Bluebell's own wrappers are unnamed or named for the tag, so `name` is the
    discriminator, and a presentation wrapper is scaffolding whatever it is named.
    """
    if tag in GENERIC_CONTAINER_TAGS and name in PRESENTATION_CONTAINER_NAMES:
        return True
    return tag in GENERIC_CONTAINER_TAGS and (name is None or name == tag)


def local_name(tag: object) -> str:
    """The element name without its namespace."""
    return str(tag).rsplit("}", 1)[-1]


def is_lifted_note(el: etree._Element) -> bool:
    """A footnote lifted out of prose, as opposed to one belonging in it.

    `enrich/notes.py` moves amendment footnotes onto the provision they annotate as
    `<authorialNote placement="bottom">`. Those are annotation, not read as part of the
    sentence, and every projection must agree or the same footnote is prose in one
    surface and a note in the next. Placement is the discriminator, not the tag: the
    FORMEX converter emits non-FOOTNOTE `authorialNote` with no placement mid-prose, and
    excluding those would take a parenthetical out of its own sentence.
    """
    return local_name(el.tag) == "authorialNote" and el.get("placement") == "bottom"


# Bluebell and the FORMEX converter derive these from a `<blockList>`, so the count
# tracks prose reflow, not legal structure: recovering list punctuation invents them,
# resolving a split list back into prose removes them, and neither is a loss. `point`
# is absent deliberately: a lettered point is a numbered unit whose loss is real.
LIST_RENDERING_TAGS = frozenset({"item", "indent"})


def count_provisions_by_kind(root: etree._Element) -> dict[str, int]:
    """Provisions carrying their own identity, counted per kind rather than as a total,
    because a total lets one kind mask another: a re-read losing 48 articles and gaining
    50 list items nets positive. Quoted amendments and `<meta>` are skipped, their
    structure belonging to another act; `is_generic_container` settles the wrappers, so a
    named `<hcontainer>` is counted where a static exclusion would miss it.
    """
    counts: dict[str, int] = {}
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        name = local_name(el.tag)
        if name not in PROVISION_ELEMENTS or name in LIST_RENDERING_TAGS:
            continue
        if is_generic_container(name, el.get("name")):
            continue
        if any(local_name(a.tag) in BORROWED_ANCESTORS for a in el.iterancestors()):
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


# Arabic-Indic (U+0660-0669) and extended/Persian (U+06F0-06F9) digits to ASCII,
# so a number set compares across the two blocks a re-OCR may disagree on (an
# old scan can render the same article as ٩٤ one pass and ۹٤ the next).
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# The level that carries the law. A dropped numbered article is dropped law; a
# regrouped chapter or a re-rendered list is not.
_BASIC_UNIT_KINDS = frozenset({"article", "section", "rule"})


def basic_unit_numbers(root: etree._Element) -> frozenset[str]:
    """The basic units (article/section/rule) as ``"kind:number"`` keys.

    Compared as a set, this survives what a per-kind count cannot: re-OCR re-renders
    lists and regroups chapters, so counts churn and a raw total masks real loss (act
    2014/23 shed 48 articles while its other counts rose). Digits are normalised across
    the Arabic-Indic and Persian blocks, so an article and a section both numbered 5 stay
    two units, and a spelled or Roman number keeps its letters rather than vanishing. A
    named `<hcontainer>` is a basic unit; only the bare Bluebell wrapper is not.

    A lone <section> numbered 1 with no article or rule is the structureless fallback and
    contributes nothing, so a candidate replacing it with real articles is not charged
    for losing it.
    """
    units: set[str] = set()
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        name = local_name(el.tag)
        # article/section/rule, plus a named hcontainer that carries law (a
        # mukrrar bis article); the bare generic wrapper is skipped.
        if name not in _BASIC_UNIT_KINDS and is_generic_container(name, el.get("name")):
            continue
        if name not in _BASIC_UNIT_KINDS and name not in GENERIC_CONTAINER_TAGS:
            continue
        if any(local_name(a.tag) in BORROWED_ANCESTORS for a in el.iterancestors()):
            continue
        num_el = next((c for c in el if local_name(c.tag) == "num"), None)
        raw = "".join(num_el.itertext()) if num_el is not None else ""
        digits = re.sub(r"[^0-9]", "", raw.translate(_ARABIC_DIGITS))
        suffix = "".join(ch for ch in raw if ch.isalpha())  # bis / Roman / spelled marker
        # A named container keys on its name (`mukrrar`), so two differently
        # named units of the same number cannot alias onto one another.
        kind = el.get("name") or name
        if digits or suffix:
            units.add(f"{kind}:{digits}{suffix}")
    if units == {"section:1"}:  # the _verbatim_single_section structureless fallback
        units = set()
    return frozenset(units)


def provision_text(el: etree._Element) -> str:
    """The element's text as a reader of the provision would read it. Lifted notes are left
    out at any depth; inline notes stay, being part of the sentence. Callers needing the
    raw serialisation want `itertext()`.
    """
    parts: list[str] = [el.text or ""]
    for child in el:
        if not is_lifted_note(child):
            parts.append(provision_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def table_row_eid(table_eid: str, position: int) -> str:
    """The `<tr>` eId for a 1-based row position.

    Bluebell puts `tr` in `id_exempt` and emits no eId, so three writers mint
    one: the Formex table emitter, the row resolver search cites, and the
    reader's document walker. This is the format they share. Each still counts
    `position` itself, and a writer that numbered differently would produce ids
    that pass every check here, so the counting is pinned by test, not by this.

    Both halves are ordinal, so an id is only meaningful against the expression
    that minted it: the same id can name a different row in another expression.
    Ids are also unpadded, so `__tr_10` sorts before `__tr_2`.
    """
    if not table_eid:
        raise ValueError("table_row_eid needs the table's eId; a row id without it cites nothing")
    if position < 1:
        raise ValueError(f"table row positions are 1-based, got {position}")
    return f"{table_eid}__tr_{position}"


__all__ = [
    "BORROWED_ANCESTORS",
    "GENERIC_CONTAINER_TAGS",
    "LIST_RENDERING_TAGS",
    "basic_unit_numbers",
    "count_provisions_by_kind",
    "is_generic_container",
    "is_lifted_note",
    "IDENTIFIED_PROVISIONS",
    "PROSE_ELEMENTS",
    "PROVISION_ELEMENTS",
    "QUOTED_AMENDMENT_ANCESTORS",
    "TAG_TO_KIND",
    "local_name",
    "provision_text",
    "table_row_eid",
]
