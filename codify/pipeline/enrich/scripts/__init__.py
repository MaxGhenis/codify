"""Per-script data and passes, selected by the jurisdiction config.

The structurer used to decide "is this Arabic?" by counting codepoints at runtime,
with Arabic tables inline in the general modules, so every ingest paid for Arabic
handling and a second script meant editing `anchors.py` again. Every config already
declares `display.script`, which is now the authoritative selector; codepoint
detection is the fallback where a document disagrees with its jurisdiction.

The convention: script data and script-only passes live in `scripts/<script>.py`,
general algorithms stay in `enrich/` and take a pack.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from codify.jurisdictions import JurisdictionConfig

logger = structlog.get_logger()

ARABIC_SCRIPT = "arabic"


@dataclass(frozen=True)
class ScriptPack:
    """Everything the structurer needs that is specific to one writing system."""

    script: str
    # Letters of the script, for the codepoint fallback and for bridging
    # OCR-inserted joiners inside a word.
    letter_re: re.Pattern[str]
    # Optional-joiner class inserted between letters of a term so a literal
    # handed straight to a compiled regex still matches unnormalised text.
    joiner_opt: str = ""
    # Ordinal words to their integer value ("الأولى" -> 1).
    ordinal_to_int: dict[str, int] = field(default_factory=dict)
    # Alphabetic enumerator sequence in legal-list order, with the standalone
    # forms that render the same letter.
    letter_order: tuple[str, ...] = ()
    letter_alias: dict[str, str] = field(default_factory=dict)
    # A digit token, including the script's own digit shapes.
    digit_re: re.Pattern[str] = re.compile(r"^\(?(\d+)\)?$")
    # Words that mark the text before a marker as prose rather than a header.
    prose_precursors: tuple[str, ...] = ()
    # Words that mark a citation only when adjacent on the same line.
    sameline_precursors: tuple[str, ...] = ()
    # Ordinal list markers ("أولاً" -> "1") for a list numbered in words.
    ordinal_list_markers: tuple[tuple[str, str], ...] = ()
    # Alphabetic-numbering rank, and the fold that collapses carrier forms
    # onto the bare letter before ranking.
    abjad_rank: dict[str, int] = field(default_factory=dict)
    abjad_fold: dict[int, str] = field(default_factory=dict)
    # Character sequences OCR produces that the script never writes.
    garble_patterns: tuple[re.Pattern[str], ...] = ()
    # The plural form of the article noun, which governs a citation list.
    # Never matches when a script has no plural marker of its own.
    plural_marker_re: re.Pattern[str] = re.compile(r"(?!)")

    def is_letter(self, ch: str) -> bool:
        return self.letter_re.match(ch) is not None


_REGISTRY: dict[str, ScriptPack] = {}


def register(pack: ScriptPack) -> ScriptPack:
    _REGISTRY[pack.script] = pack
    return pack


def _load_packs() -> dict[str, ScriptPack]:
    """The registry, with every shipped pack imported. Packs register on import, so a lookup
    must not depend on whether another module imported them first. The import sits here
    rather than at module scope because a pack imports ``ScriptPack`` from here.
    """
    if not _REGISTRY:
        from codify.pipeline.enrich.scripts import arabic  # noqa: F401
    return _REGISTRY


def pack_for(script: str | None) -> ScriptPack | None:
    """The pack a script name selects, or ``None`` when none is registered.

    ``None`` means "no script-specific handling", which is the correct state
    for Latin-script jurisdictions: the general algorithms carry them.
    """
    return _load_packs().get((script or "").strip().lower())


def pack_for_config(cfg: JurisdictionConfig | None) -> ScriptPack | None:
    """The pack a jurisdiction declares in ``display.script``."""
    if cfg is None or cfg.display is None:
        return None
    return pack_for(cfg.display.script)


def detect_pack(text: str, *, minimum_share: float = 0.2) -> ScriptPack | None:
    """The pack whose letters dominate ``text``, or ``None``. The fallback where a document
    disagrees with the script its jurisdiction declares: a wholly English page in an
    Arabic corpus, or an Arabic annex in a Latin-script corpus. Config comes first, a corpus author
    knowing the jurisdiction where a codepoint count knows only the page.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    for pack in _load_packs().values():
        hits = sum(1 for c in letters if pack.letter_re.match(c))
        if hits / len(letters) >= minimum_share:
            return pack
    return None


def resolve_pack(cfg: JurisdictionConfig | None, text: str = "") -> ScriptPack | None:
    """The pack for a document: what the config declares, else what the text
    shows. Logs when the two disagree, because that is either a mislabelled
    config or a document in the wrong corpus."""
    declared = pack_for_config(cfg)
    if declared is not None:
        return declared
    detected = detect_pack(text) if text else None
    if detected is not None and cfg is not None and cfg.display is not None:
        logger.info(
            "script_pack_from_content",
            declared=cfg.display.script,
            detected=detected.script,
        )
    return detected


def dominant_script(text: str) -> str | None:
    """The dominant Unicode script name of ``text``. General rather than per-pack because it
    answers whether two strings share a writing system, which is how parenthesised aliases
    are filtered ("Член (Чл.)" keeps both, "მუხლი (Article)" keeps the head), and that is
    asked of scripts no pack covers.
    """
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        script = name.split(" ")[0]
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda k: counts[k])


__all__ = [
    "ARABIC_SCRIPT",
    "ScriptPack",
    "detect_pack",
    "dominant_script",
    "pack_for",
    "pack_for_config",
    "register",
    "resolve_pack",
]
