"""Extract bibliographic metadata from legislation text via LLM."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from codify.calendar import (
    normalise_calendar,
    title_year_token,
    year_from_calendar,
)
from codify.core.llm import LLMClient
from codify.pipeline.dating import resolve_dating

logger = structlog.get_logger()

PROMPTS_DIR = Path(__file__).parent / "prompts"
METADATA_SYSTEM_PROMPT = (PROMPTS_DIR / "metadata_system.txt").read_text()

# Keep the opening identity and closing issuance/signature within one budget.
_METADATA_TEXT_CHARS = 8000
_METADATA_END_CHARS = 2000
_METADATA_OMISSION = "\n\n[Middle of document omitted]\n\n"


def _metadata_text(raw_text: str) -> str:
    """Include closing bibliographic evidence without duplicating short documents."""
    if len(raw_text) <= _METADATA_TEXT_CHARS:
        return raw_text
    head_chars = _METADATA_TEXT_CHARS - _METADATA_END_CHARS - len(_METADATA_OMISSION)
    return raw_text[:head_chars] + _METADATA_OMISSION + raw_text[-_METADATA_END_CHARS:]


class ExtractedMetadata(BaseModel):
    """Schema-enforced shape for the metadata LLM call."""

    title: str = ""
    date: str = ""
    year: str = ""
    number: str = ""
    chapter: str = ""
    is_amendment: bool = False
    language: str = ""
    additional_languages: list[str] = Field(default_factory=list)
    # Calendar the date/year are expressed in. "" if too short to judge.
    calendar: str = ""


# Config vocabulary to the values this prompt asks the model to emit. The two
# were written separately, so a hint built from a config verbatim would name a
# calendar the schema does not offer. `dual` is deliberately absent: two
# calendars is the thing the model must judge, not something to hint at.
_CALENDAR_FOR_PROMPT = {
    "lunar_hijri": "hijri",
    "hijri": "hijri",
    "solar_hijri": "hijri_solar",
    "buddhist_era": "buddhist",
    "gregorian": "gregorian",
    "ethiopian": "ethiopian",
    "japanese_era": "japanese_era",
    "minguo": "minguo",
}


def _as_prompt_calendar(value: str) -> str:
    """The prompt's name for a config's calendar, empty when there is none it can
    state without guessing."""
    return _CALENDAR_FOR_PROMPT.get(value, "")


def calendar_hint(config: Any) -> str:
    """What this jurisdiction's config says its instruments are dated in. The
    shared prompt carries only signals that hold anywhere."""
    if config is None:
        return ""
    parts: list[str] = []
    named = _as_prompt_calendar(getattr(config, "calendar", "") or "")
    if named:
        parts.append(f"This jurisdiction dates its instruments in the {named} calendar.")
    eras = [e for e in (getattr(config, "legal_eras", None) or []) if getattr(e, "label", "")]
    for era in eras:
        span = getattr(era, "to", "") or ""
        cal = _as_prompt_calendar(getattr(era, "calendar", None) or "")
        if cal:
            parts.append(f"{era.label}{f' (to {span})' if span else ''}: dated in {cal}.")
    return " ".join(parts)


async def extract_metadata(
    raw_text: str,
    client: LLMClient,
    filename: str = "",
    source_url: str = "",
    calendar_hint_text: str = "",
) -> dict[str, Any]:
    """Extract title, date, year, number, chapter, language from legislation
    text. Filename and source URL are contextual hints; the document text is
    authoritative."""
    context_parts = []
    if filename:
        context_parts.append(f"Filename: {filename}")
    if source_url:
        context_parts.append(f"Source URL: {source_url}")

    context = "\n".join(context_parts)
    prompt_parts = []
    if context:
        prompt_parts.append(context)
    prompt_parts.append(f"Legislation text:\n\n{_metadata_text(raw_text)}")

    try:
        result = await client.chat_schema(
            prompt="\n\n".join(prompt_parts),
            schema=ExtractedMetadata,
            system=(
                f"{METADATA_SYSTEM_PROMPT}\n\n{calendar_hint_text}"
                if calendar_hint_text
                else METADATA_SYSTEM_PROMPT
            ),
        )
    except Exception:  # noqa: BLE001
        logger.warning("metadata_extraction_failed", exc_info=True)
        return {}

    logger.info(
        "metadata_extracted",
        title=result.title,
        date=result.date,
        number=result.number,
        language=result.language,
        additional_languages=result.additional_languages,
        calendar=result.calendar,
    )
    return result.model_dump()


# `Nomor 45/PUU-IX/2011`, `No. 6 of 2025`, `Nr 12`.
_TITLE_NUMBER = re.compile(
    r"\b(?:nomor|number|nomer|nr|no)\b\.?\s*:?\s*([^\s,;()]+)",
    re.IGNORECASE,
)

# Words after which a title is naming another instrument, not itself. A
# `tentang` designation puts its own number before `tentang`; everything the
# act is about, including the act it amends, comes after.
_REFERS_ONWARD = re.compile(
    r"\b(?:tentang|atas|pelaksanaan|to\s+amend|amending|implementing|modifiant|"
    r"perubahan)\b",
    re.IGNORECASE,
)


def number_from_title(title: str) -> str:
    """The number a title states about itself, or "".

    A deterministic second source for the one metadata field that had none. A
    number quoted after a reference cue belongs to the instrument being named,
    and adopting it would file this document under that one."""
    text = title or ""
    match = _TITLE_NUMBER.search(text)
    if match is None:
        return ""
    refers = _REFERS_ONWARD.search(text)
    if refers is not None and match.start() > refers.start():
        return ""
    # OCR leaves a full stop on the token often enough that keeping it would
    # send an otherwise recoverable title to the content address.
    return match.group(1).strip().strip(".,;:·")


def _legacy_year(metadata: dict[str, Any], country: str, title: str) -> int | None:
    """What the shipped run stored: the generic conversion and the old coercion,
    the title year unconverted. A replay reproduces it rather than improves it."""
    raw_year = str(metadata.get("year") or "").strip()
    raw_date = str(metadata.get("date") or "").strip()
    cal = normalise_calendar(metadata.get("calendar"))
    candidate = raw_year or (raw_date.split("-")[0] if "-" in raw_date else raw_date)
    from_title = str(metadata.get("title") or "").strip() or title.strip()
    if not candidate and from_title:
        title_year = title_year_token(from_title, metadata.get("number"))
        if title_year:
            return int(title_year)
    if not candidate:
        return None
    if cal and cal != "gregorian":
        converted = year_from_calendar(candidate, cal, country)
        if converted is not None:
            return converted
    try:
        return int(candidate)
    except ValueError:
        return None


def gregorian_year(
    metadata: dict[str, Any],
    country: str = "",
    *,
    legacy: bool = False,
    title: str = "",
) -> int | None:
    """One reading of the shared resolution, so the stored and filed years cannot
    differ. A jurisdiction's own rules need its code; a plain year does not."""
    if legacy:
        return _legacy_year(metadata, country, title)
    return resolve_dating(metadata, country=country, title=title).stored_year
