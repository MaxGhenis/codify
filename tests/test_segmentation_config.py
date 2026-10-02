"""The `segmentation` block refuses at load what would misfire on every source."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from codify.jurisdictions import SegmentationConfig


def test_a_declared_block_loads() -> None:
    config = SegmentationConfig.model_validate(
        {
            "act_heading_patterns": [r"ACT No\. (?P<number>\d+)"],
            "issue_heading_patterns": [r"ISSUE (?P<number>\d+)"],
            "contents_keywords": ["Contents"],
            "printed_page_pattern": r"^-(\d+)-$",
        }
    )
    assert config.act_heading_patterns == [r"ACT No\. (?P<number>\d+)"]
    assert config.printed_page_pattern == r"^-(\d+)-$"


def test_an_unknown_key_is_refused() -> None:
    """Strict: a misspelt field would otherwise leave the jurisdiction unsplit."""
    with pytest.raises(ValidationError, match="act_heading_pattern"):
        SegmentationConfig.model_validate({"act_heading_pattern": ["ACT"]})


@pytest.mark.parametrize("field", ["act_heading_patterns", "issue_heading_patterns"])
@pytest.mark.parametrize(
    "pattern", ["", "   ", r"(?:ACT)?", r"\d*", r"(?:ACT)?\b", r"(?=ACT)", r"ACT|"]
)
def test_a_heading_that_matches_empty_text_is_refused(field: str, pattern: str) -> None:
    with pytest.raises(ValidationError, match="must not be empty or match empty text"):
        SegmentationConfig.model_validate({field: ["ACT", pattern]})


@pytest.mark.parametrize("field", ["act_heading_patterns", "issue_heading_patterns"])
def test_a_heading_that_does_not_compile_is_refused(field: str) -> None:
    with pytest.raises(ValidationError, match="does not compile"):
        SegmentationConfig.model_validate({field: ["ACT (No"]})


@pytest.mark.parametrize("field", ["act_heading_patterns", "issue_heading_patterns"])
def test_a_heading_that_fails_only_once_wrapped_is_refused(field: str) -> None:
    """Global flags compile alone but not inside the line-start wrapper headings run in."""
    with pytest.raises(ValidationError, match="does not compile at a line start"):
        SegmentationConfig.model_validate({field: ["(?i)ACT"]})
    # Scoped flags survive the wrapper.
    assert SegmentationConfig.model_validate({field: ["(?i:ACT)"]})


def test_a_blank_contents_keyword_is_refused() -> None:
    with pytest.raises(ValidationError, match="contents keyword must not be blank"):
        SegmentationConfig.model_validate({"contents_keywords": ["Contents", " "]})


def test_a_page_pattern_without_a_group_is_refused() -> None:
    """The group is the number; without one there is nothing to read."""
    with pytest.raises(ValidationError, match="needs a group"):
        SegmentationConfig.model_validate({"printed_page_pattern": r"Page \d+"})


def test_a_page_pattern_that_matches_empty_text_is_refused() -> None:
    with pytest.raises(ValidationError, match="must not be empty or match empty text"):
        SegmentationConfig.model_validate({"printed_page_pattern": r"(\d*)"})
