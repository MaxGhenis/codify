"""The content-filter fallback: when the primary model refuses a call with a
content-filter finish reason, chat_schema retries once on the fallback model.
Gemini's non-configurable filter false-fires on benign legal prose, so the
retry recovers the call on a model without that floor."""

from __future__ import annotations

from unittest.mock import MagicMock

import openai
import pytest
from pydantic import BaseModel

from codify.core.llm import LiteLLMClient


class _Parsed(BaseModel):
    value: str


def _client(fallback_model: str | None) -> LiteLLMClient:
    c = LiteLLMClient(
        base_url="http://x/v1", api_key="k", model="gemini-3.6-flash", fallback_model=fallback_model
    )
    c.client = MagicMock()
    c.client.chat = MagicMock()
    c.client.chat.completions = MagicMock()
    return c


async def test_chat_schema_retries_on_fallback_model() -> None:
    c = _client("gpt-5.6-sol")
    seen: list[str] = []
    good = MagicMock(
        choices=[MagicMock(message=MagicMock(parsed=_Parsed(value="ok"), content=None))]
    )

    async def parse(*, model: str, **_: object) -> MagicMock:
        seen.append(model)
        if model != "gpt-5.6-sol":
            raise openai.ContentFilterFinishReasonError()
        return good

    c.client.chat.completions.parse = parse
    out = await c.chat_schema("translate this", _Parsed)
    assert out.value == "ok"
    # Primary refused, fallback ran, in that order.
    assert seen == ["gemini-3.6-flash", "gpt-5.6-sol"]


async def test_chat_schema_reraises_when_no_fallback_configured() -> None:
    c = _client(None)

    async def parse(**_: object) -> MagicMock:
        raise openai.ContentFilterFinishReasonError()

    c.client.chat.completions.parse = parse
    with pytest.raises(openai.ContentFilterFinishReasonError):
        await c.chat_schema("x", _Parsed)


async def test_chat_schema_reraises_when_fallback_also_filtered() -> None:
    c = _client("gpt-5.6-sol")
    seen: list[str] = []

    async def parse(*, model: str, **_: object) -> MagicMock:
        seen.append(model)
        raise openai.ContentFilterFinishReasonError()

    c.client.chat.completions.parse = parse
    with pytest.raises(openai.ContentFilterFinishReasonError):
        await c.chat_schema("x", _Parsed)
    # Tried primary then fallback exactly once each; no infinite recursion.
    assert seen == ["gemini-3.6-flash", "gpt-5.6-sol"]


async def test_chat_schema_retries_on_a_suffixed_finish_reason() -> None:
    # Gemini's endpoint reports 'content_filter: RECITATION'. The SDK raises only
    # on the bare reason, so .parse() returns an empty completion instead.
    c = _client("gpt-5.6-sol")
    seen: list[str] = []

    async def parse(*, model: str, **_: object) -> MagicMock:
        seen.append(model)
        if model != "gpt-5.6-sol":
            return MagicMock(
                choices=[
                    MagicMock(
                        finish_reason="content_filter: RECITATION",
                        message=MagicMock(parsed=None, content=None),
                    )
                ]
            )
        return MagicMock(
            choices=[
                MagicMock(
                    finish_reason="stop",
                    message=MagicMock(parsed=_Parsed(value="ok"), content=None),
                )
            ]
        )

    c.client.chat.completions.parse = parse
    out = await c.chat_schema("translate this", _Parsed)
    assert out.value == "ok"
    assert seen == ["gemini-3.6-flash", "gpt-5.6-sol"]


async def test_chat_schema_raises_on_a_suffixed_reason_with_no_fallback() -> None:
    # Without this the empty completion surfaced as 'invalid structured output'.
    c = _client(None)

    async def parse(**_: object) -> MagicMock:
        return MagicMock(
            choices=[
                MagicMock(
                    finish_reason="content_filter: RECITATION",
                    message=MagicMock(parsed=None, content=None),
                )
            ]
        )

    c.client.chat.completions.parse = parse
    with pytest.raises(openai.ContentFilterFinishReasonError):
        await c.chat_schema("x", _Parsed)


_PNG = b"\x89PNG\r\n\x1a\nfake"


def _vision_resp(finish_reason: str, content: str | None) -> MagicMock:
    # vision uses .create(): a block returns finish_reason='content_filter' with
    # empty content, it does not raise the way .parse() does.
    return MagicMock(
        choices=[MagicMock(finish_reason=finish_reason, message=MagicMock(content=content))]
    )


async def test_vision_retries_on_fallback_model() -> None:
    c = _client("gpt-5.6-sol")
    seen: list[str] = []

    async def create(*, model: str, **_: object) -> MagicMock:
        seen.append(model)
        if model != "gpt-5.6-sol":
            return _vision_resp("content_filter", None)
        return _vision_resp("stop", "recovered text")

    c.client.chat.completions.create = create
    out = await c.vision("read this page", images=[_PNG])
    assert out == "recovered text"
    # Primary content-filtered, fallback ran, in that order.
    assert seen == ["gemini-3.6-flash", "gpt-5.6-sol"]


async def test_vision_retries_on_a_suffixed_finish_reason() -> None:
    # What Gemini's OpenAI-compatible endpoint returns for a scanned page of
    # published law; an equality test against 'content_filter' never matched it.
    c = _client("gpt-5.6-sol")
    seen: list[str] = []

    async def create(*, model: str, **_: object) -> MagicMock:
        seen.append(model)
        if model != "gpt-5.6-sol":
            return _vision_resp("content_filter: RECITATION", None)
        return _vision_resp("stop", "recovered text")

    c.client.chat.completions.create = create
    out = await c.vision("read this page", images=[_PNG])
    assert out == "recovered text"
    assert seen == ["gemini-3.6-flash", "gpt-5.6-sol"]


async def test_vision_does_not_retry_on_an_ordinary_stop() -> None:
    c = _client("gpt-5.6-sol")
    seen: list[str] = []

    async def create(*, model: str, **_: object) -> MagicMock:
        seen.append(model)
        return _vision_resp("stop", "page text")

    c.client.chat.completions.create = create
    assert await c.vision("x", images=[_PNG]) == "page text"
    assert seen == ["gemini-3.6-flash"]


async def test_vision_returns_empty_when_no_fallback_configured() -> None:
    c = _client(None)

    async def create(**_: object) -> MagicMock:
        return _vision_resp("content_filter", None)

    c.client.chat.completions.create = create
    assert await c.vision("x", images=[_PNG]) == ""


async def test_vision_returns_empty_when_fallback_also_filtered() -> None:
    c = _client("gpt-5.6-sol")
    seen: list[str] = []

    async def create(*, model: str, **_: object) -> MagicMock:
        seen.append(model)
        return _vision_resp("content_filter", None)

    c.client.chat.completions.create = create
    assert await c.vision("x", images=[_PNG]) == ""
    # Tried primary then fallback exactly once each; no infinite recursion.
    assert seen == ["gemini-3.6-flash", "gpt-5.6-sol"]


async def test_an_unrecovered_refusal_carries_its_finish_reason() -> None:
    # The page is empty either way; the reason is what tells a refusal from a blank leaf.
    c = _client(None)

    async def create(**_: object) -> MagicMock:
        return _vision_resp("content_filter: RECITATION", None)

    c.client.chat.completions.create = create
    out = await c.vision("x", images=[_PNG])
    assert (out, getattr(out, "finish_reason", None)) == ("", "content_filter: RECITATION")


async def test_a_recovered_read_carries_the_fallback_finish_reason() -> None:
    c = _client("gpt-5.6-sol")

    async def create(*, model: str, **_: object) -> MagicMock:
        if model != "gpt-5.6-sol":
            return _vision_resp("content_filter: RECITATION", None)
        return _vision_resp("stop", "recovered text")

    c.client.chat.completions.create = create
    out = await c.vision("x", images=[_PNG])
    assert (out, getattr(out, "finish_reason", None)) == ("recovered text", "stop")
