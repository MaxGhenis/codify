"""LLM client Protocol + LiteLLM-backed implementation."""

from __future__ import annotations

import asyncio
import base64
import functools
import inspect
import json
import time
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import contextmanager
from typing import Any, Literal, ParamSpec, Protocol, TypeVar, cast

import httpx
import openai
import structlog
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from opentelemetry import propagate
from pydantic import BaseModel, ValidationError
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from codify.core.tracing import (
    direct_observation,
    proxy_call_metadata,
    proxy_trace_headers,
    record_usage,
)


def _trace_headers() -> dict[str, str]:
    """W3C context only, for direct providers that do not run LiteLLM."""
    headers: dict[str, str] = {}
    propagate.inject(headers)
    return headers


def _finish_usage(observation: Any, model: str, response: Any) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    record_usage(
        observation,
        model,
        {
            "input": getattr(usage, "prompt_tokens", None),
            "output": getattr(usage, "completion_tokens", None),
            "total": getattr(usage, "total_tokens", None),
        },
    )


@contextmanager
def _null_observation() -> Any:
    yield None


OnChunk = Callable[[str], None | Awaitable[None]]
"""Streaming callback. May be sync or async; both are awaited as needed."""

logger = structlog.get_logger()


_MIME_SIGS: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
)


def _sniff_mime(data: bytes) -> str:
    for sig, mime in _MIME_SIGS:
        if data.startswith(sig):
            return mime
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


_SchemaT = TypeVar("_SchemaT", bound=BaseModel)


class LLMClient(Protocol):
    """Common interface for LLM backends, addressed via the LiteLLM gateway."""

    async def chat(
        self, prompt: str, system: str | None = None, model: str | None = None
    ) -> str: ...

    async def vision(
        self,
        prompt: str,
        images: list[bytes],
        system: str | None = None,
        model: str | None = None,
    ) -> str: ...

    async def ocr(
        self,
        *,
        image: bytes | None = None,
        pdf: bytes | None = None,
        model: str,
    ) -> list[dict[str, Any]]: ...

    async def chat_stream(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        on_chunk: OnChunk | None = None,
    ) -> str: ...

    async def chat_json(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        *,
        seed: int | None = None,
    ) -> dict[str, Any]: ...

    async def chat_schema(
        self,
        prompt: str,
        schema: type[_SchemaT],
        system: str | None = None,
        model: str | None = None,
    ) -> _SchemaT: ...

    async def chat_schema_stream(
        self,
        prompt: str,
        schema: type[_SchemaT],
        *,
        on_partial: Callable[[object], None] | None = None,
        system: str | None = None,
        model: str | None = None,
    ) -> _SchemaT: ...


class OcrNotConfigured(RuntimeError):
    """The direct OCR endpoint has no URL or key. Names the variables and nothing
    else, so the message is safe to log where a provider body would not be."""


class OcrStatusError(RuntimeError):
    """A non-2xx from the direct Azure Foundry OCR endpoint, carrying the status so a
    caller can tell "slow down" from "this will never work"; that route bypasses
    the OpenAI SDK, so no openai exception type reaches it. No response body: the
    handlers log `str(exc)`, and a body echoing the page put document text in logs.
    """

    def __init__(self, status_code: int, retry_after: float | None = None) -> None:
        super().__init__(f"OCR {status_code}")
        self.status_code = status_code
        self.retry_after = retry_after


class RateLimiter:
    """Caps calls to a requests-per-minute ceiling across concurrent callers,
    in-process only. The Foundry deployment's limit is per-deployment, so two API
    replicas each get the full allowance and can overrun it together; a shared
    limiter would need Redis. Sized from the deployment's stated RPM.
    """

    def __init__(self, per_minute: int) -> None:
        if per_minute < 1:
            raise ValueError(f"per_minute must be at least 1, got {per_minute}")
        self._per_minute = per_minute
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
                if len(self._calls) < self._per_minute:
                    self._calls.append(now)
                    return
                sleep_for = 60.0 - (now - self._calls[0])
            await asyncio.sleep(max(sleep_for, 0.05))


def is_connection_class_error(exc: BaseException) -> bool:
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code in (502, 503, 504):
        return True
    return False


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, OcrStatusError):
        return exc.status_code == 429 or 500 <= exc.status_code < 600
    # The direct OCR route bypasses the OpenAI SDK, so its timeouts arrive as
    # httpx types and would read as permanent. Scoped to the retry decision:
    # `is_connection_class_error` also decides failure versus unavailable
    # dependency, which is a wider claim.
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return isinstance(exc, openai.RateLimitError) or is_connection_class_error(exc)


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _wait_honouring_retry_after(max_wait: float) -> Callable[[RetryCallState], float]:
    """Back off exponentially, unless the server said how long to wait."""
    curve = wait_exponential(min=1, max=max_wait)

    def wait(state: RetryCallState) -> float:
        backoff = curve(state)
        outcome = state.outcome
        exc = outcome.exception() if outcome is not None else None
        after = getattr(exc, "retry_after", None)
        return max(backoff, min(float(after), max_wait)) if after else backoff

    return wait


def _resilient_with(
    attempts: int, max_wait: float
) -> Callable[[Callable[_P, Awaitable[_R]]], Callable[_P, Coroutine[Any, Any, _R]]]:
    """Retry on transient gateway errors; raise everything else as-is.

    Retries on the exception itself rather than a marker, so whatever finally
    reaches the caller still carries its status and the caller can say why.
    """

    def decorate(func: Callable[_P, Awaitable[_R]]) -> Callable[_P, Coroutine[Any, Any, _R]]:
        @retry(
            stop=stop_after_attempt(attempts),
            wait=_wait_honouring_retry_after(max_wait),
            retry=retry_if_exception(_is_transient),
            reraise=True,
        )
        @functools.wraps(func)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return await func(*args, **kwargs)

        return wrapper

    return decorate


_resilient = _resilient_with(3, 30)
# OCR runs behind a hard requests-per-minute ceiling, so a 429 here means wait,
# not fail. The bound is mostly a clamp on a long server-sent Retry-After.
_resilient_ocr = _resilient_with(6, 120)


_DETERMINISM_INSTRUCTION = (
    "Answer deterministically. Given identical input, produce identical output: "
    "do not vary wording, ordering or formatting between runs. Run key: {seed}."
)


def _with_determinism(system: str | None, seed: int) -> str:
    """``system`` plus an explicit request for a reproducible answer. Gemini has no
    ``seed`` and accepts ``temperature``, ``top_p`` and ``top_k`` without acting
    on them, so a sampling argument reaches the gateway and stops there.
    Instructions are the only channel the model reads.
    """
    line = _DETERMINISM_INSTRUCTION.format(seed=seed)
    return f"{system}\n\n{line}" if system else line


def _messages(prompt: str, system: str | None) -> list[ChatCompletionMessageParam]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    out.append({"role": "user", "content": prompt})
    return cast(list[ChatCompletionMessageParam], out)


def content_filtered(finish_reason: object) -> bool:
    """Gemini's OpenAI-compatible endpoint suffixes the cause, as in
    'content_filter: RECITATION', so an equality test never sees the block."""
    return isinstance(finish_reason, str) and finish_reason.startswith("content_filter")


class VisionText(str):
    """A vision read that also carries the provider's finish reason, so a page
    refused by a content filter is told apart from a blank one."""

    finish_reason: str

    def __new__(cls, text: str, finish_reason: str = "") -> VisionText:
        obj = super().__new__(cls, text)
        obj.finish_reason = finish_reason
        return obj


class LiteLLMClient:
    """Single OpenAI-SDK-backed client pointed at a LiteLLM gateway."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: httpx.Timeout | None = None,
        azure_foundry_ocr_url: str | None = None,
        azure_foundry_ocr_key: str | None = None,
        azure_foundry_ocr_rpm: int = 40,
        telemetry_mode: Literal["proxy", "direct"] = "proxy",
        fallback_model: str | None = None,
    ) -> None:
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=openai.Timeout(**(timeout or httpx.Timeout(600.0, connect=10.0)).as_dict()),
        )
        self.model = model
        self.telemetry_mode = telemetry_mode
        # Retried on when the primary returns a content-filter finish reason.
        # Gemini's filter false-fires on benign legal prose and `safety_settings`
        # cannot disable it. Empty disables the retry.
        self.fallback_model = fallback_model
        # Direct Azure Foundry Mistral OCR URL, bypasses the LiteLLM proxy
        # (which has a routing gap on this endpoint under v1.83.7-stable).
        self.azure_foundry_ocr_url = azure_foundry_ocr_url
        self.azure_foundry_ocr_key = azure_foundry_ocr_key
        self._ocr_limiter = RateLimiter(azure_foundry_ocr_rpm)

    def _headers(self) -> dict[str, str]:
        return proxy_trace_headers() if self.telemetry_mode == "proxy" else _trace_headers()

    def _body(self) -> dict[str, Any]:
        """Request-body attribution the gateway forwards to its callbacks.

        Headers carry the Langfuse contract; this carries the one LiteLLM reads
        from the body. Empty off the proxy, where there is no gateway to tell."""
        if self.telemetry_mode != "proxy":
            return {}
        metadata = proxy_call_metadata()
        return {"metadata": metadata} if metadata else {}

    @_resilient
    async def chat(self, prompt: str, system: str | None = None, model: str | None = None) -> str:
        resolved_model = model or self.model
        if self.telemetry_mode == "direct":
            with direct_observation(
                "direct.chat",
                as_type="generation",
                model=resolved_model,
                metadata={"route": "chat"},
            ) as observation:
                resp = await self.client.chat.completions.create(
                    model=resolved_model,
                    messages=_messages(prompt, system),
                    extra_headers=self._headers(),
                    extra_body=self._body(),
                )
                _finish_usage(observation, resolved_model, resp)
        else:
            resp = await self.client.chat.completions.create(
                model=resolved_model,
                messages=_messages(prompt, system),
                extra_headers=self._headers(),
                extra_body=self._body(),
            )
        return resp.choices[0].message.content or ""

    @_resilient
    async def chat_stream(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        on_chunk: OnChunk | None = None,
        timeout: float | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": _messages(prompt, system),
            "stream": True,
            "extra_headers": self._headers(),
            "extra_body": self._body(),
        }
        if self.telemetry_mode == "direct":
            kwargs["stream_options"] = {"include_usage": True}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resolved_model = model or self.model
        with (
            direct_observation(
                "direct.chat_stream",
                as_type="generation",
                model=resolved_model,
                metadata={"route": "chat_stream"},
            )
            if self.telemetry_mode == "direct"
            else _null_observation() as observation
        ):
            stream = await self.client.chat.completions.create(**kwargs)
            accumulated = ""
            last_chunk: Any = None
            async for chunk in stream:
                last_chunk = chunk
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                accumulated += delta
                if on_chunk is not None:
                    result = on_chunk(accumulated)
                    if inspect.isawaitable(result):
                        await result
            if observation is not None and last_chunk is not None:
                _finish_usage(observation, resolved_model, last_chunk)
        return accumulated

    @_resilient
    async def chat_json(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        *,
        seed: int | None = None,
    ) -> dict[str, Any]:
        # `is not None`, not truthiness: seed 0 is a seed like any other, and it
        # is forwarded below, so it must carry the instruction too.
        instructions = system if seed is None else _with_determinism(system, seed)
        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": _messages(prompt, instructions),
            "response_format": {"type": "json_object"},
            "extra_headers": self._headers(),
            "extra_body": self._body(),
        }
        if seed is not None:
            kwargs["seed"] = seed
        resolved_model = model or self.model
        with (
            direct_observation(
                "direct.chat_json",
                as_type="generation",
                model=resolved_model,
                metadata={"route": "chat_json"},
            )
            if self.telemetry_mode == "direct"
            else _null_observation() as observation
        ):
            resp = await self.client.chat.completions.create(**kwargs)
            if observation is not None:
                _finish_usage(observation, resolved_model, resp)
        text = resp.choices[0].message.content or "{}"
        try:
            return cast(dict[str, Any], json.loads(text))
        except json.JSONDecodeError as exc:
            logger.warning("invalid_json", model=model or self.model, preview=text[:500])
            raise ValueError(f"Model returned invalid JSON: {text[:200]}") from exc

    @_resilient
    async def chat_schema(
        self,
        prompt: str,
        schema: type[_SchemaT],
        system: str | None = None,
        model: str | None = None,
    ) -> _SchemaT:
        resolved_model = model or self.model
        try:
            with (
                direct_observation(
                    "direct.chat_schema",
                    as_type="generation",
                    model=resolved_model,
                    metadata={"route": "chat_schema"},
                )
                if self.telemetry_mode == "direct"
                else _null_observation() as observation
            ):
                resp = await self.client.chat.completions.parse(
                    model=resolved_model,
                    messages=_messages(prompt, system),
                    response_format=schema,
                    extra_headers=self._headers(),
                    extra_body=self._body(),
                )
                if observation is not None:
                    _finish_usage(observation, resolved_model, resp)
            # The SDK raises only on the bare 'content_filter'; a suffixed
            # reason comes back as an empty completion instead.
            if content_filtered(resp.choices[0].finish_reason):
                raise openai.ContentFilterFinishReasonError()
        except openai.ContentFilterFinishReasonError:
            if self.fallback_model and resolved_model != self.fallback_model:
                logger.warning(
                    "content_filter_fallback",
                    route="chat_schema",
                    from_model=resolved_model,
                    to_model=self.fallback_model,
                )
                return await self.chat_schema(
                    prompt, schema, system=system, model=self.fallback_model
                )
            raise
        message = resp.choices[0].message
        if message.parsed is not None:
            return message.parsed
        text = message.content or "{}"
        try:
            return schema.model_validate_json(text)
        except (ValidationError, ValueError) as exc:
            logger.warning(
                "invalid_structured_output", model=model or self.model, preview=text[:500]
            )
            raise ValueError(f"Model returned invalid structured output: {text[:200]}") from exc

    async def chat_schema_stream(
        self,
        prompt: str,
        schema: type[_SchemaT],
        *,
        on_partial: Callable[[object], None] | None = None,
        system: str | None = None,
        model: str | None = None,
    ) -> _SchemaT:
        try:
            resolved_model = model or self.model
            with (
                direct_observation(
                    "direct.chat_schema_stream",
                    as_type="generation",
                    model=resolved_model,
                    metadata={"route": "chat_schema_stream"},
                )
                if self.telemetry_mode == "direct"
                else _null_observation() as observation
            ):
                async with self.client.chat.completions.stream(
                    model=resolved_model,
                    messages=_messages(prompt, system),
                    response_format=schema,
                    extra_headers=self._headers(),
                    extra_body=self._body(),
                ) as stream:
                    async for event in stream:
                        if event.type == "content.delta" and on_partial and event.parsed:
                            on_partial(event.parsed)
                    final = await stream.get_final_completion()
                if observation is not None:
                    _finish_usage(observation, resolved_model, final)
            parsed = final.choices[0].message.parsed
            if parsed is not None:
                return parsed
            raise ValueError("stream produced no parsed output")
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.BadRequestError):
            raise
        except Exception as exc:  # noqa: BLE001, stream unsupported → non-stream path
            logger.warning("chat_schema_stream_fallback", error=str(exc)[:200])
            return await self.chat_schema(prompt, schema, system, model)

    @_resilient
    async def vision(
        self,
        prompt: str,
        images: list[bytes],
        system: str | None = None,
        model: str | None = None,
    ) -> str:
        parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            mime = _sniff_mime(img)
            b64 = base64.b64encode(img).decode("ascii")
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": parts})
        resolved_model = model or self.model
        with (
            direct_observation(
                "direct.vision",
                as_type="generation",
                model=resolved_model,
                metadata={"route": "vision", "image_count": len(images)},
            )
            if self.telemetry_mode == "direct"
            else _null_observation() as observation
        ):
            resp = await self.client.chat.completions.create(
                model=resolved_model,
                messages=cast(list[ChatCompletionMessageParam], msgs),
                extra_headers=self._headers(),
                extra_body=self._body(),
            )
            if observation is not None:
                _finish_usage(observation, resolved_model, resp)
        # Unlike `chat_schema`'s `.parse()`, `.create()` does not raise on a
        # block: a filtered page returns finish_reason='content_filter' with
        # empty content, so a RECITATION-blocked scan of published law silently
        # dropped its text. The non-Google fallback reads it.
        choice = resp.choices[0]
        if (
            content_filtered(choice.finish_reason)
            and self.fallback_model
            and resolved_model != self.fallback_model
        ):
            logger.warning(
                "content_filter_fallback",
                route="vision",
                from_model=resolved_model,
                to_model=self.fallback_model,
            )
            return await self.vision(prompt, images, system=system, model=self.fallback_model)
        return VisionText(choice.message.content or "", str(choice.finish_reason or ""))

    @_resilient_ocr
    async def ocr(
        self,
        *,
        image: bytes | None = None,
        pdf: bytes | None = None,
        model: str,
    ) -> list[dict[str, Any]]:
        """Azure Foundry Mistral OCR, called directly (LiteLLM proxy bypassed).

        Give one of image OR pdf. PDF path sends the whole document in a single
        call, which is faster and higher-quality than per-page rendering: the
        model has cross-page context and layout is preserved natively.
        Returns `data["pages"]` verbatim, `[]` if absent. Each page carries
        `index`, `markdown`, and (always requested) `header`, `footer`, `blocks`.
        """
        if not self.azure_foundry_ocr_url or not self.azure_foundry_ocr_key:
            raise OcrNotConfigured(
                "azure_foundry_ocr not configured; set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY"
            )
        if bool(image) == bool(pdf):
            raise ValueError("ocr requires exactly one of image or pdf")
        if pdf is not None:
            b64 = base64.b64encode(pdf).decode("ascii")
            document = {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{b64}",
            }
        else:
            assert image is not None
            mime = _sniff_mime(image)
            b64 = base64.b64encode(image).decode("ascii")
            document = {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"}
        headers = {
            "Authorization": f"Bearer {self.azure_foundry_ocr_key}",
            "Content-Type": "application/json",
            **_trace_headers(),
        }
        payload: dict[str, Any] = {
            "model": model,
            "document": document,
            # Lifts running header/footer out of `markdown`, so a masthead cannot
            # become anchor zero.
            "extract_header": True,
            "extract_footer": True,
            # Pinned, not inherited: the default was observed true on 84/84 pages.
            "include_blocks": True,
            # Per-word confidence with a character offset into the markdown, so
            # doubt lands on the text rather than only on the page image.
            "confidence_scores_granularity": "word",
            # Without this a table arrives flattened into the surrounding prose.
            "table_format": "markdown",
        }
        with direct_observation(
            "direct.azure_ocr",
            as_type="generation",
            model=model,
            metadata={"route": "azure_foundry_ocr", "media_type": "pdf" if pdf else "image"},
        ) as observation:
            await self._ocr_limiter.acquire()
            async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0)) as h:
                r = await h.post(self.azure_foundry_ocr_url, headers=headers, json=payload)
                if r.status_code >= 400:
                    raw = r.headers.get("retry-after")
                    raise OcrStatusError(
                        r.status_code, float(raw) if raw and raw.isdigit() else None
                    )
                data = r.json()
            usage = data.get("usage_info") or data.get("usage")
            if isinstance(usage, dict):
                # Named, not filtered. Langfuse sums unrecognised usage keys into
                # the unit total, so forwarding a provider blob wholesale is how
                # doc_size_bytes came to be counted as tokens.
                record_usage(observation, model, {"pages_processed": usage.get("pages_processed")})
                size = usage.get("doc_size_bytes")
                if isinstance(size, int):
                    observation.update(metadata={"doc_size_bytes": size})
        pages = data.get("pages") or []
        return cast(list[dict[str, Any]], pages)


def create_llm_client(
    *,
    base_url: str,
    api_key: str,
    model: str,
    azure_foundry_ocr_url: str | None = None,
    azure_foundry_ocr_key: str | None = None,
    azure_foundry_ocr_rpm: int = 40,
    telemetry_mode: Literal["proxy", "direct"] = "proxy",
    fallback_model: str | None = None,
) -> LLMClient:
    return LiteLLMClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        azure_foundry_ocr_url=azure_foundry_ocr_url,
        azure_foundry_ocr_key=azure_foundry_ocr_key,
        azure_foundry_ocr_rpm=azure_foundry_ocr_rpm,
        telemetry_mode=telemetry_mode,
        fallback_model=fallback_model,
    )


__all__ = [
    "LLMClient",
    "LiteLLMClient",
    "OcrStatusError",
    "RateLimiter",
    "VisionText",
    "content_filtered",
    "create_llm_client",
]
