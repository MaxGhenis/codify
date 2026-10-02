from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

import pytest
import structlog

from codify.core.tracing import (
    DEFAULT_ACTOR_ID,
    TelemetryContext,
    attach_trace_attribution,
    bind_telemetry,
    current_telemetry,
    direct_observation,
    error_fields,
    get_current_actor,
    get_current_session,
    make_run_telemetry,
    proxy_trace_headers,
)


class _Client:
    def __init__(self, parent_id: str | None = None) -> None:
        self.parent_id = parent_id
        self.started: dict[str, Any] | None = None
        self.context_saw_exception = False

    def create_trace_id(self, *, seed: str) -> str:
        return f"trace:{seed}"

    def get_current_observation_id(self) -> str | None:
        return self.parent_id

    @contextmanager
    def start_as_current_observation(self, **kwargs: Any) -> Iterator[_Observation]:
        self.started = kwargs
        try:
            yield _Observation()
        except Exception:
            self.context_saw_exception = True
            raise


class _Observation:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


def _context() -> TelemetryContext:
    return TelemetryContext(
        run_id="run-1",
        workflow_kind="translation",
        langfuse_trace_id="a" * 32,
        jurisdiction="xz",
        actor_id="actor@example.com",
        session_id="session-1",
        release="v1.2.3",
    )


def test_make_run_telemetry_is_deterministic(monkeypatch: Any) -> None:
    client = _Client()
    monkeypatch.setattr("langfuse.get_client", lambda: client)

    one = make_run_telemetry("run-1", "ingest", {"jurisdiction_code": "xz"})
    two = make_run_telemetry("run-1", "ingest", {"jurisdiction_code": "xz"})

    assert one.langfuse_trace_id == two.langfuse_trace_id == "trace:run-1"


def test_proxy_headers_use_existing_trace_and_parent(monkeypatch: Any) -> None:
    monkeypatch.setattr("langfuse.get_client", lambda: _Client(parent_id="b" * 16))

    with bind_telemetry(_context()):
        headers = proxy_trace_headers()

    assert headers["langfuse_existing_trace_id"] == "a" * 32
    assert headers["langfuse_parent_observation_id"] == "b" * 16
    assert headers["langfuse_run_id"] == "run-1"
    assert headers["langfuse_workflow_kind"] == "translation"
    assert headers["langfuse_jurisdiction"] == "xz"
    assert headers["langfuse_actor_id"] == "actor@example.com"
    assert headers["langfuse_codify_session_id"] == "session-1"
    assert "langfuse_trace_id" not in headers
    assert "langfuse_update_trace_keys" not in headers


def test_binding_resets_context_and_structlog() -> None:
    before = structlog.contextvars.get_contextvars()

    with bind_telemetry(_context()):
        assert current_telemetry.get() == _context()
        assert get_current_actor() == "actor@example.com"
        assert get_current_session() == "session-1"
        assert structlog.contextvars.get_contextvars()["run_id"] == "run-1"

    assert current_telemetry.get() is None
    assert get_current_actor() == DEFAULT_ACTOR_ID
    assert get_current_session() is None
    assert structlog.contextvars.get_contextvars() == before


def test_direct_observation_never_receives_payload(monkeypatch: Any) -> None:
    client = _Client()
    monkeypatch.setattr("langfuse.get_client", lambda: client)

    with bind_telemetry(_context()):
        with direct_observation(
            "direct.embedding",
            as_type="embedding",
            model="gemini-embedding-2",
            metadata={"batch_size": 2},
        ):
            pass

    assert client.started == {
        "name": "direct.embedding",
        "as_type": "embedding",
        "trace_context": {"trace_id": "a" * 32},
        "model": "gemini-embedding-2",
        "metadata": {"batch_size": 2},
    }
    assert "input" not in client.started
    assert "output" not in client.started


def test_direct_observation_records_error_type_only(monkeypatch: Any) -> None:
    client = _Client()
    observation: _Observation | None = None
    monkeypatch.setattr("langfuse.get_client", lambda: client)

    try:
        with direct_observation(
            "direct.ocr", as_type="generation", model="mistral", metadata=None
        ) as active:
            observation = active
            raise RuntimeError("secret provider response")
    except RuntimeError:
        pass

    assert observation is not None
    assert observation.updates == [{"level": "ERROR", "status_message": "RuntimeError"}]
    assert client.context_saw_exception is False


def test_trace_attribution_uses_langfuse_v4_attributes(monkeypatch: Any) -> None:
    class _Span:
        def __init__(self) -> None:
            self.attributes: dict[str, str] = {}

        def is_recording(self) -> bool:
            return True

        def set_attribute(self, key: str, value: str) -> None:
            self.attributes[key] = value

    span = _Span()
    monkeypatch.setattr("codify.core.tracing.otel_trace.get_current_span", lambda: span)

    attach_trace_attribution(
        object(),
        user_id="actor@example.com",
        session_id="session-1",
        release="v1.2.3",
        tags=["workflow:translation"],
        trace_name="codify.translation",
        metadata={"run_id": "run-1"},
    )

    assert span.attributes == {
        "langfuse.trace.name": "codify.translation",
        "user.id": "actor@example.com",
        "session.id": "session-1",
        "langfuse.release": "v1.2.3",
        "langfuse.trace.tags": '["workflow:translation"]',
        "langfuse.trace.metadata.run_id": "run-1",
    }


def test_error_fields_drops_the_provider_body() -> None:
    """An OCR failure must not put document text in the logs."""
    from codify.core.llm import OcrStatusError

    exc = OcrStatusError(429, retry_after=7.0)
    assert error_fields(exc) == {
        "error_type": "OcrStatusError",
        "status_code": 429,
        "retry_after": 7.0,
    }
    assert "مادة" not in str(exc)


def test_error_fields_carries_the_type_when_there_is_nothing_else() -> None:
    assert error_fields(ValueError("secret page text")) == {"error_type": "ValueError"}


def test_error_fields_keeps_a_message_that_cannot_carry_a_payload() -> None:
    """Type name alone left a connect failure indistinguishable from a bug."""
    import httpx

    assert error_fields(httpx.ConnectError("[Errno 61] Connection refused"))["error"] == (
        "[Errno 61] Connection refused"
    )
    assert error_fields(FileNotFoundError(2, "No such file", "gazette.pdf"))["error"]


def test_the_workspace_is_tagged_so_cost_can_be_split_by_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/admin/cost` groups by trace tags, so a workspace with no tag cannot be
    billed for what it spent. The id is tagged rather than the name because a
    rename must not fork the history."""
    from codify.core.tracing import TelemetryContext, attach_current_telemetry, bind_telemetry

    captured: dict[str, object] = {}

    def _fake(_client: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("codify.core.tracing.attach_trace_attribution", _fake)
    context = TelemetryContext(
        run_id="run-1",
        workflow_kind="ingest",
        langfuse_trace_id="trace-1",
        jurisdiction="xz",
        org_id="7f69c0c5-9d39-4fed-a364-9786769ac754",
    )
    with bind_telemetry(context):
        attach_current_telemetry(object())

    assert captured["tags"] == [
        "workflow:ingest",
        "jurisdiction:xz",
        "organisation:7f69c0c5-9d39-4fed-a364-9786769ac754",
    ]


def test_no_workspace_means_no_workspace_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Platform work (schedulers, backfills) belongs to no workspace, and an
    empty tag would collect them all under one meaningless label."""
    from codify.core.tracing import TelemetryContext, attach_current_telemetry, bind_telemetry

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "codify.core.tracing.attach_trace_attribution",
        lambda _c, **kw: captured.update(kw),
    )
    with bind_telemetry(
        TelemetryContext(run_id="r", workflow_kind="janitor", langfuse_trace_id="t")
    ):
        attach_current_telemetry(object())

    assert captured["tags"] == ["workflow:janitor"]


def test_a_pipeline_stage_adds_tags_without_dropping_the_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`set_attribute` replaces rather than merges, and ingest and compare stamp
    their own tags deeper in the run. Whichever span wrote last used to decide
    what the trace was tagged with, so the workspace a cost is billed to hung on
    ordering."""
    from codify.core.tracing import TelemetryContext, attach_trace_attribution, bind_telemetry

    class _Span:
        def __init__(self) -> None:
            self.attributes: dict[str, str] = {}

        def is_recording(self) -> bool:
            return True

        def set_attribute(self, key: str, value: str) -> None:
            self.attributes[key] = value

    span = _Span()
    monkeypatch.setattr("codify.core.tracing.otel_trace.get_current_span", lambda: span)

    with bind_telemetry(
        TelemetryContext(
            run_id="run-1",
            workflow_kind="ingest",
            langfuse_trace_id="t",
            jurisdiction="xz",
            org_id="org-1",
        )
    ):
        # What pipeline/formats/pdf.py stamps once it reaches the document.
        attach_trace_attribution(object(), tags=["jurisdiction:xz", "pipeline:ingest"])

    tags = json.loads(span.attributes["langfuse.trace.tags"])
    assert "pipeline:ingest" in tags, "the caller's own tag must survive"
    assert "organisation:org-1" in tags, "the workspace must not be dropped"
    assert tags.count("jurisdiction:xz") == 1, "a tag both sides carry is not duplicated"
