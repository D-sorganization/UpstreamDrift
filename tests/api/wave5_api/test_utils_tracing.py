"""Tests for src/api/utils/tracing.py."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.api.utils import tracing

pytestmark = pytest.mark.unit


def test_generate_request_id_has_prefix_and_uniqueness() -> None:
    a = tracing.generate_request_id()
    b = tracing.generate_request_id()
    assert a.startswith("req_")
    assert b.startswith("req_")
    assert a != b
    assert len(a) == len("req_") + 16


def test_generate_correlation_id_has_prefix_and_uniqueness() -> None:
    a = tracing.generate_correlation_id()
    b = tracing.generate_correlation_id()
    assert a.startswith("cor_")
    assert a != b


def test_get_set_reset_request_id() -> None:
    assert tracing.get_request_id() == ""
    token = tracing.set_request_id("req_abc")
    try:
        assert tracing.get_request_id() == "req_abc"
    finally:
        tracing._request_id_var.reset(token)
    assert tracing.get_request_id() == ""


def test_trace_context_to_dict_round_trip() -> None:
    ctx = tracing.TraceContext(
        request_id="req_1",
        correlation_id="cor_1",
        operation="GET /x",
        start_time=1.5,
        metadata={"k": "v"},
    )
    d = ctx.to_dict()
    assert d["request_id"] == "req_1"
    assert d["correlation_id"] == "cor_1"
    assert d["operation"] == "GET /x"
    assert d["metadata"] == {"k": "v"}


def test_get_set_trace_context() -> None:
    assert tracing.get_trace_context() is None
    ctx = tracing.TraceContext(request_id="r", correlation_id="c")
    token = tracing.set_trace_context(ctx)
    try:
        got = tracing.get_trace_context()
        assert got is ctx
    finally:
        tracing._trace_context_var.reset(token)
    assert tracing.get_trace_context() is None


def test_traced_log_requires_level() -> None:
    with pytest.raises(ValueError):
        tracing.traced_log(None, "msg")  # type: ignore[arg-type]


def test_traced_log_injects_context() -> None:
    # Should not raise; we verify by spying on logger via patching.
    logs: list[tuple[str, dict[str, Any]]] = []

    def fake_info(msg: str, extra: dict[str, Any] | None = None, **_: Any) -> None:
        logs.append((msg, extra or {}))

    orig = tracing.logger.info
    tracing.logger.info = fake_info  # type: ignore[assignment]
    rid_token = tracing.set_request_id("req_xyz")
    ctx_token = tracing.set_trace_context(
        tracing.TraceContext(request_id="req_xyz", correlation_id="cor_xyz")
    )
    try:
        tracing.traced_log("info", "hello", extra_key="ek")
    finally:
        tracing.logger.info = orig  # type: ignore[assignment]
        tracing._request_id_var.reset(rid_token)
        tracing._trace_context_var.reset(ctx_token)

    assert logs
    _msg, extra = logs[0]
    assert extra["request_id"] == "req_xyz"
    assert extra["correlation_id"] == "cor_xyz"
    assert extra["extra_key"] == "ek"


def test_traced_log_unknown_level_falls_back_to_info() -> None:
    # Just ensure no crash for unknown level.
    tracing.traced_log("noplevel", "msg")


class _FakeReq:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}
        self.method = "GET"

        class _U:
            path = "/x"

        self.url = _U()

        class _C:
            host = "1.2.3.4"

        self.client = _C()


def test_request_tracer_requires_request() -> None:
    tracer = tracing.RequestTracer()

    async def go() -> None:
        with pytest.raises(ValueError):
            await tracer.trace_request(None, lambda r: r)  # type: ignore[arg-type]

    asyncio.run(go())


def test_request_tracer_happy_path_sets_headers() -> None:
    tracer = tracing.RequestTracer()

    class _Resp:
        def __init__(self) -> None:
            self.status_code = 200
            self.headers: dict[str, str] = {}

    async def call_next(_req: Any) -> _Resp:
        return _Resp()

    req = _FakeReq(headers={"X-Correlation-ID": "cor_existing"})

    async def go() -> _Resp:
        return await tracer.trace_request(req, call_next)

    resp = asyncio.run(go())
    assert resp.headers[tracing.CORRELATION_ID_HEADER] == "cor_existing"
    assert resp.headers[tracing.REQUEST_ID_HEADER].startswith("req_")
    assert "X-Response-Time-Ms" in resp.headers


def test_request_tracer_generates_correlation_if_missing() -> None:
    tracer = tracing.RequestTracer()

    class _Resp:
        def __init__(self) -> None:
            self.status_code = 200
            self.headers: dict[str, str] = {}

    async def call_next(_req: Any) -> _Resp:
        return _Resp()

    async def go() -> _Resp:
        return await tracer.trace_request(_FakeReq(), call_next)

    resp = asyncio.run(go())
    assert resp.headers[tracing.CORRELATION_ID_HEADER].startswith("cor_")


def test_request_tracer_propagates_errors_and_resets_context() -> None:
    tracer = tracing.RequestTracer()

    async def call_next(_req: Any) -> Any:
        raise RuntimeError("boom")

    async def go() -> None:
        with pytest.raises(RuntimeError):
            await tracer.trace_request(_FakeReq(), call_next)

    asyncio.run(go())
    # Context must be cleared
    assert tracing.get_request_id() == ""
    assert tracing.get_trace_context() is None
