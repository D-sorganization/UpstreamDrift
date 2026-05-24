"""Tests for ws_pubsub backend selection and Rust-path delegation.

These tests avoid spinning up a real HTTP/WS server. The Rust backend path
is exercised by injecting a fake ``upstream_realtime`` module via
``sys.modules`` for the duration of each test.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

from src.shared.python.realtime import ws_pubsub as ws_mod

# ----------------------------- helpers ----------------------------------------


class _FakeRustServer:
    """In-process stand-in for ``upstream_realtime.Server``."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port or 12345
        self.published: list[tuple[str, str]] = []
        self.stopped = False
        self._subs: dict[str, list[_FakeSubscriber]] = {}

    def bound_port(self) -> int:
        return self.port

    def publish(self, channel: str, payload_json: str) -> None:
        self.published.append((channel, payload_json))
        for sub in self._subs.get(channel, []):
            sub.feed(payload_json)

    def subscribe(self, channel: str) -> _FakeSubscriber:
        sub = _FakeSubscriber()
        self._subs.setdefault(channel, []).append(sub)
        return sub

    def stop(self) -> None:
        self.stopped = True


class _FakeSubscriber:
    def __init__(self) -> None:
        self._queue: list[str] = []
        self._cv = threading.Condition()

    def feed(self, payload_json: str) -> None:
        with self._cv:
            self._queue.append(payload_json)
            self._cv.notify()

    def recv(self, timeout: float) -> str | None:
        deadline = time.monotonic() + timeout
        with self._cv:
            while not self._queue:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(timeout=remaining)
            return self._queue.pop(0)


@pytest.fixture
def fake_rust(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``upstream_realtime`` module."""
    mod = types.ModuleType("upstream_realtime")
    mod.Server = _FakeRustServer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "upstream_realtime", mod)
    yield mod


# ----------------------------- _has_rust_wheel --------------------------------


def test_has_rust_wheel_true_when_importable(fake_rust) -> None:
    assert ws_mod._has_rust_wheel() is True


def test_has_rust_wheel_false_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pretend the wheel is not importable
    monkeypatch.setitem(sys.modules, "upstream_realtime", None)
    assert ws_mod._has_rust_wheel() is False


# ----------------------------- _resolve_backend -------------------------------


class TestResolveBackend:
    def test_explicit_rust_with_wheel(
        self, monkeypatch: pytest.MonkeyPatch, fake_rust
    ) -> None:
        monkeypatch.setenv("UD_REALTIME_BACKEND", "rust")
        assert ws_mod._resolve_backend() == "rust"

    def test_explicit_rust_without_wheel_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("UD_REALTIME_BACKEND", "rust")
        monkeypatch.setitem(sys.modules, "upstream_realtime", None)
        assert ws_mod._resolve_backend() == "python"

    def test_explicit_python(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("UD_REALTIME_BACKEND", "python")
        assert ws_mod._resolve_backend() == "python"

    def test_unset_with_wheel_picks_rust(
        self, monkeypatch: pytest.MonkeyPatch, fake_rust
    ) -> None:
        monkeypatch.delenv("UD_REALTIME_BACKEND", raising=False)
        assert ws_mod._resolve_backend() == "rust"

    def test_unset_without_wheel_picks_python(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("UD_REALTIME_BACKEND", raising=False)
        monkeypatch.setitem(sys.modules, "upstream_realtime", None)
        assert ws_mod._resolve_backend() == "python"


# ----------------------------- BackoffSleeper ---------------------------------


class TestBackoffSleeper:
    def test_doubles_until_cap(self) -> None:
        b = ws_mod._BackoffSleeper(cap=5.0)
        # Use a stop event that is already set so wait() returns immediately
        stop = threading.Event()
        stop.set()
        assert b._delay == 1.0
        b.wait(stop)
        assert b._delay == 2.0
        b.wait(stop)
        assert b._delay == 4.0
        b.wait(stop)
        assert b._delay == 5.0  # capped
        b.wait(stop)
        assert b._delay == 5.0  # stays at cap

    def test_reset_returns_to_one(self) -> None:
        b = ws_mod._BackoffSleeper()
        stop = threading.Event()
        stop.set()
        b.wait(stop)
        b.wait(stop)
        assert b._delay > 1.0
        b.reset()
        assert b._delay == 1.0

    def test_wait_returns_false_when_stop_set(self) -> None:
        b = ws_mod._BackoffSleeper()
        stop = threading.Event()
        stop.set()
        assert b.wait(stop) is False


# ----------------------------- port_in_use ------------------------------------


def test_port_in_use_deterministic() -> None:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        assert ws_mod._port_in_use("127.0.0.1", port) is True

    # After closing, the port should be free
    assert ws_mod._port_in_use("127.0.0.1", port) is False


# ----------------------------- WSPubSub (rust) --------------------------------


class TestWSPubSubRust:
    def test_init_rust_backend(self, fake_rust) -> None:
        ps = ws_mod.WSPubSub(port=0, backend="rust")
        try:
            assert ps.backend == "rust"
            assert ps._rust_server is not None
            assert ps.port == 12345
        finally:
            ps.stop()

    def test_backend_resolution_is_lazy_until_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def fake_resolve() -> str:
            calls.append("resolve")
            return "python"

        monkeypatch.setattr(ws_mod, "_resolve_backend", fake_resolve)
        monkeypatch.setattr(ws_mod, "_port_in_use", lambda _h, _p: True)

        ps = ws_mod.WSPubSub(port=12357, autostart=False)
        assert ps.backend == "auto"
        assert calls == []

        ps.start()
        assert ps.backend == "python"
        assert calls == ["resolve"]

    def test_publish_resolves_backend_lazily(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def fake_resolve() -> str:
            calls.append("resolve")
            return "python"

        captured: dict[str, object] = {}

        class FakeResp:
            def raise_for_status(self) -> None:
                pass

        class FakeClient:
            def __init__(self, timeout=None) -> None:
                captured["timeout"] = timeout

            def post(self, url: str, json):
                captured["url"] = url
                captured["body"] = json
                return FakeResp()

        monkeypatch.setattr(ws_mod, "_resolve_backend", fake_resolve)
        monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=FakeClient))
        ws_mod.WSPubSub._http_client = None

        ps = ws_mod.WSPubSub(port=12358, autostart=False)
        ps.publish("scope/topic", {"v": 1})

        assert calls == ["resolve"]
        assert ps.backend == "python"
        assert captured["body"] == {"channel": "scope/topic", "payload": {"v": 1}}

    def test_publish_via_rust(self, fake_rust) -> None:
        ps = ws_mod.WSPubSub(port=0, backend="rust")
        try:
            ps.publish("scope/topic", {"v": 1})
            assert ps._rust_server.published == [("scope/topic", json.dumps({"v": 1}))]
        finally:
            ps.stop()

    def test_publish_invalid_channel(self, fake_rust) -> None:
        ps = ws_mod.WSPubSub(port=0, backend="rust")
        try:
            with pytest.raises(ValueError):
                ps.publish("BAD", {"v": 1})
        finally:
            ps.stop()

    def test_publish_non_dict(self, fake_rust) -> None:
        ps = ws_mod.WSPubSub(port=0, backend="rust")
        try:
            with pytest.raises(TypeError):
                ps.publish("scope/topic", "string")  # type: ignore[arg-type]
        finally:
            ps.stop()

    def test_subscribe_rust_delivers(self, fake_rust) -> None:
        ps = ws_mod.WSPubSub(port=0, backend="rust")
        received: list = []
        evt = threading.Event()

        def cb(payload) -> None:
            received.append(payload)
            evt.set()

        sub = ps.subscribe("scope/topic", cb)
        try:
            ps.publish("scope/topic", {"v": 7})
            assert evt.wait(2.0)
            assert received[-1] == {"v": 7}
        finally:
            sub.unsubscribe()
            ps.stop()

    def test_subscribe_invalid_channel(self, fake_rust) -> None:
        ps = ws_mod.WSPubSub(port=0, backend="rust")
        try:
            with pytest.raises(ValueError):
                ps.subscribe("BAD", lambda p: None)
        finally:
            ps.stop()

    def test_stop_calls_rust_stop(self, fake_rust) -> None:
        ps = ws_mod.WSPubSub(port=0, backend="rust")
        server = ps._rust_server
        ps.stop()
        assert server.stopped is True
        # Double-stop is safe
        ps.stop()

    def test_subscribe_skips_non_dict_payloads(self, fake_rust) -> None:
        """Non-JSON / non-decodable lines should be silently dropped."""
        ps = ws_mod.WSPubSub(port=0, backend="rust")
        received: list = []
        evt = threading.Event()

        def cb(payload) -> None:
            received.append(payload)
            evt.set()

        sub = ps.subscribe("scope/topic", cb)
        try:
            # Inject a bad message directly via the fake subscriber's feed,
            # then a good one. The bad one should be discarded.
            fake_sub = ps._rust_server._subs["scope/topic"][0]
            fake_sub.feed("not json")
            fake_sub.feed(json.dumps({"v": 9}))
            assert evt.wait(2.0)
            assert received[-1] == {"v": 9}
        finally:
            sub.unsubscribe()
            ps.stop()


# ----------------------------- start failure fallback -------------------------


class TestRustStartFailure:
    def test_rust_server_construction_error_falls_back_to_python(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = types.ModuleType("upstream_realtime")

        class _BoomServer:
            def __init__(self, host, port) -> None:
                raise RuntimeError("bad rust day")

        mod.Server = _BoomServer  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "upstream_realtime", mod)

        # Patch _port_in_use so we don't try to spawn the python server
        monkeypatch.setattr(ws_mod, "_port_in_use", lambda h, p: True)

        ps = ws_mod.WSPubSub(port=12346, backend="rust")
        assert ps.backend == "python"
        assert ps._rust_server is None

    def test_rust_import_failure_falls_back_to_python(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "upstream_realtime", None)
        monkeypatch.setattr(ws_mod, "_port_in_use", lambda h, p: True)
        ps = ws_mod.WSPubSub(port=12347, backend="rust")
        assert ps.backend == "python"


# ----------------------------- URL helpers ------------------------------------


def test_publish_url_format(fake_rust) -> None:
    ps = ws_mod.WSPubSub(port=0, backend="rust")
    try:
        url = ps._publish_url()
        assert url.startswith("http://127.0.0.1:")
        assert url.endswith("/realtime/publish")
    finally:
        ps.stop()


def test_subscribe_url_format(fake_rust) -> None:
    ps = ws_mod.WSPubSub(port=0, backend="rust")
    try:
        url = ps._subscribe_url("scope/topic")
        assert "ws://127.0.0.1:" in url
        assert url.endswith("/realtime/subscribe?channel=scope/topic")
    finally:
        ps.stop()


# ----------------------------- python publish path ----------------------------


class TestPythonPublishPath:
    def test_publish_requires_httpx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Make a python-backend WSPubSub without autostart so we don't try
        # to bind a port.
        ps = ws_mod.WSPubSub(port=12348, backend="python", autostart=False)
        # Force the httpx import to fail
        monkeypatch.setitem(sys.modules, "httpx", None)
        with pytest.raises(RuntimeError, match="httpx is required"):
            ps.publish("scope/topic", {"v": 1})

    def test_publish_uses_httpx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ps = ws_mod.WSPubSub(port=12349, backend="python", autostart=False)
        ws_mod.WSPubSub._http_client = None

        captured: dict = {}

        class FakeResp:
            def raise_for_status(self) -> None:
                pass

        class FakeClient:
            def __init__(self, timeout=None) -> None:
                captured["timeout"] = timeout

            def __enter__(self) -> FakeClient:
                return self

            def __exit__(self, *a) -> None:
                pass

            def post(self, url: str, json):
                captured["url"] = url
                captured["body"] = json
                return FakeResp()

        fake_httpx = SimpleNamespace(Client=FakeClient)
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

        ps.publish("scope/topic", {"v": 1})
        assert captured["body"] == {"channel": "scope/topic", "payload": {"v": 1}}
        assert "/realtime/publish" in captured["url"]


# ----------------------------- python subscribe path --------------------------


class TestPythonSubscribePath:
    def test_subscribe_without_websockets_exits_cleanly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If ``websockets`` is not importable the subscribe thread should
        log and exit; the returned Subscription must still unsubscribe cleanly.
        """
        ps = ws_mod.WSPubSub(port=12350, backend="python", autostart=False)
        monkeypatch.setitem(sys.modules, "websockets", None)
        sub = ps.subscribe("scope/topic", lambda p: None)
        # Give the worker a moment to import-fail
        time.sleep(0.05)
        sub.unsubscribe()


# ----------------------------- rust subscribe failure mode --------------------


class _BoomSubscriber:
    """Fake rust subscriber whose recv always raises."""

    def __init__(self) -> None:
        self.calls = 0

    def recv(self, timeout: float) -> str | None:
        self.calls += 1
        raise RuntimeError("rust broke")


class _BoomServer:
    def __init__(self, host: str, port: int) -> None:
        self.port = port or 12345

    def bound_port(self) -> int:
        return self.port

    def publish(self, channel: str, payload_json: str) -> None:
        pass

    def subscribe(self, channel: str) -> _BoomSubscriber:
        return _BoomSubscriber()

    def stop(self) -> None:
        pass


def test_rust_subscribe_recv_exception_exits_after_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = types.ModuleType("upstream_realtime")
    mod.Server = _BoomServer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "upstream_realtime", mod)
    ps = ws_mod.WSPubSub(port=0, backend="rust")
    sub = ps.subscribe("scope/topic", lambda p: None)
    # The recv loop raises immediately; unsubscribe must still tear it down
    # cleanly within a short timeout.
    sub.unsubscribe()
    ps.stop()


def test_rust_subscribe_callback_exception_is_isolated(fake_rust) -> None:
    ps = ws_mod.WSPubSub(port=0, backend="rust")
    received_good: list = []
    evt_good = threading.Event()

    def bad(_p) -> None:
        raise RuntimeError("nope")

    def good(p) -> None:
        received_good.append(p)
        evt_good.set()

    sub_bad = ps.subscribe("scope/topic", bad)
    sub_good = ps.subscribe("scope/topic", good)
    try:
        ps.publish("scope/topic", {"v": 1})
        assert evt_good.wait(2.0)
        assert received_good[-1] == {"v": 1}
    finally:
        sub_bad.unsubscribe()
        sub_good.unsubscribe()
        ps.stop()


# ----------------------------- python subscribe deeper -----------------------


class TestPythonSubscribeLoop:
    """Drive the asyncio consumer loop via a fake ``websockets`` module."""

    def test_consume_delivers_then_unsubscribes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio as _aio

        delivered: list = []
        delivered_evt = threading.Event()

        class FakeWS:
            def __init__(self) -> None:
                self._msgs = [json.dumps({"v": 1}), "not json", json.dumps({"v": 2})]

            async def __aenter__(self) -> FakeWS:
                return self

            async def __aexit__(self, *a) -> None:
                pass

            async def recv(self) -> str:
                if self._msgs:
                    return self._msgs.pop(0)
                # Stall — let the timeout fire occasionally
                await _aio.sleep(0.05)
                return "stall"

        fake_websockets = types.ModuleType("websockets")

        def connect(_url: str) -> FakeWS:
            return FakeWS()

        fake_websockets.connect = connect
        monkeypatch.setitem(sys.modules, "websockets", fake_websockets)

        ps = ws_mod.WSPubSub(port=12352, backend="python", autostart=False)

        def cb(payload) -> None:
            delivered.append(payload)
            if len(delivered) >= 2:
                delivered_evt.set()

        sub = ps.subscribe("scope/topic", cb)
        try:
            assert delivered_evt.wait(3.0)
            assert {"v": 1} in delivered
            assert {"v": 2} in delivered
        finally:
            sub.unsubscribe()

    def test_consume_callback_exception_isolated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio as _aio

        class FakeWS:
            def __init__(self) -> None:
                self._sent = False

            async def __aenter__(self) -> FakeWS:
                return self

            async def __aexit__(self, *a) -> None:
                pass

            async def recv(self) -> str:
                if not self._sent:
                    self._sent = True
                    return json.dumps({"v": 1})
                await _aio.sleep(0.05)
                return json.dumps({"v": 2})

        fake_websockets = types.ModuleType("websockets")
        fake_websockets.connect = lambda url: FakeWS()
        monkeypatch.setitem(sys.modules, "websockets", fake_websockets)

        ps = ws_mod.WSPubSub(port=12353, backend="python", autostart=False)

        def cb(_payload) -> None:
            raise RuntimeError("nope")

        sub = ps.subscribe("scope/topic", cb)
        time.sleep(0.2)  # let it raise at least once
        sub.unsubscribe()


# ----------------------------- autostart spawn server -------------------------


class TestSpawnServer:
    def test_spawn_server_without_fastapi_is_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If FastAPI/uvicorn can't be imported, _spawn_server should warn
        and return without raising.
        """
        monkeypatch.setitem(sys.modules, "upstream_realtime", None)
        monkeypatch.setitem(sys.modules, "uvicorn", None)
        monkeypatch.setitem(sys.modules, "fastapi", None)
        # Force port-in-use to false to push autostart path
        monkeypatch.setattr(ws_mod, "_port_in_use", lambda h, p: False)
        ps = ws_mod.WSPubSub(port=12351, backend="python")
        # Server thread should not be alive
        assert ps._server_thread is None or not ps._server_thread.is_alive()

    def test_spawn_server_router_import_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If FastAPI imports but the router import fails, spawn returns
        without launching a thread."""
        fake_uvicorn = types.ModuleType("uvicorn")

        class _Config:
            def __init__(self, *a, **kw) -> None:
                pass

        class _Server:
            def __init__(self, cfg) -> None:
                pass

            def run(self) -> None:
                pass

        fake_uvicorn.Config = _Config
        fake_uvicorn.Server = _Server

        fake_fastapi = types.ModuleType("fastapi")

        class _App:
            def __init__(self, **kw) -> None:
                pass

            def include_router(self, r) -> None:
                pass

        fake_fastapi.FastAPI = _App

        monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
        monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi)
        # Force router import to fail
        monkeypatch.setitem(sys.modules, "src.api.routes.realtime", None)
        monkeypatch.setitem(sys.modules, "upstream_realtime", None)
        monkeypatch.setattr(ws_mod, "_port_in_use", lambda h, p: False)
        ps = ws_mod.WSPubSub(port=12354, backend="python")
        assert ps._server_thread is None

    def test_spawn_server_starts_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When fastapi/uvicorn/router are all importable, _spawn_server
        starts a daemon thread. We never wait for the port to bind (uses
        the 5s deadline branch but skips by stubbing _port_in_use=True after
        start).
        """
        fake_uvicorn = types.ModuleType("uvicorn")
        started = {"value": False}

        class _Config:
            def __init__(self, *a, **kw) -> None:
                pass

        class _Server:
            def __init__(self, cfg) -> None:
                pass

            def run(self) -> None:
                started["value"] = True
                # Return immediately

        fake_uvicorn.Config = _Config
        fake_uvicorn.Server = _Server

        fake_fastapi = types.ModuleType("fastapi")

        class _App:
            def __init__(self, **kw) -> None:
                pass

            def include_router(self, r) -> None:
                pass

        fake_fastapi.FastAPI = _App

        fake_routes_mod = types.ModuleType("src.api.routes.realtime")
        fake_routes_mod.router = object()

        monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
        monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi)
        monkeypatch.setitem(sys.modules, "src.api.routes.realtime", fake_routes_mod)
        monkeypatch.setitem(sys.modules, "upstream_realtime", None)

        # Have _port_in_use return True immediately so the bind-wait loop
        # exits at once.
        monkeypatch.setattr(ws_mod, "_port_in_use", lambda h, p: True)
        ps = ws_mod.WSPubSub(port=12355, backend="python")
        # The autostart skipped because port-in-use=True. Reset to False so
        # _spawn_server is called explicitly.
        monkeypatch.setattr(
            ws_mod,
            "_port_in_use",
            lambda h, p: True,  # for bind-wait to short-circuit
        )
        ps._spawn_server()
        assert ps._server_thread is not None


# ----------------------------- WSPubSub publish python fallback ---------------


def test_publish_rust_no_server_falls_back_to_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If backend is rust but rust_server is None (failed start), publish
    must use the python httpx code path."""
    ps = ws_mod.WSPubSub(port=12356, backend="python", autostart=False)
    ws_mod.WSPubSub._http_client = None
    # Force the "rust but no server" condition
    ps.backend = "rust"
    ps._rust_server = None

    captured: dict = {}

    class FakeResp:
        def raise_for_status(self):
            pass

    class FakeClient:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url, json):
            captured["url"] = url
            captured["body"] = json
            return FakeResp()

    fake_httpx = SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    ps.publish("scope/topic", {"v": 1})
    assert captured["body"] == {"channel": "scope/topic", "payload": {"v": 1}}
