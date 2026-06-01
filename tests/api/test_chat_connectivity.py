"""TDD coverage for the desktop-launcher chat-connectivity contract.

These tests guard the four invariants that broke the chat in PR #6239:

1. **WS route is reachable at every public prefix** — the chat dock defaults
   to ``/api/ws/chat/{session_id}``; the API server must mount that path in
   addition to ``""`` and ``/api/v1``. (DRY-mounted via ``_PUBLIC_PREFIXES``.)
2. **Local mode bypasses Bearer auth** — when ``GOLF_SUITE_MODE=local`` (or
   ``GOLF_AUTH_DISABLED=true``) the WS handshake must succeed without a
   token. The desktop launcher relies on this because it has no browser
   to obtain a token from.
3. **Stream-response surfaces timeouts as structured errors** — if the
   provider thread produces nothing in time, the consumer must receive a
   structured ``{"type": "error", ...}`` chunk instead of the previous
   silent ``queue.Empty`` → "Internal server error" pathway.
4. **The chat dock default WS URL honours ``GOLF_API_PORT``** — the
   launcher probes a free port; without env-driven default the client
   would dial the wrong port on a machine where 8000 is already in use.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Invariant 4 — WS URL default reads ``GOLF_API_PORT`` / ``API_PORT``
# ---------------------------------------------------------------------------


def test_chat_dock_default_server_honours_golf_api_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOLF_API_PORT", "8137")
    monkeypatch.delenv("UD_CHAT_WS_URL", raising=False)

    from src.shared.python.chat import chat_dock_widget

    assert chat_dock_widget._resolve_default_server() == "ws://127.0.0.1:8137"


def test_chat_dock_default_server_explicit_override_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UD_CHAT_WS_URL", "wss://chat.example/ws")
    monkeypatch.setenv("GOLF_API_PORT", "9999")

    from src.shared.python.chat import chat_dock_widget

    assert chat_dock_widget._resolve_default_server() == "wss://chat.example/ws"


def test_chat_dock_default_server_falls_back_to_8000(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in ("UD_CHAT_WS_URL", "GOLF_API_PORT", "API_PORT", "GOLF_PORT"):
        monkeypatch.delenv(var, raising=False)

    from src.shared.python.chat import chat_dock_widget

    assert chat_dock_widget._resolve_default_server() == "ws://127.0.0.1:8000"


def test_chat_dock_default_server_rejects_invalid_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOLF_API_PORT", "not-a-port")
    monkeypatch.delenv("UD_CHAT_WS_URL", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)
    monkeypatch.delenv("GOLF_PORT", raising=False)

    from src.shared.python.chat import chat_dock_widget

    assert chat_dock_widget._resolve_default_server() == "ws://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# Invariant 2 — Local-mode WS auth bypass
# ---------------------------------------------------------------------------


def test_is_auth_disabled_picks_up_GOLF_SUITE_MODE_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOLF_SUITE_MODE", "local")
    monkeypatch.delenv("GOLF_AUTH_DISABLED", raising=False)

    from src.shared.python.config.environment import is_auth_disabled

    assert is_auth_disabled() is True


def test_is_auth_disabled_default_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOLF_SUITE_MODE", raising=False)
    monkeypatch.delenv("GOLF_AUTH_DISABLED", raising=False)

    from src.shared.python.config.environment import is_auth_disabled

    assert is_auth_disabled() is False


# ---------------------------------------------------------------------------
# Invariant 1 — WS route is mounted at every public prefix
# ---------------------------------------------------------------------------


_EXPECTED_WS_PATHS: tuple[str, ...] = (
    "/ws/chat/{session_id}",
    "/api/ws/chat/{session_id}",
    "/api/v1/ws/chat/{session_id}",
)


def _collect_ws_paths(app: Any) -> set[str]:
    return {
        route.path
        for route in app.routes
        if type(route).__name__ == "APIWebSocketRoute"
    }


def test_chat_ws_mounted_at_every_public_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOLF_SUITE_MODE", "local")

    from src.api.server import app

    ws_paths = _collect_ws_paths(app)
    missing = [p for p in _EXPECTED_WS_PATHS if p not in ws_paths]
    assert not missing, (
        f"chat_ws is missing from prefixes: {missing}. Got: {sorted(ws_paths)}"
    )


# ---------------------------------------------------------------------------
# Invariant — LOD helper: state lookup goes through a single accessor
# ---------------------------------------------------------------------------


def test_chat_service_from_returns_attached_service() -> None:
    from src.api.routes.chat_ws import _chat_service_from

    class _State:
        chat_service = object()

    class _App:
        state = _State()

    class _Holder:
        app = _App()

    assert _chat_service_from(_Holder()) is _State.chat_service


def test_chat_service_from_raises_when_unset() -> None:
    from src.api.routes.chat_ws import _chat_service_from

    class _App:
        class state:  # noqa: N801 - mimic FastAPI ``app.state``
            pass

    class _Holder:
        app = _App()

    with pytest.raises(RuntimeError, match="ChatService not initialised"):
        _chat_service_from(_Holder())


def test_chat_service_from_rejects_none_holder() -> None:
    from src.api.routes.chat_ws import _chat_service_from

    with pytest.raises(ValueError, match="holder must be provided"):
        _chat_service_from(None)


# ---------------------------------------------------------------------------
# Invariant 3 — Stream timeout surfaces a structured error chunk
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_response_yields_timeout_error_when_queue_stays_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Producer thread that dies silently must not hang the consumer.

    Construct a ChatService with a stub adapter whose ``stream_response``
    blocks forever (simulating a wedged provider). The two-stage timeout
    in ``stream_response`` must yield a structured error chunk in well
    under the historical 60s queue.Empty timeout.
    """
    from src.api.services.chat_service import ChatService

    class _BlockingAdapter:
        def stream_response(self, *_a: Any, **_kw: Any) -> Iterator[Any]:
            import threading

            threading.Event().wait()  # block forever
            yield  # pragma: no cover - unreachable

    svc = ChatService()
    svc._adapter = _BlockingAdapter()
    ctx = svc.get_or_create_session(None)
    svc.add_user_message(ctx.session_id, "hi")

    # Shrink the timeouts so the test completes quickly.
    monkeypatch.setattr(
        "src.api.services.chat_service.threading.Thread.join",
        lambda self, timeout=None: None,
        raising=False,
    )

    # Patch the timeouts at the function scope via local symbols.
    # The cleanest hook is to monkey-patch ``asyncio.to_thread`` so the
    # first call raises ``queue.Empty`` immediately — preserving the
    # production code path while bounding test runtime.
    import asyncio
    from queue import Empty

    real_to_thread = asyncio.to_thread

    async def _fast_to_thread(func: Any, *args: Any, **kw: Any) -> Any:
        # Simulate the queue staying empty past the timeout window.
        if getattr(func, "__name__", "") == "get":
            raise Empty
        return await real_to_thread(func, *args, **kw)

    monkeypatch.setattr("asyncio.to_thread", _fast_to_thread)

    chunks: list[Any] = []
    async for chunk in svc.stream_response(ctx.session_id):
        chunks.append(chunk)
        if isinstance(chunk, dict) and chunk.get("type") == "error":
            break

    assert chunks, "expected at least one chunk before the stream closes"
    last = chunks[-1]
    assert isinstance(last, dict), f"expected dict error chunk, got {type(last)}"
    assert last.get("type") == "error", f"expected type=error, got {last}"
    assert "provider" in last.get("detail", "").lower(), (
        f"expected human-readable provider hint in detail, got {last!r}"
    )


# ---------------------------------------------------------------------------
# Invariant — get_or_create_session postcondition
# ---------------------------------------------------------------------------


def test_get_or_create_session_postcondition_registered_in_sessions() -> None:
    from src.api.services.chat_service import ChatService

    svc = ChatService()
    ctx = svc.get_or_create_session(None)
    assert ctx is not None
    assert ctx.session_id
    assert ctx.session_id in svc._sessions


@pytest.mark.asyncio
async def test_stream_cancellation_pairs_unexecuted_tool_calls() -> None:
    from src.api.services.chat_service import ChatService
    from src.shared.python.ai.types import AgentChunk, ToolResult

    release_first_tool = threading.Event()

    class TwoToolAdapter:
        def stream_response(self, *_a: Any, **_kw: Any) -> Iterator[AgentChunk]:
            yield AgentChunk(
                tool_call_delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "tool-1",
                            "function": {"name": "one", "arguments": "{}"},
                        },
                        {
                            "index": 1,
                            "id": "tool-2",
                            "function": {"name": "two", "arguments": "{}"},
                        },
                    ]
                }
            )

    class BlockingRegistry:
        def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
            assert name == "one"
            release_first_tool.wait(timeout=1.0)
            time.sleep(0.05)
            return ToolResult(tool_call_id="tool-1", success=True, result="done")

    svc = ChatService()
    svc._adapter = TwoToolAdapter()  # type: ignore[assignment]
    svc._tool_registry = BlockingRegistry()  # type: ignore[assignment]
    svc._persist_session = lambda session_id: None  # type: ignore[method-assign]
    ctx = svc.get_or_create_session(None)
    ctx.add_user_message("run tools")

    stream = svc.stream_response(ctx.session_id)
    first = await stream.__anext__()
    assert first == {"type": "tool_call_started", "tool": "one"}

    release_first_tool.set()
    await stream.aclose()

    tool_results = [msg for msg in ctx.messages if msg.role == "tool"]
    assert [msg.tool_call_id for msg in tool_results] == ["tool-1", "tool-2"]
    assert "disconnected" in tool_results[1].content
