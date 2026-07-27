"""Regression coverage for the shared chat WebSocket protocol loop."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.websockets import WebSocketDisconnect

import chat.router_factory as shared_router_factory
import chat.websocket_protocol as shared_protocol
import src.api.routes.chat_ws as api_chat_ws


pytestmark = [pytest.mark.anyio, pytest.mark.unit]

_Entrypoint = Callable[[Any, str], Awaitable[None]]


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


class _Session:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.metadata: dict[str, Any] = {}

    def add_message(self, _role: str, _content: str) -> None:
        raise AssertionError("context injection must be disabled in protocol tests")


class _ChatService:
    def __init__(self, *, add_error: bool = False) -> None:
        self.add_error = add_error
        self.added: list[tuple[str, str, str | None]] = []
        self._new_session_count = 0

    def get_or_create_session(self, session_id: str | None) -> _Session:
        if session_id is None:
            self._new_session_count += 1
            return _Session(f"new-{self._new_session_count}")
        return _Session(session_id)

    def get_session_history(self, session_id: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": f"history:{session_id}"}]

    def add_user_message(
        self,
        session_id: str,
        message: str,
        app_context: str | None,
    ) -> None:
        if self.add_error:
            raise ValueError("cannot add")
        self.added.append((session_id, message, app_context))

    async def stream_response(self, _session_id: str):
        yield {"type": "chunk", "content": "dict-chunk"}
        yield "text-chunk"


class _FakeWebSocket:
    def __init__(
        self,
        messages: list[dict[str, Any]],
        chat_service: _ChatService,
        *,
        receive_error: BaseException | None = None,
    ) -> None:
        self._messages = list(messages)
        self._receive_error = receive_error
        self.accepted = False
        self.sent: list[dict[str, Any]] = []
        self.app = SimpleNamespace(state=SimpleNamespace(chat_service=chat_service))

    async def accept(self) -> None:
        self.accepted = True

    async def receive_json(self) -> dict[str, Any]:
        if self._receive_error is not None:
            raise self._receive_error
        if not self._messages:
            raise WebSocketDisconnect(code=1000)
        return self._messages.pop(0)

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


def _router_factory_entrypoint() -> _Entrypoint:
    router = shared_router_factory.create_chat_router(authorize_fn=lambda _ws: True)
    route = next(
        route
        for route in router.routes
        if getattr(route, "path", None) == "/ws/chat/{session_id}"
    )
    return route.endpoint


def _assert_session_info_and_events(
    sent: list[dict[str, Any]],
    expected_events: list[dict[str, Any]],
) -> None:
    """Validate the shared session contract while allowing declared extensions."""
    session_info, *events = sent
    assert session_info["type"] == "session_info"
    assert session_info["session_id"] == "session-1"
    assert set(session_info) <= {"type", "session_id", "capabilities"}
    if "capabilities" in session_info:
        assert session_info["capabilities"] == {"terminal_runtime": False}
    assert events == expected_events


@pytest.fixture(
    params=[
        pytest.param(api_chat_ws.chat_stream, id="api_route"),
        pytest.param(_router_factory_entrypoint(), id="router_factory"),
    ]
)
def entrypoint(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> _Entrypoint:
    monkeypatch.setenv("UPSTREAMDRIFT_SIDEKICK_CONTEXT", "0")

    async def _authorized(_websocket: Any) -> object:
        return object()

    monkeypatch.setattr(api_chat_ws, "resolve_ws_user", _authorized)
    return request.param


async def test_core_protocol_success_path_matches_entrypoints(
    entrypoint: _Entrypoint,
) -> None:
    service = _ChatService()
    websocket = _FakeWebSocket(
        [
            {"action": "send", "message": "hello", "app_context": "desktop"},
            {"action": "history"},
            {"action": "new_session"},
            {"action": "unknown_action"},
        ],
        service,
    )

    await entrypoint(websocket, "session-1")

    assert websocket.accepted is True
    _assert_session_info_and_events(
        websocket.sent,
        [
            {"type": "chunk", "content": "dict-chunk"},
            {"type": "chunk", "content": "text-chunk"},
            {"type": "complete", "session_id": "session-1"},
            {
                "type": "history",
                "messages": [{"role": "user", "content": "history:session-1"}],
            },
            {"type": "session_created", "session_id": "new-1"},
            {"type": "error", "detail": "Unknown action: unknown_action"},
        ],
    )
    assert service.added == [("session-1", "hello", "desktop")]


async def test_add_user_error_matches_entrypoints_and_socket_survives(
    entrypoint: _Entrypoint,
) -> None:
    service = _ChatService(add_error=True)
    websocket = _FakeWebSocket(
        [
            {"action": "send", "message": "hello"},
            {"action": "history"},
        ],
        service,
    )

    await entrypoint(websocket, "session-1")

    _assert_session_info_and_events(
        websocket.sent,
        [
            {"type": "error", "detail": "cannot add"},
            {
                "type": "history",
                "messages": [{"role": "user", "content": "history:session-1"}],
            },
        ],
    )


async def test_connection_error_is_sanitized_for_both_entrypoints(
    entrypoint: _Entrypoint,
) -> None:
    service = _ChatService()
    websocket = _FakeWebSocket([], service, receive_error=OSError("secret"))

    await entrypoint(websocket, "session-1")

    _assert_session_info_and_events(
        websocket.sent,
        [{"type": "error", "detail": "Connection error"}],
    )


def test_protocol_loop_lives_only_in_shared_helper() -> None:
    assert "while True:" not in Path(api_chat_ws.__file__).read_text()
    assert "while True:" not in Path(shared_router_factory.__file__).read_text()
    assert Path(shared_protocol.__file__).read_text().count("while True:") == 1


def test_api_route_imports_parent_owned_protocol_surface() -> None:
    """The Upstream route must depend on the canonical Tools package name."""
    source = Path(api_chat_ws.__file__).read_text(encoding="utf-8")

    assert "from chat.websocket_protocol import (" in source
    assert "from src.shared.python.chat.websocket_protocol import (" not in source
