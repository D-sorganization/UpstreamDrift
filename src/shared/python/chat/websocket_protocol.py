"""Shared chat WebSocket protocol loop.

Application routes and the portable router factory speak the same core chat
protocol. Host-specific behavior is supplied through hooks so this module
remains reusable and has no application imports.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

__all__ = [
    "ChatWebSocketState",
    "DisconnectLogConfig",
    "run_chat_websocket_protocol",
]

_INTERNAL_ERROR_DETAIL = "Internal server error"
_CONNECTION_ERROR_DETAIL = "Connection error"


@dataclass
class ChatWebSocketState:
    """Mutable per-connection chat session state."""

    chat_service: Any
    session_id: str
    session: Any


AuthorizeFn: TypeAlias = Callable[[WebSocket], Awaitable[Any] | Any]
BeforeSendHook: TypeAlias = Callable[[Any], Awaitable[Any] | Any]
ChatServiceGetter: TypeAlias = Callable[[WebSocket], Any]
DisconnectLogArgsFn: TypeAlias = Callable[[str], tuple[Any, ...]]
MessageContextGetter: TypeAlias = Callable[[dict[str, Any]], Any]
SessionInfoExtraFn: TypeAlias = Callable[
    [WebSocket, ChatWebSocketState],
    Awaitable[Mapping[str, Any]] | Mapping[str, Any],
]
ActionHandler: TypeAlias = Callable[
    [WebSocket, dict[str, Any], ChatWebSocketState],
    Awaitable[None],
]


@dataclass(frozen=True)
class DisconnectLogConfig:
    """Route-specific disconnect logging behavior."""

    message: str = "Chat WebSocket disconnected"
    args_fn: DisconnectLogArgsFn | None = None


_DEFAULT_DISCONNECT_LOG = DisconnectLogConfig()


async def _maybe_await(value: Awaitable[Any] | Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _default_chat_service_getter(websocket: WebSocket) -> Any:
    return websocket.app.state.chat_service


def _default_message_context(message: dict[str, Any]) -> Any:
    return message.get("engine_context") or message.get("app_context")


async def run_chat_websocket_protocol(
    websocket: WebSocket,
    session_id: str = "new",
    *,
    authorize_fn: AuthorizeFn | None = None,
    chat_service_getter: ChatServiceGetter = _default_chat_service_getter,
    before_send: BeforeSendHook | None = None,
    action_handlers: Mapping[str, ActionHandler] | None = None,
    message_context_getter: MessageContextGetter = _default_message_context,
    session_info_extra: SessionInfoExtraFn | None = None,
    log: logging.Logger = logger,
    disconnect_log: DisconnectLogConfig = _DEFAULT_DISCONNECT_LOG,
) -> None:
    """Run the shared chat WebSocket protocol.

    Route-specific behavior belongs in injected hooks. The shared skeleton
    owns authorization, acceptance, session setup, core actions, custom action
    dispatch, streaming error handling, and transport error handling.
    """
    if websocket is None:
        raise ValueError("websocket must be provided")

    if authorize_fn is not None and not await _maybe_await(authorize_fn(websocket)):
        return

    await websocket.accept()

    chat_service = chat_service_getter(websocket)
    session = chat_service.get_or_create_session(
        None if session_id == "new" else session_id
    )
    state = ChatWebSocketState(
        chat_service=chat_service,
        session_id=session.session_id,
        session=session,
    )

    session_info: dict[str, Any] = {
        "type": "session_info",
        "session_id": state.session_id,
    }
    if session_info_extra is not None:
        session_info.update(
            dict(await _maybe_await(session_info_extra(websocket, state)))
        )
        session_info["type"] = "session_info"
        session_info["session_id"] = state.session_id
    await websocket.send_json(session_info)

    try:
        while True:
            message = await websocket.receive_json()
            action = message.get("action")

            if action == "send":
                await _handle_send_action(
                    websocket,
                    message,
                    state,
                    before_send,
                    message_context_getter,
                    log,
                )
            elif action == "history":
                messages = chat_service.get_session_history(state.session_id)
                await websocket.send_json({"type": "history", "messages": messages})
            elif action == "new_session":
                state.session = chat_service.get_or_create_session(None)
                state.session_id = state.session.session_id
                await websocket.send_json(
                    {"type": "session_created", "session_id": state.session_id}
                )
            elif action_handlers is not None and action in action_handlers:
                await action_handlers[action](websocket, message, state)
            else:
                await websocket.send_json(
                    {"type": "error", "detail": f"Unknown action: {action}"}
                )
    except WebSocketDisconnect:
        args = (
            disconnect_log.args_fn(state.session_id)
            if disconnect_log.args_fn is not None
            else ()
        )
        log.debug(disconnect_log.message, *args)
    except (ConnectionError, TimeoutError, OSError):
        log.exception("Chat WebSocket connection error")
        with contextlib.suppress(ConnectionError, TimeoutError, OSError):
            await websocket.send_json(
                {"type": "error", "detail": _CONNECTION_ERROR_DETAIL}
            )


async def _handle_send_action(
    websocket: WebSocket,
    message: dict[str, Any],
    state: ChatWebSocketState,
    before_send: BeforeSendHook | None,
    message_context_getter: MessageContextGetter,
    log: logging.Logger,
) -> None:
    user_message = message.get("message", "").strip()
    if not user_message:
        await websocket.send_json({"type": "error", "detail": "Empty message"})
        return

    if before_send is not None:
        await _maybe_await(before_send(state.session))

    try:
        state.chat_service.add_user_message(
            state.session_id,
            user_message,
            message_context_getter(message),
        )
    except ValueError as exc:
        await websocket.send_json({"type": "error", "detail": str(exc)})
        return

    try:
        async for chunk in state.chat_service.stream_response(state.session_id):
            if isinstance(chunk, dict):
                await websocket.send_json(chunk)
            else:
                await websocket.send_json({"type": "chunk", "content": str(chunk)})
        await websocket.send_json({"type": "complete", "session_id": state.session_id})
    except WebSocketDisconnect:
        raise
    except Exception:  # noqa: BLE001
        log.exception("Error during streaming response")
        await websocket.send_json({"type": "error", "detail": _INTERNAL_ERROR_DETAIL})
