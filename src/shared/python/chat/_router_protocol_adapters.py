"""Private adapters between the chat router and shared WebSocket protocol."""

from __future__ import annotations

from typing import Any

from fastapi import WebSocket

from .terminal_runtime import TerminalRuntimeError
from .websocket_protocol import ChatWebSocketState


def router_message_context(message: dict[str, Any]) -> Any:
    """Preserve the portable router's app-context precedence."""
    return message.get("app_context") or message.get("engine_context")


def router_session_info_extra(
    websocket: WebSocket,
    _state: ChatWebSocketState,
) -> dict[str, Any]:
    """Preserve capability discovery in the router handshake."""
    return {
        "capabilities": {
            "terminal_runtime": getattr(
                websocket.app.state,
                "terminal_runtime",
                None,
            )
            is not None
        }
    }


def terminal_runtime(websocket: WebSocket) -> Any:
    """Return the configured terminal runtime or fail with its stable error."""
    runtime = getattr(websocket.app.state, "terminal_runtime", None)
    if runtime is None:
        raise TerminalRuntimeError("Terminal runtime is not configured")
    return runtime
