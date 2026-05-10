"""Shared WebSocket router factory for AI chat streaming.

Creates a FastAPI APIRouter with the standard chat WebSocket protocol
that any application can mount. The router delegates to a
``ChatServiceBase`` subclass provided via ``app.state.chat_service``.

Protocol (identical for all consumers):
    Client -> Server:
        {"action": "send", "message": "...", "app_context": "gasification"}
        {"action": "history"}
        {"action": "new_session"}

    Server -> Client:
        {"type": "session_info", "session_id": "..."}
        {"type": "chunk", "content": "..."}
        {"type": "complete", "session_id": "..."}
        {"type": "history", "messages": [...]}
        {"type": "error", "detail": "..."}

Usage::

    from chat.router_factory import create_chat_router

    router = create_chat_router()
    app.include_router(router, prefix="/api")

This module has ZERO application-specific imports.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


def create_chat_router(
    prefix: str = "",
    authorize_fn: Any = None,
) -> APIRouter:
    """Create a chat WebSocket + REST router.

    Args:
        prefix: Optional URL prefix for routes.
        authorize_fn: Optional async callable ``(WebSocket) -> bool``
            that returns False to reject the connection. If None, all
            connections are accepted.

    Returns:
        Configured FastAPI APIRouter.
    """
    router = APIRouter(prefix=prefix)

    @router.websocket("/ws/chat/{session_id}")
    async def chat_stream(  # noqa: C901
        websocket: WebSocket,
        session_id: str = "new",
    ) -> None:
        """Stream AI chat over WebSocket."""
        # Optional authorization hook
        if authorize_fn is not None and not await authorize_fn(websocket):
            return

        await websocket.accept()

        chat_service = websocket.app.state.chat_service

        # Resolve or create session
        if session_id == "new":
            session = chat_service.get_or_create_session(None)
            session_id = session.session_id
        else:
            session = chat_service.get_or_create_session(session_id)
            session_id = session.session_id

        await websocket.send_json({"type": "session_info", "session_id": session_id})

        try:
            while True:
                msg = await websocket.receive_json()
                action = msg.get("action")

                if action == "send":
                    user_message = msg.get("message", "").strip()
                    if not user_message:
                        await websocket.send_json(
                            {"type": "error", "detail": "Empty message"}
                        )
                        continue

                    app_context = msg.get("app_context") or msg.get("engine_context")

                    try:
                        chat_service.add_user_message(
                            session_id, user_message, app_context
                        )
                    except ValueError as e:
                        await websocket.send_json({"type": "error", "detail": str(e)})
                        continue

                    # Stream response chunks
                    async for chunk in chat_service.stream_response(session_id):
                        if isinstance(chunk, dict):
                            await websocket.send_json(chunk)
                        else:
                            await websocket.send_json(
                                {"type": "chunk", "content": str(chunk)}
                            )

                    await websocket.send_json(
                        {"type": "complete", "session_id": session_id}
                    )

                elif action == "history":
                    messages = chat_service.get_session_history(session_id)
                    await websocket.send_json({"type": "history", "messages": messages})

                elif action == "new_session":
                    session = chat_service.get_or_create_session(None)
                    session_id = session.session_id
                    await websocket.send_json(
                        {"type": "session_created", "session_id": session_id}
                    )

                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "detail": f"Unknown action: {action}",
                        }
                    )

        except WebSocketDisconnect:
            logger.debug("Chat WebSocket disconnected: session=%s", session_id)
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("Chat WebSocket error: %s", e)
            with contextlib.suppress(ConnectionError, TimeoutError, OSError):
                await websocket.send_json({"type": "error", "detail": str(e)})

    # ── REST fallback endpoints ──────────────────────────────────────

    @router.get("/chat/sessions")
    async def list_sessions(request: Request) -> list[dict[str, Any]]:
        """List all active chat sessions."""
        return request.app.state.chat_service.list_sessions()  # type: ignore[no-any-return]

    @router.get("/chat/sessions/{session_id}/history")
    async def get_history(request: Request, session_id: str) -> dict[str, Any]:
        """Get message history for a session."""
        if not session_id or not session_id.strip():
            return {"session_id": session_id, "messages": []}
        messages = request.app.state.chat_service.get_session_history(session_id)
        return {"session_id": session_id, "messages": messages}

    return router
