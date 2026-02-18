"""WebSocket and REST routes for AI chat streaming."""

from __future__ import annotations

import contextlib

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.websocket("/ws/chat/{session_id}")
async def chat_stream(websocket: WebSocket, session_id: str = "new") -> None:
    """Stream AI chat over WebSocket.

    Protocol:
        Client -> Server:
            {"action": "send", "message": "...", "engine_context": "mujoco"}
            {"action": "history"}
            {"action": "new_session"}

        Server -> Client:
            {"type": "session_info", "session_id": "..."}
            {"type": "chunk", "content": "..."}
            {"type": "complete", "session_id": "..."}
            {"type": "history", "messages": [...]}
            {"type": "error", "detail": "..."}
    """
    await websocket.accept()

    chat_service = websocket.app.state.chat_service

    # Resolve or create session
    if session_id == "new":
        ctx = chat_service.get_or_create_session(None)
        session_id = ctx.session_id
    else:
        ctx = chat_service.get_or_create_session(session_id)
        session_id = ctx.session_id

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

                engine_context = msg.get("engine_context")

                try:
                    chat_service.add_user_message(
                        session_id, user_message, engine_context
                    )
                except ValueError as e:
                    await websocket.send_json({"type": "error", "detail": str(e)})
                    continue

                # Stream response chunks
                async for chunk in chat_service.stream_response(session_id):
                    await websocket.send_json({"type": "chunk", "content": chunk})

                await websocket.send_json(
                    {"type": "complete", "session_id": session_id}
                )

            elif action == "history":
                messages = chat_service.get_session_history(session_id)
                await websocket.send_json({"type": "history", "messages": messages})

            elif action == "new_session":
                ctx = chat_service.get_or_create_session(None)
                session_id = ctx.session_id
                await websocket.send_json(
                    {"type": "session_created", "session_id": session_id}
                )

            else:
                await websocket.send_json(
                    {"type": "error", "detail": f"Unknown action: {action}"}
                )

    except WebSocketDisconnect:
        logger.debug("Chat WebSocket disconnected: session=%s", session_id)
    except (ConnectionError, TimeoutError, OSError) as e:
        logger.error("Chat WebSocket error: %s", e)
        with contextlib.suppress(ConnectionError, TimeoutError, OSError):
            await websocket.send_json({"type": "error", "detail": str(e)})


# ── REST fallback endpoints ──────────────────────────────────────────


@router.get("/chat/sessions")
async def list_sessions(request: Request) -> list[dict]:
    """List all active chat sessions."""
    return request.app.state.chat_service.list_sessions()  # type: ignore[no-any-return]


@router.get("/chat/sessions/{session_id}/history")
@precondition(
    lambda request, session_id: session_id is not None and len(session_id.strip()) > 0,
    "Session ID must be a non-empty string",
)
async def get_history(request: Request, session_id: str) -> dict:
    """Get message history for a session."""
    messages = request.app.state.chat_service.get_session_history(session_id)
    return {"session_id": session_id, "messages": messages}
