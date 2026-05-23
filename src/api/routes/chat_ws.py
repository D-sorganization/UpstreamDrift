"""WebSocket and REST routes for AI chat streaming.

Context injection
-----------------
When ``UPSTREAMDRIFT_SIDEKICK_CONTEXT`` is **not** set to ``"0"``,
:func:`_maybe_inject_chat_context` prepends a compact "recent app state"
system message into the session before the assistant replies.

Privacy / size cap
    The payload is built by :mod:`src.shared.python.ai.chat_context` which
    strips keys and values matching secrets/PII patterns and caps the
    serialised size at ~4 KB (last N events).  No file paths or credentials
    reach the assistant.

Deduplication
    A truncated SHA-256 digest is stored in ``session.metadata[_CONTEXT_HASH_KEY]``
    so that unchanged app state is not re-injected on every consecutive send.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()
_INTERNAL_ERROR_DETAIL = "Internal server error"
_CONNECTION_ERROR_DETAIL = "Connection error"

# ── Chat-context injection helpers ───────────────────────────────────

#: ``session.metadata`` key used to store the last-seen context digest.
_CONTEXT_HASH_KEY: str = "_sidekick_ctx_hash"

#: Environment-variable name: set to "0" to disable context injection.
_SIDEKICK_CONTEXT_ENV: str = "UPSTREAMDRIFT_SIDEKICK_CONTEXT"


def _context_section_hash(section: str) -> str:
    """Return a 16-character hex digest of *section*.

    Args:
        section: The formatted context section string.

    Returns:
        16-character lowercase hex string (truncated SHA-256).

    Raises:
        TypeError: If *section* is not a :class:`str`.
    """
    if not isinstance(section, str):
        raise TypeError("section must be a str")
    return hashlib.sha256(section.encode("utf-8")).hexdigest()[:16]


def _maybe_inject_chat_context(session: Any) -> str | None:
    """Inject recent app-state context into *session* as a system message.

    Context injection is skipped when:
    * ``UPSTREAMDRIFT_SIDEKICK_CONTEXT=0`` is set.
    * The ring buffer is empty.
    * *session* does not expose an ``add_message(role, content)`` method.
    * The context payload is identical to the last-injected payload
      (deduplication via ``session.metadata[_CONTEXT_HASH_KEY]``).

    Args:
        session: Chat session object.  Must expose ``add_message(role, content)``.
            Optionally exposes a ``metadata`` :class:`dict` for deduplication.

    Returns:
        The injected section string, or ``None`` when injection is skipped.

    Postconditions:
        * If a non-``None`` value is returned, ``session.add_message`` has been
          called exactly once with ``role="system"`` and ``content=<return value>``.
        * The 16-character digest is stored in ``session.metadata[_CONTEXT_HASH_KEY]``
          when ``session.metadata`` is a mutable mapping.
    """
    if os.environ.get(_SIDEKICK_CONTEXT_ENV) == "0":
        return None

    add_message = getattr(session, "add_message", None)
    if not callable(add_message):
        return None

    try:
        from src.shared.python.ai.chat_context import (
            format_context_section,
            get_chat_context,
        )
    except ImportError:
        logger.debug("chat_context unavailable — skipping context injection")
        return None

    payload = get_chat_context()
    section = format_context_section(payload)
    if not section:
        return None
    section_text = str(section)

    # Deduplication: skip if state unchanged since last injection.
    digest = _context_section_hash(section_text)
    metadata: dict[str, Any] | None = getattr(session, "metadata", None)
    if isinstance(metadata, dict):
        if metadata.get(_CONTEXT_HASH_KEY) == digest:
            return None
        metadata[_CONTEXT_HASH_KEY] = digest

    add_message("system", section_text)
    return section_text


@router.websocket("/ws/chat/{session_id}")
async def chat_stream(websocket: WebSocket, session_id: str = "new") -> None:  # noqa: C901
    """Stream AI chat over WebSocket.

    Protocol:
        Client -> Server:
            {"action": "send", "message": "...", "engine_context": "mujoco"}
            {"action": "history"}
            {"action": "new_session"}
            {"action": "refresh_models"}
            {"action": "index_codebase"}

        Server -> Client:
            {"type": "session_info", "session_id": "..."}
            {"type": "chunk", "content": "..."}
            {"type": "complete", "session_id": "..."}
            {"type": "history", "messages": [...]}
            {"type": "model_list", "models": [...], "refreshed_at": "..."}
            {"type": "index_status", "state": "running"|"complete"|"error", ...}
            {"type": "error", "detail": "..."}
    """
    if not (websocket is not None):
        raise ValueError("websocket must be provided")
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

                # Accept both keys: React clients send ``engine_context``;
                # PyQt clients send ``app_context``. Mirrors router_factory.py.
                engine_context = msg.get("engine_context") or msg.get("app_context")

                # Inject recent app-state context into the session before the
                # assistant replies (skipped when env var is "0" or buffer empty).
                _maybe_inject_chat_context(ctx)

                try:
                    chat_service.add_user_message(
                        session_id, user_message, engine_context
                    )
                except ValueError as e:
                    await websocket.send_json({"type": "error", "detail": str(e)})
                    continue

                # Stream response chunks
                try:
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
                except Exception:  # noqa: BLE001
                    logger.exception("Error during streaming response")
                    await websocket.send_json(
                        {"type": "error", "detail": _INTERNAL_ERROR_DETAIL}
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

            elif action == "refresh_models":
                # Tools issue #2547 / PR #2566: poll the configured provider
                # for available models and ship the result over the chat
                # socket so the dock widget can repopulate its dropdown.
                payload = await asyncio.to_thread(chat_service.refresh_models)
                await websocket.send_json(
                    {
                        "type": "model_list",
                        "models": payload["models"],
                        "refreshed_at": payload["refreshed_at"],
                    }
                )

            elif action == "index_codebase":
                # Tools issue #2549 / PR #2567. Run the existing
                # codemap.indexer.rebuild pathway in a worker thread and
                # ship a ``running`` event up front, then a final
                # ``complete`` (or ``error``) event when the rebuild
                # finishes. We deliberately don't reimplement the
                # indexer here — see ``src/shared/python/codemap``.
                await websocket.send_json(
                    {
                        "type": "index_status",
                        "state": "running",
                        "files_parsed": 0,
                        "symbols_inserted": 0,
                    }
                )
                payload = await chat_service.run_codemap_rebuild()
                await websocket.send_json({"type": "index_status", **payload})

            else:
                await websocket.send_json(
                    {"type": "error", "detail": f"Unknown action: {action}"}
                )

    except WebSocketDisconnect:
        logger.debug("Chat WebSocket disconnected")
    except (ConnectionError, TimeoutError, OSError):
        logger.exception("Chat WebSocket connection error")
        with contextlib.suppress(ConnectionError, TimeoutError, OSError):
            await websocket.send_json(
                {"type": "error", "detail": _CONNECTION_ERROR_DETAIL}
            )


# ── REST fallback endpoints ──────────────────────────────────────────


@router.get("/chat/sessions")
async def list_sessions(request: Request) -> list[dict[str, Any]]:
    """List all active chat sessions."""
    return request.app.state.chat_service.list_sessions()  # type: ignore[no-any-return]


@router.get("/chat/sessions/{session_id}/history")
@precondition(
    lambda request, session_id: session_id is not None and len(session_id.strip()) > 0,
    "Session ID must be a non-empty string",
)
async def get_history(request: Request, session_id: str) -> dict[str, Any]:
    """Get message history for a session."""
    if not (request is not None):
        raise ValueError("request must be provided")
    messages = request.app.state.chat_service.get_session_history(session_id)
    return {"session_id": session_id, "messages": messages}
