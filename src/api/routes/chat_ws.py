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

from src.shared.python.chat.websocket_protocol import (
    ChatWebSocketState,
    DisconnectLogConfig,
    run_chat_websocket_protocol,
)
from src.api.auth.ws_auth import resolve_ws_user
from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _chat_service_from(holder: Any) -> Any:
    """Extract ChatService from holder (e.g. Request or WebSocket)."""
    if holder is None:
        raise ValueError("holder must be provided")
    app = getattr(holder, "app", None)
    if app is None:
        raise ValueError("holder must have app attribute")
    state = getattr(app, "state", None)
    if state is None:
        raise ValueError("app must have state attribute")
    chat_service = getattr(state, "chat_service", None)
    if chat_service is None:
        raise RuntimeError("ChatService not initialised")
    return chat_service


_INTERNAL_ERROR_DETAIL = "Internal server error"

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


def _session_log_token(session_id: str) -> str:
    """Return a non-reversible token for logging chat session identifiers."""
    if not isinstance(session_id, str):
        raise TypeError("session_id must be a str")
    return _context_section_hash(session_id)


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
async def chat_stream(websocket: WebSocket, session_id: str = "new") -> None:
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
    await run_chat_websocket_protocol(
        websocket,
        session_id,
        authorize_fn=_authorize_chat_websocket,
        chat_service_getter=_chat_service_from,
        before_send=_maybe_inject_chat_context,
        action_handlers={
            "refresh_models": _handle_refresh_models,
            "index_codebase": _handle_index_codebase,
        },
        log=logger,
        disconnect_log=DisconnectLogConfig(
            message="Chat WebSocket disconnected: session_token=%s",
            args_fn=lambda current_session_id: (
                _session_log_token(current_session_id),
            ),
        ),
    )


async def _authorize_chat_websocket(websocket: WebSocket) -> bool:
    return await resolve_ws_user(websocket) is not None


async def _handle_refresh_models(
    websocket: WebSocket,
    _msg: dict[str, Any],
    state: ChatWebSocketState,
) -> None:
    # Tools issue #2547 / PR #2566: poll the configured provider for available
    # models and ship the result over the chat socket so the dock widget can
    # repopulate its dropdown. Issue #7687: failures must surface as frames.
    try:
        payload = await asyncio.to_thread(state.chat_service.refresh_models)
        await websocket.send_json(
            {
                "type": "model_list",
                "models": payload["models"],
                "refreshed_at": payload["refreshed_at"],
            }
        )
    except WebSocketDisconnect:
        raise
    except Exception:  # noqa: BLE001
        logger.exception("Error refreshing models")
        await websocket.send_json({"type": "error", "detail": _INTERNAL_ERROR_DETAIL})


async def _handle_index_codebase(
    websocket: WebSocket,
    _msg: dict[str, Any],
    state: ChatWebSocketState,
) -> None:
    # Tools issue #2549 / PR #2567. Run the codemap rebuild pathway and ship
    # running/final status frames without reimplementing the indexer here.
    await websocket.send_json(
        {
            "type": "index_status",
            "state": "running",
            "files_parsed": 0,
            "symbols_inserted": 0,
        }
    )
    try:
        payload = await state.chat_service.run_codemap_rebuild()
        await websocket.send_json({"type": "index_status", **payload})
    except WebSocketDisconnect:
        raise
    except Exception:  # noqa: BLE001
        logger.exception("Error indexing codebase")
        # The recovery frame is best-effort: the socket may already be torn
        # down, in which case ``send_json`` raises (RuntimeError once the
        # ASGI send channel is closed, or a transport error). Suppress those
        # so the original failure isn't masked by a secondary exception.
        with contextlib.suppress(
            WebSocketDisconnect,
            ConnectionError,
            TimeoutError,
            OSError,
            RuntimeError,
        ):
            await websocket.send_json(
                {
                    "type": "index_status",
                    "state": "error",
                    "detail": _INTERNAL_ERROR_DETAIL,
                }
            )


# ── REST fallback endpoints ──────────────────────────────────────────


@router.get("/chat/context")
async def get_chat_context_endpoint(request: Request) -> dict[str, Any]:
    """Return the live app/engine context visible to the chat assistant.

    Issue #7453: lets the web UI display a "context chip" (engine · model ·
    last run) mirroring the desktop Sidekick's awareness of app state. The
    payload is the shared ``ChatAppContext`` schema; missing services
    degrade to empty/null fields rather than erroring.
    """
    if not (request is not None):
        raise ValueError("request must be provided")
    from src.api.services.chat_app_context import build_chat_app_context

    state = request.app.state
    return build_chat_app_context(
        engine_manager=getattr(state, "engine_manager", None),
        simulation_service=getattr(state, "simulation_service", None),
    ).model_dump()


@router.get("/chat/sessions")
async def list_sessions(request: Request) -> list[dict[str, Any]]:
    """List all active chat sessions."""
    return _chat_service_from(request).list_sessions()  # type: ignore[no-any-return]


@router.get("/chat/sessions/{session_id}/history")
@precondition(
    lambda request, session_id: session_id is not None and len(session_id.strip()) > 0,
    "Session ID must be a non-empty string",
)
async def get_history(request: Request, session_id: str) -> dict[str, Any]:
    """Get message history for a session."""
    if not (request is not None):
        raise ValueError("request must be provided")
    messages = _chat_service_from(request).get_session_history(session_id)
    return {"session_id": session_id, "messages": messages}


_ALLOWED_VOICE_MIMES: frozenset[str] = frozenset(
    {
        "audio/wav",
        "audio/x-wav",
        "audio/webm",
        "audio/ogg",
        "audio/mp4",
        "audio/mpeg",
    }
)
_MAX_VOICE_BYTES: int = 10 * 1024 * 1024


@router.get("/chat/voice/status")
async def get_chat_voice_status() -> Any:
    """Return local Whisper voice-to-text availability (Cache-Control: no-store)."""
    from fastapi.responses import JSONResponse

    payload: dict[str, Any] = {
        "available": False,
        "engine": "none",
        "model": "base.en",
        "max_bytes": _MAX_VOICE_BYTES,
        "allowed_mimes": sorted(_ALLOWED_VOICE_MIMES),
    }
    try:
        from src.shared.python.sidekick.voice import VoiceTranscriber

        capability = VoiceTranscriber().detect_capability()
        payload.update(
            {
                "available": capability.available,
                "engine": capability.engine,
                "model": capability.model,
            }
        )
    except ImportError:
        import importlib.util

        if importlib.util.find_spec("faster_whisper") is not None:
            payload.update({"available": True, "engine": "faster-whisper"})
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})


@router.post("/chat/voice/transcribe")
async def transcribe_chat_voice(request: Request) -> Any:
    """Transcribe uploaded audio locally via Whisper for composer insertion."""
    from fastapi.responses import JSONResponse

    content_type = (
        (
            request.headers.get("x-audio-mime")
            or request.headers.get("content-type", "audio/wav")
        )
        .split(";")[0]
        .strip()
        .lower()
    )
    raw_bytes = await request.body()
    if not raw_bytes:
        return JSONResponse(
            status_code=400,
            content={"detail": "Audio payload must not be empty."},
            headers={"Cache-Control": "no-store"},
        )
    if len(raw_bytes) > _MAX_VOICE_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": "Audio payload exceeds 10 MiB limit."},
            headers={"Cache-Control": "no-store"},
        )
    if content_type not in _ALLOWED_VOICE_MIMES:
        return JSONResponse(
            status_code=415,
            content={"detail": f"Unsupported audio MIME type: {content_type}"},
            headers={"Cache-Control": "no-store"},
        )
    try:
        from src.shared.python.sidekick.voice import (
            VoiceTranscriber,
            VoiceUnavailableError,
        )

        res = VoiceTranscriber().transcribe_bytes(raw_bytes, mime_type=content_type)
        body = {
            "text": res.text,
            "duration_s": res.duration_s,
            "language": getattr(res, "language", None),
            "engine": res.engine,
            "auto_send": False,
        }
        return JSONResponse(content=body, headers={"Cache-Control": "no-store"})
    except ImportError:
        return JSONResponse(
            status_code=503,
            content={"detail": "Local Whisper engine is not installed."},
            headers={"Cache-Control": "no-store"},
        )
    except VoiceUnavailableError:
        return JSONResponse(
            status_code=503,
            content={"detail": "Local Whisper engine is not installed."},
            headers={"Cache-Control": "no-store"},
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)},
            headers={"Cache-Control": "no-store"},
        )
