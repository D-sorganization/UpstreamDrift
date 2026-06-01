"""FastAPI routes for the realtime IPC layer (issue #4997).

Provides:

- ``POST /realtime/publish`` — body ``{channel, payload}``; broadcasts to all
  WebSocket subscribers on that channel.
- ``WS /realtime/subscribe?channel=...`` — keeps the connection open and
  pushes JSON messages as they are published.

Subscriber bookkeeping is in-memory and protected by an :class:`asyncio.Lock`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from src.api.auth.ws_auth import resolve_ws_user
from src.api.rate_limit import get_limit, limiter
from src.shared.python.realtime.protocol import validate_channel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["realtime"])

# Upper bound on a channel name. Channel names are short ``scope/topic``
# identifiers; this cap rejects pathological inputs before validation.
_MAX_CHANNEL_LENGTH = 256

# Upper bound on the JSON-serialized publish payload. A single publish is
# fanned out to every subscriber, so an unbounded payload is an amplification
# vector (issue #6928). 256 KiB is generous for realtime control messages.
_MAX_PAYLOAD_BYTES = 256 * 1024


class PublishRequest(BaseModel):
    """Request body for ``POST /realtime/publish``."""

    channel: str = Field(
        ...,
        max_length=_MAX_CHANNEL_LENGTH,
        description="Channel name (scope/topic pattern)",
    )
    payload: dict[str, Any] = Field(default_factory=dict)


class _SubscriberRegistry:
    """In-memory mapping of channel name → set of WebSocket connections."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._subs: dict[str, set[WebSocket]] = {}

    async def add(self, channel: str, ws: WebSocket) -> None:
        async with self._lock:
            self._subs.setdefault(channel, set()).add(ws)

    async def remove(self, channel: str, ws: WebSocket) -> None:
        async with self._lock:
            bucket = self._subs.get(channel)
            if bucket is None:
                return
            bucket.discard(ws)
            if not bucket:
                self._subs.pop(channel, None)

    async def snapshot(self, channel: str) -> list[WebSocket]:
        async with self._lock:
            bucket = self._subs.get(channel)
            return list(bucket) if bucket else []


# Module-level registry: shared across all requests served by this app.
_registry = _SubscriberRegistry()


@router.post("/realtime/publish")
@limiter.limit(get_limit("API_LIMIT_REALTIME_PUBLISH", "60/minute"))
async def publish(request: Request, req: PublishRequest) -> dict[str, Any]:
    """Broadcast ``req.payload`` to all subscribers of ``req.channel``.

    Args:
        request: FastAPI request object (used by the rate limiter).
        req: Publish body with the target ``channel`` and ``payload``.
    """
    try:
        validate_channel(req.channel)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Amplification guard (issue #6928): a single publish fans out to every
    # subscriber, so cap the serialized payload size. ``default=str`` mirrors
    # the lenient encoding used when broadcasting non-JSON-native values.
    serialized = json.dumps(req.payload, default=str)
    if len(serialized.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(f"payload exceeds maximum size of {_MAX_PAYLOAD_BYTES} bytes"),
        )

    targets = await _registry.snapshot(req.channel)
    delivered = 0
    for ws in targets:
        try:
            await ws.send_json(req.payload)
            delivered += 1
        except Exception:  # noqa: BLE001
            # Best-effort: drop dead sockets but do not fail the publish.
            logger.debug("realtime: dropping dead subscriber on %s", req.channel)
            await _registry.remove(req.channel, ws)

    return {"channel": req.channel, "delivered": delivered}


@router.websocket("/realtime/subscribe")
async def subscribe(websocket: WebSocket, channel: str) -> None:
    """Stream messages published to ``channel`` to the connected client."""
    try:
        validate_channel(channel)
    except ValueError:
        await websocket.close(code=1008)  # policy violation
        return

    user = await resolve_ws_user(websocket)
    if user is None:
        return
    await websocket.accept()
    await _registry.add(channel, websocket)
    try:
        # Keep the connection open until the client disconnects. We do not
        # expect inbound traffic, but draining the receive queue prevents the
        # transport from buffering indefinitely.
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        logger.debug("realtime: subscriber loop ended for %s", channel)
    finally:
        await _registry.remove(channel, websocket)
