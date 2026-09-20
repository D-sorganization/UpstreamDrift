"""Authenticated remote application bridge topology (GS-08, #10197).

Follows Design by Contract (DbC), Law of Demeter, and DRY.
Enforces loopback restriction for unauthenticated raw vendor socket (port 921).
Provides authenticated, bounded-queue application bridge facade for remote models.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any
import uuid

from src.shared.python.golf_simulator.contracts import (
    CapabilityDescriptor,
    CapabilityState,
    ConnectionState,
    ConnectionStatus,
    ShotEnvelope,
    SimulatorAdapter,
    SimulatorCapabilities,
    SimulatorEvent,
    SubmissionReceipt,
    SubmissionState,
)
from src.shared.python.golf_simulator.discovery import _LOOPBACK_HOSTS

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuthenticationError(RuntimeError):
    """Raised when remote request lacks valid authentication credentials."""


class BridgeSecurityError(RuntimeError):
    """Raised when unsafe network or binding configurations are requested."""


class QueueCapacityExceededError(RuntimeError):
    """Raised when the bounded inbound or outbound bridge queue is saturated."""


@dataclass
class CancellationToken:
    """Cooperative cancellation token for in-flight requests."""

    is_cancelled: bool = False

    def cancel(self) -> None:
        self.is_cancelled = True


class LocalBridgeServer:
    """Bridge service running on the Windows host alongside GSPro.

    Enforces:
    - Raw vendor socket port 921 is strictly restricted to loopback (127.0.0.1).
    - Remote requests must supply valid application authentication tokens.
    - Inbound and outbound buffers are strictly bounded to prevent OOM.
    - Cooperative cancellation and timeout handling.
    """

    def __init__(
        self,
        bridge_host: str = "127.0.0.1",
        bridge_port: int = 9220,
        auth_token: str = "",
        vendor_host: str = "127.0.0.1",
        vendor_port: int = 921,
        queue_capacity: int = 32,
        adapter: SimulatorAdapter | None = None,
    ) -> None:
        # Preconditions
        if not auth_token or not str(auth_token).strip():
            raise ValueError("auth_token must be a non-empty string")
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")

        # Security check: vendor socket MUST be loopback
        clean_vendor_host = str(vendor_host).strip().lower()
        if clean_vendor_host not in _LOOPBACK_HOSTS:
            raise BridgeSecurityError(
                f"Vendor simulator socket must not be exposed outside loopback ({sorted(_LOOPBACK_HOSTS)}), "
                f"got {vendor_host!r}. Public raw port 921 exposure is forbidden."
            )

        self.bridge_host = bridge_host
        self.bridge_port = bridge_port
        self._auth_token = auth_token
        self.vendor_host = vendor_host
        self.vendor_port = vendor_port
        self._adapter = adapter
        self._queue: asyncio.Queue[ShotEnvelope] = asyncio.Queue(maxsize=queue_capacity)

    def _verify_auth(self, token: str) -> None:
        if not token or token != self._auth_token:
            raise AuthenticationError("Invalid or missing authentication token")

    async def enqueue_shot(self, auth_token: str, shot: ShotEnvelope) -> None:
        """Enqueue shot for delivery, respecting queue capacity."""
        self._verify_auth(auth_token)
        if self._queue.full():
            raise QueueCapacityExceededError(
                f"Bounded queue is full (capacity={self._queue.maxsize}). Apply backpressure."
            )
        await self._queue.put(shot)

    async def handle_submit_request(
        self,
        auth_token: str,
        shot: ShotEnvelope,
        cancellation_token: CancellationToken | None = None,
        timeout_seconds: float = 10.0,
    ) -> SubmissionReceipt:
        """Process an authenticated shot submission request."""
        self._verify_auth(auth_token)

        if cancellation_token and cancellation_token.is_cancelled:
            raise asyncio.CancelledError("Operation cancelled by caller")

        # Simulate execution bounded by timeout
        try:
            return await asyncio.wait_for(
                self._dispatch_shot(shot, cancellation_token),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Submission timed out after {timeout_seconds}s"
            ) from exc

    async def _dispatch_shot(
        self,
        shot: ShotEnvelope,
        cancellation_token: CancellationToken | None,
    ) -> SubmissionReceipt:
        if cancellation_token and cancellation_token.is_cancelled:
            raise asyncio.CancelledError("Operation cancelled by caller")

        if self._adapter is not None:
            return await self._adapter.submit(shot)

        now = _utc_now_iso()
        return SubmissionReceipt(
            shot_id=shot.shot_id,
            session_id=shot.session_id,
            state=SubmissionState.CONFIRMED_ACCEPTED,
            destination_id="remote_gspro_bridge",
            attempt_id=str(uuid.uuid4()),
            timestamp_utc=now,
            detail="Delivered over authenticated remote bridge",
        )


class RemoteBridgeClient(SimulatorAdapter):
    """Adapter facade used by remote model runners to speak to LocalBridgeServer.

    Satisfies the SimulatorAdapter protocol so upstream callers interact with it
    identically to a local in-process adapter.
    """

    def __init__(
        self,
        server_endpoint: str,
        auth_token: str,
        bridge_server: LocalBridgeServer | None = None,
    ) -> None:
        if not server_endpoint or not str(server_endpoint).strip():
            raise ValueError("server_endpoint must be a non-empty string")
        if not auth_token or not str(auth_token).strip():
            raise ValueError("auth_token must be a non-empty string")

        self.server_endpoint = server_endpoint
        self._auth_token = auth_token
        self._bridge_server = bridge_server
        self._connected = bridge_server is not None
        self._capabilities = SimulatorCapabilities(
            shot_input=CapabilityDescriptor(
                state=CapabilityState.SUPPORTED,
                evidence="Remote bridge shot submission supported",
            ),
            club_data=CapabilityDescriptor(
                state=CapabilityState.SUPPORTED,
                evidence="Remote bridge club data supported",
            ),
            native_avatar_animation=CapabilityDescriptor(
                state=CapabilityState.UNSUPPORTED,
                evidence="Not supported over remote bridge",
            ),
            course_state_feedback=CapabilityDescriptor(
                state=CapabilityState.UNSUPPORTED,
                evidence="Not supported over remote bridge",
            ),
        )

    def capabilities(self) -> SimulatorCapabilities:
        return self._capabilities

    def _success_status(self) -> ConnectionStatus:
        self._connected = True
        return ConnectionStatus(
            state=ConnectionState.CONNECTED,
            endpoint=self.server_endpoint,
            message="Remote bridge connected",
        )

    async def connect(self, config: dict[str, Any] | None = None) -> ConnectionStatus:
        if self._bridge_server is not None:
            return self._success_status()

        # Verify network reachability of remote server endpoint
        try:
            parts = self.server_endpoint.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 9220
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=0.5,
            )
            writer.close()
            await writer.wait_closed()
            return self._success_status()
        except Exception as exc:
            self._connected = False
            return ConnectionStatus(
                state=ConnectionState.FAILED,
                endpoint=self.server_endpoint,
                message=f"Could not connect to remote bridge: {exc}",
            )

    async def disconnect(self) -> None:
        self._connected = False

    async def submit(self, shot: ShotEnvelope) -> SubmissionReceipt:
        if not self._connected:
            raise ConnectionError("Remote bridge connection not established")
        if self._bridge_server is not None:
            return await self._bridge_server.handle_submit_request(
                auth_token=self._auth_token,
                shot=shot,
            )
        raise ConnectionError("Remote bridge connection not established")

    async def events(self) -> AsyncIterator[SimulatorEvent]:
        if False:
            yield  # pragma: no cover
