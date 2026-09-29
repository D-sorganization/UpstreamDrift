"""Async TCP transport and bounded stream framer for GSPro Open Connect v1 (GS-03, #10192).

Enforces DbC invariants:
- Bounded framing buffer: rejects frames > max_buffer_bytes (default 64 KiB).
- Integrates with ShotJournal to record delivery intent BEFORE physical socket write.
- Network drops / timeouts mid-flight record AMBIGUOUS delivery status without auto-resending.
- Strict response code mapping to delivery status.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from src.shared.python.golf_simulator.adapters.gspro.codec import (
    ResponseCategory,
    ResponseReceipt,
    decode_simulator_response,
)
from src.shared.python.golf_simulator.adapters.gspro.profile import (
    DEFAULT_GSPRO_PROFILE,
    GSProProfile,
)

if TYPE_CHECKING:
    from src.shared.python.golf_simulator.journal import ShotJournal

logger = logging.getLogger(__name__)


class TransportError(RuntimeError):
    """Base exception for simulator transport failures."""


class FramingBufferOverflowError(TransportError):
    """Raised when the incoming stream exceeds the maximum allowable buffer size."""


class FramingProtocolError(TransportError):
    """Raised when incoming stream data violates protocol structure."""


class GSProTransport:
    """Async TCP transport for GSPro Open Connect v1."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 921,
        max_buffer_bytes: int = 65536,
        timeout_s: float = 5.0,
        profile: GSProProfile = DEFAULT_GSPRO_PROFILE,
        journal: ShotJournal | None = None,
    ) -> None:
        if max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be strictly positive")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be strictly positive")

        self.host = host
        self.port = port
        self.max_buffer_bytes = max_buffer_bytes
        self.timeout_s = timeout_s
        self.profile = profile
        self.journal = journal

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._buffer = bytearray()
        self._lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        """Check if transport currently has an active connection."""
        return self._writer is not None and not self._writer.is_closing()

    async def connect(self) -> None:
        """Establish outbound TCP connection to GSPro."""
        async with self._lock:
            if self.is_connected:
                return
            try:
                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_connection(self.host, self.port),
                    timeout=self.timeout_s,
                )
                self._buffer.clear()
                logger.info("Connected to GSPro at %s:%d", self.host, self.port)
            except Exception as exc:
                logger.error(
                    "Failed to connect to GSPro at %s:%d: %s", self.host, self.port, exc
                )
                raise TransportError(
                    f"Cannot connect to {self.host}:{self.port}: {exc}"
                ) from exc

    async def disconnect(self) -> None:
        """Close outbound connection cleanly."""
        async with self._lock:
            if self._writer is not None:
                try:
                    self._writer.close()
                    await self._writer.wait_closed()
                except Exception as exc:
                    logger.debug("Error while closing writer: %s", exc)
                finally:
                    self._writer = None
                    self._reader = None
                    self._buffer.clear()
                    logger.info("Disconnected from GSPro")

    async def send_shot(self, shot_id: str, payload_bytes: bytes) -> ResponseReceipt:
        """Send shot payload to GSPro and wait for response receipt.

        Enforces:
        - Intent recorded in journal (status: PENDING) before socket write.
        - Network drop or timeout during in-flight wait marks status: AMBIGUOUS.
        - No automatic retransmission on ambiguity.
        - Successful reply marks status: ACKNOWLEDGED.
        - Simulator rejection marks status: REJECTED.
        """
        if not self.is_connected or self._writer is None or self._reader is None:
            raise TransportError("Not connected to simulator")
        if not payload_bytes or len(payload_bytes) > self.max_buffer_bytes:
            raise ValueError(
                f"Payload size {len(payload_bytes)} invalid or exceeds max {self.max_buffer_bytes}"
            )

        # 1. Record intent prior to wire write
        if self.journal is not None:
            self.journal.record_intent(shot_id, payload_bytes)

        # 2. Write payload to TCP stream
        try:
            self._writer.write(payload_bytes)
            await self._writer.drain()
        except Exception as exc:
            reason = f"Write failed: {exc}"
            logger.error("Failed to write shot %s to socket: %s", shot_id, exc)
            if self.journal is not None:
                self.journal.record_ambiguity(shot_id, reason)
            await self.disconnect()
            raise TransportError(reason) from exc

        # 3. Read and frame response
        try:
            raw_response = await asyncio.wait_for(
                self._read_complete_frame(),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError as exc:
            reason = f"Timeout waiting for response to shot {shot_id}"
            logger.error(reason)
            if self.journal is not None:
                self.journal.record_ambiguity(shot_id, reason)
            await self.disconnect()
            raise TransportError(reason) from exc
        except Exception as exc:
            reason = f"Connection closed or failed while waiting for response: {exc}"
            logger.error(reason)
            if self.journal is not None:
                self.journal.record_ambiguity(shot_id, reason)
            await self.disconnect()
            if isinstance(exc, FramingBufferOverflowError):
                raise
            raise TransportError(reason) from exc

        # 4. Decode response and update journal
        receipt = decode_simulator_response(raw_response, profile=self.profile)
        if self.journal is not None:
            if receipt.category == ResponseCategory.CONFIRMED_ACCEPTED:
                self.journal.record_acknowledgment(
                    shot_id, receipt.code, receipt.message
                )
            else:
                self.journal.record_rejection(shot_id, receipt.code, receipt.message)

        return receipt

    async def _read_complete_frame(self) -> str:
        """Accumulate bytes and extract next balanced JSON object or newline frame."""
        assert self._reader is not None

        while True:
            # Check if existing buffer contains a complete JSON object
            frame = self._extract_frame_from_buffer()
            if frame is not None:
                return frame

            # Read next chunk from network
            chunk = await self._reader.read(4096)
            if not chunk:
                # EOF reached
                frame = self._extract_frame_from_buffer()
                if frame is not None:
                    return frame
                raise TransportError(
                    "Connection closed by peer before complete response frame received"
                )

            if len(self._buffer) + len(chunk) > self.max_buffer_bytes:
                self._buffer.clear()
                raise FramingBufferOverflowError(
                    f"Incoming stream exceeded buffer limit of {self.max_buffer_bytes} bytes"
                )

            self._buffer.extend(chunk)

    def _extract_frame_from_buffer(self) -> str | None:
        """Attempt to extract a complete JSON frame from the internal buffer."""
        if not self._buffer:
            return None

        # Search for first opening brace '{'
        start_idx = self._buffer.find(b"{")
        if start_idx == -1:
            # Discard any leading whitespace/junk if too long
            if len(self._buffer) > 1024:
                self._buffer.clear()
            return None

        # Discard any bytes before start_idx
        if start_idx > 0:
            del self._buffer[:start_idx]

        # Scan for balanced braces respecting string quotes
        in_string = False
        escape = False
        depth = 0
        for i, byte_val in enumerate(self._buffer):
            char = chr(byte_val)
            if escape:
                escape = False
                continue
            if char == "\\":
                if in_string:
                    escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        # Found complete JSON object
                        frame_bytes = bytes(self._buffer[: i + 1])
                        del self._buffer[: i + 1]
                        return frame_bytes.decode("utf-8")

        return None
