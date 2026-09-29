from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from src.unreal_integration._streaming_config import StreamingState
from src.unreal_integration.data_models import UnrealDataFrame


class StreamingProtocol:
    """Protocol message formatters for streaming.

    Provides static methods for creating protocol-compliant messages.
    """

    @staticmethod
    def create_frame_message(frame: UnrealDataFrame) -> dict[str, Any]:
        """Create frame message for streaming.

        Args:
            frame: Frame data to send.

        Returns:
            Protocol-compliant message dictionary.
        """
        return {
            "type": "frame",
            "data": frame.to_dict(),
        }

    @staticmethod
    def create_status_message(
        state: StreamingState,
        fps: float,
        frames_sent: int,
        buffer_size: int = 0,
    ) -> dict[str, Any]:
        """Create status message.

        Args:
            state: Current streaming state.
            fps: Current frames per second.
            frames_sent: Total frames sent.
            buffer_size: Current buffer size.

        Returns:
            Protocol-compliant status message.
        """
        return {
            "type": "status",
            "state": state.name.lower(),
            "fps": fps,
            "frames_sent": frames_sent,
            "buffer_size": buffer_size,
        }

    @staticmethod
    def create_error_message(
        error_code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create error message.

        Args:
            error_code: Error code identifier.
            message: Human-readable error message.
            details: Additional error details.

        Returns:
            Protocol-compliant error message.
        """
        if error_code is None:
            raise ValueError("error_code must be provided")
        msg: dict[str, Any] = {
            "type": "error",
            "error_code": error_code,
            "message": message,
        }
        if details:
            msg["details"] = details
        return msg

    @staticmethod
    def create_ack_message(
        frame_number: int,
        timestamp: float,
    ) -> dict[str, Any]:
        """Create acknowledgment message.

        Args:
            frame_number: Acknowledged frame number.
            timestamp: Frame timestamp.

        Returns:
            Protocol-compliant acknowledgment message.
        """
        return {
            "type": "ack",
            "frame_number": frame_number,
            "timestamp": timestamp,
        }

    @staticmethod
    def create_heartbeat_message() -> dict[str, Any]:
        """Create heartbeat message.

        Returns:
            Protocol-compliant heartbeat message.
        """
        return {
            "type": "heartbeat",
            "server_time": time.time(),
        }


class _StreamClient:
    """Thin wrapper around an asyncio StreamWriter that exposes ``send()``.

    Provides the same interface that ``broadcast()`` expects (``await
    client.send(str)``), backed by a real asyncio TCP connection.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._reader = reader
        self._writer = writer

    async def send(self, data: str) -> None:
        """Write *data* to the underlying TCP stream."""
        self._writer.write(data.encode())
        await self._writer.drain()

    async def close(self) -> None:
        """Close the underlying TCP connection."""
        self._writer.close()
        with contextlib.suppress(OSError):
            await self._writer.wait_closed()
