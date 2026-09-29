from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from typing import Any

from src.unreal_integration._streaming_buffer import FrameBuffer
from src.unreal_integration._streaming_config import (
    ControlAction,
    ControlMessage,
    StreamingConfig,
    StreamingState,
)
from src.unreal_integration._streaming_protocol import StreamingProtocol, _StreamClient
from src.unreal_integration.data_models import UnrealDataFrame

logger = logging.getLogger(__name__)


class UnrealStreamingServer:
    """WebSocket server for streaming to Unreal Engine.

    Provides real-time streaming of physics data to Unreal Engine
    visualization frontend.

    Design by Contract:
        Preconditions:
            - start() requires STOPPED state
            - stop() requires active state
            - broadcast() requires RUNNING state

        Postconditions:
            - start() transitions to RUNNING
            - stop() transitions to STOPPED

        Invariants:
            - client_count >= 0
            - frames_sent >= 0

    Example:
        >>> server = UnrealStreamingServer()
        >>> async with server:
        ...     await server.broadcast(frame)
    """

    def __init__(self, config: StreamingConfig | None = None) -> None:
        """Initialize streaming server.

        Args:
            config: Server configuration (uses defaults if not provided).
        """
        self.config = config or StreamingConfig()
        self._state = StreamingState.STOPPED
        self._clients: set[Any] = set()
        self._buffer = FrameBuffer(max_size=self.config.buffer_size)
        self._server: asyncio.Server | None = None
        self._playback_speed = 1.0
        self._current_time = 0.0
        self._start_time: float | None = None
        self._frames_sent = 0
        self._last_frame_time = 0.0
        self._on_client_connect: Callable[[Any], None] | None = None
        self._on_client_disconnect: Callable[[Any], None] | None = None
        self._on_control_message: Callable[[ControlMessage], None] | None = None

    @property
    def state(self) -> StreamingState:
        """Get current server state."""
        return self._state

    @property
    def client_count(self) -> int:
        """Get number of connected clients."""
        return len(self._clients)

    @property
    def playback_speed(self) -> float:
        """Get current playback speed."""
        return self._playback_speed

    @property
    def bound_port(self) -> int:
        """Return the actual port the server is bound to.

        Useful when config.port == 0 (OS-assigned port).  Returns 0 if the
        server has not been started yet.
        """
        if self._server is None:
            return 0
        sockets = self._server.sockets
        if not sockets:
            return 0
        return int(sockets[0].getsockname()[1])

    def get_statistics(self) -> dict[str, Any]:
        """Get server statistics.

        Returns:
            Dictionary of server statistics.
        """
        uptime = 0.0
        if self._start_time is not None:
            uptime = time.time() - self._start_time

        average_fps = 0.0
        if uptime > 0:
            average_fps = self._frames_sent / uptime

        return {
            "state": self._state.name,
            "clients_connected": self.client_count,
            "frames_sent": self._frames_sent,
            "uptime": uptime,
            "average_fps": average_fps,
            "buffer_size": len(self._buffer),
            "playback_speed": self._playback_speed,
            "current_time": self._current_time,
        }

    async def __aenter__(self) -> UnrealStreamingServer:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

    async def _handle_new_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Accept an incoming TCP connection and track it as a client.

        The client object exposes an async ``send(data: str)`` method so it is
        compatible with the existing ``broadcast()`` call pattern.
        """
        client = _StreamClient(reader, writer)
        await self._add_client(client)
        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break
        except (OSError, asyncio.IncompleteReadError):
            pass
        finally:
            await self._remove_client(client)
            await client.close()

    async def start(self) -> None:
        """Start the streaming server.

        Binds to the configured host/port before transitioning to RUNNING.

        Precondition: Server must be in STOPPED state.
        Postcondition: Server transitions to RUNNING state with an active socket.

        Raises:
            RuntimeError: If server is not in STOPPED state or socket cannot
                be bound.
        """
        if self._state != StreamingState.STOPPED:
            raise RuntimeError(f"Cannot start server in {self._state} state")

        self._state = StreamingState.STARTING
        self._start_time = time.time()
        self._frames_sent = 0

        try:
            self._server = await asyncio.start_server(
                self._handle_new_connection,
                self.config.host,
                self.config.port,
            )
            self._state = StreamingState.RUNNING
            actual_port = self.bound_port
            logger.info(
                "Streaming server started on %s:%d", self.config.host, actual_port
            )
        except OSError as e:
            self._state = StreamingState.ERROR
            logger.error("Failed to bind streaming server: %s", e)
            raise RuntimeError(
                f"Cannot bind streaming server to "
                f"{self.config.host}:{self.config.port}: {e}"
            ) from e
        except (RuntimeError, TypeError, ValueError) as e:
            self._state = StreamingState.ERROR
            logger.error("Failed to start streaming server: %s", e)
            raise

    async def stop(self) -> None:
        """Stop the streaming server.

        Precondition: Server must be in active state.
        Postcondition: Server transitions to STOPPED state and the socket is closed.
        """
        if not self._state.is_active and self._state != StreamingState.STARTING:
            return

        self._state = StreamingState.STOPPING

        for client in list(self._clients):
            await self._remove_client(client)

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        self._buffer.clear()

        self._state = StreamingState.STOPPED
        logger.info("Streaming server stopped")

    def queue_frame(self, frame: UnrealDataFrame) -> None:
        """Add frame to streaming buffer.

        Args:
            frame: Frame to queue for streaming.
        """
        self._buffer.add(frame)

    async def broadcast(self, frame: UnrealDataFrame) -> None:
        """Broadcast frame to all connected clients.

        Args:
            frame: Frame to broadcast.
        """
        if frame is None:
            raise ValueError("frame must be provided")
        if self._state != StreamingState.RUNNING:
            return

        message = StreamingProtocol.create_frame_message(frame)
        json_msg = json.dumps(message)

        disconnected = []
        for client in self._clients:
            try:
                await client.send(json_msg)
            except (RuntimeError, ValueError, OSError):
                disconnected.append(client)

        for client in disconnected:
            await self._remove_client(client)

        self._frames_sent += 1
        self._last_frame_time = time.time()
        self._current_time = frame.timestamp

    async def _add_client(self, client: Any) -> None:
        """Add a new client connection.

        Args:
            client: WebSocket client connection.
        """
        if len(self._clients) >= self.config.max_clients:
            logger.warning("Max clients reached, rejecting new connection")
            return

        self._clients.add(client)
        logger.info(f"Client connected. Total clients: {self.client_count}")

        if self._on_client_connect:
            self._on_client_connect(client)

    async def _remove_client(self, client: Any) -> None:
        """Remove a client connection.

        Args:
            client: WebSocket client connection.
        """
        self._clients.discard(client)
        logger.info(f"Client disconnected. Total clients: {self.client_count}")

        if self._on_client_disconnect:
            self._on_client_disconnect(client)

    async def _handle_control(self, message: ControlMessage) -> None:  # noqa: C901
        """Handle control message from client.

        Args:
            message: Control message to handle.
        """
        if message is None:
            raise ValueError("message must be provided")
        if self._on_control_message:
            self._on_control_message(message)

        if message.action == ControlAction.PAUSE:
            if self._state == StreamingState.RUNNING:
                self._state = StreamingState.PAUSED
                logger.info("Streaming paused")

        elif message.action == ControlAction.PLAY:
            if self._state == StreamingState.PAUSED:
                self._state = StreamingState.RUNNING
                logger.info("Streaming resumed")

        elif message.action == ControlAction.SET_SPEED:
            if message.value is not None and isinstance(message.value, (int, float)):
                self._playback_speed = float(message.value)
                logger.info(f"Playback speed set to {self._playback_speed}")

        elif message.action == ControlAction.SEEK:
            if message.value is not None and isinstance(message.value, (int, float)):
                self._current_time = float(message.value)
                logger.info(f"Seeked to {self._current_time}")

        elif message.action == ControlAction.STOP:
            await self.stop()

        elif message.action == ControlAction.RESET:
            self._buffer.clear()
            self._current_time = 0.0
            self._frames_sent = 0
            logger.info("Streaming reset")

    def on_client_connect(self, callback: Callable[[Any], None]) -> None:
        """Register callback for client connections.

        Args:
            callback: Function to call when client connects.
        """
        self._on_client_connect = callback

    def on_client_disconnect(self, callback: Callable[[Any], None]) -> None:
        """Register callback for client disconnections.

        Args:
            callback: Function to call when client disconnects.
        """
        self._on_client_disconnect = callback

    def on_control_message(self, callback: Callable[[ControlMessage], None]) -> None:
        """Register callback for control messages.

        Args:
            callback: Function to call when control message received.
        """
        self._on_control_message = callback
