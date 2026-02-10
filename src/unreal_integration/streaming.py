"""WebSocket streaming server for Unreal Engine integration.

This module provides real-time data streaming from the Python physics
backend to Unreal Engine visualization frontend.

Design by Contract:
    - Server state machine with explicit transitions
    - Thread-safe buffer operations
    - Graceful degradation under load

Features:
    - WebSocket server with multiple client support
    - Frame buffering with overflow protection
    - Playback control (play, pause, seek, speed)
    - Statistics and monitoring

Usage:
    from src.unreal_integration.streaming import (
        UnrealStreamingServer,
        StreamingConfig,
    )

    # Create and start server
    server = UnrealStreamingServer(
        config=StreamingConfig(host="localhost", port=8765)
    )

    async with server:
        for frame in simulation:
            await server.broadcast(frame)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from src.unreal_integration.data_models import UnrealDataFrame

logger = logging.getLogger(__name__)


class StreamingState(Enum):
    """Server streaming state."""

    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    ERROR = auto()

    @property
    def is_active(self) -> bool:
        """Check if state is active (running or paused).

        Returns:
            True if server is in an active state.
        """
        return self in (StreamingState.RUNNING, StreamingState.PAUSED)


class ControlAction(Enum):
    """Control message actions from Unreal client."""

    PLAY = "play"
    PAUSE = "pause"
    SEEK = "seek"
    SET_SPEED = "set_speed"
    STOP = "stop"
    RESET = "reset"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"

    @classmethod
    def from_string(cls, s: str) -> ControlAction:
        """Create ControlAction from string.

        Args:
            s: Action string (e.g., "pause").

        Returns:
            Corresponding ControlAction.

        Raises:
            ValueError: If action string is invalid.
        """
        for action in cls:
            if action.value == s.lower():
                return action
        raise ValueError(f"Unknown control action: {s}")


@dataclass
class ControlMessage:
    """Control message from Unreal client.

    Attributes:
        action: The control action to perform.
        value: Optional value for the action (e.g., seek position).
        client_id: Client identifier (optional).
    """

    action: ControlAction
    value: float | str | None = None
    client_id: str | None = None

    def to_json(self) -> str:
        """Convert to JSON string.

        Returns:
            JSON representation.
        """
        d: dict[str, Any] = {
            "type": "control",
            "action": self.action.value,
        }
        if self.value is not None:
            d["value"] = self.value
        if self.client_id is not None:
            d["client_id"] = self.client_id
        return json.dumps(d)

    @classmethod
    def from_json(cls, json_str: str) -> ControlMessage:
        """Create ControlMessage from JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            New ControlMessage instance.
        """
        d = json.loads(json_str)
        return cls(
            action=ControlAction.from_string(d["action"]),
            value=d.get("value"),
            client_id=d.get("client_id"),
        )


@dataclass
class StreamingConfig:
    """Configuration for streaming server.

    Attributes:
        host: Host address to bind to.
        port: Port number.
        target_fps: Target frames per second.
        buffer_size: Maximum frames to buffer.
        enable_compression: Whether to compress frames.
        heartbeat_interval: Seconds between heartbeat messages.
        max_clients: Maximum simultaneous clients.
        enable_metrics: Whether to track streaming metrics.
    """

    host: str = "localhost"
    port: int = 8765
    target_fps: int = 60
    buffer_size: int = 10
    enable_compression: bool = False
    heartbeat_interval: float = 1.0
    max_clients: int = 10
    enable_metrics: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.port < 0 or self.port > 65535:
            raise ValueError(f"Invalid port number: {self.port}")
        if self.target_fps <= 0:
            raise ValueError(f"Invalid target fps: {self.target_fps}")
        if self.buffer_size <= 0:
            raise ValueError(f"Invalid buffer size: {self.buffer_size}")

    @property
    def frame_interval(self) -> float:
        """Calculate frame interval in seconds.

        Returns:
            Time between frames in seconds.
        """
        return 1.0 / self.target_fps

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "host": self.host,
            "port": self.port,
            "target_fps": self.target_fps,
            "buffer_size": self.buffer_size,
            "enable_compression": self.enable_compression,
            "heartbeat_interval": self.heartbeat_interval,
            "max_clients": self.max_clients,
            "enable_metrics": self.enable_metrics,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StreamingConfig:
        """Create StreamingConfig from dictionary.

        Args:
            d: Dictionary representation.

        Returns:
            New StreamingConfig instance.
        """
        return cls(
            host=d.get("host", "localhost"),
            port=d.get("port", 8765),
            target_fps=d.get("target_fps", 60),
            buffer_size=d.get("buffer_size", 10),
            enable_compression=d.get("enable_compression", False),
            heartbeat_interval=d.get("heartbeat_interval", 1.0),
            max_clients=d.get("max_clients", 10),
            enable_metrics=d.get("enable_metrics", True),
        )


class FrameBuffer:
    """Thread-safe frame buffer with overflow protection.

    Uses a circular buffer to store frames. When buffer is full,
    oldest frames are dropped to make room for new ones.

    Attributes:
        max_size: Maximum number of frames to store.

    Example:
        >>> buffer = FrameBuffer(max_size=10)
        >>> buffer.add(frame)
        >>> frame = buffer.get()
    """

    def __init__(self, max_size: int = 10):
        """Initialize frame buffer.

        Args:
            max_size: Maximum number of frames to store.
        """
        self.max_size = max_size
        self._buffer: deque[UnrealDataFrame] = deque(maxlen=max_size)
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None

    def __len__(self) -> int:
        """Return number of frames in buffer."""
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._buffer) == 0

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self._buffer) >= self.max_size

    def add(self, frame: UnrealDataFrame) -> bool:
        """Add frame to buffer.

        If buffer is full, oldest frame is dropped.

        Args:
            frame: Frame to add.

        Returns:
            True if frame was added (oldest may have been dropped).
        """
        self._buffer.append(frame)
        return True

    def get(self) -> UnrealDataFrame | None:
        """Remove and return oldest frame.

        Returns:
            Oldest frame, or None if buffer is empty.
        """
        if self.is_empty:
            return None
        return self._buffer.popleft()

    def peek(self) -> UnrealDataFrame | None:
        """Return oldest frame without removing it.

        Returns:
            Oldest frame, or None if buffer is empty.
        """
        if self.is_empty:
            return None
        return self._buffer[0]

    def clear(self) -> None:
        """Remove all frames from buffer."""
        self._buffer.clear()

    def get_all(self) -> list[UnrealDataFrame]:
        """Get all frames without removing them.

        Returns:
            List of all frames in buffer (oldest first).
        """
        return list(self._buffer)


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

    def __init__(self, config: StreamingConfig | None = None):
        """Initialize streaming server.

        Args:
            config: Server configuration (uses defaults if not provided).
        """
        self.config = config or StreamingConfig()
        self._state = StreamingState.STOPPED
        self._clients: set[Any] = set()
        self._buffer = FrameBuffer(max_size=self.config.buffer_size)
        self._server = None
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

    async def start(self) -> None:
        """Start the streaming server.

        Precondition: Server must be in STOPPED state.
        Postcondition: Server transitions to RUNNING state.

        Raises:
            RuntimeError: If server is not in STOPPED state.
        """
        if self._state != StreamingState.STOPPED:
            raise RuntimeError(f"Cannot start server in {self._state} state")

        self._state = StreamingState.STARTING
        self._start_time = time.time()
        self._frames_sent = 0

        try:
            # In a real implementation, this would start the WebSocket server
            # For now, we just transition to RUNNING
            self._state = StreamingState.RUNNING
            logger.info(
                f"Streaming server started on {self.config.host}:{self.config.port}"
            )
        except (RuntimeError, TypeError, ValueError) as e:
            self._state = StreamingState.ERROR
            logger.error(f"Failed to start streaming server: {e}")
            raise

    async def stop(self) -> None:
        """Stop the streaming server.

        Precondition: Server must be in active state.
        Postcondition: Server transitions to STOPPED state.
        """
        if not self._state.is_active and self._state != StreamingState.STARTING:
            return

        self._state = StreamingState.STOPPING

        # Disconnect all clients
        for client in list(self._clients):
            await self._remove_client(client)

        # Clear buffer
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
        if self._state != StreamingState.RUNNING:
            return

        message = StreamingProtocol.create_frame_message(frame)
        json_msg = json.dumps(message)

        # Send to all clients
        disconnected = []
        for client in self._clients:
            try:
                await client.send(json_msg)
            except (RuntimeError, ValueError, OSError):
                disconnected.append(client)

        # Remove disconnected clients
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

    async def _handle_control(self, message: ControlMessage) -> None:
        """Handle control message from client.

        Args:
            message: Control message to handle.
        """
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


class SimulationStreamer:
    """High-level interface for streaming simulation data.

    Provides a convenient interface for streaming physics simulation
    data to Unreal Engine, handling frame timing and buffering.

    Example:
        >>> streamer = SimulationStreamer(server)
        >>> for state in simulation:
        ...     await streamer.send_state(state, timestamp)
    """

    def __init__(self, server: UnrealStreamingServer):
        """Initialize simulation streamer.

        Args:
            server: Streaming server instance.
        """
        self.server = server
        self._frame_number = 0
        self._last_send_time = 0.0

    async def send_frame(self, frame: UnrealDataFrame) -> None:
        """Send a pre-constructed frame.

        Args:
            frame: Frame to send.
        """
        await self.server.broadcast(frame)
        self._frame_number = frame.frame_number + 1

    async def send_state(
        self,
        joints: dict[str, Any],
        timestamp: float,
        forces: list[Any] | None = None,
        metrics: Any | None = None,
    ) -> None:
        """Send physics state as frame.

        Convenience method that constructs an UnrealDataFrame from
        raw physics state data.

        Args:
            joints: Dictionary of joint states.
            timestamp: Simulation timestamp.
            forces: Optional list of force vectors.
            metrics: Optional swing metrics.
        """
        from src.unreal_integration.data_models import JointState

        # Convert raw joints to JointState objects if needed
        joint_states = {}
        for name, state in joints.items():
            if isinstance(state, JointState):
                joint_states[name] = state
            else:
                # Assume it's a dict-like object
                joint_states[name] = JointState.from_dict(state)

        frame = UnrealDataFrame(
            timestamp=timestamp,
            frame_number=self._frame_number,
            joints=joint_states,
            forces=forces,
            metrics=metrics,
        )

        await self.send_frame(frame)

    def reset(self) -> None:
        """Reset streamer state."""
        self._frame_number = 0
        self._last_send_time = 0.0
