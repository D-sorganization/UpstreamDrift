from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


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
        if json_str is None:
            raise ValueError("json_str must be provided")
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
