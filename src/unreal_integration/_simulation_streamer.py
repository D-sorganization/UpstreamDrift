from __future__ import annotations

from typing import Any

from src.unreal_integration._streaming_server import UnrealStreamingServer
from src.unreal_integration.data_models import UnrealDataFrame


class SimulationStreamer:
    """High-level interface for streaming simulation data.

    Provides a convenient interface for streaming physics simulation
    data to Unreal Engine, handling frame timing and buffering.

    Example:
        >>> streamer = SimulationStreamer(server)
        >>> for state in simulation:
        ...     await streamer.send_state(state, timestamp)
    """

    def __init__(self, server: UnrealStreamingServer) -> None:
        """Initialize simulation streamer.

        Args:
            server: Streaming server instance.
        """
        if server is None:
            raise ValueError("server must be provided")
        self.server = server
        self._frame_number = 0
        self._last_send_time = 0.0

    async def send_frame(self, frame: UnrealDataFrame) -> None:
        """Send a pre-constructed frame.

        Args:
            frame: Frame to send.
        """
        if frame is None:
            raise ValueError("frame must be provided")
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
        if joints is None:
            raise ValueError("joints must be provided")
        from src.unreal_integration.data_models import JointState

        joint_states = {}
        for name, state in joints.items():
            if isinstance(state, JointState):
                joint_states[name] = state
            else:
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
