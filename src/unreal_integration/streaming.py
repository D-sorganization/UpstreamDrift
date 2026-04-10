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

from src.unreal_integration._simulation_streamer import SimulationStreamer
from src.unreal_integration._streaming_buffer import FrameBuffer
from src.unreal_integration._streaming_config import (
    ControlAction,
    ControlMessage,
    StreamingConfig,
    StreamingState,
)
from src.unreal_integration._streaming_protocol import StreamingProtocol, _StreamClient
from src.unreal_integration._streaming_server import UnrealStreamingServer

__all__ = [
    "ControlAction",
    "ControlMessage",
    "FrameBuffer",
    "SimulationStreamer",
    "StreamingConfig",
    "StreamingProtocol",
    "StreamingState",
    "UnrealStreamingServer",
    "_StreamClient",
]
