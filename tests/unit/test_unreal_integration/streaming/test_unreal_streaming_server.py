"""Unit tests for Unreal Engine WebSocket streaming.

TDD tests for the streaming server that sends data to Unreal Engine.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Generator
from unittest.mock import AsyncMock

import pytest
from src.unreal_integration.data_models import (  # noqa: E402
    JointState,
    Quaternion,
    UnrealDataFrame,
    Vector3,
)
from src.unreal_integration.streaming import (  # noqa: E402
    ControlAction,
    ControlMessage,
    FrameBuffer,
    StreamingConfig,
    StreamingProtocol,
    StreamingState,
    UnrealStreamingServer,
)

# Ensure pytest-asyncio is available for async test classes;
# skip the entire module if it is not installed.
pytest.importorskip("pytest_asyncio", reason="pytest-asyncio not installed")


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Provide an explicit event loop for sync buffer tests on Python 3.14+."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@pytest.mark.asyncio
class TestUnrealStreamingServer:
    """Tests for the streaming server."""

    async def test_server_creation(self) -> None:
        """Test server creation."""
        config = StreamingConfig(host="localhost", port=8765)
        server = UnrealStreamingServer(config=config)
        assert server.state == StreamingState.STOPPED
        assert server.config.port == 8765

    async def test_server_state_transitions(self) -> None:
        """Test server state transitions."""
        server = UnrealStreamingServer()
        assert server.state == StreamingState.STOPPED

        # Start should transition to STARTING then RUNNING
        # (In tests we mock the actual server start)

    async def test_server_broadcast_frame(self) -> None:
        """Test broadcasting frame to clients."""
        server = UnrealStreamingServer()
        server._state = StreamingState.RUNNING  # Must be running to broadcast
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()
        server._clients = {mock_client1, mock_client2}

        frame = UnrealDataFrame(
            timestamp=0.0,
            frame_number=0,
            joints={},
        )

        await server.broadcast(frame)

        # Verify all clients received the frame
        mock_client1.send.assert_called_once()
        mock_client2.send.assert_called_once()

    async def test_server_queue_frame(self) -> None:
        """Test queuing frame for streaming."""
        server = UnrealStreamingServer()
        frame = UnrealDataFrame(timestamp=0.0, frame_number=0, joints={})

        server.queue_frame(frame)

        assert len(server._buffer) == 1

    async def test_server_statistics(self) -> None:
        """Test server statistics."""
        server = UnrealStreamingServer()
        stats = server.get_statistics()

        assert "frames_sent" in stats
        assert "clients_connected" in stats
        assert "uptime" in stats
        assert "average_fps" in stats

    async def test_server_client_management(self) -> None:
        """Test client connection management."""
        server = UnrealStreamingServer()
        assert server.client_count == 0

        mock_client = AsyncMock()
        await server._add_client(mock_client)
        assert server.client_count == 1

        await server._remove_client(mock_client)
        assert server.client_count == 0

    async def test_server_handle_control_message(self) -> None:
        """Test handling control messages."""
        server = UnrealStreamingServer()
        server._state = StreamingState.RUNNING

        # Test pause
        await server._handle_control(ControlMessage(action=ControlAction.PAUSE))
        assert server.state == StreamingState.PAUSED

        # Test play
        await server._handle_control(ControlMessage(action=ControlAction.PLAY))
        assert server.state == StreamingState.RUNNING

    async def test_server_playback_speed(self) -> None:
        """Test playback speed control."""
        server = UnrealStreamingServer()
        assert server.playback_speed == 1.0

        await server._handle_control(
            ControlMessage(action=ControlAction.SET_SPEED, value=0.5)
        )
        assert server.playback_speed == 0.5

    async def test_server_seek(self) -> None:
        """Test seek functionality."""
        server = UnrealStreamingServer()

        # Queue some frames
        for i in range(10):
            server.queue_frame(
                UnrealDataFrame(timestamp=float(i) * 0.1, frame_number=i, joints={})
            )

        # Seek to timestamp 0.5
        await server._handle_control(
            ControlMessage(action=ControlAction.SEEK, value=0.5)
        )

        # Buffer should be at appropriate position
        assert server._current_time == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Issue #2475: server must bind a real socket before reporting RUNNING
# ---------------------------------------------------------------------------
