"""Unit tests for Unreal Engine WebSocket streaming.

TDD tests for the streaming server that sends data to Unreal Engine.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.unreal_integration.data_models import (
    JointState,
    Quaternion,
    UnrealDataFrame,
    Vector3,
)
from src.unreal_integration.streaming import (
    ControlAction,
    ControlMessage,
    FrameBuffer,
    StreamingConfig,
    StreamingProtocol,
    StreamingState,
    UnrealStreamingServer,
)


class TestStreamingConfig:
    """Tests for StreamingConfig."""

    def test_default_config(self):
        """Test default streaming configuration."""
        config = StreamingConfig()
        assert config.host == "localhost"
        assert config.port == 8765
        assert config.target_fps == 60
        assert config.buffer_size == 10

    def test_custom_config(self):
        """Test custom streaming configuration."""
        config = StreamingConfig(
            host="0.0.0.0",
            port=9000,
            target_fps=120,
            buffer_size=20,
        )
        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.target_fps == 120
        assert config.buffer_size == 20

    def test_config_frame_interval(self):
        """Test frame interval calculation."""
        config = StreamingConfig(target_fps=60)
        assert config.frame_interval == pytest.approx(1 / 60)

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError, match="port"):
            StreamingConfig(port=-1)
        with pytest.raises(ValueError, match="fps"):
            StreamingConfig(target_fps=0)
        with pytest.raises(ValueError, match="buffer"):
            StreamingConfig(buffer_size=0)

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = StreamingConfig()
        d = config.to_dict()
        assert "host" in d
        assert "port" in d
        assert "target_fps" in d

    def test_config_from_dict(self):
        """Test configuration deserialization."""
        d = {"host": "192.168.1.1", "port": 8080, "target_fps": 30}
        config = StreamingConfig.from_dict(d)
        assert config.host == "192.168.1.1"
        assert config.port == 8080


class TestStreamingState:
    """Tests for StreamingState enum."""

    def test_streaming_states(self):
        """Test all streaming states exist."""
        assert StreamingState.STOPPED is not None
        assert StreamingState.STARTING is not None
        assert StreamingState.RUNNING is not None
        assert StreamingState.PAUSED is not None
        assert StreamingState.STOPPING is not None
        assert StreamingState.ERROR is not None

    def test_streaming_state_is_active(self):
        """Test is_active property."""
        assert StreamingState.RUNNING.is_active
        assert StreamingState.PAUSED.is_active
        assert not StreamingState.STOPPED.is_active
        assert not StreamingState.ERROR.is_active


class TestControlMessage:
    """Tests for control message handling."""

    def test_create_control_message(self):
        """Test control message creation."""
        msg = ControlMessage(
            action=ControlAction.PAUSE,
            value=None,
        )
        assert msg.action == ControlAction.PAUSE

    def test_control_actions(self):
        """Test all control actions exist."""
        assert ControlAction.PLAY is not None
        assert ControlAction.PAUSE is not None
        assert ControlAction.SEEK is not None
        assert ControlAction.SET_SPEED is not None
        assert ControlAction.STOP is not None

    def test_control_message_from_json(self):
        """Test parsing control message from JSON."""
        json_str = '{"type": "control", "action": "pause"}'
        msg = ControlMessage.from_json(json_str)
        assert msg.action == ControlAction.PAUSE

    def test_control_message_with_value(self):
        """Test control message with value."""
        json_str = '{"type": "control", "action": "seek", "value": 0.5}'
        msg = ControlMessage.from_json(json_str)
        assert msg.action == ControlAction.SEEK
        assert msg.value == 0.5

    def test_control_message_to_json(self):
        """Test serializing control message."""
        msg = ControlMessage(action=ControlAction.SET_SPEED, value=2.0)
        json_str = msg.to_json()
        data = json.loads(json_str)
        assert data["action"] == "set_speed"
        assert data["value"] == 2.0


class TestFrameBuffer:
    """Tests for frame buffering."""

    def test_create_buffer(self):
        """Test buffer creation."""
        buffer = FrameBuffer(max_size=10)
        assert buffer.max_size == 10
        assert len(buffer) == 0

    def test_buffer_add_frame(self):
        """Test adding frames to buffer."""
        buffer = FrameBuffer(max_size=10)
        frame = UnrealDataFrame(
            timestamp=0.0,
            frame_number=0,
            joints={},
        )
        buffer.add(frame)
        assert len(buffer) == 1

    def test_buffer_overflow(self):
        """Test buffer overflow handling."""
        buffer = FrameBuffer(max_size=3)
        for i in range(5):
            buffer.add(UnrealDataFrame(timestamp=float(i), frame_number=i, joints={}))
        assert len(buffer) == 3
        # Oldest frames should be dropped
        assert buffer.peek().frame_number == 2

    def test_buffer_get_frame(self):
        """Test getting frame from buffer."""
        buffer = FrameBuffer(max_size=10)
        frame = UnrealDataFrame(timestamp=0.0, frame_number=0, joints={})
        buffer.add(frame)
        retrieved = buffer.get()
        assert retrieved.frame_number == 0
        assert len(buffer) == 0

    def test_buffer_peek(self):
        """Test peeking at buffer without removing."""
        buffer = FrameBuffer(max_size=10)
        frame = UnrealDataFrame(timestamp=0.0, frame_number=0, joints={})
        buffer.add(frame)
        peeked = buffer.peek()
        assert peeked.frame_number == 0
        assert len(buffer) == 1  # Frame still in buffer

    def test_buffer_clear(self):
        """Test clearing buffer."""
        buffer = FrameBuffer(max_size=10)
        for i in range(5):
            buffer.add(UnrealDataFrame(timestamp=float(i), frame_number=i, joints={}))
        buffer.clear()
        assert len(buffer) == 0

    def test_buffer_is_empty(self):
        """Test empty buffer check."""
        buffer = FrameBuffer(max_size=10)
        assert buffer.is_empty
        buffer.add(UnrealDataFrame(timestamp=0.0, frame_number=0, joints={}))
        assert not buffer.is_empty

    def test_buffer_is_full(self):
        """Test full buffer check."""
        buffer = FrameBuffer(max_size=2)
        assert not buffer.is_full
        buffer.add(UnrealDataFrame(timestamp=0.0, frame_number=0, joints={}))
        buffer.add(UnrealDataFrame(timestamp=1.0, frame_number=1, joints={}))
        assert buffer.is_full


class TestStreamingProtocol:
    """Tests for streaming protocol messages."""

    def test_frame_message_format(self):
        """Test frame message format."""
        frame = UnrealDataFrame(
            timestamp=0.0167,
            frame_number=1,
            joints={},
        )
        msg = StreamingProtocol.create_frame_message(frame)
        assert msg["type"] == "frame"
        assert "data" in msg
        assert msg["data"]["timestamp"] == 0.0167

    def test_status_message_format(self):
        """Test status message format."""
        msg = StreamingProtocol.create_status_message(
            state=StreamingState.RUNNING,
            fps=59.8,
            frames_sent=1000,
        )
        assert msg["type"] == "status"
        assert msg["state"] == "running"
        assert msg["fps"] == 59.8

    def test_error_message_format(self):
        """Test error message format."""
        msg = StreamingProtocol.create_error_message(
            error_code="BUFFER_OVERFLOW",
            message="Frame buffer overflow",
        )
        assert msg["type"] == "error"
        assert msg["error_code"] == "BUFFER_OVERFLOW"

    def test_ack_message_format(self):
        """Test acknowledgment message format."""
        msg = StreamingProtocol.create_ack_message(
            frame_number=100,
            timestamp=1.667,
        )
        assert msg["type"] == "ack"
        assert msg["frame_number"] == 100


@pytest.mark.asyncio
class TestUnrealStreamingServer:
    """Tests for the streaming server."""

    async def test_server_creation(self):
        """Test server creation."""
        config = StreamingConfig(host="localhost", port=8765)
        server = UnrealStreamingServer(config=config)
        assert server.state == StreamingState.STOPPED
        assert server.config.port == 8765

    async def test_server_state_transitions(self):
        """Test server state transitions."""
        server = UnrealStreamingServer()
        assert server.state == StreamingState.STOPPED

        # Start should transition to STARTING then RUNNING
        # (In tests we mock the actual server start)

    async def test_server_broadcast_frame(self):
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

    async def test_server_queue_frame(self):
        """Test queuing frame for streaming."""
        server = UnrealStreamingServer()
        frame = UnrealDataFrame(timestamp=0.0, frame_number=0, joints={})

        server.queue_frame(frame)

        assert len(server._buffer) == 1

    async def test_server_statistics(self):
        """Test server statistics."""
        server = UnrealStreamingServer()
        stats = server.get_statistics()

        assert "frames_sent" in stats
        assert "clients_connected" in stats
        assert "uptime" in stats
        assert "average_fps" in stats

    async def test_server_client_management(self):
        """Test client connection management."""
        server = UnrealStreamingServer()
        assert server.client_count == 0

        mock_client = AsyncMock()
        await server._add_client(mock_client)
        assert server.client_count == 1

        await server._remove_client(mock_client)
        assert server.client_count == 0

    async def test_server_handle_control_message(self):
        """Test handling control messages."""
        server = UnrealStreamingServer()
        server._state = StreamingState.RUNNING

        # Test pause
        await server._handle_control(ControlMessage(action=ControlAction.PAUSE))
        assert server.state == StreamingState.PAUSED

        # Test play
        await server._handle_control(ControlMessage(action=ControlAction.PLAY))
        assert server.state == StreamingState.RUNNING

    async def test_server_playback_speed(self):
        """Test playback speed control."""
        server = UnrealStreamingServer()
        assert server.playback_speed == 1.0

        await server._handle_control(
            ControlMessage(action=ControlAction.SET_SPEED, value=0.5)
        )
        assert server.playback_speed == 0.5

    async def test_server_seek(self):
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


@pytest.mark.asyncio
class TestStreamingServerIntegration:
    """Integration-style tests for streaming server."""

    async def test_full_streaming_cycle(self):
        """Test complete streaming cycle."""
        server = UnrealStreamingServer(
            config=StreamingConfig(
                host="localhost",
                port=0,  # Use any available port
                target_fps=30,
            )
        )

        # Create test frames
        frames = [
            UnrealDataFrame(
                timestamp=i / 30.0,
                frame_number=i,
                joints={
                    "root": JointState(
                        name="root",
                        position=Vector3(x=0.0, y=float(i), z=0.0),
                        rotation=Quaternion.identity(),
                    )
                },
            )
            for i in range(100)
        ]

        # Queue all frames
        for frame in frames:
            server.queue_frame(frame)

        assert len(server._buffer) <= server.config.buffer_size

    async def test_streaming_with_metrics(self):
        """Test streaming with swing metrics."""
        from src.unreal_integration.data_models import SwingMetrics

        server = UnrealStreamingServer()

        frame = UnrealDataFrame(
            timestamp=0.5,
            frame_number=30,
            joints={},
            metrics=SwingMetrics(
                club_head_speed=45.0,
                x_factor=52.0,
            ),
        )

        server.queue_frame(frame)
        assert len(server._buffer) == 1


class TestStreamingPerformance:
    """Performance tests for streaming."""

    def test_frame_serialization_speed(self):
        """Test frame serialization is fast enough for real-time."""
        import time

        frame = UnrealDataFrame(
            timestamp=0.0,
            frame_number=0,
            joints={
                f"joint_{i}": JointState(
                    name=f"joint_{i}",
                    position=Vector3(x=float(i), y=0.0, z=0.0),
                    rotation=Quaternion.identity(),
                )
                for i in range(50)
            },
        )

        # Serialize many times
        start = time.perf_counter()
        for _ in range(1000):
            _ = frame.to_json()
        elapsed = time.perf_counter() - start

        # Should serialize 1000 frames in less than 1 second
        assert elapsed < 1.0
        # Average should be < 1ms per frame
        assert elapsed / 1000 < 0.001

    def test_buffer_throughput(self):
        """Test buffer can handle high throughput."""
        import time

        buffer = FrameBuffer(max_size=1000)

        start = time.perf_counter()
        for i in range(10000):
            buffer.add(UnrealDataFrame(timestamp=float(i), frame_number=i, joints={}))
        elapsed = time.perf_counter() - start

        # Should handle 10000 frames in less than 1 second
        assert elapsed < 1.0
