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


class TestFrameBuffer:
    """Tests for frame buffering."""

    def test_create_buffer(self, event_loop) -> None:
        """Test buffer creation."""
        buffer = FrameBuffer(max_size=10)
        assert buffer.max_size == 10
        assert len(buffer) == 0

    def test_buffer_add_frame(self, event_loop) -> None:
        """Test adding frames to buffer."""
        buffer = FrameBuffer(max_size=10)
        frame = UnrealDataFrame(
            timestamp=0.0,
            frame_number=0,
            joints={},
        )
        buffer.add(frame)
        assert len(buffer) == 1

    def test_buffer_overflow(self, event_loop) -> None:
        """Test buffer overflow handling."""
        buffer = FrameBuffer(max_size=3)
        for i in range(5):
            buffer.add(UnrealDataFrame(timestamp=float(i), frame_number=i, joints={}))
        assert len(buffer) == 3
        # Oldest frames should be dropped
        assert buffer.peek().frame_number == 2

    def test_buffer_get_frame(self, event_loop) -> None:
        """Test getting frame from buffer."""
        buffer = FrameBuffer(max_size=10)
        frame = UnrealDataFrame(timestamp=0.0, frame_number=0, joints={})
        buffer.add(frame)
        retrieved = buffer.get()
        assert retrieved.frame_number == 0
        assert len(buffer) == 0

    def test_buffer_peek(self, event_loop) -> None:
        """Test peeking at buffer without removing."""
        buffer = FrameBuffer(max_size=10)
        frame = UnrealDataFrame(timestamp=0.0, frame_number=0, joints={})
        buffer.add(frame)
        peeked = buffer.peek()
        assert peeked.frame_number == 0
        assert len(buffer) == 1  # Frame still in buffer

    def test_buffer_clear(self, event_loop) -> None:
        """Test clearing buffer."""
        buffer = FrameBuffer(max_size=10)
        for i in range(5):
            buffer.add(UnrealDataFrame(timestamp=float(i), frame_number=i, joints={}))
        buffer.clear()
        assert len(buffer) == 0

    def test_buffer_is_empty(self, event_loop) -> None:
        """Test empty buffer check."""
        buffer = FrameBuffer(max_size=10)
        assert buffer.is_empty
        buffer.add(UnrealDataFrame(timestamp=0.0, frame_number=0, joints={}))
        assert not buffer.is_empty

    def test_buffer_is_full(self, event_loop) -> None:
        """Test full buffer check."""
        buffer = FrameBuffer(max_size=2)
        assert not buffer.is_full
        buffer.add(UnrealDataFrame(timestamp=0.0, frame_number=0, joints={}))
        buffer.add(UnrealDataFrame(timestamp=1.0, frame_number=1, joints={}))
        assert buffer.is_full


# ---------------------------------------------------------------------------
# Issue #2475: server must bind a real socket before reporting RUNNING
# ---------------------------------------------------------------------------
