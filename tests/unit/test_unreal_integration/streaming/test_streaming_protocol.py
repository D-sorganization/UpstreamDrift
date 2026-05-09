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


class TestStreamingProtocol:
    """Tests for streaming protocol messages."""

    def test_frame_message_format(self) -> None:
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

    def test_status_message_format(self) -> None:
        """Test status message format."""
        msg = StreamingProtocol.create_status_message(
            state=StreamingState.RUNNING,
            fps=59.8,
            frames_sent=1000,
        )
        assert msg["type"] == "status"
        assert msg["state"] == "running"
        assert msg["fps"] == 59.8

    def test_error_message_format(self) -> None:
        """Test error message format."""
        msg = StreamingProtocol.create_error_message(
            error_code="BUFFER_OVERFLOW",
            message="Frame buffer overflow",
        )
        assert msg["type"] == "error"
        assert msg["error_code"] == "BUFFER_OVERFLOW"

    def test_ack_message_format(self) -> None:
        """Test acknowledgment message format."""
        msg = StreamingProtocol.create_ack_message(
            frame_number=100,
            timestamp=1.667,
        )
        assert msg["type"] == "ack"
        assert msg["frame_number"] == 100


# ---------------------------------------------------------------------------
# Issue #2475: server must bind a real socket before reporting RUNNING
# ---------------------------------------------------------------------------
