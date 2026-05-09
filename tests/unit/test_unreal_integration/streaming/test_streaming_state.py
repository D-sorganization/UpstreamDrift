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


class TestStreamingState:
    """Tests for StreamingState enum."""

    def test_streaming_states(self) -> None:
        """Test all streaming states exist."""
        assert StreamingState.STOPPED is not None
        assert StreamingState.STARTING is not None
        assert StreamingState.RUNNING is not None
        assert StreamingState.PAUSED is not None
        assert StreamingState.STOPPING is not None
        assert StreamingState.ERROR is not None

    def test_streaming_state_is_active(self) -> None:
        """Test is_active property."""
        assert StreamingState.RUNNING.is_active
        assert StreamingState.PAUSED.is_active
        assert not StreamingState.STOPPED.is_active
        assert not StreamingState.ERROR.is_active


# ---------------------------------------------------------------------------
# Issue #2475: server must bind a real socket before reporting RUNNING
# ---------------------------------------------------------------------------
