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


class TestControlMessage:
    """Tests for control message handling."""

    def test_create_control_message(self) -> None:
        """Test control message creation."""
        msg = ControlMessage(
            action=ControlAction.PAUSE,
            value=None,
        )
        assert msg.action == ControlAction.PAUSE

    def test_control_actions(self) -> None:
        """Test all control actions exist."""
        assert ControlAction.PLAY is not None
        assert ControlAction.PAUSE is not None
        assert ControlAction.SEEK is not None
        assert ControlAction.SET_SPEED is not None
        assert ControlAction.STOP is not None

    def test_control_message_from_json(self) -> None:
        """Test parsing control message from JSON."""
        json_str = '{"type": "control", "action": "pause"}'
        msg = ControlMessage.from_json(json_str)
        assert msg.action == ControlAction.PAUSE

    def test_control_message_with_value(self) -> None:
        """Test control message with value."""
        json_str = '{"type": "control", "action": "seek", "value": 0.5}'
        msg = ControlMessage.from_json(json_str)
        assert msg.action == ControlAction.SEEK
        assert msg.value == 0.5

    def test_control_message_to_json(self) -> None:
        """Test serializing control message."""
        msg = ControlMessage(action=ControlAction.SET_SPEED, value=2.0)
        json_str = msg.to_json()
        data = json.loads(json_str)
        assert data["action"] == "set_speed"
        assert data["value"] == 2.0


# ---------------------------------------------------------------------------
# Issue #2475: server must bind a real socket before reporting RUNNING
# ---------------------------------------------------------------------------
