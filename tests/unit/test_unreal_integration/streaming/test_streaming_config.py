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


class TestStreamingConfig:
    """Tests for StreamingConfig."""

    def test_streaming_default_config(self) -> None:
        """Test default streaming configuration."""
        config = StreamingConfig()
        assert config.host == "localhost"
        assert config.port == 8765
        assert config.target_fps == 60
        assert config.buffer_size == 10

    def test_streaming_custom_config(self) -> None:
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

    def test_config_frame_interval(self) -> None:
        """Test frame interval calculation."""
        config = StreamingConfig(target_fps=60)
        assert config.frame_interval == pytest.approx(1 / 60)

    def test_streaming_config_validation(self) -> None:
        """Test configuration validation."""
        with pytest.raises(ValueError, match="port"):
            StreamingConfig(port=-1)
        with pytest.raises(ValueError, match="fps"):
            StreamingConfig(target_fps=0)
        with pytest.raises(ValueError, match="buffer"):
            StreamingConfig(buffer_size=0)

    def test_config_to_dict(self) -> None:
        """Test configuration serialization."""
        config = StreamingConfig()
        d = config.to_dict()
        assert "host" in d
        assert "port" in d
        assert "target_fps" in d

    def test_config_from_dict(self) -> None:
        """Test configuration deserialization."""
        d = {"host": "192.168.1.1", "port": 8080, "target_fps": 30}
        config = StreamingConfig.from_dict(d)
        assert config.host == "192.168.1.1"
        assert config.port == 8080


# ---------------------------------------------------------------------------
# Issue #2475: server must bind a real socket before reporting RUNNING
# ---------------------------------------------------------------------------
