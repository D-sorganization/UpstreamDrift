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
class TestStreamingServerIntegration:
    """Integration-style tests for streaming server."""

    async def test_full_streaming_cycle(self) -> None:
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

    async def test_streaming_with_metrics(self) -> None:
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


# ---------------------------------------------------------------------------
# Issue #2475: server must bind a real socket before reporting RUNNING
# ---------------------------------------------------------------------------
