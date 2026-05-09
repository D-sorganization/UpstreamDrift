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


class TestStreamingPerformance:
    """Performance tests for streaming."""

    def test_frame_serialization_speed(self) -> None:
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

        # CI runners vary noticeably on JSON-heavy serialization work.
        # Generous threshold: 3s for 1000 iterations on slow GH Actions runners.
        assert elapsed < 3.0
        assert elapsed / 1000 < 0.003

    def test_buffer_throughput(self, event_loop) -> None:
        """Test buffer can handle high throughput."""
        import time

        buffer = FrameBuffer(max_size=1000)

        start = time.perf_counter()
        for i in range(10000):
            buffer.add(UnrealDataFrame(timestamp=float(i), frame_number=i, joints={}))
        elapsed = time.perf_counter() - start

        # Should handle 10000 frames in less than 1 second
        assert elapsed < 1.0


# ---------------------------------------------------------------------------
# Issue #2475: server must bind a real socket before reporting RUNNING
# ---------------------------------------------------------------------------
