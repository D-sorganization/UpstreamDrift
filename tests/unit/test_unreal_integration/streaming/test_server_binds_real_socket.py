"""Unit tests for Unreal Engine WebSocket streaming.

TDD tests for the streaming server that sends data to Unreal Engine.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator

import pytest
from src.unreal_integration.streaming import (  # noqa: E402
    StreamingConfig,
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


# ---------------------------------------------------------------------------
# Issue #2475: server must bind a real socket before reporting RUNNING
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestServerBindsRealSocket:
    """Verify server actually opens a socket before transitioning to RUNNING.

    Before the fix, start() set state = RUNNING without ever calling
    asyncio.start_server() or any equivalent. A client would therefore
    never be able to connect despite the backend claiming it was ready.
    """

    async def test_start_transitions_to_running_only_after_bind(self) -> None:
        """Server must be in RUNNING state after start()."""
        server = UnrealStreamingServer(config=StreamingConfig(host="127.0.0.1", port=0))
        assert server.state == StreamingState.STOPPED
        await server.start()
        try:
            assert server.state == StreamingState.RUNNING, (
                "Server is not RUNNING after start()"
            )
        finally:
            await server.stop()

    async def test_client_can_connect_after_start(self) -> None:
        """A raw TCP client must be able to connect to the bound port.

        This is the regression test for #2475: previously the server claimed
        RUNNING but no socket was ever opened, so every connection attempt
        would be refused.
        """
        server = UnrealStreamingServer(config=StreamingConfig(host="127.0.0.1", port=0))
        await server.start()
        try:
            port = server.bound_port
            assert port > 0, "Server must expose its bound port"

            # A plain TCP connection must succeed (not be refused)
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await writer.wait_closed()
        finally:
            await server.stop()

    async def test_start_fails_cleanly_on_port_conflict(self) -> None:
        """If the port is already in use, start() must set ERROR state and raise."""
        import socket as _socket

        # Bind a raw socket to steal the port
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 0)
        sock.bind(("127.0.0.1", 0))
        stolen_port = sock.getsockname()[1]
        sock.listen(1)

        server = UnrealStreamingServer(
            config=StreamingConfig(host="127.0.0.1", port=stolen_port)
        )
        try:
            with pytest.raises((RuntimeError, OSError)):
                await server.start()
            assert server.state == StreamingState.ERROR, (
                "Server must be in ERROR state when binding fails"
            )
        finally:
            sock.close()
