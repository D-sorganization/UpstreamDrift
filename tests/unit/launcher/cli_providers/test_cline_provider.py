"""Tests for ClineProvider."""

from __future__ import annotations

import asyncio
import socket

import pytest

from src.shared.python.ai.cli_providers import base as base_mod
from src.shared.python.ai.cli_providers.cline_provider import ClineProvider

pytestmark = pytest.mark.unit


def _find_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _echo_replies(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    received: list[bytes],
) -> None:
    """Echo ``reply-a``/``reply-b`` once for each received payload line.

    ``ClineProvider.is_available()`` probes the port with a real TCP
    connection that it closes immediately, so the server sees connections
    that end in EOF before any payload arrives. Those probe connections
    must not kill the handler task: an exception escaping
    ``client_connected_cb`` leaves the connection open, and on Python 3.12
    ``Server.wait_closed()`` then blocks forever waiting for it — the
    60-second hang seen in the CI Standard ``tests (3.12)`` lane (#9431).
    """

    try:
        line = await reader.readline()
        received.append(line)
        if not line:
            # A probe connection (``is_available()``): nothing to echo.
            return
        writer.write(b"reply-a\nreply-b\n")
        await writer.drain()
        writer.write_eof()
    except OSError:
        # The probe socket may already be reset by the time we reply; the
        # probe exchange is not what this server exists to test.
        return
    finally:
        writer.close()


def test_is_available_false_when_no_server() -> None:
    provider = ClineProvider(host="127.0.0.1", port=1)  # privileged port
    assert provider.is_available() is False


def test_send_raises_when_no_server() -> None:
    provider = ClineProvider(host="127.0.0.1", port=1)

    async def run() -> None:
        async for _ in provider.send("hi"):
            pass

    with pytest.raises(base_mod.CliProviderUnavailableError):
        asyncio.run(run())


def test_send_streams_socket_lines() -> None:
    port = _find_free_port()
    received: list[bytes] = []

    async def run() -> list[str]:
        server = await asyncio.start_server(
            lambda reader, writer: _echo_replies(reader, writer, received),
            "127.0.0.1",
            port,
        )
        provider = ClineProvider(host="127.0.0.1", port=port)
        assert provider.is_available() is True
        chunks = [c.text async for c in provider.send("ping")]
        server.close()
        await server.wait_closed()
        return chunks

    chunks = asyncio.run(run())
    assert "reply-a\n" in chunks
    assert "reply-b\n" in chunks
    # is_available() probes the port (which also accepts a connection),
    # so the meaningful payload may not be the first received frame.
    payloads = [r.rstrip(b"\n") for r in received]
    assert b"ping" in payloads


def test_cancel_idempotent() -> None:
    provider = ClineProvider(host="127.0.0.1", port=1)
    asyncio.run(provider.cancel())
    asyncio.run(provider.cancel())


class _ProbeCrashWriter:
    """A StreamWriter stub whose peer already closed: writes fail.

    Emulates the Linux CI runner behaviour that killed the handler task:
    ``writer.write_eof()`` raised ``OSError`` (``ENOTCONN``) after the
    probe socket's FIN, leaking the server connection and hanging
    ``Server.wait_closed()`` on Python 3.12 (issue #9431).
    """

    def __init__(self) -> None:
        self.closed = False

    def write_eof(self) -> None:
        raise OSError(107, "Transport endpoint is not connected")

    def write(self, data: bytes) -> None:  # pragma: no cover - unreachable
        raise OSError(107, "Transport endpoint is not connected")

    async def drain(self) -> None:  # pragma: no cover - unreachable
        raise OSError(107, "Transport endpoint is not connected")

    def close(self) -> None:
        self.closed = True


def test_echo_replies_survives_already_closed_peer() -> None:
    """RED→GREEN for #9431: a crashed probe must not leak the connection."""

    received: list[bytes] = []

    async def run() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(b"ping\n")
        writer: asyncio.StreamWriter = _ProbeCrashWriter()  # type: ignore[assignment]
        await _echo_replies(reader, writer, received)
        assert writer.closed is True

    asyncio.run(run())

    assert received == [b"ping\n"]


def test_echo_replies_ignores_probe_eof() -> None:
    received: list[bytes] = []

    async def run() -> None:
        reader = asyncio.StreamReader()
        reader.feed_eof()
        writer: asyncio.StreamWriter = _ProbeCrashWriter()  # type: ignore[assignment]
        await _echo_replies(reader, writer, received)
        assert writer.closed is True

    asyncio.run(run())

    assert received == [b""]


def test_echo_replies_wired_into_start_server_signature() -> None:
    """The helper must remain directly adaptable to ``client_connected_cb``."""

    async def check() -> None:
        received: list[bytes] = []
        port = _find_free_port()
        server = await asyncio.start_server(
            lambda reader, writer: _echo_replies(reader, writer, received),
            "127.0.0.1",
            port,
        )
        probe = socket.socket()
        probe.connect(("127.0.0.1", port))
        probe.close()

        client_reader, client_writer = await asyncio.open_connection("127.0.0.1", port)
        client_writer.write(b"ping\n")
        await client_writer.drain()
        lines = [await client_reader.readline(), await client_reader.readline()]
        assert lines == [b"reply-a\n", b"reply-b\n"]
        client_writer.close()

        server.close()
        await asyncio.wait_for(server.wait_closed(), timeout=5.0)

    asyncio.run(check())
