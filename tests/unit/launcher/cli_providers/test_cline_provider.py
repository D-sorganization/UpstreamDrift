"""Tests for ClineProvider."""

from __future__ import annotations

import asyncio
import socket

import pytest

from src.shared.python.ai.cli_providers import base as base_mod
from src.shared.python.ai.cli_providers.cline_provider import ClineProvider


def _find_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


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

    async def _handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        line = await reader.readline()
        received.append(line)
        writer.write(b"reply-a\nreply-b\n")
        await writer.drain()
        writer.write_eof()

    async def run() -> list[str]:
        server = await asyncio.start_server(_handle, "127.0.0.1", port)
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
