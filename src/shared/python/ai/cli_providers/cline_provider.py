"""Cline provider — connects to the local Cline VS Code server.

Cline is a VS Code extension that exposes a local socket for
external chat clients. Unlike Claude CLI / Codex CLI it is not a
subprocess we spawn — discovery probes a TCP port to decide whether
it's running.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator, Mapping
from typing import Any

from src.shared.python.ai.cli_providers.base import (
    CliProvider,
    CliProviderUnavailableError,
    ResponseChunk,
)
from src.shared.python.ai.cli_providers.contracts import CliProviderDescriptor

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 47261


class ClineProvider(CliProvider):
    """Socket-backed adapter for the local Cline VS Code server.

    Falls back to a clear "Cline not running" error when the server
    socket is not reachable, so the chat header can show a helpful
    error widget instead of a silent failure.
    """

    PROVIDER_ID = "cline"

    def __init__(
        self,
        descriptor: CliProviderDescriptor | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        """Initialize with explicit or default socket coordinates.

        Args:
            descriptor: Optional pre-built descriptor.
            host: Cline server hostname.
            port: Cline server TCP port.
        """
        if descriptor is None:
            descriptor = CliProviderDescriptor(
                id=self.PROVIDER_ID,
                name="Cline",
                executable_path="",
                transport="socket",
                required_env=(),
                working_dir_aware=False,
            )
        super().__init__(descriptor)
        self._host = host
        self._port = port
        self._writer: asyncio.StreamWriter | None = None
        self._reader: asyncio.StreamReader | None = None

    def is_available(self) -> bool:
        """Probe the TCP port; True when something accepts a connect.

        Uses a fast non-blocking check so this can be called from
        synchronous discovery code without an event loop.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)
        try:
            return sock.connect_ex((self._host, self._port)) == 0
        except OSError:
            return False
        finally:
            sock.close()

    async def send(
        self,
        message: str,
        context: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[ResponseChunk]:
        """Send ``message`` over the socket and stream reply lines.

        Raises:
            CliProviderUnavailableError: If Cline is not reachable.

        Contract:
            Pre: ``is_available()`` is True.
            Post: socket is closed.
        """
        if not self.is_available():
            raise CliProviderUnavailableError(
                f"Cline server not running on {self._host}:{self._port}",
            )

        self._reader, self._writer = await asyncio.open_connection(
            self._host, self._port
        )

        try:
            self._writer.write(message.encode("utf-8") + b"\n")
            await self._writer.drain()

            while True:
                line = await self._reader.readline()
                if not line:
                    break
                yield ResponseChunk(text=line.decode("utf-8"), kind="text")
        finally:
            await self.cancel()

    async def cancel(self) -> None:
        """Close the socket connection (idempotent)."""
        writer = self._writer
        self._writer = None
        self._reader = None
        if writer is None:
            return
        try:
            writer.close()
            await writer.wait_closed()
        except (OSError, RuntimeError):
            return
