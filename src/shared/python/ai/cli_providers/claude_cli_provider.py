"""Claude CLI provider — wraps Anthropic's ``claude`` command.

Streams text from the subprocess's stdout. Tool-use events are detected
on stderr via a known ``"tool_use:"`` JSON marker that the Claude CLI
emits when it invokes a local hook.

Used by the chat header dropdown when the user picks the "Claude CLI"
entry under the "CLI Agents" section.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import AsyncIterator, Mapping
from typing import Any

from src.shared.python.ai.cli_providers.base import (
    CliProvider,
    CliProviderUnavailableError,
    ResponseChunk,
)
from src.shared.python.ai.cli_providers.contracts import CliProviderDescriptor


class ClaudeCliProvider(CliProvider):
    """Subprocess-backed adapter for the ``claude`` CLI tool.

    Contract:
        - ``send()`` precondition: ``is_available()`` returns True.
        - ``cancel()`` is idempotent.
    """

    PROVIDER_ID = "claude-cli"

    def __init__(self, descriptor: CliProviderDescriptor | None = None) -> None:
        """Initialize with an explicit or auto-discovered descriptor.

        Args:
            descriptor: Optional pre-built descriptor. When omitted, a
                default descriptor is constructed via ``shutil.which``.
        """
        if descriptor is None:
            descriptor = CliProviderDescriptor(
                id=self.PROVIDER_ID,
                name="Claude CLI",
                executable_path=shutil.which("claude") or "",
                transport="stdio",
                required_env=(),
                working_dir_aware=True,
            )
        super().__init__(descriptor)
        self._proc: Any = None

    def is_available(self) -> bool:
        """Available when the descriptor points at an existing file.

        Returns:
            True if ``executable_path`` is set and the file exists.
        """
        path = self._descriptor.executable_path
        return bool(path) and os.path.exists(path)

    async def send(
        self,
        message: str,
        context: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[ResponseChunk]:
        """Send ``message`` to ``claude`` and stream stdout chunks.

        Args:
            message: User prompt; written to the subprocess's stdin.
            context: Optional context dict; ``"working_dir"`` is the
                only honored key.

        Yields:
            ``ResponseChunk(text=..., kind="text"|"tool_use"|"error")``
            in arrival order.

        Raises:
            CliProviderUnavailableError: If the executable is missing.

        Contract:
            Pre: ``is_available()`` is True.
            Post: subprocess is terminated and ``self._proc`` is None.
        """
        if not self.is_available():
            raise CliProviderUnavailableError(
                "Claude CLI executable not found on PATH",
            )

        cwd = None
        if context is not None:
            cwd = context.get("working_dir")

        self._proc = await self._spawn(
            args=("--stream", "--no-color"),
            cwd=cwd,
        )

        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write(message.encode("utf-8"))
            await self._proc.stdin.drain()
            self._proc.stdin.close()

            async for chunk in _read_streams(self._proc):
                yield chunk
        finally:
            await self.cancel()

    async def cancel(self) -> None:
        """Terminate the running subprocess (idempotent)."""
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        if proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                return
            try:
                await proc.wait()
            except ChildProcessError:
                return


async def _read_streams(proc: Any) -> AsyncIterator[ResponseChunk]:
    """Interleave stdout/stderr line-reads into ``ResponseChunk``s.

    Stderr lines starting with ``"tool_use:"`` become ``tool_use``
    chunks; other stderr lines become ``error`` chunks. All stdout
    lines become ``text`` chunks. The helper exits once both streams
    are drained.
    """
    assert proc.stdout is not None
    assert proc.stderr is not None

    while True:
        stdout_line = await proc.stdout.readline()
        if stdout_line:
            yield ResponseChunk(text=stdout_line.decode("utf-8"), kind="text")

        stderr_line = await proc.stderr.readline()
        if stderr_line:
            decoded = stderr_line.decode("utf-8")
            if decoded.startswith("tool_use:"):
                yield ResponseChunk(text=decoded, kind="tool_use")
            else:
                yield ResponseChunk(text=decoded, kind="error")

        if not stdout_line and not stderr_line:
            return
