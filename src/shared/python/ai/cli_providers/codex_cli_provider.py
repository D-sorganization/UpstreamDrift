"""Codex CLI provider — wraps OpenAI's ``codex`` command-line agent.

Same subprocess/stdio contract as Claude CLI. Reuses the shared
``_spawn_subprocess`` helper to avoid duplicated launch logic.
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


class CodexCliProvider(CliProvider):
    """Subprocess-backed adapter for the ``codex`` CLI tool.

    Contract:
        - ``send()`` precondition: ``is_available()`` returns True.
        - ``cancel()`` is idempotent.
    """

    PROVIDER_ID = "codex-cli"

    def __init__(self, descriptor: CliProviderDescriptor | None = None) -> None:
        """Initialize with an explicit or auto-discovered descriptor."""
        if descriptor is None:
            descriptor = CliProviderDescriptor(
                id=self.PROVIDER_ID,
                name="Codex CLI",
                executable_path=shutil.which("codex") or "",
                transport="stdio",
                required_env=(),
                working_dir_aware=True,
            )
        super().__init__(descriptor)
        self._proc: Any = None

    def is_available(self) -> bool:
        """Available when the descriptor points at an existing file."""
        path = self._descriptor.executable_path
        return bool(path) and os.path.exists(path)

    async def send(
        self,
        message: str,
        context: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[ResponseChunk]:
        """Send ``message`` to ``codex`` and stream stdout chunks.

        Raises:
            CliProviderUnavailableError: If the executable is missing.

        Contract:
            Pre: ``is_available()`` is True.
            Post: subprocess is terminated.
        """
        if not self.is_available():
            raise CliProviderUnavailableError(
                "Codex CLI executable not found on PATH",
            )

        cwd = None
        if context is not None:
            cwd = context.get("working_dir")

        self._proc = await self._spawn(
            args=("--stream",),
            cwd=cwd,
        )

        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write(message.encode("utf-8"))
            await self._proc.stdin.drain()
            self._proc.stdin.close()

            assert self._proc.stdout is not None
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                yield ResponseChunk(text=line.decode("utf-8"), kind="text")
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
