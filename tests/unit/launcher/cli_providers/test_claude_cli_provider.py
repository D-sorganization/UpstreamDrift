"""Tests for ClaudeCliProvider."""

from __future__ import annotations

import asyncio

import pytest

from src.shared.python.ai.cli_providers import base as base_mod
from src.shared.python.ai.cli_providers.claude_cli_provider import (
    ClaudeCliProvider,
)
from src.shared.python.ai.cli_providers.contracts import CliProviderDescriptor


class _FakeStream:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    async def readline(self) -> bytes:
        if not self._lines:
            return b""
        return self._lines.pop(0)


class _FakeStdin:
    def __init__(self) -> None:
        self.written = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.written += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeProc:
    def __init__(
        self,
        stdout_lines: list[bytes],
        stderr_lines: list[bytes] | None = None,
    ) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStream(stdout_lines)
        self.stderr = _FakeStream(stderr_lines or [])
        self.returncode: int | None = None
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    async def wait(self) -> int:
        return self.returncode or 0


def _make_descriptor(tmp_path) -> CliProviderDescriptor:
    exe = tmp_path / "claude"
    exe.write_text("#!/bin/sh\n")
    return CliProviderDescriptor(
        id="claude-cli",
        name="Claude CLI",
        executable_path=str(exe),
        transport="stdio",
    )


def test_is_available_false_when_path_missing() -> None:
    provider = ClaudeCliProvider(
        descriptor=CliProviderDescriptor(
            id="claude-cli", name="Claude CLI", executable_path=""
        )
    )
    assert provider.is_available() is False


def test_is_available_true_when_file_exists(tmp_path) -> None:
    provider = ClaudeCliProvider(descriptor=_make_descriptor(tmp_path))
    assert provider.is_available() is True


def test_send_raises_when_unavailable() -> None:
    provider = ClaudeCliProvider(
        descriptor=CliProviderDescriptor(
            id="claude-cli", name="Claude CLI", executable_path=""
        )
    )

    async def run() -> None:
        async for _ in provider.send("hi"):
            pass

    with pytest.raises(base_mod.CliProviderUnavailableError):
        asyncio.run(run())


def test_send_streams_stdout_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake = _FakeProc(stdout_lines=[b"hello\n", b"world\n"])

    async def fake_spawn(*_a, **_kw):
        return fake

    monkeypatch.setattr(base_mod, "_spawn_subprocess", fake_spawn)

    provider = ClaudeCliProvider(descriptor=_make_descriptor(tmp_path))

    async def run() -> list[str]:
        return [chunk.text async for chunk in provider.send("ping")]

    chunks = asyncio.run(run())
    assert "hello\n" in chunks
    assert "world\n" in chunks
    assert fake.stdin.written == b"ping"
    assert fake.terminated is True


def test_send_classifies_tool_use_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fake = _FakeProc(
        stdout_lines=[],
        stderr_lines=[b'tool_use:{"name":"read_file"}\n'],
    )

    async def fake_spawn(*_a, **_kw):
        return fake

    monkeypatch.setattr(base_mod, "_spawn_subprocess", fake_spawn)

    provider = ClaudeCliProvider(descriptor=_make_descriptor(tmp_path))

    async def run() -> list[tuple[str, str]]:
        return [(chunk.kind, chunk.text) async for chunk in provider.send("x")]

    chunks = asyncio.run(run())
    kinds = {kind for kind, _ in chunks}
    assert "tool_use" in kinds


def test_cancel_idempotent() -> None:
    provider = ClaudeCliProvider(
        descriptor=CliProviderDescriptor(
            id="claude-cli", name="Claude CLI", executable_path=""
        )
    )
    asyncio.run(provider.cancel())
    asyncio.run(provider.cancel())  # second call must be a no-op
