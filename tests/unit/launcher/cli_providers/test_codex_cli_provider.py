"""Tests for CodexCliProvider."""

from __future__ import annotations

import asyncio

import pytest

from src.shared.python.ai.cli_providers import base as base_mod
from src.shared.python.ai.cli_providers.codex_cli_provider import (
    CodexCliProvider,
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

    def write(self, data: bytes) -> None:
        self.written += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeProc:
    def __init__(self, lines: list[bytes]) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStream(lines)
        self.stderr = _FakeStream([])
        self.returncode: int | None = None

    def terminate(self) -> None:
        self.returncode = -15

    async def wait(self) -> int:
        return 0


def _make_descriptor(tmp_path) -> CliProviderDescriptor:
    exe = tmp_path / "codex"
    exe.write_text("#!/bin/sh\n")
    return CliProviderDescriptor(
        id="codex-cli", name="Codex CLI", executable_path=str(exe)
    )


def test_is_available_false_when_path_missing() -> None:
    provider = CodexCliProvider(
        descriptor=CliProviderDescriptor(
            id="codex-cli", name="Codex CLI", executable_path=""
        )
    )
    assert provider.is_available() is False


def test_send_raises_when_unavailable() -> None:
    provider = CodexCliProvider(
        descriptor=CliProviderDescriptor(
            id="codex-cli", name="Codex CLI", executable_path=""
        )
    )

    async def run() -> None:
        async for _ in provider.send("hi"):
            pass

    with pytest.raises(base_mod.CliProviderUnavailableError):
        asyncio.run(run())


def test_send_streams_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake = _FakeProc(lines=[b"a\n", b"b\n"])

    async def fake_spawn(*_a, **_kw):
        return fake

    monkeypatch.setattr(base_mod, "_spawn_subprocess", fake_spawn)

    provider = CodexCliProvider(descriptor=_make_descriptor(tmp_path))

    async def run() -> list[str]:
        return [c.text async for c in provider.send("prompt")]

    chunks = asyncio.run(run())
    assert chunks == ["a\n", "b\n"]
    assert fake.stdin.written == b"prompt"


def test_cancel_idempotent() -> None:
    provider = CodexCliProvider(
        descriptor=CliProviderDescriptor(
            id="codex-cli", name="Codex CLI", executable_path=""
        )
    )
    asyncio.run(provider.cancel())
    asyncio.run(provider.cancel())
