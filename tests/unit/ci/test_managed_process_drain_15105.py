"""Regression for #15105: drain_process_streams must not spin on MagicMock readline."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from src.launchers.managed_process import drain_process_streams

pytestmark = pytest.mark.unit

_DRAIN_TIMEOUT_S = 0.5


def test_drain_process_streams_exits_promptly_with_magic_mock_readline() -> None:
    stderr = MagicMock()
    stderr.readline = MagicMock(return_value=MagicMock())

    proc = MagicMock()
    proc.stdout = MagicMock()
    proc.stdout.readline = MagicMock(return_value=MagicMock())
    proc.stderr = stderr

    registry: dict[str, object] = {}
    lock = threading.RLock()
    emitted: list[str] = []

    start = time.monotonic()
    drain_process_streams(
        "slot",
        proc,
        managed_id=None,
        registry=registry,  # type: ignore[arg-type]
        lock=lock,
        emit_output=lambda _slot, line: emitted.append(line),
    )
    elapsed = time.monotonic() - start

    assert elapsed < _DRAIN_TIMEOUT_S, (
        f"drain_process_streams blocked for {elapsed:.2f}s; "
        "MagicMock readline must not spin forever"
    )
