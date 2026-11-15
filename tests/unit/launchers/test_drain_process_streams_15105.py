"""Regression for #15105 — drain_process_streams must not spin on MagicMock pipes."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from src.launchers.managed_process import drain_process_streams

pytestmark = pytest.mark.unit

_MAX_DRAIN_SECONDS = 2.0


def test_drain_process_streams_terminates_on_magicmock_readline() -> None:
    """Unconfigured MagicMock readline never equals b\"\"; draining must exit promptly."""
    mock_stdout = MagicMock()
    mock_stdout.readline.return_value = MagicMock(name="non_bytes_readline")
    mock_stdout.close = MagicMock()

    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = None

    registry: dict[str, object] = {}
    lock = threading.RLock()
    emitted: list[tuple[str, str]] = []

    started = time.monotonic()
    drain_process_streams(
        "mock-slot",
        mock_process,
        managed_id=None,
        registry=registry,
        lock=lock,
        emit_output=lambda slot, line: emitted.append((slot, line)),
    )
    elapsed = time.monotonic() - started

    assert elapsed < _MAX_DRAIN_SECONDS, (
        "drain_process_streams did not return; likely infinite readline loop (#15105)"
    )
    mock_stdout.close.assert_called_once()
