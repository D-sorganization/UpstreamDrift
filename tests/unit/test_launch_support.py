"""Tests for the shared launch-error handler (issue #7168)."""

from __future__ import annotations

import logging

import pytest

from src.shared.python.launch_support import EXIT_LAUNCH_FAILURE, run_launch

pytestmark = pytest.mark.unit


def test_run_launch_passes_through_success() -> None:
    calls = []
    run_launch(lambda: calls.append("ran"))
    assert calls == ["ran"]


def test_run_launch_converts_failure_to_clean_exit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _boom() -> None:
        raise RuntimeError("engine not available")

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as excinfo:
        run_launch(_boom, hint="Run with --help.")

    assert excinfo.value.code == EXIT_LAUNCH_FAILURE
    messages = " ".join(r.getMessage() for r in caplog.records)
    # Actionable message: exception type + message + hint, no raw traceback.
    assert "RuntimeError" in messages
    assert "engine not available" in messages
    assert "--help" in messages
    # No traceback rendered at default (ERROR) verbosity.
    assert "Traceback" not in messages


def test_run_launch_propagates_keyboard_interrupt() -> None:
    def _interrupt() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_launch(_interrupt)


def test_run_launch_propagates_systemexit_unchanged() -> None:
    def _exit() -> None:
        raise SystemExit(2)  # argparse-style bad-args code must survive

    with pytest.raises(SystemExit) as excinfo:
        run_launch(_exit)
    assert excinfo.value.code == 2


def test_run_launch_emits_traceback_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _boom() -> None:
        raise ValueError("detail")

    with caplog.at_level(logging.DEBUG), pytest.raises(SystemExit):
        run_launch(_boom)

    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any(r.exc_info for r in debug_records)
