"""Tests for async/background output I/O."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.shared.python.data_io._async_io import (
    get_io_executor,
    shutdown_executor,
    submit_async_save,
    submit_background_save,
)
from src.shared.python.data_io._format_handlers import OutputFormat


@pytest.fixture(autouse=True)
def _cleanup_executor():
    yield
    shutdown_executor()


def _fake_save(results, filename, format_type, engine, metadata) -> Path:
    p = Path(filename)
    p.write_text("done")
    return p


def _failing_save(results, filename, format_type, engine, metadata) -> Path:
    raise ValueError("boom")


def test_get_io_executor_returns_thread_pool():
    ex = get_io_executor()
    assert isinstance(ex, ThreadPoolExecutor)


def test_get_io_executor_is_singleton():
    a = get_io_executor()
    b = get_io_executor()
    assert a is b


def test_shutdown_executor_resets_singleton():
    a = get_io_executor()
    shutdown_executor()
    b = get_io_executor()
    assert a is not b


def test_submit_async_save_resolves_to_path(tmp_path: Path):
    target = tmp_path / "out.csv"
    fut = submit_async_save(
        _fake_save,
        results={"k": 1},
        filename=str(target),
        format_type=OutputFormat.CSV,
        engine="mujoco",
        metadata=None,
    )
    result = fut.result(timeout=5)
    assert result == target
    assert target.exists()


def test_submit_async_save_invokes_success_callback(tmp_path: Path):
    target = tmp_path / "out.csv"
    received: list = []
    fut = submit_async_save(
        _fake_save,
        results=None,
        filename=str(target),
        format_type=OutputFormat.CSV,
        engine="mj",
        metadata=None,
        callback=received.append,
    )
    fut.result(timeout=5)
    assert received == [target]


def test_submit_async_save_invokes_error_callback(tmp_path: Path):
    received: list = []
    fut = submit_async_save(
        _failing_save,
        results=None,
        filename=str(tmp_path / "x.csv"),
        format_type=OutputFormat.CSV,
        engine="mj",
        metadata=None,
        callback=received.append,
    )
    with pytest.raises(ValueError, match="boom"):
        fut.result(timeout=5)
    assert len(received) == 1
    assert isinstance(received[0], ValueError)


def test_submit_background_save_on_complete(tmp_path: Path):
    target = tmp_path / "bg.csv"
    completed: list = []
    submit_background_save(
        _fake_save,
        results=None,
        filename=str(target),
        format_type=OutputFormat.CSV,
        engine="mj",
        metadata=None,
        on_complete=completed.append,
    )
    # Drain
    get_io_executor().shutdown(wait=True)
    assert completed == [target]


def test_submit_background_save_on_error(tmp_path: Path):
    errors: list = []
    submit_background_save(
        _failing_save,
        results=None,
        filename=str(tmp_path / "x.csv"),
        format_type=OutputFormat.CSV,
        engine="mj",
        metadata=None,
        on_error=errors.append,
    )
    get_io_executor().shutdown(wait=True)
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)


def test_submit_background_save_default_error_logged(tmp_path: Path):
    # No on_error supplied: should not raise, error is logged.
    submit_background_save(
        _failing_save,
        results=None,
        filename=str(tmp_path / "x.csv"),
        format_type=OutputFormat.CSV,
        engine="mj",
        metadata=None,
    )
    get_io_executor().shutdown(wait=True)
