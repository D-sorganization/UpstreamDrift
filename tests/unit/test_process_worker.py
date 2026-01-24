"""Unit tests for ProcessWorker."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.engine_availability import PYQT6_AVAILABLE, skip_if_unavailable

# Skip entire module if PyQt6 is not available - the ProcessWorker tests require
# proper Qt mocking that doesn't work reliably when PyQt6 is missing
pytestmark = skip_if_unavailable("pyqt6")


@pytest.fixture
def mock_pyqt6():
    """Provide ProcessWorker with mocked signals for testing."""
    # Clear any cached module to ensure fresh import
    if "src.shared.python.process_worker" in sys.modules:
        del sys.modules["src.shared.python.process_worker"]

    from src.shared.python.process_worker import ProcessWorker

    yield ProcessWorker


@pytest.fixture
def worker(mock_pyqt6):
    # Since we use fallback, we can instantiate it directly
    ProcessWorker = mock_pyqt6
    return ProcessWorker(["echo", "hello"])


def test_initialization(worker):
    assert worker.cmd == ["echo", "hello"]
    assert worker._is_running is True


@patch("subprocess.Popen")
def test_run_success(mock_popen, worker):
    # Setup mock process
    process = MagicMock()
    # readline returns lines then empty string
    process.stdout.readline.side_effect = ["line1\n", "line2\n", ""]
    process.communicate.return_value = ("", "")
    process.returncode = 0
    process.poll.return_value = 0

    mock_popen.return_value = process

    # Use signal spies to capture emitted signals
    # Since PyQt6 signals have read-only emit, we connect to slots
    log_messages: list[str] = []
    finished_calls: list[tuple[int, str]] = []

    worker.log_signal.connect(log_messages.append)
    worker.finished_signal.connect(lambda code, msg: finished_calls.append((code, msg)))

    worker.run()

    # Verify log messages were emitted
    assert "Running command: echo hello" in log_messages
    assert "line1" in log_messages
    assert "line2" in log_messages

    # Verify finished signal was emitted
    assert len(finished_calls) == 1
    assert finished_calls[0] == (0, "")


@patch("subprocess.Popen")
def test_run_failure(mock_popen, worker):
    mock_popen.side_effect = OSError("Command not found")

    # Use signal spies to capture emitted signals
    finished_calls: list[tuple[int, str]] = []
    worker.finished_signal.connect(lambda code, msg: finished_calls.append((code, msg)))

    worker.run()

    # Verify finished signal was emitted with error
    assert len(finished_calls) == 1
    assert finished_calls[0] == (-1, "Command not found")


@patch("subprocess.Popen")
def test_stop(mock_popen, worker):
    process = MagicMock()
    mock_popen.return_value = process

    worker.process = process
    worker.stop()

    assert worker._stop_event.is_set()
    process.terminate.assert_called()
