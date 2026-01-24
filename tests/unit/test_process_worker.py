# ruff: noqa: E402
"""Unit tests for ProcessWorker."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.engine_availability import PYQT6_AVAILABLE

# Skip entire module if PyQt6 is not available - the ProcessWorker tests require
# proper Qt mocking that doesn't work reliably when PyQt6 is missing
if not PYQT6_AVAILABLE:
    pytest.skip("PyQt6 not installed", allow_module_level=True)


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

    # Check that signals are emitted
    # In fallback mode, pyqtSignal is a class with emit method that does nothing by default.
    # We need to mock the emit method on the instance signals to verify calls.

    # The class defines log_signal = pyqtSignal(str)
    # worker.log_signal is an instance of the fallback pyqtSignal class

    # We can patch the emit method on the instance attributes
    worker.log_signal.emit = MagicMock()
    worker.finished_signal.emit = MagicMock()

    worker.run()

    worker.log_signal.emit.assert_any_call("Running command: echo hello")
    worker.log_signal.emit.assert_any_call("line1")
    worker.log_signal.emit.assert_any_call("line2")
    worker.finished_signal.emit.assert_called_with(0, "")


@patch("subprocess.Popen")
def test_run_failure(mock_popen, worker):
    mock_popen.side_effect = OSError("Command not found")

    worker.log_signal.emit = MagicMock()
    worker.finished_signal.emit = MagicMock()

    worker.run()

    worker.finished_signal.emit.assert_called_with(-1, "Command not found")


@patch("subprocess.Popen")
def test_stop(mock_popen, worker):
    process = MagicMock()
    mock_popen.return_value = process

    worker.process = process
    worker.stop()

    assert worker._stop_event.is_set()
    process.terminate.assert_called()
