"""Unit tests for ProcessWorker."""

import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

# Force ImportError for PyQt6 to use fallback classes defined in process_worker.py
# This ensures ProcessWorker is a real class, not a Mock subclass
sys.modules["PyQt6.QtCore"] = None

# Now import ProcessWorker
from shared.python.process_worker import ProcessWorker


@pytest.fixture
def worker():
    # Since we use fallback, we can instantiate it directly
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
