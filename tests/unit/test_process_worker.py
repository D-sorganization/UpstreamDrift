"""Unit tests for ProcessWorker.

Uses the fallback signal stubs (connect/emit pattern) so tests run
regardless of whether PyQt6 is installed.
"""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _force_fallback_signals():
    """Force ProcessWorker to use the fallback (non-PyQt6) signal stubs.

    This patches PYQT6_AVAILABLE to False and clears the cached module
    so the fallback pyqtSignal/QThread stubs are used instead of real Qt.
    """
    mod_key = "src.shared.python.ui.qt.process_worker"
    saved = sys.modules.pop(mod_key, None)

    import src.shared.python.engine_core.engine_availability as ea_mod

    orig = ea_mod.PYQT6_AVAILABLE
    ea_mod.PYQT6_AVAILABLE = False
    try:
        yield
    finally:
        ea_mod.PYQT6_AVAILABLE = orig
        if saved is not None:
            sys.modules[mod_key] = saved
        elif mod_key in sys.modules:
            del sys.modules[mod_key]


@pytest.fixture
def worker_cls():
    """Import ProcessWorker using fallback signals."""
    mod_key = "src.shared.python.ui.qt.process_worker"
    if mod_key in sys.modules:
        del sys.modules[mod_key]

    import src.shared.python.ui.qt.process_worker as pw_mod

    importlib.reload(pw_mod)
    return pw_mod.ProcessWorker


@pytest.fixture
def worker(worker_cls):
    """Create a ProcessWorker instance with a simple command."""
    return worker_cls(["echo", "hello"])


def test_initialization(worker):
    assert worker.cmd == ["echo", "hello"]
    assert worker._is_running is True


@patch("subprocess.Popen")
def test_run_success(mock_popen, worker):
    # Setup mock process
    process = MagicMock()
    process.stdout.readline.side_effect = ["line1\n", "line2\n", ""]
    process.stderr.read.return_value = ""
    process.returncode = 0
    process.poll.return_value = 0

    mock_popen.return_value = process

    log_messages: list[str] = []
    finished_calls: list[tuple[int, str]] = []

    worker.log_signal.connect(log_messages.append)
    worker.finished_signal.connect(lambda code, msg: finished_calls.append((code, msg)))

    worker.run()

    assert "Running command: echo hello" in log_messages
    assert "line1" in log_messages
    assert "line2" in log_messages

    assert len(finished_calls) == 1
    assert finished_calls[0] == (0, "")


@patch("subprocess.Popen")
def test_run_failure(mock_popen, worker):
    mock_popen.side_effect = OSError("Command not found")

    finished_calls: list[tuple[int, str]] = []
    worker.finished_signal.connect(lambda code, msg: finished_calls.append((code, msg)))

    worker.run()

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
