"""
Unit tests for basic golf launcher functionality (Docker threads).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# Define Dummy Qt classes to avoid inheriting from Mock
class MockQThread:
    def __init__(self, parent=None):
        pass

    def start(self):
        self.run()

    def run(self):
        pass

    def wait(self):
        pass


def mock_pyqt_signal(*args):
    return MagicMock()


# Setup sys.modules for PyQt6
# We need to construct the module structure so imports work
mock_qt_core = MagicMock()
mock_qt_core.QThread = MockQThread
mock_qt_core.pyqtSignal = mock_pyqt_signal
mock_qt_core.Qt = MagicMock()

mock_qt_widgets = MagicMock()


# Define widget mocks
class MockQWidget:
    def __init__(self, parent=None):
        self._window_title = ""

    def setWindowTitle(self, title):
        self._window_title = title

    def windowTitle(self):
        return self._window_title

    def resize(self, w, h):
        pass

    def setLayout(self, layout):
        pass


class MockQDialog(MockQWidget):
    def accept(self):
        pass


class MockQTextEdit(MockQWidget):
    def setReadOnly(self, b):
        pass

    def setMarkdown(self, t):
        pass


class MockQVBoxLayout:
    def __init__(self, parent=None):
        pass

    def addWidget(self, w):
        pass


mock_qt_widgets.QDialog = MockQDialog
mock_qt_widgets.QTextEdit = MockQTextEdit
mock_qt_gui = MagicMock()
mock_qt_widgets.QVBoxLayout = MockQVBoxLayout
mock_qt_widgets.QWidget = MockQWidget
mock_qt_widgets.QLabel = MagicMock()

sys.modules["PyQt6"] = MagicMock()
sys.modules["PyQt6.QtCore"] = mock_qt_core
sys.modules["PyQt6.QtGui"] = mock_qt_gui
sys.modules["PyQt6.QtWidgets"] = mock_qt_widgets

# Now we can safely import
# Note: We must enable import of launchers.golf_launcher
# We import HelpDialog too
from launchers.golf_launcher import (  # noqa: E402, I001
    DockerBuildThread,
    DockerCheckThread,
    HelpDialog,
)


class TestDockerThreads:
    """Test Docker-related threads in golf_launcher."""

    @patch("subprocess.run")
    def test_docker_check_thread_success(self, mock_run):
        """Test DockerCheckThread success."""
        # subprocess.run return value
        mock_run.return_value.returncode = 0

        thread = DockerCheckThread()
        # Mock the signal (it's a MagicMock from mock_pyqt_signal)
        # We replace it with a fresh Mock to assert calls easily
        thread.result = Mock()

        thread.run()

        mock_run.assert_called_once()
        thread.result.emit.assert_called_with(True)

    @patch("subprocess.run")
    def test_docker_check_thread_failure(self, mock_run):
        """Test DockerCheckThread failure."""
        mock_run.side_effect = FileNotFoundError

        thread = DockerCheckThread()
        thread.result = Mock()

        thread.run()

        thread.result.emit.assert_called_with(False)

    @patch("subprocess.Popen")
    def test_docker_build_thread_success(self, mock_popen):
        """Test DockerBuildThread success."""
        # Setup mock process with file-like stdout
        process_mock = Mock()

        # Create a mock file-like object for stdout that behaves like a real file
        # After the lines are exhausted, readline() should return empty string
        stdout_lines_iter = iter(["Step 1/5\n", "Successfully built\n", ""])

        def readline_side_effect():
            try:
                return next(stdout_lines_iter)
            except StopIteration:
                return ""  # After exhausting lines, return empty string

        stdout_mock = Mock()
        stdout_mock.readline = Mock(side_effect=readline_side_effect)

        process_mock.stdout = stdout_mock
        # poll() returns None while running, then 0 when done
        # The loop calls poll() after each readline(), so we need enough None values
        process_mock.poll = Mock(side_effect=[None, None, 0, 0, 0])
        process_mock.wait = Mock(return_value=None)
        process_mock.returncode = 0

        mock_popen.return_value = process_mock

        thread = DockerBuildThread(target_stage="mujoco")
        thread.log_signal = Mock()
        thread.finished_signal = Mock()

        thread.run()

        # Check that it tried to build
        mock_popen.assert_called()
        args = mock_popen.call_args[0][0]
        assert "docker" in args
        assert "build" in args
        assert "mujoco" in args

        # Check signals
        assert thread.log_signal.emit.call_count >= 2
        thread.finished_signal.emit.assert_called_with(True, "Build successful.")

    @patch("subprocess.Popen")
    def test_docker_build_thread_failure(self, mock_popen):
        """Test DockerBuildThread failure."""
        # Setup mock process with file-like stdout
        process_mock = Mock()

        # Create a mock file-like object for stdout that behaves like a real file
        stdout_lines_iter = iter(["Error building\n", ""])

        def readline_side_effect():
            try:
                return next(stdout_lines_iter)
            except StopIteration:
                return ""  # After exhausting lines, return empty string

        stdout_mock = Mock()
        stdout_mock.readline = Mock(side_effect=readline_side_effect)

        process_mock.stdout = stdout_mock
        # poll() returns None while running, then 1 when done with error
        process_mock.poll = Mock(side_effect=[None, 1, 1, 1])
        process_mock.wait = Mock(return_value=None)
        process_mock.returncode = 1

        mock_popen.return_value = process_mock

        thread = DockerBuildThread(target_stage="mujoco")
        thread.log_signal = Mock()
        thread.finished_signal = Mock()

        thread.run()

        thread.finished_signal.emit.assert_called_with(
            False, "Build failed with code 1"
        )

    def test_docker_build_thread_missing_path(self):
        """Test DockerBuildThread with missing path (mocking exists)."""
        with patch("pathlib.Path.exists", return_value=False):
            thread = DockerBuildThread()
            thread.finished_signal = Mock()

            thread.run()

            # verify it emitted failure immediately
            thread.finished_signal.emit.assert_called_once()
            args = thread.finished_signal.emit.call_args[0]
            assert args[0] is False
            assert "Path not found" in args[1]

    @patch("pathlib.Path.read_text", return_value="# Help")
    @patch("pathlib.Path.exists", return_value=True)
    def test_help_dialog(self, mock_exists, mock_read):
        """Test HelpDialog initialization and content loading."""
        dialog = HelpDialog()
        assert dialog is not None
        # Verify text was loaded (mock read_text called)
        mock_read.assert_called_once()
        # Verify title
        assert dialog.windowTitle() == "Golf Suite - Help"
        # Verify text was set
        # Since we mocked QTextEdit, we can't easily verify internal state without
        # capturing the instance. But simpler is just ensuring it runs without error.
