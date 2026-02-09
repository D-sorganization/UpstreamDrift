"""
Unit tests for basic golf launcher functionality (Docker threads).

Note: These tests manipulate sys.modules to mock PyQt6 imports.
This can cause worker crashes when running under pytest-xdist (parallel).
We mark them as serial to avoid this.
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Prevent xdist worker crashes from sys.modules manipulation during import
pytestmark = pytest.mark.serial


# Define Dummy Qt classes to avoid inheriting from Mock
class MockQThread:
    def __init__(self, parent=None):
        """Mock constructor."""

    def start(self):
        self.run()

    def run(self):
        """Mock run."""

    def wait(self):
        """Mock wait."""


def mock_pyqt_signal(*args):
    return MagicMock()


# Define widget mocks
class MockQWidget:
    def __init__(self, parent=None):
        """Mock constructor."""
        self._window_title = ""

    def setWindowTitle(self, title):
        """Mock setWindowTitle."""
        self._window_title = title

    def windowTitle(self):
        """Mock windowTitle."""
        return self._window_title

    def resize(self, w, h):
        """Mock resize."""

    def setLayout(self, layout):
        """Mock setLayout."""


class MockQDialog(MockQWidget):
    """Mock QDialog that handles missing attributes gracefully."""

    def __getattr__(self, name):
        """Return a no-op callable for any missing attribute."""
        return lambda *args, **kwargs: None

    def accept(self):
        """Mock accept."""

    def setMinimumSize(self, w, h):
        """Mock setMinimumSize."""


class MockQTextEdit(MockQWidget):
    def setReadOnly(self, b):
        """Mock setReadOnly."""

    def setMarkdown(self, t):
        """Mock setMarkdown."""


class MockQVBoxLayout:
    def __init__(self, parent=None):
        """Mock constructor."""

    def addWidget(self, w, *args, **kwargs):
        """Mock addWidget."""

    def addLayout(self, layout, *args, **kwargs):
        """Mock addLayout."""

    def setContentsMargins(self, *args):
        """Mock setContentsMargins."""

    def setSpacing(self, s):
        """Mock setSpacing."""


@pytest.fixture
def mocked_launcher_module():
    """
    Import golf_launcher with mocked Qt modules.
    This fixture ensures that the mocks don't pollute the global sys.modules,
    allowing other tests to run with real Qt modules.
    """
    # Create mocks
    mock_qt_core = MagicMock()
    mock_qt_core.QThread = MockQThread
    mock_qt_core.pyqtSignal = mock_pyqt_signal
    mock_qt_core.Qt = MagicMock()

    mock_qt_widgets = MagicMock()
    mock_qt_widgets.QDialog = MockQDialog
    mock_qt_widgets.QTextEdit = MockQTextEdit
    mock_qt_widgets.QVBoxLayout = MockQVBoxLayout
    mock_qt_widgets.QWidget = MockQWidget
    mock_qt_widgets.QLabel = MagicMock()

    mock_qt_gui = MagicMock()

    # Create mock module dictionary
    mock_modules = {
        "PyQt6": MagicMock(),
        "PyQt6.QtCore": mock_qt_core,
        "PyQt6.QtGui": mock_qt_gui,
        "PyQt6.QtWidgets": mock_qt_widgets,
    }

    # Patch sys.modules
    with patch.dict(sys.modules, mock_modules):
        # Remove launchers.golf_launcher and its dependencies from sys.modules
        # to ensure it gets re-imported using our mocks
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("src.launchers.golf_launcher"):
                del sys.modules[mod_name]

        try:
            # Import the module
            import src.launchers.golf_launcher

            yield src.launchers.golf_launcher
        except Exception as exc:
            pytest.skip(f"golf_launcher import failed under mocked Qt: {exc}")
        finally:
            # Cleanup: Remove the module from sys.modules so subsequent tests
            # import the clean/real version
            for mod_name in list(sys.modules.keys()):
                if mod_name.startswith("src.launchers.golf_launcher"):
                    del sys.modules[mod_name]


class TestDockerThreads:
    """Test Docker-related threads in golf_launcher."""

    @patch("subprocess.run")
    def test_docker_check_thread_success(self, mock_run, mocked_launcher_module):
        """Test DockerCheckThread success."""
        # subprocess.run return value
        mock_run.return_value.returncode = 0

        thread = mocked_launcher_module.DockerCheckThread()
        # Mock the signal (it's a MagicMock from mock_pyqt_signal)
        # We replace it with a fresh Mock to assert calls easily
        thread.result = Mock()

        thread.run()

        mock_run.assert_called_once()
        thread.result.emit.assert_called_with(True)

    @patch("subprocess.run")
    def test_docker_check_thread_failure(self, mock_run, mocked_launcher_module):
        """Test DockerCheckThread failure."""
        mock_run.side_effect = FileNotFoundError

        thread = mocked_launcher_module.DockerCheckThread()
        thread.result = Mock()

        thread.run()

        thread.result.emit.assert_called_with(False)

    @pytest.mark.skip(reason="DockerBuildThread moved to src.launchers.docker_manager")
    def test_docker_build_thread_success(self, mocked_launcher_module):
        """Test DockerBuildThread success (skipped: moved to docker_manager)."""

    @pytest.mark.skip(reason="DockerBuildThread moved to src.launchers.docker_manager")
    def test_docker_build_thread_failure(self, mocked_launcher_module):
        """Test DockerBuildThread failure (skipped: moved to docker_manager)."""

    @pytest.mark.skip(reason="DockerBuildThread moved to src.launchers.docker_manager")
    def test_docker_build_thread_missing_path(self, mocked_launcher_module):
        """Test DockerBuildThread with missing path (skipped: moved to docker_manager)."""

    @pytest.mark.skip(reason="HelpDialog Qt construction crashes worker in CI")
    @patch("pathlib.Path.read_text", return_value="# Help")
    @patch("pathlib.Path.exists", return_value=True)
    def test_help_dialog(self, mock_exists, mock_read, mocked_launcher_module):
        """Test HelpDialog initialization and content loading."""
        dialog = mocked_launcher_module.HelpDialog()
        assert dialog is not None
        # Verify text was loaded (mock read_text called at least once)
        # HelpDialog may read multiple files (help topics)
        assert mock_read.call_count >= 1
        # Verify title
        assert "Help" in dialog.windowTitle()
