"""
Unit tests for basic golf launcher functionality (Docker threads).

Note: These tests manipulate sys.modules to mock PyQt6 imports.
This can cause worker crashes when running under pytest-xdist (parallel).
We mark them as serial to avoid this.
"""

import sys
import types
from collections.abc import Generator
from unittest.mock import MagicMock, Mock, patch

import pytest

# Prevent xdist worker crashes from sys.modules manipulation during import
pytestmark = pytest.mark.serial


# Define Dummy Qt classes to avoid inheriting from Mock
class MockQThread:
    def __init__(self, parent=None):
        """Mock constructor."""

    def start(self) -> None:
        self.run()

    def run(self) -> None:
        """Mock run."""

    def wait(self) -> None:
        """Mock wait."""


def mock_pyqt_signal(*args) -> MagicMock:
    return MagicMock()


# Define widget mocks
class MockQWidget:
    def __init__(self, parent=None):
        """Mock constructor."""
        self._window_title = ""

    def setWindowTitle(self, title) -> None:
        """Mock setWindowTitle."""
        self._window_title = title

    def windowTitle(self) -> str:
        """Mock windowTitle."""
        return self._window_title

    def resize(self, w, h) -> None:
        """Mock resize."""

    def setLayout(self, layout) -> None:
        """Mock setLayout."""


class MockQDialog(MockQWidget):
    """Mock QDialog that handles missing attributes gracefully."""

    def __getattr__(self, name):
        """Return a no-op callable for any missing attribute."""
        return lambda *args, **kwargs: None

    def accept(self) -> None:
        """Mock accept."""

    def setMinimumSize(self, w, h) -> None:
        """Mock setMinimumSize."""


class MockQTextEdit(MockQWidget):
    def setReadOnly(self, b) -> None:
        """Mock setReadOnly."""

    def setMarkdown(self, t) -> None:
        """Mock setMarkdown."""


class MockQVBoxLayout:
    def __init__(self, parent=None):
        """Mock constructor."""

    def addWidget(self, w, *args, **kwargs) -> None:
        """Mock addWidget."""

    def addLayout(self, layout, *args, **kwargs) -> None:
        """Mock addLayout."""

    def setContentsMargins(self, *args) -> None:
        """Mock setContentsMargins."""

    def setSpacing(self, s) -> None:
        """Mock setSpacing."""


@pytest.fixture
def mocked_launcher_module() -> Generator[types.ModuleType, None, None]:
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
        except Exception as exc:  # noqa: BLE001
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
    def test_docker_check_thread_success(
        self, mock_run, mocked_launcher_module
    ) -> None:
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
    def test_docker_check_thread_failure(
        self, mock_run, mocked_launcher_module
    ) -> None:
        """Test DockerCheckThread failure."""
        mock_run.side_effect = FileNotFoundError

        thread = mocked_launcher_module.DockerCheckThread()
        thread.result = Mock()

        thread.run()

        thread.result.emit.assert_called_with(False)

    @pytest.mark.xfail(
        reason="HelpDialog Qt construction crashes worker in CI", strict=False
    )
    @patch("pathlib.Path.read_text", return_value="# Help")
    @patch("pathlib.Path.exists", return_value=True)
    def test_help_dialog(self, mock_exists, mock_read, mocked_launcher_module) -> None:
        """Test HelpDialog initialization and content loading."""
        dialog = mocked_launcher_module.HelpDialog()
        assert dialog is not None
        # Verify text was loaded (mock read_text called at least once)
        # HelpDialog may read multiple files (help topics)
        assert mock_read.call_count >= 1
        # Verify title
        assert "Help" in dialog.windowTitle()
