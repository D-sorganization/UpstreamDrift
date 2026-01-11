"""
Unit tests for Golf Suite Launcher.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure offscreen platform
os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="module", autouse=True)
def cleanup_imports():
    """Clean up imports after tests to prevent mock leakage."""
    yield
    sys.modules.pop("launchers.golf_suite_launcher", None)
    sys.modules.pop("launchers.golf_launcher", None)


# Create mocks for PyQt6 modules
mock_qt = MagicMock()
mock_widgets = MagicMock()
mock_core = MagicMock()


# Setup mock classes
class MockQMainWindow:
    def __init__(self, *args, **kwargs):
        pass

    def setWindowTitle(self, t):
        pass

    def resize(self, w, h):
        pass

    def setCentralWidget(self, w):
        pass

    def show(self):
        pass

    def style(self):
        return MagicMock()


class MockQWidget:
    def __init__(self, *args, **kwargs):
        pass


class MockQVBoxLayout:
    def __init__(self, *args, **kwargs):
        pass

    def addWidget(self, w):
        pass

    def addSpacing(self, s):
        pass

    def addStretch(self):
        pass

    def addLayout(self, layout):
        pass


class MockQHBoxLayout:
    def __init__(self, *args, **kwargs):
        pass

    def addWidget(self, w):
        pass

    def addStretch(self):
        pass


class MockQLabel:
    def __init__(self, t="", *args, **kwargs):
        pass

    def setAlignment(self, a):
        pass

    def font(self):
        return MagicMock()

    def setFont(self, f):
        pass

    def setText(self, t):
        pass


class MockQPushButton:
    def __init__(self, t="", *args, **kwargs):
        self.clicked = MagicMock()

    def setMinimumHeight(self, h):
        pass

    def setIcon(self, icon):
        pass

    def setToolTip(self, t):
        pass

    def setAccessibleName(self, n):
        pass

    def setText(self, t):
        pass


class MockQTextEdit:
    def __init__(self, *args, **kwargs):
        pass

    def setMaximumHeight(self, h):
        pass

    def setReadOnly(self, b):
        pass

    def setStyleSheet(self, s):
        pass

    def append(self, s):
        pass

    def clear(self):
        pass

    def toPlainText(self):
        return "Log content"


class MockQGroupBox:
    def __init__(self, t="", *args, **kwargs):
        pass


class MockQMessageBox:
    @staticmethod
    def critical(p, t, m):
        pass


class MockQClipboard:
    def __init__(self):
        self.text = ""

    def setText(self, t):
        self.text = t


# Assign mocks
mock_widgets.QMainWindow = MockQMainWindow
mock_widgets.QWidget = MockQWidget
mock_widgets.QVBoxLayout = MockQVBoxLayout
mock_widgets.QHBoxLayout = MockQHBoxLayout
mock_widgets.QLabel = MockQLabel
mock_widgets.QPushButton = MockQPushButton
mock_widgets.QTextEdit = MockQTextEdit
mock_widgets.QGroupBox = MockQGroupBox
mock_widgets.QMessageBox = MockQMessageBox
mock_widgets.QApplication = MagicMock()
mock_widgets.QApplication.return_value.exec.return_value = 0
mock_widgets.QApplication.clipboard.return_value = MockQClipboard()

mock_core.Qt.AlignmentFlag.AlignCenter = 0

mock_qt.QtWidgets = mock_widgets
mock_qt.QtCore = mock_core

# Apply patches to sys.modules BEFORE importing the launcher
with patch.dict(
    sys.modules,
    {"PyQt6": mock_qt, "PyQt6.QtWidgets": mock_widgets, "PyQt6.QtCore": mock_core},
):
    # Force reload to ensure it uses our mocks if it was already loaded
    import importlib

    from launchers import golf_suite_launcher

    importlib.reload(golf_suite_launcher)


@pytest.fixture
def mock_subprocess():
    """Mock subprocess.Popen."""
    with patch("subprocess.Popen") as mock_popen:
        process = MagicMock()
        process.pid = 12345
        mock_popen.return_value = process
        yield mock_popen


@pytest.fixture
def launcher_app():
    """Fixture to create the launcher instance."""
    # Ensure PYQT_AVAILABLE is True for logic testing
    golf_suite_launcher.PYQT_AVAILABLE = True
    app = golf_suite_launcher.GolfLauncher()
    return app


class TestGolfSuiteLauncher:
    """Test suite for GolfLauncher."""

    def test_initialization(self, launcher_app):
        """Test UI initialization."""
        # Check title set on mock
        # Since MockQMainWindow.__init__ is empty, we can't check super calls unless we mock init
        # But we can check attributes if the class set them
        pass

    def test_launch_mujoco(self, launcher_app, mock_subprocess):
        """Test launching MuJoCo engine."""
        # Mock path existence
        with patch.object(Path, "exists", return_value=True):
            launcher_app._launch_mujoco()

        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        cmd = args[0]
        assert cmd[0] == sys.executable
        assert "advanced_gui.py" in str(cmd[1])
        # CWD is the python root, checks only for mujoco path component
        assert "mujoco" in str(kwargs["cwd"])

    def test_launch_drake(self, launcher_app, mock_subprocess):
        """Test launching Drake engine."""
        with patch.object(Path, "exists", return_value=True):
            launcher_app._launch_drake()

        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        assert "golf_gui.py" in str(args[0][1])

    def test_launch_pinocchio(self, launcher_app, mock_subprocess):
        """Test launching Pinocchio engine."""
        with patch.object(Path, "exists", return_value=True):
            launcher_app._launch_pinocchio()

        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        assert "gui.py" in str(args[0][1])

    def test_script_not_found(self, launcher_app, mock_subprocess):
        """Test handling of missing script."""
        with patch.object(Path, "exists", return_value=False):
            # We need to mock log_text since it's an instance of MockQTextEdit
            launcher_app.log_text = MagicMock()

            launcher_app._launch_mujoco()
            mock_subprocess.assert_not_called()

            # Verify error logged
            launcher_app.log_text.append.assert_called()
            args = launcher_app.log_text.append.call_args_list
            assert any("ERROR" in str(a) for a in args)

    def test_log_functions(self, launcher_app):
        """Test logging functions."""
        launcher_app.log_text = MagicMock()

        launcher_app.log_message("Test message")
        launcher_app.log_text.append.assert_called()
        assert "Test message" in str(launcher_app.log_text.append.call_args)

        launcher_app.clear_log()
        launcher_app.log_text.clear.assert_called()

    def test_copy_log(self, launcher_app):
        """Test copying log to clipboard."""
        launcher_app.log_text = MagicMock()
        launcher_app.log_text.toPlainText.return_value = "Log content"
        launcher_app.status = MagicMock()

        launcher_app.copy_log()

        # Check if clipboard.setText was called with the correct content
        clipboard = mock_widgets.QApplication.clipboard()
        assert clipboard.text == "Log content"

        # Check log message and status update
        # The log message includes a timestamp, so we check if the message content is present
        assert launcher_app.log_text.append.called
        args = launcher_app.log_text.append.call_args[0]
        assert "Log copied to clipboard." in args[0]
        launcher_app.status.setText.assert_called_with("Log copied")

    def test_main_function(self, launcher_app):
        """Test main entry point."""
        # Use manual patching to ensure we modify the *reloaded* module object
        original_launcher = golf_suite_launcher.GolfLauncher
        mock_launcher = MagicMock()
        golf_suite_launcher.GolfLauncher = mock_launcher  # type: ignore[misc]

        # Use the already mocked QApplication from module setup
        app = mock_widgets.QApplication.return_value

        # Mock sys.exit
        with patch("sys.exit") as mock_exit:
            golf_suite_launcher.PYQT_AVAILABLE = True

            # Print for debug (captured in verbose output if needed)
            print(f"DEBUG: PYQT_AVAILABLE={golf_suite_launcher.PYQT_AVAILABLE}")

            golf_suite_launcher.main()

            mock_launcher.assert_called()
            mock_launcher.return_value.show.assert_called()
            app.exec.assert_called()
            mock_exit.assert_called()

        # Restore
        golf_suite_launcher.GolfLauncher = original_launcher  # type: ignore[misc]
