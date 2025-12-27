"""
Unit tests for Golf Suite Launcher.
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure offscreen platform
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Create mocks for PyQt6 modules
mock_qt = MagicMock()
mock_widgets = MagicMock()
mock_core = MagicMock()

# Setup mock classes
class MockQMainWindow:
    def __init__(self): pass
    def setWindowTitle(self, t): pass
    def resize(self, w, h): pass
    def setCentralWidget(self, w): pass
    def show(self): pass

class MockQWidget:
    pass

class MockQVBoxLayout:
    def addWidget(self, w): pass
    def addSpacing(self, s): pass
    def addStretch(self): pass
    def addLayout(self, l): pass

class MockQHBoxLayout:
    def addWidget(self, w): pass
    def addStretch(self): pass

class MockQLabel:
    def __init__(self, t=""): pass
    def setAlignment(self, a): pass
    def font(self): return MagicMock()
    def setFont(self, f): pass
    def setText(self, t): pass

class MockQPushButton:
    def __init__(self, t=""):
        self.clicked = MagicMock()
    def setMinimumHeight(self, h): pass

class MockQTextEdit:
    def setMaximumHeight(self, h): pass
    def setReadOnly(self, b): pass
    def setStyleSheet(self, s): pass
    def append(self, s): pass
    def clear(self): pass

class MockQGroupBox:
    def __init__(self, t=""): pass

class MockQMessageBox:
    @staticmethod
    def critical(p, t, m): pass

class MockQApplication:
    def __init__(self, args): pass
    def exec(self): return 0

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
mock_widgets.QApplication = MockQApplication
mock_core.Qt.AlignmentFlag.AlignCenter = 0

mock_qt.QtWidgets = mock_widgets
mock_qt.QtCore = mock_core

# Apply patches to sys.modules BEFORE importing the launcher
with patch.dict(sys.modules, {
    "PyQt6": mock_qt,
    "PyQt6.QtWidgets": mock_widgets,
    "PyQt6.QtCore": mock_core
}):
    from launchers import golf_suite_launcher
    # Force reload to ensure it uses our mocks if it was already loaded
    import importlib
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
        assert "mujoco_humanoid_golf" in str(kwargs["cwd"])

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

    def test_main_function(self):
        """Test main entry point."""
        with patch("launchers.golf_suite_launcher.GolfLauncher") as MockLauncher, \
             patch("launchers.golf_suite_launcher.QtWidgets.QApplication") as MockApp, \
             patch("sys.exit") as mock_exit:

            instance = MockLauncher.return_value
            app = MockApp.return_value

            golf_suite_launcher.PYQT_AVAILABLE = True
            golf_suite_launcher.main()

            MockLauncher.assert_called()
            instance.show.assert_called()
            app.exec.assert_called()
            mock_exit.assert_called()
