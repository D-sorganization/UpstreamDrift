"""
Unit tests for GolfLauncher GUI logic (Model selection, Launching).
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# --- Mock PyQt6 Modules ---
class MockQWidget:
    def __init__(self, parent=None):
        self._window_title = ""
        self._style_sheet = ""

    def setWindowTitle(self, title):
        self._window_title = title

    def windowTitle(self):
        return self._window_title

    def setWindowIcon(self, icon):
        pass

    def setFont(self, f):
        pass

    def resize(self, w, h):
        pass

    def setCentralWidget(self, w):
        pass

    def setCursor(self, c):
        pass

    def setStyleSheet(self, s):
        self._style_sheet = s

    def styleSheet(self):
        return self._style_sheet

    def setFixedSize(self, w, h):
        pass

    def setAlignment(self, a):
        pass

    def setWordWrap(self, b):
        pass

    def setAttribute(self, a):
        pass

    def setLayout(self, layout):
        pass

    def setSpacing(self, s):
        pass

    def setContentsMargins(self, left, top, right, bottom):
        pass

    def addWidget(self, w, *args):
        pass

    def addLayout(self, layout, *args):
        pass

    def addStretch(self):
        pass

    def setWidget(self, w):
        pass

    def setWidgetResizable(self, b):
        pass

    def setFrameShape(self, s):
        pass

    def objectName(self):
        return ""

    def setObjectName(self, n):
        pass


class MockQMainWindow(MockQWidget):
    pass


class MockQPushButton(MockQWidget):
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._text = text
        self.clicked = MagicMock()
        self._enabled = True

    def setText(self, t):
        self._text = t

    def text(self):
        return self._text

    def setEnabled(self, b):
        self._enabled = bool(b)

    def isEnabled(self):
        return self._enabled

    def setFont(self, f):
        pass

    def setFixedHeight(self, h):
        pass


class MockQCheckBox(MockQWidget):
    def __init__(self, text="", parent=None):
        self.checked = False

    def setChecked(self, b):
        self.checked = b

    def isChecked(self):
        return self.checked

    def setToolTip(self, t):
        pass


class MockQFrame(MockQWidget):
    class Shape:
        NoFrame = 0


class MockQGridLayout(MockQWidget):
    pass


class MockQVBoxLayout(MockQWidget):
    pass


class MockQHBoxLayout(MockQWidget):
    pass


class MockQScrollArea(MockQWidget):
    pass


class MockQLabel(MockQWidget):
    def __init__(self, text="", parent=None):
        self._text = text

    def setText(self, t):
        self._text = t

    def setPixmap(self, p):
        pass


mock_qt_widgets = MagicMock()
mock_qt_widgets.QMainWindow = MockQMainWindow
mock_qt_widgets.QWidget = MockQWidget
mock_qt_widgets.QPushButton = MockQPushButton
mock_qt_widgets.QCheckBox = MockQCheckBox
mock_qt_widgets.QLabel = MockQLabel
mock_qt_widgets.QFrame = MockQFrame
mock_qt_widgets.QGridLayout = MockQGridLayout
mock_qt_widgets.QVBoxLayout = MockQVBoxLayout
mock_qt_widgets.QHBoxLayout = MockQHBoxLayout
mock_qt_widgets.QScrollArea = MockQScrollArea
mock_qt_widgets.QApplication = MagicMock()
mock_qt_widgets.QComboBox = MagicMock()
mock_qt_widgets.QDialog = MagicMock()
mock_qt_widgets.QTextEdit = MagicMock()
mock_qt_widgets.QTabWidget = MagicMock()

mock_qt_core = MagicMock()
mock_qt_core.Qt = MagicMock()
mock_qt_core.QThread = MagicMock()
mock_qt_core.pyqtSignal = MagicMock()

mock_qt_gui = MagicMock()
mock_qt_gui.QFont = MagicMock()
mock_qt_gui.QIcon = MagicMock()
mock_qt_gui.QPixmap = MagicMock()

sys.modules["PyQt6"] = MagicMock()
sys.modules["PyQt6.QtCore"] = mock_qt_core
sys.modules["PyQt6.QtGui"] = mock_qt_gui
sys.modules["PyQt6.QtWidgets"] = mock_qt_widgets

# Import after mocking
import launchers.golf_launcher  # noqa: E402

importlib.reload(launchers.golf_launcher)
from launchers.golf_launcher import GolfLauncher  # noqa: E402


class TestGolfLauncherLogic:

    @patch("shared.python.model_registry.ModelRegistry")
    @patch("launchers.golf_launcher.DockerCheckThread")
    def test_initialization(self, mock_thread, mock_registry):
        """Test proper initialization of the launcher."""
        # Setup mock registry
        registry_instance = mock_registry.return_value
        registry_instance.get_all_models.return_value = []

        # Setup mock thread
        thread_instance = mock_thread.return_value
        thread_instance.result = MagicMock()

        launcher = GolfLauncher()

        assert launcher.windowTitle() == "Golf Modeling Suite - GolfingRobot"
        mock_thread.return_value.start.assert_called_once()

        # Verify UI components exist
        assert hasattr(launcher, "grid_layout")
        assert hasattr(launcher, "btn_launch")

    @patch("shared.python.model_registry.ModelRegistry")
    @patch("launchers.golf_launcher.DockerCheckThread")
    def test_model_selection_updates_ui(self, mock_thread, mock_registry):
        """Test that selecting a model updates the launch button."""
        # Setup registry with one model
        mock_model = Mock()
        mock_model.name = "Test Model"
        mock_model.description = "Desc"
        mock_model.id = "test_model"

        registry_instance = mock_registry.return_value
        registry_instance.get_all_models.return_value = [mock_model]
        registry_instance.get_model.return_value = mock_model

        launcher = GolfLauncher()

        # Initial state: No Docker, No Model
        assert launcher.btn_launch.isEnabled() is False

        # Simulate Docker becoming available
        launcher.on_docker_check_complete(True)
        assert launcher.docker_available is True

        # Initial selection logic might have selected test_model since it's the only one
        # Let's verify or reset
        launcher.selected_model = None
        launcher.btn_launch.setEnabled(False)
        launcher.btn_launch.setText("SELECT A MODEL")

        # Select by ID
        launcher.select_model("test_model")

        assert launcher.selected_model == "test_model"
        assert launcher.btn_launch.isEnabled() is True
        # The button text should contain the NAME, upper case
        assert "TEST MODEL" in launcher.btn_launch.text()

    @patch("shared.python.model_registry.ModelRegistry")
    @patch("launchers.golf_launcher.DockerCheckThread")
    def test_launch_simulation_constructs_command(self, mock_thread, mock_registry):
        """Test launch simulation logic."""
        mock_model = Mock()
        mock_model.name = "Test Model"
        mock_model.path = "engines/test"
        mock_model.id = "test_model"
        mock_model.type = "docker"

        registry_instance = mock_registry.return_value
        registry_instance.get_all_models.return_value = [mock_model]
        registry_instance.get_model.return_value = mock_model

        launcher = GolfLauncher()
        launcher.docker_available = True
        launcher.select_model("test_model")

        # Mock subprocess
        with patch("launchers.golf_launcher.subprocess.Popen") as mock_popen:
            with patch.object(Path, "exists", return_value=True):
                with patch("os.name", "posix"):

                    # We need to verify _launch_docker_container is called essentially
                    # because type is "docker" (not custom)

                    launcher.launch_simulation()

                    mock_popen.assert_called()
                    args = mock_popen.call_args[0][0]
                    assert args[0] == "docker"
                    assert args[1] == "run"
                    # Verify volume mount path logic: args[5] should be the
                    # '-v host_path:container_path' argument containing the model path,
                    # allowing for platform-specific path separators.
                    assert "engines/test" in args[5] or "engines\\test" in args[5]

    @patch("shared.python.model_registry.ModelRegistry")
    @patch("launchers.golf_launcher.DockerCheckThread")
    def test_launch_generic_mjcf(self, mock_thread, mock_registry):
        """Test launching a generic MJCF file."""
        mock_model = Mock()
        mock_model.name = "Generic MJCF"
        mock_model.path = "engines/test/model.xml"
        mock_model.id = "generic_mjcf"
        mock_model.type = "mjcf"

        registry_instance = mock_registry.return_value
        registry_instance.get_all_models.return_value = [mock_model]
        registry_instance.get_model.return_value = mock_model

        launcher = GolfLauncher()
        launcher.docker_available = True
        launcher.select_model("generic_mjcf")

        with patch("launchers.golf_launcher.subprocess.Popen") as mock_popen:
            with patch.object(Path, "exists", return_value=True):
                launcher.launch_simulation()

                mock_popen.assert_called()
                args = mock_popen.call_args[0][0]
                # Should use sys.executable
                assert args[0] == sys.executable
                assert "mujoco.viewer" in args[2]
