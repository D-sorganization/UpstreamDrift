"""
Unit tests for GolfLauncher GUI logic (Model selection, Launching).
"""

import sys
from importlib import reload
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

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
        self.text = text
        self.clicked = MagicMock()
        self.enabled = True

    def setText(self, t):
        self.text = t

    def text(self):
        return self.text

    def setEnabled(self, b):
        self.enabled = b

    def isEnabled(self):
        return self.enabled

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
        self.text = text

    def setText(self, t):
        self.text = t

    def setPixmap(self, p):
        pass


@pytest.fixture
def mocked_launcher_module():
    """Import golf_launcher with mocked Qt modules within a clean sys.modules context."""
    # Create mock module dictionary
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

    mock_modules = {
        "PyQt6": MagicMock(),
        "PyQt6.QtCore": mock_qt_core,
        "PyQt6.QtGui": mock_qt_gui,
        "PyQt6.QtWidgets": mock_qt_widgets,
    }

    # Patch sys.modules
    with patch.dict(sys.modules, mock_modules):
        # Remove launchers.golf_launcher from sys.modules if it exists
        # to ensure it gets re-imported using our mocks
        if "launchers.golf_launcher" in sys.modules:
            del sys.modules["launchers.golf_launcher"]

        # Import the module
        import launchers.golf_launcher

        # If it was already imported before, we might need to reload it to ensure
        # it picks up the mocked Qt modules
        reload(launchers.golf_launcher)

        yield launchers.golf_launcher

        # Cleanup: Remove the module from sys.modules so subsequent tests
        # import the clean/real version
        if "launchers.golf_launcher" in sys.modules:
            del sys.modules["launchers.golf_launcher"]


class TestGolfLauncherLogic:

    @patch("shared.python.model_registry.ModelRegistry")
    def test_initialization(self, mock_registry, mocked_launcher_module):
        """Test proper initialization of the launcher."""
        # Setup mock registry
        registry_instance = mock_registry.return_value
        registry_instance.get_all_models.return_value = []

        # Patch DockerCheckThread in the mocked module
        with patch.object(mocked_launcher_module, "DockerCheckThread") as mock_thread:
            thread_instance = mock_thread.return_value
            thread_instance.result = MagicMock()

            launcher = mocked_launcher_module.GolfLauncher()

            assert launcher.windowTitle() == "Golf Modeling Suite - GolfingRobot"
            mock_thread.return_value.start.assert_called_once()

            # Verify UI components exist
            assert hasattr(launcher, "grid_layout")
            assert hasattr(launcher, "btn_launch")

    @patch("shared.python.model_registry.ModelRegistry")
    def test_model_selection_updates_ui(self, mock_registry, mocked_launcher_module):
        """Test that selecting a model updates the launch button."""
        # Setup registry with one model
        mock_model = Mock()
        mock_model.name = "Test Model"
        mock_model.description = "Desc"
        mock_model.id = "test_model"

        registry_instance = mock_registry.return_value
        registry_instance.get_all_models.return_value = [mock_model]

        with patch.object(mocked_launcher_module, "DockerCheckThread"):
            launcher = mocked_launcher_module.GolfLauncher()

            # Initial state: No Docker, No Model
            assert launcher.btn_launch.isEnabled() is False

            # Simulate Docker becoming available
            launcher.on_docker_check_complete(True)
            assert launcher.docker_available is True

            # Let's manually select "Test Model"
            launcher.select_model("Test Model")

            assert launcher.selected_model == "Test Model"
            assert launcher.btn_launch.isEnabled() is True
            assert "TEST MODEL" in launcher.btn_launch.text

    @patch("shared.python.model_registry.ModelRegistry")
    def test_launch_simulation_constructs_command(
        self, mock_registry, mocked_launcher_module
    ):
        """Test launch simulation logic."""
        mock_model = Mock()
        mock_model.name = "Test Model"
        mock_model.path = "engines/test"
        mock_model.id = "test_model"

        registry_instance = mock_registry.return_value
        registry_instance.get_all_models.return_value = [mock_model]
        registry_instance.get_model_by_name.return_value = mock_model

        with patch.object(mocked_launcher_module, "DockerCheckThread"):
            launcher = mocked_launcher_module.GolfLauncher()
            launcher.docker_available = True
            launcher.select_model("Test Model")

            # Mock subprocess within the mocked module namespace
            with patch(
                f"{mocked_launcher_module.__name__}.subprocess.Popen"
            ) as mock_popen:
                with patch.object(Path, "exists", return_value=True):
                    with patch("os.name", "posix"):
                        launcher.launch_simulation()

                        mock_popen.assert_called()
                        args = mock_popen.call_args[0][0]
                        assert args[0] == "docker"
                        assert args[1] == "run"
                        assert "engines/test" in args[5] or "engines\\test" in args[5]

    @patch("shared.python.model_registry.ModelRegistry")
    def test_launch_generic_mjcf(self, mock_registry, mocked_launcher_module):
        """Test launching a generic MJCF file."""
        mock_model = Mock()
        mock_model.name = "Generic MJCF"
        mock_model.path = "engines/test/model.xml"
        mock_model.id = "generic_mjcf"

        registry_instance = mock_registry.return_value
        registry_instance.get_all_models.return_value = [mock_model]
        registry_instance.get_model_by_name.return_value = mock_model

        with patch.object(mocked_launcher_module, "DockerCheckThread"):
            launcher = mocked_launcher_module.GolfLauncher()
            launcher.docker_available = True
            launcher.select_model("Generic MJCF")

            with patch(
                f"{mocked_launcher_module.__name__}.subprocess.Popen"
            ) as mock_popen:
                with patch.object(Path, "exists", return_value=True):
                    launcher.launch_simulation()

                    mock_popen.assert_called()
                    args = mock_popen.call_args[0][0]
                    assert args[0] == sys.executable
                    assert "mujoco.viewer" in args[2]
