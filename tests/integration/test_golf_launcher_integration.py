"""Integration tests for UpstreamDriftLauncher."""

import os
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from src.shared.python.config.model_registry import ModelRegistry

# Mock PyQt6 for headless/CI environment where DLLs are broken/missing
# This must happen BEFORE importing modules that use PyQt6
mock_qt = MagicMock()
mock_widgets = MagicMock()
mock_core = MagicMock()
mock_gui = MagicMock()


# Define robust Mock classes to avoid recursion issues when inheriting from MagicMock
class MockQtBase:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return MagicMock()

    def setWindowTitle(self, title) -> None:
        pass

    def resize(self, *args) -> None:
        pass

    def setStyleSheet(self, s) -> None:
        pass

    def setWindowIcon(self, i) -> None:
        pass

    def show(self) -> None:
        pass

    def setCentralWidget(self, w) -> None:
        pass

    def setLayout(self, layout) -> None:
        pass

    def exec(self) -> None:
        pass

    def setFixedSize(self, w, h) -> None:
        pass

    def setAlignment(self, a) -> None:
        pass

    def setWordWrap(self, b) -> None:
        pass

    def font(self) -> MagicMock:
        return MagicMock()


class MockQWidget(MockQtBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enabled = True

    def setEnabled(self, b) -> None:
        self._enabled = bool(b)

    def isEnabled(self) -> bool:
        if isinstance(self._enabled, bool):
            return self._enabled
        return False


class MockQMainWindow(MockQtBase):
    pass


class MockQDialog(MockQtBase):
    pass


class MockQFrame(MockQtBase):
    class Shape:
        NoFrame = 0


class MockQLayout(MockQtBase):
    """Mock for QLayout subclasses (QVBoxLayout, QHBoxLayout, QGridLayout).

    The key override is ``count()`` returning ``0`` so that
    ``while grid_layout.count(): ...`` loops in production code
    terminate immediately instead of spinning forever on a truthy
    MagicMock.
    """

    def count(self) -> int:
        return 0

    def addWidget(self, *args) -> None:
        pass

    def addLayout(self, *args) -> None:
        pass

    def setSpacing(self, s) -> None:
        pass

    def setContentsMargins(self, *args) -> None:
        pass


class MockQThread(MockQtBase):
    def start(self) -> None:
        pass

    def wait(self) -> None:
        pass

    def run(self) -> None:
        pass


# Setup Constants and Classes
mock_core.Qt.AlignmentFlag.AlignCenter = 0
mock_core.Qt.AspectRatioMode.KeepAspectRatio = 0
mock_core.Qt.TransformationMode.SmoothTransformation = 0
mock_core.Qt.CursorShape.PointingHandCursor = 0
mock_core.QThread = MockQThread
# Use lambda to ignore arguments so they aren't treated as 'spec' by MagicMock
mock_core.pyqtSignal = lambda *args, **kwargs: MagicMock()

# QSettings must return real strings from .value() so that downstream code
# (e.g. ThemeManager._load_custom_themes) calling json.loads() does not
# receive a MagicMock and raise TypeError.
_mock_qsettings_instance = MagicMock()
_mock_qsettings_instance.value.return_value = ""
mock_core.QSettings = MagicMock(return_value=_mock_qsettings_instance)

mock_widgets.QApplication = MagicMock()
mock_widgets.QApplication.instance.return_value = None
mock_widgets.QMainWindow = MockQMainWindow
mock_widgets.QWidget = MockQWidget
mock_widgets.QDialog = MockQDialog
mock_widgets.QFrame = MockQFrame
mock_widgets.QLabel = MockQWidget
mock_widgets.QPushButton = MockQWidget
mock_widgets.QCheckBox = MockQWidget
mock_widgets.QComboBox = MockQWidget
mock_widgets.QTextEdit = MockQWidget
mock_widgets.QScrollArea = MockQWidget
mock_widgets.QTabWidget = MockQWidget
mock_widgets.QVBoxLayout = MockQLayout
mock_widgets.QHBoxLayout = MockQLayout
mock_widgets.QGridLayout = MockQLayout
mock_widgets.QMessageBox = MagicMock()
mock_widgets.QDockWidget = MagicMock()
mock_widgets.QLineEdit = MockQWidget


@pytest.fixture(scope="module", autouse=True)
def mock_pyqt_modules() -> Generator[None, None, None]:
    """Patch PyQt6 modules in sys.modules for the duration of this test module."""
    with patch.dict(
        sys.modules,
        {
            "PyQt6": mock_qt,
            "PyQt6.QtWidgets": mock_widgets,
            "PyQt6.QtCore": mock_core,
            "PyQt6.QtGui": mock_gui,
        },
    ):
        yield


# Note: We do not import UpstreamDriftLauncher at the top level to avoid
# importing it before the sys.modules patch is active, and to avoid
# referencing a "stale" class definition if we reload it later.


@pytest.fixture(scope="session")
def qapp() -> Generator[Any, None, None]:
    """Create QApplication instance."""
    app = mock_widgets.QApplication.instance()
    if app is None:
        app = mock_widgets.QApplication(sys.argv)
    yield app


@pytest.fixture
def launcher_env(qapp) -> Generator[Any, None, None]:
    """Setup launcher environment with temp config."""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a fake models.yaml
        config_dir = temp_path / "config"
        config_dir.mkdir()
        models_yaml = config_dir / "models.yaml"

        # Create a dummy model file
        engines_dir = temp_path / "engines" / "test"
        engines_dir.mkdir(parents=True)
        model_xml = engines_dir / "test_model.xml"
        model_xml.write_text("<mujoco/>", encoding="utf-8")

        config_content = f"""
models:
  - id: test_integration_model
    name: Integration Test Model
    description: A model for integration testing
    type: mjcf
    path: {str(model_xml).replace(os.sep, "/")}
"""
        models_yaml.write_text(config_content, encoding="utf-8")

        # Patch ModelRegistry to use our temp file
        temp_registry = ModelRegistry(models_yaml)

        # Override ASSETS_DIR to point to our empty temp dir.
        # In the real application, UpstreamDriftLauncher creates QIcon instances from
        # files under ASSETS_DIR. In headless/CI test runs, fully initializing
        # QIcon with real image assets can trigger Qt plugin loading and C++
        # type handling that is fragile without a complete desktop environment
        # (e.g. missing platform plugins or windowing system), which has
        # previously resulted in hard-to-debug C++-side errors.
        #
        # Instead of mocking QIcon itself (which would bypass the real
        # filesystem-based icon lookup that we want to exercise), we point
        # ASSETS_DIR at an empty temporary directory. This makes icon lookups
        # resolve via genuine filesystem checks but guarantees that no real
        # icon files are found, so Qt never attempts heavy-weight icon
        # initialization while the logic we care about (graceful handling of
        # missing assets) is still executed.
        # Skip in CI - mixed mock/real Qt causes hangs and segfaults
        is_ci = (
            os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"
        )

        if is_ci:
            pytest.skip(
                "UpstreamDriftLauncher construction unreliable in CI (mock/real Qt mix)"
            )

        # Mock AIAssistantPanel before importing to prevent Qt crashes
        # Clear modules first so patches take effect
        sys.modules.pop("src.launchers.upstream_drift_launcher", None)
        sys.modules.pop("src.launchers.upstream_drift_launcher", None)
        sys.modules.pop("src.launchers.ui_components", None)
        sys.modules.pop("src.shared.python.ai.gui.assistant_panel", None)
        sys.modules.pop("src.shared.python.ai.gui", None)
        sys.modules.pop("shared.python.ai.gui", None)
        sys.modules.pop("shared.python.ai.gui.assistant_panel", None)

        mock_ai_panel = MagicMock()
        mock_ai_panel.settings_requested = MagicMock()

        # Patch AIAssistantPanel BEFORE importing upstream_drift_launcher
        # The import happens at module level, so we need the patch in place first
        ai_panel_patcher = patch(
            "src.shared.python.ai.gui.AIAssistantPanel",
            return_value=mock_ai_panel,
        )
        ai_panel_patcher.start()

        # Patch ContextHelpDock to avoid TypeError from real QDockWidget parent
        # ContextHelpDock was refactored from upstream_drift_launcher into ui_components
        context_help_patcher = patch(
            "src.launchers.ui_components.ContextHelpDock",
            MagicMock(),
        )
        context_help_patcher.start()

        # Mock _setup_process_console to prevent fatal abort from
        # QPlainTextEdit / QDockWidget instantiation in headless environments.
        # The method creates real Qt C++ widgets (QPlainTextEdit, QToolButton,
        # QDockWidget) that are not covered by the module-level PyQt6 mocks,
        # causing a SIGABRT when Qt tries to initialise them without a display.
        # We provide stub attributes that downstream code expects.
        def _mock_setup_process_console(self_arg) -> None:
            self_arg._console_text = MagicMock()
            self_arg._console_dock = MagicMock()

        console_patcher = patch(
            "src.launchers.launcher_ui_setup.UISetupManager._setup_process_console",
            _mock_setup_process_console,
        )
        console_patcher.start()

        # Patch the entire init_ui to prevent SIGABRT from real Qt widget
        # creation (QWidget, QSplitter, etc.) in headless WSL environments.
        # In the full test suite, launcher_ui_setup is already imported with
        # real C++ Qt classes cached at module scope, so the sys.modules mock
        # for PyQt6 has no effect on those cached references.
        def _mock_init_ui(self_arg) -> None:
            self_arg.grid_layout = MockQLayout()
            self_arg.btn_launch = MagicMock()
            self_arg.lbl_status = MagicMock()
            self_arg.content_splitter = MagicMock()
            self_arg.search_bar = MagicMock()

        init_ui_patcher = patch(
            "src.launchers.launcher_ui_setup.UISetupManager.init_ui",
            _mock_init_ui,
        )
        init_ui_patcher.start()

        # Patch _init_ui_components to prevent QShortcut creation which
        # rejects UpstreamDriftLauncher (inheriting MockQMainWindow) as parent
        # when the real PyQt6.QtGui.QShortcut is cached at module scope.
        ui_components_patcher = patch(
            "src.launchers.launcher_dialogs.DialogsManager._init_ui_components",
            lambda self_arg: setattr(self_arg, "toast_manager", None),
        )
        ui_components_patcher.start()

        try:
            with (
                patch(
                    "src.shared.python.config.model_registry.ModelRegistry"
                ) as MockRegistry,
                patch(
                    "src.launchers.upstream_drift_launcher.ASSETS_DIR", new=temp_path
                ),
            ):
                MockRegistry.return_value = temp_registry

                # Create Launcher
                # Import after patches are in place
                from src.launchers.upstream_drift_launcher import (
                    UpstreamDriftLauncher as FreshUpstreamDriftLauncher,
                )

                launcher = FreshUpstreamDriftLauncher()
                yield launcher, model_xml
        finally:
            ui_components_patcher.stop()
            init_ui_patcher.stop()
            console_patcher.stop()
            ai_panel_patcher.stop()
            context_help_patcher.stop()


def test_launcher_handles_missing_file_on_launch(launcher_env) -> None:
    """Test launching a model where the file was deleted after load."""
    launcher, model_path = launcher_env

    launcher.select_model("test_integration_model")

    # DELETE the file
    os.remove(model_path)
    assert not model_path.exists(), "Assertion failed: not model_path.exists()"

    # Attempt launch while mocking the low-level launch helpers to avoid
    # real subprocess execution or mujoco imports.  The launcher should
    # surface an error (via show_toast or status label) rather than
    # proceeding successfully.
    #
    # _launch_generic_mjcf is the fallback for mjcf models and will
    # fail when the file is missing.  We also mock secure_popen and
    # subprocess.Popen in the simulation mixin to prevent any real
    # process creation.
    with (
        patch("src.launchers.launcher_simulation.subprocess.Popen") as mock_popen,
        patch("src.launchers.launcher_simulation.secure_popen") as mock_secure_popen,
        patch("src.launchers.launcher_simulation.QMessageBox"),
    ):
        launcher.launch_simulation()

        # Should NOT call Popen/secure_popen because file is missing
        mock_popen.assert_not_called()
        mock_secure_popen.assert_not_called()
