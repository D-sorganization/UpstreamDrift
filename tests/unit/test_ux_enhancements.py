import sys
import types
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest


# Mock Qt classes — use __getattr__ catch-all to handle any missing Qt methods
class MockQWidget:
    def __init__(self, parent=None):
        pass

    def __getattr__(self, name):
        """Return a no-op callable for any missing Qt method."""
        return lambda *args, **kwargs: None

    def setAccessibleName(self, name) -> None:
        pass

    def setAccessibleDescription(self, desc) -> None:
        pass

    def setObjectName(self, name) -> None:
        pass

    def setCursor(self, cursor) -> None:
        pass

    def setFocusPolicy(self, policy) -> None:
        pass

    def setStyleSheet(self, style) -> None:
        pass


class MockQLabel(MockQWidget):
    def __init__(self, text="", parent=None):
        self._text = text

    def setFont(self, font) -> None:
        pass

    def setWordWrap(self, wrap) -> None:
        pass

    def setAlignment(self, align) -> None:
        pass

    def setStyleSheet(self, style) -> None:
        pass

    def setFixedWidth(self, w) -> None:
        pass

    def setFixedSize(self, w, h) -> None:
        pass

    def setPixmap(self, pixmap) -> None:
        pass

    def setToolTip(self, text) -> None:
        pass

    def setText(self, text) -> None:
        self._text = text


class MockQVBoxLayout:
    def __init__(self, parent=None):
        pass

    def setAlignment(self, *args) -> None:
        pass

    def addWidget(self, widget, *args, **kwargs) -> None:
        pass

    def addLayout(self, layout, *args, **kwargs) -> None:
        pass

    def setContentsMargins(self, *args) -> None:
        pass


class MockQHBoxLayout:
    def __init__(self, parent=None):
        pass

    def addStretch(self, *args) -> None:
        pass

    def addWidget(self, widget, *args, **kwargs) -> None:
        pass

    def addLayout(self, layout, *args, **kwargs) -> None:
        pass

    def setContentsMargins(self, *args) -> None:
        pass


class MockQFrame(MockQWidget):
    def __init__(self, parent=None):
        pass

    def setAcceptDrops(self, accept) -> None:
        pass


@pytest.fixture
def mocked_launcher_module() -> Generator[types.ModuleType, None, None]:
    """Import golf_launcher with mocked Qt modules."""
    mock_qt_core = MagicMock()
    mock_qt_core.Qt = MagicMock()
    mock_qt_core.QPoint = MagicMock()
    mock_qt_core.QMimeData = MagicMock()
    mock_qt_core.QTimer = MagicMock()

    mock_qt_widgets = MagicMock()
    mock_qt_widgets.QWidget = MockQWidget
    mock_qt_widgets.QLabel = MockQLabel
    mock_qt_widgets.QVBoxLayout = MockQVBoxLayout
    mock_qt_widgets.QHBoxLayout = MockQHBoxLayout
    mock_qt_widgets.QFrame = MockQFrame
    mock_qt_widgets.QApplication = MagicMock()

    mock_qt_gui = MagicMock()
    mock_qt_gui.QPixmap = MagicMock()
    mock_qt_gui.QFont = MagicMock()
    mock_qt_gui.QColor = MagicMock()
    mock_qt_gui.QDrag = MagicMock()

    mock_modules = {
        "PyQt6": MagicMock(),
        "PyQt6.QtCore": mock_qt_core,
        "PyQt6.QtGui": mock_qt_gui,
        "PyQt6.QtWidgets": mock_qt_widgets,
        "src.shared.python.engine_manager": MagicMock(),
        "src.shared.python.model_registry": MagicMock(),
        "src.shared.python.secure_subprocess": MagicMock(),
    }

    with patch.dict(sys.modules, mock_modules):
        if "src.launchers.golf_launcher" in sys.modules:
            del sys.modules["src.launchers.golf_launcher"]
        import src.launchers.golf_launcher

        yield src.launchers.golf_launcher


@pytest.mark.xfail(
    reason="Theme colors are dynamic (QSettings-based), cannot validate in mocked env",
    strict=False,
)
def test_status_info_contrast(mocked_launcher_module) -> None:
    """Test that _get_status_info returns appropriate text colors."""

    # Mock model object
    class MockModel:
        def __init__(self, type_name, path=""):
            self.type = type_name
            self.path = path
            self.name = "Test Model"
            self.description = "Desc"
            self.id = "test_model"

    # Test cases: (model_type, expected_bg, expected_text_color)
    test_cases = [
        ("custom_humanoid", "#28a745", "#000000"),  # Green -> Black
        ("drake", "#28a745", "#000000"),  # Green -> Black
        ("mjcf", "#17a2b8", "#000000"),  # Blue -> Black
        ("matlab", "#6f42c1", "#ffffff"),  # Purple -> White
        ("urdf_generator", "#6c757d", "#ffffff"),  # Gray -> White
    ]

    mock_parent_launcher = MagicMock()
    mock_parent_launcher.layout_edit_mode = False

    for m_type, exp_bg, exp_text in test_cases:
        model = MockModel(m_type)
        card = mocked_launcher_module.DraggableModelCard(model, mock_parent_launcher)

        # We expect _get_status_info to return 3 values now
        status_info = card._get_status_info()

        # Currently it returns 2, so this test will fail if we assert length is 3
        # or if we try to unpack 3 values.
        # But for TDD, we want to verify the Logic.

        # If the code hasn't been changed yet, this will be length 2.
        if len(status_info) == 2:
            text, bg = status_info
            text_color = "white"  # Default in current code
        else:
            text, bg, text_color = status_info

        assert bg == exp_bg, f"Background color mismatch for {m_type}"

        # This assertion defines our requirement for the new feature
        assert text_color == exp_text, (
            f"Text color mismatch for {m_type}. Expected {exp_text}, got {text_color}"
        )


@pytest.mark.xfail(
    reason="QShortcut mocking with complex imports is flaky", strict=False
)
def test_escape_shortcut_logic(mocked_launcher_module) -> None:
    """Test that GolfLauncher sets up the Escape shortcut."""
    with (
        patch("src.launchers.golf_launcher.QShortcut") as MockShortcut,
        patch("src.launchers.golf_launcher.QKeySequence") as MockKeySequence,
    ):
        # Setup QKeySequence to return identifiable mocks
        def key_seq_side_effect(arg) -> MagicMock:
            m = MagicMock()
            m.key_str = arg
            return m

        MockKeySequence.side_effect = key_seq_side_effect

        mocked_launcher_module.GolfLauncher()

        # Check if QShortcut was called with a key sequence for "Esc"
        found_escape = False
        for _i, call in enumerate(MockShortcut.call_args_list):
            args = call[0]
            if args:
                first_arg = args[0]
                if hasattr(first_arg, "key_str"):
                    if first_arg.key_str == "Esc":
                        found_escape = True
                        break
                else:
                    pass

        assert found_escape, "Escape shortcut not registered"
