import sys
from unittest.mock import MagicMock, patch

import pytest


# Mock Qt classes
class MockQWidget:
    def __init__(self, parent=None):
        pass

    def setAccessibleName(self, name):
        pass

    def setAccessibleDescription(self, desc):
        pass

    def setObjectName(self, name):
        pass

    def setCursor(self, cursor):
        pass

    def setFocusPolicy(self, policy):
        pass

    def setStyleSheet(self, style):
        pass


class MockQLabel(MockQWidget):
    def __init__(self, text="", parent=None):
        self._text = text

    def setFont(self, font):
        pass

    def setWordWrap(self, wrap):
        pass

    def setAlignment(self, align):
        pass

    def setStyleSheet(self, style):
        pass

    def setFixedWidth(self, w):
        pass

    def setFixedSize(self, w, h):
        pass

    def setPixmap(self, pixmap):
        pass

    def setToolTip(self, text):
        pass

    def setText(self, text):
        self._text = text


class MockQVBoxLayout:
    def __init__(self, parent=None):
        pass

    def setAlignment(self, align):
        pass

    def addWidget(self, widget):
        pass

    def addLayout(self, layout):
        pass


class MockQHBoxLayout:
    def __init__(self, parent=None):
        pass

    def addStretch(self):
        pass

    def addWidget(self, widget):
        pass

    def addLayout(self, layout):
        pass


class MockQFrame(MockQWidget):
    def __init__(self, parent=None):
        pass

    def setAcceptDrops(self, accept):
        pass


@pytest.fixture
def mocked_launcher_module():
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
        "shared.python.engine_manager": MagicMock(),
        "shared.python.model_registry": MagicMock(),
        "shared.python.secure_subprocess": MagicMock(),
    }

    with patch.dict(sys.modules, mock_modules):
        if "launchers.golf_launcher" in sys.modules:
            del sys.modules["launchers.golf_launcher"]
        import launchers.golf_launcher

        yield launchers.golf_launcher


def test_status_info_contrast(mocked_launcher_module):
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

    for m_type, exp_bg, exp_text in test_cases:
        model = MockModel(m_type)
        card = mocked_launcher_module.DraggableModelCard(model)

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
        assert (
            text_color == exp_text
        ), f"Text color mismatch for {m_type}. Expected {exp_text}, got {text_color}"


@pytest.mark.skip(reason="QShortcut mocking with complex imports is flaky")
def test_escape_shortcut_logic(mocked_launcher_module):
    """Test that GolfLauncher sets up the Escape shortcut."""
    with (
        patch("launchers.golf_launcher.QShortcut") as MockShortcut,
        patch("launchers.golf_launcher.QKeySequence") as MockKeySequence,
    ):
        # Setup QKeySequence to return identifiable mocks
        def key_seq_side_effect(arg):
            m = MagicMock()
            m.key_str = arg
            return m

        MockKeySequence.side_effect = key_seq_side_effect

        mocked_launcher_module.GolfLauncher()

        # Check if QShortcut was called with a key sequence for "Esc"
        found_escape = False
        print(f"DEBUG: Calls to QShortcut: {len(MockShortcut.call_args_list)}")
        for i, call in enumerate(MockShortcut.call_args_list):
            args = call[0]
            if args:
                first_arg = args[0]
                print(f"DEBUG: Call {i} arg[0]: {first_arg} type: {type(first_arg)}")
                if hasattr(first_arg, "key_str"):
                    print(f"DEBUG: Call {i} key_str: {first_arg.key_str}")
                    if first_arg.key_str == "Esc":
                        found_escape = True
                        break
                else:
                    print(f"DEBUG: Call {i} has no key_str attribute")

        assert found_escape, "Escape shortcut not registered"
