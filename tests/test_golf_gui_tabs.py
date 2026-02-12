import sys
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch as _patch

from src.shared.python.data_io.path_utils import get_repo_root

# Simscape GUI directory contains spaces and cannot be a proper Python package.
# This is the one intentional sys.path.insert remaining in the codebase.
_SIMSCAPE_GUI_DIR = str(
    get_repo_root()
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "matlab"
    / "src"
    / "apps"
    / "golf_gui"
    / "Simscape Multibody Data Plotters"
    / "Python Version"
    / "integrated_golf_gui_r0"
)
if _SIMSCAPE_GUI_DIR not in sys.path:
    sys.path.insert(0, _SIMSCAPE_GUI_DIR)


# Define Mock Base Classes
class MockQObject:
    def __init__(self, parent=None):
        self.parent = parent


class MockQWidget(MockQObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = None

    def setLayout(self, layout):
        self.layout = layout

    def show(self):
        pass

    def setFocusPolicy(self, policy):
        pass


class MockQOpenGLWidget(MockQWidget):
    def __init__(self, parent=None):
        super().__init__(parent)


class MockLayout(MockQObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.widgets = []
        self.layouts = []

    def addWidget(self, widget, *args):
        self.widgets.append(widget)

    def addLayout(self, layout, *args):
        self.layouts.append(layout)


class MockQVBoxLayout(MockLayout):
    pass


class MockQHBoxLayout(MockLayout):
    pass


class MockQGridLayout(MockLayout):
    pass


class MockQGroupBox(MockQWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.title = title


class MockQPushButton(MockQWidget):
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.clicked = MagicMock()
        self.setMaximumWidth = MagicMock()


class MockQLabel(MockQWidget):
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.text = text
        self.setStyleSheet = MagicMock()
        self.setAlignment = MagicMock()

    def setText(self, text):
        self.text = text


class MockQComboBox(MockQWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = []
        self.currentText = MagicMock(return_value="")

    def addItems(self, items):
        self.items.extend(items)


class MockQSlider(MockQWidget):
    def __init__(self, orientation=None, parent=None):
        super().__init__(parent)
        self.valueChanged = MagicMock()
        self.setMinimum = MagicMock()
        self.setMaximum = MagicMock()
        self.setValue = MagicMock()
        self.blockSignals = MagicMock()


class MockQCheckBox(MockQWidget):
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.stateChanged = MagicMock()
        self.setChecked = MagicMock()
        self.isChecked = MagicMock(return_value=True)


class MockQPropertyAnimation(MockQObject):
    def __init__(self, target, property_name, parent=None):
        super().__init__(parent)
        self.setEasingCurve = MagicMock()
        self.valueChanged = MagicMock()
        self.finished = MagicMock()
        self.setStartValue = MagicMock()
        self.setEndValue = MagicMock()
        self.setDuration = MagicMock()
        self.start = MagicMock()
        self.pause = MagicMock()
        self.stop = MagicMock()


class MockProperty:
    def __init__(self, fget=None, fset=None):
        self.fget = fget
        self.fset = fset

    def setter(self, fset):
        self.fset = fset
        return self


def mock_pyqt_property(type_):
    def decorator(func):
        return MockProperty(fget=func)

    return decorator


# -------------------------------------------------------------------------
# MOCK MODULES – scoped to the import of golf_gui_application only.
#
# We temporarily inject mock modules into sys.modules so that
# golf_gui_application picks them up during import, then immediately
# restore the originals.  This prevents poisoning later tests that depend
# on real scipy, PyQt6, etc.
# -------------------------------------------------------------------------

# Build the mock module dict
_mock_qt_widgets = MagicMock()
_mock_qt_widgets.QWidget = MockQWidget
_mock_qt_widgets.QOpenGLWidget = MockQOpenGLWidget
_mock_qt_widgets.QVBoxLayout = MockQVBoxLayout
_mock_qt_widgets.QHBoxLayout = MockQHBoxLayout
_mock_qt_widgets.QGridLayout = MockQGridLayout
_mock_qt_widgets.QGroupBox = MockQGroupBox
_mock_qt_widgets.QPushButton = MockQPushButton
_mock_qt_widgets.QLabel = MockQLabel
_mock_qt_widgets.QComboBox = MockQComboBox
_mock_qt_widgets.QSlider = MockQSlider
_mock_qt_widgets.QCheckBox = MockQCheckBox
_mock_qt_widgets.QMainWindow = MockQWidget
_mock_qt_widgets.QApplication = MagicMock()
_mock_qt_widgets.QMessageBox = MagicMock()
_mock_qt_widgets.QStatusBar = MockQWidget
_mock_qt_widgets.QTabWidget = MockQWidget  # Simplified

_mock_qt_core = MagicMock()
_mock_qt_core.QObject = MockQObject
_mock_qt_core.Qt = MagicMock()
_mock_qt_core.Qt.Orientation.Horizontal = 1
_mock_qt_core.Qt.AlignmentFlag.AlignCenter = 1
_mock_qt_core.Qt.FocusPolicy.StrongFocus = 1
_mock_qt_core.Qt.Key = MagicMock()
_mock_qt_core.Qt.MouseButton = MagicMock()
_mock_qt_core.pyqtSignal = MagicMock(return_value=MagicMock())
_mock_qt_core.pyqtProperty = MagicMock(side_effect=mock_pyqt_property)
_mock_qt_core.QPropertyAnimation = MockQPropertyAnimation
_mock_qt_core.QEasingCurve = MagicMock()

_mock_qt_opengl = MagicMock()
_mock_qt_opengl.QOpenGLWidget = MockQOpenGLWidget

_MOCK_MODULES = {
    "PyQt6": MagicMock(),
    "PyQt6.QtCore": _mock_qt_core,
    "PyQt6.QtGui": MagicMock(),
    "PyQt6.QtWidgets": _mock_qt_widgets,
    "PyQt6.QtOpenGLWidgets": _mock_qt_opengl,
    "moderngl": MagicMock(),
    "golf_opengl_renderer": MagicMock(),
    "golf_video_export": MagicMock(),
    "numba": MagicMock(),
    "scipy": MagicMock(),
    "scipy.io": MagicMock(),
    "golf_inverse_dynamics": MagicMock(),
    "wiffle_data_loader": MagicMock(),
}

# Import golf_gui_application inside a patch.dict context so that
# sys.modules is automatically restored when the import completes.
with _patch.dict("sys.modules", _MOCK_MODULES):
    import golf_gui_application  # noqa: E402


class TestGolfGuiTabs(unittest.TestCase):
    def test_simulink_model_tab_structure(self):
        # Instantiate the tab
        tab = golf_gui_application.SimulinkModelTab()

        # Check layout
        self.assertTrue(hasattr(tab, "layout"))
        self.assertIsInstance(tab.layout, MockQVBoxLayout)

        # Check widgets
        # Layout structure: Control Panel, Visualizer, Status Bar
        self.assertEqual(len(tab.layout.widgets), 3)
        self.assertIsInstance(tab.layout.widgets[0], golf_gui_application.QGroupBox)
        self.assertIsInstance(
            tab.layout.widgets[1], golf_gui_application.GolfVisualizerWidget
        )
        self.assertIsInstance(tab.layout.widgets[2], golf_gui_application.QLabel)

    def test_comparison_tab_structure(self):
        # Instantiate the tab
        tab = golf_gui_application.ComparisonTab()

        # Check layout
        self.assertTrue(hasattr(tab, "layout"))
        self.assertIsInstance(tab.layout, MockQVBoxLayout)

        # Widgets: Control Panel, Metrics Label
        # Layouts: Split Layout

        control_panel_found = False
        metrics_found = False
        split_layout_found = False

        for w in tab.layout.widgets:
            if isinstance(w, golf_gui_application.QGroupBox):
                control_panel_found = True
            if isinstance(w, golf_gui_application.QLabel) and "Metrics" in w.text:
                metrics_found = True

        for layout in tab.layout.layouts:
            if isinstance(layout, MockQGridLayout):
                split_layout_found = True
                # Check for 2 visualizers
                visualizers = [
                    w
                    for w in layout.widgets
                    if isinstance(w, golf_gui_application.GolfVisualizerWidget)
                ]
                self.assertEqual(len(visualizers), 2)

        self.assertTrue(control_panel_found)
        self.assertTrue(metrics_found)
        self.assertTrue(split_layout_found)


if __name__ == "__main__":
    unittest.main()
