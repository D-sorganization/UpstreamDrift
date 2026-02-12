import sys
import os
import unittest
from unittest.mock import MagicMock

# Add the directory to sys.path
gui_dir = os.path.abspath("src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Python Version/integrated_golf_gui_r0/")
if gui_dir not in sys.path:
    sys.path.insert(0, gui_dir)

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

    def show(self): pass

    def setFocusPolicy(self, policy): pass

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

class MockQVBoxLayout(MockLayout): pass
class MockQHBoxLayout(MockLayout): pass
class MockQGridLayout(MockLayout): pass

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
# MOCK MODULES
# -------------------------------------------------------------------------

# Create mocks for the modules
mock_qt_widgets = MagicMock()
mock_qt_widgets.QWidget = MockQWidget
mock_qt_widgets.QOpenGLWidget = MockQOpenGLWidget
mock_qt_widgets.QVBoxLayout = MockQVBoxLayout
mock_qt_widgets.QHBoxLayout = MockQHBoxLayout
mock_qt_widgets.QGridLayout = MockQGridLayout
mock_qt_widgets.QGroupBox = MockQGroupBox
mock_qt_widgets.QPushButton = MockQPushButton
mock_qt_widgets.QLabel = MockQLabel
mock_qt_widgets.QComboBox = MockQComboBox
mock_qt_widgets.QSlider = MockQSlider
mock_qt_widgets.QCheckBox = MockQCheckBox
mock_qt_widgets.QMainWindow = MockQWidget
mock_qt_widgets.QApplication = MagicMock()
mock_qt_widgets.QMessageBox = MagicMock()
mock_qt_widgets.QStatusBar = MockQWidget
mock_qt_widgets.QTabWidget = MockQWidget # Simplified

mock_qt_core = MagicMock()
mock_qt_core.QObject = MockQObject
mock_qt_core.Qt = MagicMock()
mock_qt_core.Qt.Orientation.Horizontal = 1
mock_qt_core.Qt.AlignmentFlag.AlignCenter = 1
mock_qt_core.Qt.FocusPolicy.StrongFocus = 1
mock_qt_core.Qt.Key = MagicMock()
mock_qt_core.Qt.MouseButton = MagicMock()
mock_qt_core.pyqtSignal = MagicMock(return_value=MagicMock())
mock_qt_core.pyqtProperty = MagicMock(side_effect=mock_pyqt_property) # Decorator mock
mock_qt_core.QPropertyAnimation = MockQPropertyAnimation
mock_qt_core.QEasingCurve = MagicMock()

# Inject into sys.modules
sys.modules["PyQt6"] = MagicMock()
sys.modules["PyQt6.QtCore"] = mock_qt_core
sys.modules["PyQt6.QtGui"] = MagicMock()
sys.modules["PyQt6.QtWidgets"] = mock_qt_widgets
sys.modules["PyQt6.QtOpenGLWidgets"] = MagicMock()
sys.modules["PyQt6.QtOpenGLWidgets"].QOpenGLWidget = MockQOpenGLWidget # Important

sys.modules["moderngl"] = MagicMock()
sys.modules["golf_opengl_renderer"] = MagicMock()
sys.modules["golf_video_export"] = MagicMock()

# Import the module under test
import golf_gui_application

class TestGolfGuiTabs(unittest.TestCase):
    def test_simulink_model_tab_structure(self):
        # Instantiate the tab
        tab = golf_gui_application.SimulinkModelTab()

        # Check layout
        self.assertTrue(hasattr(tab, 'layout'))
        self.assertIsInstance(tab.layout, MockQVBoxLayout)

        # Check widgets
        # Layout structure: Control Panel, Visualizer, Status Bar
        self.assertEqual(len(tab.layout.widgets), 3)
        self.assertIsInstance(tab.layout.widgets[0], golf_gui_application.QGroupBox)
        self.assertIsInstance(tab.layout.widgets[1], golf_gui_application.GolfVisualizerWidget)
        self.assertIsInstance(tab.layout.widgets[2], golf_gui_application.QLabel)

    def test_comparison_tab_structure(self):
        # Instantiate the tab
        tab = golf_gui_application.ComparisonTab()

        # Check layout
        self.assertTrue(hasattr(tab, 'layout'))
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

        for l in tab.layout.layouts:
            if isinstance(l, MockQGridLayout):
                split_layout_found = True
                # Check for 2 visualizers
                visualizers = [w for w in l.widgets if isinstance(w, golf_gui_application.GolfVisualizerWidget)]
                self.assertEqual(len(visualizers), 2)

        self.assertTrue(control_panel_found)
        self.assertTrue(metrics_found)
        self.assertTrue(split_layout_found)

if __name__ == "__main__":
    unittest.main()
