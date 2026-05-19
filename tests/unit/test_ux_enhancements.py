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
    """Import upstream_drift_launcher with mocked Qt modules."""
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
        if "src.launchers.upstream_drift_launcher" in sys.modules:
            del sys.modules["src.launchers.upstream_drift_launcher"]
        import src.launchers.upstream_drift_launcher

        yield src.launchers.upstream_drift_launcher
