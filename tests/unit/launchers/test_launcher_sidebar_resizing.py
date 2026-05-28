import pytest
from typing import Any
from unittest.mock import MagicMock
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QSizePolicy, QSplitter
from src.launchers.launcher_ui_setup import UISetupManager


class DummyLauncher(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui_setup_manager = UISetupManager(self)

        self.layout_manager = MagicMock()
        self._action_buttons = []
        self._top_bar = None
        self.grid_layout = None
        self.content_splitter = None

    def _setup_top_bar(self):
        pass

    def _setup_grid_area(self, *args):
        pass

    def _setup_bottom_bar(self, *args):
        pass

    def _setup_ai_panel(self, *args):
        pass

    def _show_preferences(self):
        pass

    def toggle_process_console(self):
        pass

    def apply_styles(self):
        pass

    def __getattr__(self, name: str) -> Any:
        if "ui_setup_manager" in self.__dict__:
            manager = self.ui_setup_manager
            if name in manager.__dict__ or hasattr(type(manager), name):
                import types

                attr = getattr(manager, name)
                if isinstance(attr, types.MethodType):
                    return types.MethodType(attr.__func__, self)
                return attr
        return MagicMock()


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_sidebar_is_resizable(qapp):
    """Test that the global sidebar is configured to be resizable."""
    from src.shared.python.theme.style_constants import Styles

    launcher = DummyLauncher()
    sidebar = launcher._setup_global_sidebar()

    # Mocked widgets won't return Styles.SIDEBAR_MIN_WIDTH, so we assert the mock was called correctly if it's a mock
    if hasattr(sidebar.setMinimumWidth, "assert_called_with"):
        sidebar.setMinimumWidth.assert_called_with(Styles.SIDEBAR_MIN_WIDTH)
    else:
        assert sidebar.minimumWidth() == Styles.SIDEBAR_MIN_WIDTH

    # Size policy must allow horizontal resizing (Preferred allows it)
    inner_widget = sidebar.widget()
    if hasattr(inner_widget.setSizePolicy, "assert_called_with"):
        inner_widget.setSizePolicy.assert_called_with(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
    else:
        policy = inner_widget.sizePolicy()
        assert policy.horizontalPolicy() == QSizePolicy.Policy.Preferred
        assert policy.verticalPolicy() == QSizePolicy.Policy.Expanding


class MockSplitter(QSplitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.called_setHandleWidth = []
        self.called_setStretchFactor = []

    def setHandleWidth(self, width: int):
        self.called_setHandleWidth.append(width)
        super().setHandleWidth(width)

    def setStretchFactor(self, index: int, stretch: int):
        self.called_setStretchFactor.append((index, stretch))
        super().setStretchFactor(index, stretch)


def test_main_layout_is_splitter_with_handle(qapp, monkeypatch):
    """Test that the main layout is a QSplitter to allow sidebar resizing."""
    launcher = DummyLauncher()

    # Mock complex GUI setup to isolate layout building
    monkeypatch.setattr(launcher, "_setup_top_bar", lambda: QVBoxLayout())
    monkeypatch.setattr(launcher, "_setup_grid_area", lambda *args: None)
    monkeypatch.setattr(launcher, "_setup_bottom_bar", lambda *args: None)
    monkeypatch.setattr(launcher, "_setup_ai_panel", lambda *args: None)
    monkeypatch.setattr(launcher, "apply_styles", lambda: None)

    # Patch QSplitter with MockSplitter factory to verify it's used and configured without raising TypeError
    created_instances = []

    def factory(*args, **kwargs):
        inst = MockSplitter(*args, **kwargs)
        created_instances.append(inst)
        return inst

    monkeypatch.setattr("src.launchers.launcher_ui_setup.QSplitter", factory)

    # Run init_ui which builds the widget tree
    launcher.init_ui()

    # Verify QSplitter was instantiated
    assert created_instances, (
        "A QSplitter must be used for the main layout to allow sidebar resizing"
    )

    # Verify configuration on the created instance
    inst = created_instances[0]
    assert 4 in inst.called_setHandleWidth

    # Verify stretch factors were set (0 for sidebar, 1 for content)
    assert (0, 0) in inst.called_setStretchFactor
    assert (1, 1) in inst.called_setStretchFactor
