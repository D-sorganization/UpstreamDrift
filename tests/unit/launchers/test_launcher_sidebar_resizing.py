import pytest
from unittest.mock import MagicMock
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QSizePolicy
from src.launchers.launcher_ui_setup import UISetupManager


class DummyLauncher(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manager = UISetupManager(self)

        self.layout_manager = MagicMock()
        self._action_buttons = []
        self._top_bar = None
        self.grid_layout = None
        self.content_splitter = None


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_sidebar_is_resizable(qapp):
    """Test that the global sidebar is configured to be resizable."""
    launcher = DummyLauncher()
    sidebar = launcher._setup_global_sidebar()

    # Mocked widgets won't return 85, so we assert the mock was called correctly if it's a mock
    if hasattr(sidebar.setMinimumWidth, "assert_called_with"):
        sidebar.setMinimumWidth.assert_called_with(85)
    else:
        assert sidebar.minimumWidth() == 85

    # Size policy must allow horizontal resizing (Preferred allows it)
    if hasattr(sidebar.setSizePolicy, "assert_called_with"):
        sidebar.setSizePolicy.assert_called_with(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
    else:
        policy = sidebar.sizePolicy()
        assert policy.horizontalPolicy() == QSizePolicy.Policy.Preferred
        assert policy.verticalPolicy() == QSizePolicy.Policy.Expanding


def test_main_layout_is_splitter_with_handle(qapp, monkeypatch):
    """Test that the main layout is a QSplitter to allow sidebar resizing."""
    launcher = DummyLauncher()

    # Mock complex GUI setup to isolate layout building
    monkeypatch.setattr(launcher, "_setup_top_bar", lambda: QVBoxLayout())
    monkeypatch.setattr(launcher, "_setup_grid_area", lambda *args: None)
    monkeypatch.setattr(launcher, "_setup_bottom_bar", lambda *args: None)
    monkeypatch.setattr(launcher, "_setup_ai_panel", lambda *args: None)
    monkeypatch.setattr(launcher, "apply_styles", lambda: None)

    # Patch QSplitter to verify it's used and configured
    mock_splitter = MagicMock()
    monkeypatch.setattr("src.launchers.launcher_ui_setup.QSplitter", mock_splitter)

    # Run init_ui which builds the widget tree
    launcher.init_ui()

    # Verify QSplitter was instantiated
    assert mock_splitter.called, (
        "A QSplitter must be used for the main layout to allow sidebar resizing"
    )

    # Verify configuration on the created instance
    splitter_instance = mock_splitter.return_value
    splitter_instance.setHandleWidth.assert_any_call(4)

    # Verify stretch factors were set (0 for sidebar, 1 for content)
    splitter_instance.setStretchFactor.assert_any_call(0, 0)
    splitter_instance.setStretchFactor.assert_any_call(1, 1)
