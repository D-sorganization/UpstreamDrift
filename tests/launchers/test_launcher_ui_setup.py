"""Tests for launcher_ui_setup.py."""

from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QVBoxLayout  # noqa: E402

from src.launchers.launcher_ui_setup import LauncherUISetupMixin  # noqa: E402


class DummyLauncher(QMainWindow, LauncherUISetupMixin):
    def __init__(self):
        super().__init__()
        self.apply_styles = MagicMock()
        self._show_preferences = MagicMock()
        self._toggle_layout_mode_from_menu = MagicMock()
        self._toggle_context_help = MagicMock()
        self._setup_theme_menu = MagicMock()
        self._open_settings = MagicMock()
        self._show_help_dialog = MagicMock()
        self._open_project_map = MagicMock()
        self._show_shortcuts_overlay = MagicMock()
        self._show_about_dialog = MagicMock()
        self.update_search_filter = MagicMock()
        self._on_docker_mode_changed = MagicMock()
        self._on_wsl_mode_changed = MagicMock()
        self.toggle_layout_mode = MagicMock()
        self.open_layout_manager = MagicMock()
        self.toggle_ai_assistant = MagicMock()
        self.launch_simulation = MagicMock()
        self._open_ai_settings = MagicMock()


@pytest.fixture
def launcher(qapp) -> DummyLauncher:
    return DummyLauncher()


def test_init_ui(launcher) -> None:
    with patch("src.launchers.launcher_constants.AI_AVAILABLE", False):
        launcher.init_ui()
        assert launcher.content_splitter is not None
        launcher.apply_styles.assert_called_once()


@patch("src.launchers.launcher_constants.AI_AVAILABLE", True)
def test_init_ui_with_ai(launcher) -> None:
    with patch.object(launcher, "_setup_ai_panel") as mock_ai:
        launcher.init_ui()
        mock_ai.assert_called_once()


def test_setup_menu_bar(launcher) -> None:
    launcher._setup_menu_bar()
    menubar = launcher.menuBar()
    actions = menubar.actions()
    # File, View, Tools, Help
    assert len(actions) == 4


def test_setup_top_bar(launcher) -> None:
    with patch("src.launchers.launcher_constants.HELP_SYSTEM_AVAILABLE", False):
        top_bar = launcher._setup_top_bar()
        assert isinstance(top_bar, QHBoxLayout)


@patch("src.launchers.launcher_constants.HELP_SYSTEM_AVAILABLE", True)
@patch("src.shared.python.gui_pkg.help_system.TooltipManager.register_tooltip")
def test_setup_top_bar_with_help(mock_register, launcher) -> None:
    with patch("src.launchers.launcher_constants.AI_AVAILABLE", True):
        top_bar = launcher._setup_top_bar()
        assert isinstance(top_bar, QHBoxLayout)
        assert mock_register.call_count >= 4
        assert hasattr(launcher, "btn_ai")


def test_setup_grid_area(launcher) -> None:
    layout = QVBoxLayout()
    launcher._setup_grid_area(layout)
    assert hasattr(launcher, "scroll_area")


def test_setup_bottom_bar(launcher) -> None:
    launcher._setup_bottom_bar()
    assert hasattr(launcher, "btn_launch")


def test_setup_search_shortcuts(launcher) -> None:
    with patch("src.launchers.launcher_ui_setup.QShortcut") as mock_shortcut:
        launcher._setup_search_shortcuts()
        assert mock_shortcut.call_count == 2


def test_focus_search(launcher) -> None:
    launcher.search_input = MagicMock()
    launcher._focus_search()
    launcher.search_input.setFocus.assert_called_once()
    launcher.search_input.selectAll.assert_called_once()


def test_clear_search(launcher) -> None:
    launcher.search_input = MagicMock()
    launcher.search_input.hasFocus.return_value = True
    launcher._clear_search()
    launcher.search_input.clear.assert_called_once()
    launcher.search_input.clearFocus.assert_called_once()

    launcher.search_input.reset_mock()
    launcher.search_input.hasFocus.return_value = False
    launcher._clear_search()
    launcher.search_input.clear.assert_not_called()


def test_process_console(launcher) -> None:
    launcher._setup_process_console()
    assert hasattr(launcher, "_console_dock")

    # replace the dock with a mock to avoid Qt show/hide state issues
    dock_mock = MagicMock()
    dock_mock.isVisible.return_value = False
    launcher._console_dock = dock_mock

    # toggle
    launcher.toggle_process_console()
    dock_mock.setVisible.assert_called_with(True)

    # append_console_line
    launcher._console_dock.isVisible.return_value = False
    # mock the plain text edit to avoid needing a full Qt window text append test
    launcher._console_text = MagicMock()
    launcher._append_console_line("engine", "test message")
    launcher._console_dock.show.assert_called_once()
    launcher._console_text.appendPlainText.assert_called()


@patch("src.launchers.launcher_ui_setup.QTimer.singleShot")
def test_on_process_output(mock_timer, launcher) -> None:
    launcher._on_process_output("engine", "test")
    mock_timer.assert_called_once()


def test_setup_ai_panel_disabled(launcher) -> None:
    launcher.content_splitter = MagicMock()
    with patch("src.launchers.launcher_constants.AI_AVAILABLE", False):
        launcher._setup_ai_panel()
        assert not hasattr(launcher, "ai_panel")


@patch("src.launchers.launcher_constants.AI_AVAILABLE", True)
def test_setup_ai_panel(launcher) -> None:
    with patch("src.shared.python.ai.gui.AIAssistantPanel"):
        launcher.content_splitter = MagicMock()
        with patch.object(launcher, "_sync_chat_session") as mock_sync:
            launcher._setup_ai_panel()
            mock_sync.assert_called_once()
            assert hasattr(launcher, "ai_panel")


@patch("src.launchers.launcher_constants.AI_AVAILABLE", True)
def test_setup_ai_panel_error(launcher) -> None:
    original_import = __import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0) -> object:
        if name == "src.shared.python.ai.gui":
            raise ImportError("test error")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=mock_import):
        launcher.btn_ai = MagicMock()
        launcher._setup_ai_panel()
        launcher.btn_ai.setEnabled.assert_called_with(False)


@patch("urllib.request.urlopen")
def test_sync_chat_session(mock_urlopen, launcher) -> None:
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'[{"session_id": "test_id"}]'
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    with (
        patch("pathlib.Path.home", return_value=MagicMock()),
        patch("pathlib.Path.mkdir"),
        patch("pathlib.Path.write_text"),
    ):
        # Test success
        launcher._sync_chat_session()
        # Test exception
        mock_urlopen.side_effect = OSError("err")
        launcher._sync_chat_session()


def test_init_overlay(launcher) -> None:
    with patch("src.shared.python.ui.overlay.OverlayWidget"):
        launcher._init_overlay()
        assert hasattr(launcher, "overlay")

        launcher._toggle_overlay()
        launcher.overlay.toggle.assert_called_once()


def test_init_overlay_error(launcher) -> None:
    original_import = __import__

    def mock_import(name, *args, **kwargs) -> object:
        if "vendor" in name or "overlay" in name:
            raise ImportError("test")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        launcher._init_overlay()
        assert not hasattr(launcher, "overlay")
