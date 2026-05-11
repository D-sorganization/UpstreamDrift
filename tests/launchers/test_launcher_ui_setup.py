"""Tests for launcher_ui_setup.py."""

from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QToolButton,
    QVBoxLayout,
)  # noqa: E402
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


def test_setup_global_sidebar_uses_icon_navigation(launcher) -> None:
    sidebar = launcher._setup_global_sidebar()
    buttons = sidebar.findChildren(QToolButton)
    assert len(buttons) == 5

    accessible_names = {button.accessibleName() for button in buttons}
    assert accessible_names == {
        "Home",
        "Engines",
        "Biomechanics",
        "Settings",
        "Documentation",
    }

    for button in buttons:
        assert not button.icon().isNull()
        assert button.accessibleName() in accessible_names


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
    assert "Outfit" in launcher.btn_launch.font().families()


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
        assert "ai_panel" not in launcher.__dict__


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
        assert "overlay" not in launcher.__dict__


# ---------------------------------------------------------------------------
# Biomechanics sidebar button + routing (issue #5180)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sidebar_includes_biomechanics_button(launcher) -> None:
    """The sidebar build invokes ``_build_sidebar_button`` for the Biomechanics
    label, demonstrating that the third sidebar control is wired in."""
    build_button_spy = MagicMock(wraps=launcher._build_sidebar_button)
    with (
        patch.object(launcher, "_build_sidebar_button", build_button_spy),
        patch(
            "src.launchers.launcher_ui_setup.QWidget.setTabOrder",
            new=MagicMock(),
            create=True,
        ),
    ):
        launcher._setup_global_sidebar()

    labels_used = [call.args[0] for call in build_button_spy.call_args_list]
    assert "Biomechanics" in labels_used
    # And the canonical Home/Engines labels are still wired in.
    assert "Home" in labels_used
    assert "Engines" in labels_used


@pytest.mark.unit
def test_sidebar_biomechanics_button_registered_with_id_two(launcher) -> None:
    """``QButtonGroup`` mutual-exclusion is preserved and the Biomechanics
    button is registered under ``id=2`` per the issue spec.

    Spies on ``_build_sidebar_button`` so we can identify which button
    object corresponds to the Biomechanics label, then verifies that exact
    object is registered into the QButtonGroup with ``id=2``.
    """
    built: dict[str, object] = {}

    real_builder = launcher._build_sidebar_button

    def _spy_builder(label, icon_name, *, checkable=False):
        button = real_builder(label, icon_name, checkable=checkable)
        built[label] = button
        return button

    add_button_calls: list[tuple[object, int]] = []

    class _RecordingGroup:
        def __init__(self, *_args, **_kwargs):
            self._buttons: list[object] = []
            self._exclusive = True

        def addButton(self, button, button_id):  # noqa: N802 - Qt API
            add_button_calls.append((button, button_id))
            self._buttons.append(button)

        @property
        def idClicked(self):  # noqa: N802 - Qt API
            return MagicMock()

        def buttons(self):
            return list(self._buttons)

        def setExclusive(self, value):  # noqa: N802 - Qt API
            self._exclusive = bool(value)

        def exclusive(self):
            return self._exclusive

    with (
        patch.object(launcher, "_build_sidebar_button", _spy_builder),
        patch(
            "src.launchers.launcher_ui_setup.QButtonGroup",
            new=_RecordingGroup,
        ),
        patch(
            "src.launchers.launcher_ui_setup.QWidget.setTabOrder",
            new=MagicMock(),
            create=True,
        ),
    ):
        launcher._setup_global_sidebar()

    ids_registered = {bid for _, bid in add_button_calls}
    # Home (id=0), Engines (id=1), Biomech (id=2) all registered
    assert {0, 1, 2}.issubset(ids_registered)
    # Mutual-exclusion is still on by default
    assert launcher.sidebar_group.exclusive() is True

    # The Biomechanics button must be the one registered at id=2.
    button_at_id_two = next(b for b, bid in add_button_calls if bid == 2)
    assert button_at_id_two is built["Biomechanics"]


@pytest.mark.unit
def test_on_sidebar_routed_biomechanics_sets_filter(launcher) -> None:
    """Clicking the Biomechanics button (``id=2``) routes to the ``Biomechanics``
    category filter on the layout manager."""
    launcher.layout_manager = MagicMock()
    launcher._rebuild_grid = MagicMock()
    launcher._on_sidebar_routed(2)
    assert launcher.layout_manager.current_category_filter == "Biomechanics"
    launcher._rebuild_grid.assert_called_once()


@pytest.mark.unit
def test_on_sidebar_routed_existing_ids_still_work(launcher) -> None:
    """Existing Home (id=0) and Engines (id=1) routes are unchanged."""
    launcher.layout_manager = MagicMock()
    launcher._rebuild_grid = MagicMock()

    launcher._on_sidebar_routed(0)
    assert launcher.layout_manager.current_category_filter == "All"

    launcher._on_sidebar_routed(1)
    assert launcher.layout_manager.current_category_filter == "Physics Engines"
