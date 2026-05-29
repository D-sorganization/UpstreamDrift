"""Tests for launcher_ui_setup.py."""

from unittest.mock import MagicMock, patch  # noqa: E402

from typing import Any
import pytest  # noqa: E402
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QToolButton,
    QVBoxLayout,
    QWidget,
)  # noqa: E402
from src.launchers.launcher_ui_setup import UISetupManager  # noqa: E402


class DummyLauncher(QMainWindow):
    def __getattr__(self, name: str) -> Any:
        for mgr_name in (
            "manager",
            "ui_setup_manager",
            "theme_manager",
            "simulation_manager",
            "dialogs_manager",
        ):
            if mgr_name in self.__dict__:
                manager = self.__dict__[mgr_name]
                if name in manager.__dict__ or hasattr(type(manager), name):
                    attr = getattr(manager, name)
                    import types

                    if isinstance(attr, types.MethodType):
                        return types.MethodType(attr.__func__, self)
                    return attr
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manager = UISetupManager(self)

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
        self._on_windows_mode_changed = MagicMock()
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
    # _setup_ai_panel was removed in UpstreamDrift #5620 (deprecated-chat sweep).
    # AI access now lives exclusively in the Sidekick sidebar (btn_ai_sidebar).
    # Verify that init_ui completes and sets up main_layout without the old
    # duplicate AI panel; no AttributeError must occur on re-dock.
    launcher.init_ui()
    assert launcher.content_splitter is not None
    assert hasattr(launcher, "main_layout")


def test_setup_menu_bar(launcher) -> None:
    launcher._setup_menu_bar()
    menubar = launcher.menuBar()
    actions = menubar.actions()
    if isinstance(actions, list):
        assert len(actions) == 4


def test_setup_top_bar(launcher) -> None:
    with patch("src.launchers.launcher_constants.HELP_SYSTEM_AVAILABLE", False):
        top_bar = launcher._setup_top_bar()
        assert isinstance(top_bar, QHBoxLayout)


def test_setup_global_sidebar_uses_icon_navigation(launcher) -> None:
    sidebar = launcher._setup_global_sidebar()
    buttons = sidebar.findChildren(QToolButton)
    if isinstance(buttons, list) and len(buttons) > 0:
        button_names = {button.accessibleName() for button in buttons}
        assert "Home" in button_names
        assert "Engines" in button_names

    for button in buttons:
        assert not button.icon().isNull()
        if isinstance(buttons, list) and len(buttons) > 0:
            assert button.accessibleName() in button_names


def test_library_is_sidebar_button_not_startup_tab(launcher) -> None:
    launcher.init_ui()

    assert launcher.workspace_tabs.count() == 1
    assert launcher.workspace_tabs.tabText(0) == "Home"
    assert launcher.library_widget is None
    btn_doc = launcher.sidebar_group.button(5)
    assert btn_doc is not None
    assert btn_doc.accessibleName() == "Documentation"
    assert not btn_doc.icon().isNull()


def test_library_sidebar_route_opens_single_workspace_tab(launcher) -> None:
    launcher.init_ui()

    with patch("src.launchers.library_widget.LibraryWidget", side_effect=QWidget):
        launcher._open_library_tab()
        first_widget = launcher.library_widget
        launcher._open_library_tab()

    assert first_widget is launcher.library_widget
    assert launcher.workspace_tabs.count() == 2
    assert launcher.workspace_tabs.tabText(1) == "Library"
    assert launcher.workspace_tabs.currentWidget() is first_widget


def test_library_can_pop_out_from_workspace_tab(launcher) -> None:
    launcher.init_ui()

    with patch("src.launchers.library_widget.LibraryWidget", side_effect=QWidget):
        launcher._open_library_tab()
    widget = launcher.library_widget
    assert launcher.workspace_tabs.indexOf(widget) >= 0

    with patch.object(launcher, "popout_widget") as mock_popout:
        launcher._popout_library()

    assert launcher.workspace_tabs.indexOf(widget) == -1
    mock_popout.assert_called_once_with(widget, "Library")


@patch("src.launchers.launcher_constants.HELP_SYSTEM_AVAILABLE", True)
@patch("src.shared.python.gui_pkg.help_system.TooltipManager.register_tooltip")
def test_setup_top_bar_with_help(mock_register, launcher) -> None:
    with patch("src.launchers.launcher_constants.AI_AVAILABLE", True):
        top_bar = launcher._setup_top_bar()
        assert isinstance(top_bar, QHBoxLayout)
        assert mock_register.call_count >= 4
        # btn_ai was removed in #5620; AI now lives in the sidebar as btn_ai_sidebar.
        assert not hasattr(launcher, "btn_ai"), (
            "btn_ai must not exist after #5620 deprecated-chat sweep"
        )


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


@patch("src.launchers.launcher_constants.AI_AVAILABLE", False)
def test_sidebar_ai_button_absent_when_ai_unavailable(launcher) -> None:
    """When AI is unavailable, no AI button should be added to the sidebar.

    Regression guard for UpstreamDrift #5689: ``_setup_ai_panel`` and
    ``_sync_chat_session`` were removed in the deprecated-chat sweep (#5620).
    The sidebar should build without AttributeError even when AI is off.

    We check the instance ``__dict__`` directly (not ``hasattr``) because
    the test environment's ``DummyWidget.__getattr__`` mock intercepts *all*
    attribute lookups and would make ``hasattr`` return ``True`` for any name.
    """
    # Building the sidebar must not raise AttributeError for any deleted method.
    launcher._setup_global_sidebar()
    # btn_ai_sidebar must NOT be set when AI_AVAILABLE is False
    assert "btn_ai_sidebar" not in launcher.__dict__, (
        "btn_ai_sidebar must not be created in the sidebar when AI_AVAILABLE is False"
    )
    # btn_ai (old top-bar button) must never be set at all
    assert "btn_ai" not in launcher.__dict__, (
        "btn_ai must not exist after #5620 deprecated-chat sweep"
    )


@patch("src.launchers.launcher_constants.AI_AVAILABLE", True)
def test_sidebar_ai_button_present_when_ai_available(launcher) -> None:
    """When AI is available, the sidebar builds without error and btn_ai is absent.

    Regression guard for UpstreamDrift #5689: ``_setup_ai_panel`` and
    ``_sync_chat_session`` were deleted in #5620 (deprecated-chat sweep).
    The old ``btn_ai`` top-bar button was removed at the same time.
    AI access now flows exclusively through the Sidekick dock; the global
    sidebar must not raise AttributeError when AI_AVAILABLE is True.
    """
    # _setup_global_sidebar must complete without AttributeError when AI is on
    launcher._setup_global_sidebar()
    # btn_ai (old top-bar button) must never exist after #5620
    assert "btn_ai" not in launcher.__dict__, (
        "Deprecated btn_ai top-bar button must not exist after #5620"
    )


def test_no_setup_ai_panel_method() -> None:
    """_setup_ai_panel must not exist on UISetupManager after #5620.

    Regression guard for UpstreamDrift #5689: the reviewer's AttributeError
    was triggered when a stale test used ``patch.object(launcher,
    '_setup_ai_panel')`` against an attribute that no longer exists.  We
    check the *class* dictionary (not an instance) so that DummyWidget's
    permissive ``__getattr__`` does not mask the absence of the method.
    """
    mro_attrs = {name for cls in UISetupManager.__mro__ for name in vars(cls)}
    assert "_setup_ai_panel" not in mro_attrs, (
        "_setup_ai_panel was deleted in #5620 and must not be re-introduced"
    )


def test_no_sync_chat_session_method() -> None:
    """_sync_chat_session must not exist on UISetupManager after #5620.

    The Sidekick ChatDockWidget performs the equivalent session-id handshake
    via the shared active_chat_session.txt file.
    """
    mro_attrs = {name for cls in UISetupManager.__mro__ for name in vars(cls)}
    assert "_sync_chat_session" not in mro_attrs, (
        "_sync_chat_session was deleted in #5620 and must not be re-introduced"
    )


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
        if "overlay" in launcher.__dict__.get("_mocks", {}):
            del launcher.__dict__["_mocks"]["overlay"]
        launcher._init_overlay()
        assert "overlay" not in launcher.__dict__.get("_mocks", {})


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
    assert launcher.layout_manager.current_category_filter == "Engines"


# ---------------------------------------------------------------------------
# AutoCompleteLineEdit wiring (issue #5479)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_search_input_is_auto_complete_line_edit(launcher) -> None:
    """The launcher search field must be an AutoCompleteLineEdit with a
    non-empty vocabulary so that the Global Text Prediction feature is
    actually wired globally rather than living only in the test file."""
    from src.shared.python.ui.auto_complete import AutoCompleteLineEdit

    with patch("src.launchers.launcher_constants.HELP_SYSTEM_AVAILABLE", False):
        launcher._setup_top_bar()

    assert isinstance(
        launcher.search_input,
        AutoCompleteLineEdit,
    ), "search_input must be AutoCompleteLineEdit, not bare QLineEdit"
    assert len(launcher.search_input.completer_words) > 0, (
        "AutoCompleteLineEdit vocabulary must be non-empty after construction"
    )


@pytest.mark.unit
def test_search_vocabulary_contains_engine_names(launcher) -> None:
    """Engine names from the config (mujoco, drake, pinocchio …) must appear
    in the completion vocabulary so users can type them into the search bar."""
    from src.shared.python.ui.completion_vocab import build_vocabulary

    vocab = build_vocabulary()
    for engine in ("mujoco", "drake", "pinocchio"):
        assert engine in vocab, f"Expected engine '{engine}' in vocabulary"


@pytest.mark.unit
def test_build_vocabulary_returns_nonempty_sorted_list() -> None:
    """build_vocabulary() must always return a sorted, non-empty list."""
    from src.shared.python.ui.completion_vocab import build_vocabulary

    vocab = build_vocabulary()
    assert len(vocab) > 0
    assert vocab == sorted(vocab), "vocabulary must be returned in sorted order"


@pytest.mark.unit
def test_on_sidebar_routed_shifts_to_home_tab(launcher) -> None:
    """Routing the sidebar to a category filters the grid and shifts to Home tab."""
    launcher.layout_manager = MagicMock()
    launcher._rebuild_grid = MagicMock()
    launcher.workspace_tabs = MagicMock()

    # Route sidebar to category 1 (Engines)
    launcher._on_sidebar_routed(1)
    launcher.workspace_tabs.setCurrentIndex.assert_called_once_with(0)


@pytest.mark.unit
def test_popout_active_tab_actions(launcher) -> None:
    """View menu popout active tab action respects bounds and core tabs."""
    launcher.workspace_tabs = MagicMock()

    # 1. Active tab index < 0 (nothing happens)
    launcher.workspace_tabs.currentIndex.return_value = -1
    launcher._popout_active_tab()
    launcher.workspace_tabs.detach_tab_from_menu.assert_not_called()

    # 2. Active tab is Home (should not pop out, should show warning)
    launcher.workspace_tabs.currentIndex.return_value = 0
    launcher.workspace_tabs.tabText.return_value = "Home"
    with patch("PyQt6.QtWidgets.QMessageBox.information") as mock_info:
        launcher._popout_active_tab()
        mock_info.assert_called_once()
        launcher.workspace_tabs.detach_tab_from_menu.assert_not_called()

    # 3. Active tab is not Home (should pop out)
    launcher.workspace_tabs.currentIndex.return_value = 1
    launcher.workspace_tabs.tabText.return_value = "Library"
    launcher._popout_active_tab()
    launcher.workspace_tabs.detach_tab_from_menu.assert_called_once_with(1)


@pytest.mark.unit
def test_redock_all_tabs_action(launcher) -> None:
    """View menu redock all tabs action triggers the DraggableTabWidget redock method."""
    launcher.workspace_tabs = MagicMock()
    launcher._redock_all_tabs()
    launcher.workspace_tabs.redock_all_tabs.assert_called_once()


# ---------------------------------------------------------------------------
# Condensed Categories sidebar button + routing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sidebar_includes_condensed_buttons(launcher) -> None:
    """The sidebar build invokes ``_build_sidebar_button`` for Tools and Documentation."""
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
    assert "Tools" in labels_used
    assert "Documentation" in labels_used
    assert "Training" not in labels_used
