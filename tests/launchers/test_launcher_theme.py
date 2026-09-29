"""Tests for launcher_theme mixin."""

from typing import Any
from unittest.mock import MagicMock, patch  # noqa: E402

from PyQt6.QtWidgets import QMenu, QWidget  # noqa: E402
from src.launchers.launcher_theme import ThemeManager  # noqa: E402


class DummyLauncher(QWidget):
    def __getattr__(self, name: str) -> Any:
        if hasattr(self, "manager"):
            manager = self.manager
            if name in manager.__dict__ or hasattr(type(manager), name):
                attr = getattr(manager, name)
                import types

                if isinstance(attr, types.MethodType):
                    return types.MethodType(attr.__func__, self)
                return attr
        raise AttributeError(name)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manager = ThemeManager(self)

        self.model_cards = {}
        self.selected_model = None

    def select_model(self, model) -> None:
        pass

    def update_launch_button(self) -> None:
        pass

    def apply_styles(self) -> None:
        self.manager.apply_styles()

    def _apply_theme_system(self) -> None:
        self.manager._apply_theme_system()

    def _on_theme_changed(self, colors: object = None) -> None:
        self.manager._on_theme_changed(colors)

    def _setup_theme_menu(self, theme_menu) -> None:
        self.manager._setup_theme_menu(theme_menu)

    def _set_plot_theme(self, theme_name: str) -> None:
        self.manager._set_plot_theme(theme_name)

    def _open_theme_manager_dialog(self) -> None:
        self.manager._open_theme_manager_dialog()


def test_apply_styles_success(qapp) -> None:
    launcher = DummyLauncher()

    mock_manager = MagicMock()
    mock_manager.get_current_colors.return_value = {
        "bg_elevated": "#111",
        "border_default": "#222",
        "bg_highlight": "#333",
        "border_strong": "#444",
        "text_secondary": "#555",
    }
    mock_manager.get_current_stylesheet.return_value = "QWidget { color: red; }"

    with patch(
        "src.shared.python.theme.ThemeManager.instance", return_value=mock_manager
    ):
        launcher.apply_styles()

    style = launcher.styleSheet()
    assert "QWidget { color: red; }" in style
    assert "background-color: #111" in style


def test_apply_styles_fallback(qapp) -> None:
    launcher = DummyLauncher()

    with patch(
        "src.shared.python.theme.ThemeManager.instance",
        side_effect=ImportError("No theme"),
    ):
        launcher.apply_styles()

    style = launcher.styleSheet()
    assert "background-color: #1E1E1E" in style
    assert "Outfit" in style


def test_apply_theme_system(qapp) -> None:
    launcher = DummyLauncher()

    mock_manager = MagicMock()
    # Expose themeChanged as a mock Qt signal with .connect()
    mock_manager.themeChanged = MagicMock()

    with (
        patch(
            "src.shared.python.theme.ThemeManager.instance", return_value=mock_manager
        ),
        patch(
            "src.shared.python.theme.apply_golf_suite_style", create=True
        ) as mock_apply,
    ):
        launcher._apply_theme_system()

        mock_apply.assert_called_once()
        mock_manager.themeChanged.connect.assert_called_once()
        assert launcher._theme_manager == mock_manager


@patch("src.launchers.launcher_theme.ThemeManager.apply_styles")
def test_on_theme_changed(mock_apply, qapp) -> None:
    launcher = DummyLauncher()

    mock_card = MagicMock()
    mock_card.refresh_theme = MagicMock()

    # Second card without refresh_theme
    mock_card_no_refresh = MagicMock()
    del mock_card_no_refresh.refresh_theme

    launcher.model_cards = {"test": mock_card, "missing": mock_card_no_refresh}
    launcher.selected_model = "test_model"
    launcher.select_model = MagicMock()

    # Test setting checked state of actions
    action_1 = MagicMock()
    action_1.text.return_value = "Dark"
    action_2 = MagicMock()
    action_2.text.return_value = "Light"
    launcher._theme_actions = [action_1, action_2]

    with patch("src.shared.python.theme.ThemeManager.instance") as mock_instance:
        mock_instance().get_current_theme_name.return_value = "Dark"
        launcher._on_theme_changed({})

    mock_apply.assert_called_once()
    mock_card.refresh_theme.assert_called_once()
    launcher.select_model.assert_called_once_with("test_model")

    action_1.setChecked.assert_called_with(True)
    action_2.setChecked.assert_called_with(False)

    # Test no selected_model
    launcher.update_launch_button = MagicMock()
    launcher.selected_model = None
    with patch("src.shared.python.theme.ThemeManager.instance") as mock_instance2:
        mock_instance2().get_current_theme_name.return_value = "Dark"
        launcher._on_theme_changed({})

    launcher.update_launch_button.assert_called_once()


def test_setup_theme_menu_and_plot(qapp) -> None:
    launcher = DummyLauncher()
    menu = QMenu()

    mock_manager = MagicMock()
    mock_manager.get_current_theme_name.return_value = "Dark"
    # get_available_themes returns all built-in themes (including extras beyond presets)
    mock_manager.get_available_themes.return_value = [
        "Dark",
        "Light",
        "High Contrast",
        "Monokai",
    ]
    mock_manager.get_custom_theme_names.return_value = ["Custom1"]

    with (
        patch(
            "src.shared.python.theme.ThemeManager.instance", return_value=mock_manager
        ),
        patch("matplotlib.pyplot.style.available", ["_classic", "classic", "ggplot"]),
    ):
        launcher._setup_theme_menu(menu)

    actions = menu.actions()
    assert len(actions) > 5  # Check that several actions were added

    # Check if we built _theme_actions
    assert hasattr(launcher, "_theme_actions")
    names = [a.text() for a in launcher._theme_actions]
    # "Monokai" is an extra theme beyond the core 3 presets
    assert "Monokai" in names
    assert "Custom1" in names

    # Test import error for ThemeManager
    with patch(
        "src.shared.python.theme.ThemeManager.instance", side_effect=ImportError("Boom")
    ):
        menu2 = QMenu()
        launcher._setup_theme_menu(menu2)
        assert len(menu2.actions()) == 1
        assert menu2.actions()[0].text() == "(Theme system unavailable)"


def test_setup_theme_menu_empty_lists(qapp) -> None:
    launcher = DummyLauncher()
    menu = QMenu()

    with patch("src.shared.python.theme.ThemeManager.instance") as mock_inst:
        mock_manager = mock_inst.return_value
        mock_manager.get_current_theme_name.return_value = "Dark"
        mock_manager.get_available_themes.return_value = []
        mock_manager.get_custom_theme_names.return_value = []

        with patch.object(menu, "addMenu", return_value=None):
            launcher._setup_theme_menu(menu)

    # test on_theme_changed without _theme_actions
    if hasattr(launcher.manager, "_theme_actions"):
        del launcher.manager._theme_actions
    launcher.update_launch_button = MagicMock()
    launcher._on_theme_changed({})


def test_set_plot_theme(qapp) -> None:
    launcher = DummyLauncher()

    with (
        patch("PyQt6.QtCore.QSettings") as mock_settings_class,
        patch(
            "src.shared.python.theme.apply_golf_suite_style", create=True
        ) as mock_apply,
    ):
        mock_settings = MagicMock()
        mock_settings_class.return_value = mock_settings

        launcher._set_plot_theme("follow_ui")

        mock_settings.setValue.assert_called_once_with("plot_theme", "follow_ui")
        mock_apply.assert_called_once()

        # Handle import error gracefully
        mock_apply.side_effect = ImportError("Boom")
        launcher._set_plot_theme("follow_ui")  # Should not crash

    with patch("PyQt6.QtCore.QSettings") as mock_settings_class:
        mock_settings = MagicMock()
        mock_settings_class.return_value = mock_settings
        with patch("matplotlib.pyplot.style.use") as mock_use:
            launcher._set_plot_theme("ggplot")
            mock_use.assert_called_once_with("ggplot")

        # Exception case
        with patch("matplotlib.pyplot.style.use", side_effect=ImportError("Boom")):
            launcher._set_plot_theme("ggplot")  # Shouldn't crash


@patch("src.shared.python.theme.dialogs.ThemeManagerDialog", create=True)
def test_open_theme_manager_dialog(mock_dialog_class, qapp) -> None:
    launcher = DummyLauncher()

    mock_dialog = MagicMock()
    mock_dialog_class.return_value = mock_dialog

    with patch("src.shared.python.theme.ThemeManager.instance"):
        launcher._open_theme_manager_dialog()
        mock_dialog.exec.assert_called_once()

    with patch(
        "src.shared.python.theme.ThemeManager.instance", side_effect=ImportError("Boom")
    ):
        launcher._open_theme_manager_dialog()  # Shouldn't crash
