"""Tests for launcher_theme mixin."""

from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QMenu, QWidget

from src.launchers.launcher_theme import LauncherThemeMixin


class DummyLauncher(QWidget, LauncherThemeMixin):
    def __init__(self):
        super().__init__()
        self.model_cards = {}
        self.selected_model = None

    def select_model(self, model):
        pass

    def update_launch_button(self):
        pass


def test_apply_styles_success(qapp):
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


def test_apply_styles_fallback(qapp):
    launcher = DummyLauncher()

    with patch(
        "src.shared.python.theme.ThemeManager.instance",
        side_effect=ImportError("No theme"),
    ):
        launcher.apply_styles()

    style = launcher.styleSheet()
    assert "background-color: #1E1E1E" in style


def test_apply_theme_system(qapp):
    launcher = DummyLauncher()

    mock_manager = MagicMock()

    with (
        patch(
            "src.shared.python.theme.ThemeManager.instance", return_value=mock_manager
        ),
        patch(
            "src.shared.python.theme.apply_golf_suite_style", create=True
        ) as mock_apply,
    ):
        launcher._apply_theme_system()

        mock_manager.load_saved_theme.assert_called_once()
        mock_apply.assert_called_once()
        mock_manager.on_theme_changed.assert_called_once()
        assert launcher._theme_manager == mock_manager


@patch.object(DummyLauncher, "apply_styles")
def test_on_theme_changed(mock_apply, qapp):
    launcher = DummyLauncher()

    mock_card = MagicMock()
    launcher.model_cards = {"test": mock_card}
    launcher.selected_model = "test_model"
    launcher.select_model = MagicMock()

    with patch("src.shared.python.theme.ThemeManager.instance"):
        launcher._on_theme_changed({})

    mock_apply.assert_called_once()
    mock_card.refresh_theme.assert_called_once()
    launcher.select_model.assert_called_once_with("test_model")


def test_setup_theme_menu_and_plot(qapp):
    launcher = DummyLauncher()
    menu = QMenu()

    mock_manager = MagicMock()
    mock_manager.theme_name = "Dark"
    mock_manager.get_available_fleet_themes.return_value = ["Fleet1"]
    mock_manager.get_custom_theme_names.return_value = ["Custom1"]

    with (
        patch(
            "src.shared.python.theme.ThemeManager.instance", return_value=mock_manager
        ),
        patch("src.shared.python.theme.ThemePreset", create=True) as MockPreset,
        patch("matplotlib.pyplot.style.available", ["classic", "ggplot"]),
    ):
        MockPreset.DARK = "Dark"
        MockPreset.LIGHT = "Light"
        MockPreset.HIGH_CONTRAST = "HC"

        launcher._setup_theme_menu(menu)

    actions = menu.actions()
    assert len(actions) > 5  # Check that several actions were added

    # Check if we built _theme_actions
    assert hasattr(launcher, "_theme_actions")


def test_set_plot_theme(qapp):
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
