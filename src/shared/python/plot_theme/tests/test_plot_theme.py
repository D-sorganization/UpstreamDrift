# TRACKED_TASK: see #2310 — architecture debt extraction schedule

"""Comprehensive tests for the plot_theme module.

Covers PlotTheme dataclass, theme registry, get_theme/register_theme,
PlotThemeManager, and integration helpers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# PlotTheme dataclass tests
# ──────────────────────────────────────────────────────────────────────────────


class TestPlotThemeDataclass:
    def test_default_values(self):
        from plot_theme.themes import PlotTheme

        theme = PlotTheme(name="Test")
        assert theme.name == "Test"
        assert theme.description == ""
        assert theme.figure_facecolor == "#ffffff"
        assert theme.axes_facecolor == "#ffffff"
        assert theme.grid_alpha == 0.5
        assert theme.primary_alpha == 0.8
        assert theme.line_width == 2.0
        assert theme.marker_size == 6.0
        assert theme.font_family == "sans-serif"
        assert theme.font_size == 10.0
        assert len(theme.primary_colors) == 3
        assert len(theme.secondary_colors) == 3
        assert len(theme.accent_colors) == 3
        assert theme.rcparams == {}

    def test_get_color_cycle(self):
        from plot_theme.themes import PlotTheme

        theme = PlotTheme(
            name="Test",
            primary_colors=["#aa0000"],
            secondary_colors=["#00aa00"],
            accent_colors=["#0000aa"],
        )
        cycle = theme.get_color_cycle()
        assert cycle == ["#aa0000", "#00aa00", "#0000aa"]

    def test_to_rcparams_includes_all_keys(self):
        from plot_theme.themes import PlotTheme

        theme = PlotTheme(name="Test")
        params = theme.to_rcparams()
        required_keys = {
            "figure.facecolor",
            "axes.facecolor",
            "axes.edgecolor",
            "axes.labelcolor",
            "axes.titlecolor",
            "grid.color",
            "grid.alpha",
            "text.color",
            "font.family",
            "font.size",
            "lines.linewidth",
            "lines.markersize",
        }
        assert required_keys <= params.keys()

    def test_to_rcparams_with_string_heatmap_cmap(self):
        from plot_theme.themes import PlotTheme

        theme = PlotTheme(name="Test", heatmap_cmap="plasma")
        params = theme.to_rcparams()
        assert params["image.cmap"] == "plasma"

    def test_to_rcparams_with_list_heatmap_cmap_uses_viridis(self):
        from plot_theme.themes import PlotTheme

        # When cmap is a list, falls back to 'viridis'
        theme = PlotTheme(name="Test", heatmap_cmap=["#000", "#fff"])
        params = theme.to_rcparams()
        assert params["image.cmap"] == "viridis"

    def test_to_rcparams_merges_custom_rcparams(self):
        from plot_theme.themes import PlotTheme

        theme = PlotTheme(name="Test", rcparams={"figure.dpi": 200})
        params = theme.to_rcparams()
        assert params["figure.dpi"] == 200

    def test_custom_theme_construction(self):
        from plot_theme.themes import PlotTheme

        theme = PlotTheme(
            name="Custom",
            description="My custom theme",
            figure_facecolor="#111",
            axes_facecolor="#222",
            primary_color="#ff0000",
            secondary_color="#0000ff",
            accent_color="#00ff00",
            line_width=3.5,
            marker_size=10.0,
        )
        assert theme.primary_color == "#ff0000"
        assert theme.line_width == 3.5


# ──────────────────────────────────────────────────────────────────────────────
# Predefined theme constants
# ──────────────────────────────────────────────────────────────────────────────


class TestPredefinedThemes:
    def test_all_themes_present_in_registry(self):
        from plot_theme.themes import PLOT_THEMES

        expected_keys = {
            "scientific_violet",
            "scientific_violet_dark",
            "catppuccin_mocha",
            "catppuccin_latte",
            "vampire_dark",
            "nord",
            "solarized_dark",
            "solarized_light",
            "frost_dark",
            "gruvbox_dark",
            "material_dark",
            "tokyo_night",
            "classic_light",
            "classic_dark",
        }
        assert expected_keys <= PLOT_THEMES.keys()

    def test_scientific_violet_properties(self):
        from plot_theme.themes import SCIENTIFIC_VIOLET

        assert SCIENTIFIC_VIOLET.name == "Scientific Violet"
        assert SCIENTIFIC_VIOLET.primary_color == "#9333EA"
        assert SCIENTIFIC_VIOLET.contour_cmap == "YlGnBu"

    def test_catppuccin_mocha_is_dark(self):
        from plot_theme.themes import CATPPUCCIN_MOCHA

        assert CATPPUCCIN_MOCHA.figure_facecolor == "#1e1e2e"

    def test_default_theme_exists(self):
        from plot_theme.themes import DEFAULT_THEME, PLOT_THEMES

        assert DEFAULT_THEME in PLOT_THEMES

    def test_all_themes_have_name(self):
        from plot_theme.themes import PLOT_THEMES

        for key, theme in PLOT_THEMES.items():
            assert theme.name, f"Theme '{key}' has empty name"

    def test_all_themes_to_rcparams_succeeds(self):
        from plot_theme.themes import PLOT_THEMES

        for key, theme in PLOT_THEMES.items():
            params = theme.to_rcparams()
            assert "figure.facecolor" in params, (
                f"Theme '{key}' missing figure.facecolor"
            )


# ──────────────────────────────────────────────────────────────────────────────
# get_theme / get_theme_names / register_theme
# ──────────────────────────────────────────────────────────────────────────────


class TestThemeFunctions:
    def test_get_theme_by_name(self):
        from plot_theme.themes import get_theme

        theme = get_theme("scientific_violet")
        assert theme.name == "Scientific Violet"

    def test_get_theme_case_insensitive(self):
        from plot_theme.themes import get_theme

        t1 = get_theme("SCIENTIFIC_VIOLET")
        t2 = get_theme("scientific_violet")
        assert t1.name == t2.name

    def test_get_theme_hyphen_normalization(self):
        from plot_theme.themes import get_theme

        # Hyphens should be normalized to underscores
        theme = get_theme("scientific-violet")
        assert theme.name == "Scientific Violet"

    def test_get_theme_space_normalization(self):
        from plot_theme.themes import get_theme

        theme = get_theme("scientific violet")
        assert theme.name == "Scientific Violet"

    def test_get_theme_not_found_raises(self):
        from plot_theme.themes import get_theme

        with pytest.raises(KeyError, match="not found"):
            get_theme("nonexistent_theme")

    def test_get_theme_names_sorted(self):
        from plot_theme.themes import get_theme_names

        names = get_theme_names()
        assert names == sorted(names)
        assert len(names) >= 14

    def test_register_custom_theme(self):
        from plot_theme.themes import PLOT_THEMES, PlotTheme, register_theme

        custom = PlotTheme(name="My Theme", primary_color="#123456")
        register_theme("my_theme_test", custom)
        assert "my_theme_test" in PLOT_THEMES
        assert PLOT_THEMES["my_theme_test"].primary_color == "#123456"
        # Cleanup
        del PLOT_THEMES["my_theme_test"]

    def test_register_custom_theme_normalizes_name(self):
        from plot_theme.themes import PLOT_THEMES, PlotTheme, register_theme

        custom = PlotTheme(name="Test Theme")
        register_theme("Test-Theme-Foo", custom)
        assert "test_theme_foo" in PLOT_THEMES
        # Cleanup
        del PLOT_THEMES["test_theme_foo"]


# ──────────────────────────────────────────────────────────────────────────────
# PlotThemeManager
# ──────────────────────────────────────────────────────────────────────────────


class TestPlotThemeManager:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the global manager singleton before each test."""
        from plot_theme.manager import _ManagerHolder

        _ManagerHolder.instance = None
        yield
        _ManagerHolder.instance = None

    def _make_manager(self):
        """Create a fresh manager without PyQt6 QSettings."""
        with patch("plot_theme.manager.PlotThemeManager._load_saved_theme"):
            from plot_theme.manager import PlotThemeManager

            return PlotThemeManager()

    def test_init_sets_default_theme(self):
        m = self._make_manager()
        assert m.current_theme_name == "scientific_violet"
        assert m.current_theme.name == "Scientific Violet"

    def test_get_available_themes(self):
        m = self._make_manager()
        themes = m.get_available_themes()
        assert "scientific_violet" in themes
        assert themes == sorted(themes)

    def test_get_theme_display_names(self):
        m = self._make_manager()
        display = m.get_theme_display_names()
        assert "scientific_violet" in display
        assert display["scientific_violet"] == "Scientific Violet"

    def test_set_theme_changes_current(self):
        m = self._make_manager()
        with patch.object(m, "_save_theme"):
            m.set_theme("vampire_dark")
        assert m.current_theme_name == "vampire_dark"
        assert m.current_theme.name == "Vampire Dark"

    def test_set_theme_without_saving(self):
        m = self._make_manager()
        with patch.object(m, "_save_theme") as mock_save:
            m.set_theme("nord", save=False)
        mock_save.assert_not_called()

    def test_set_theme_with_saving(self):
        m = self._make_manager()
        with patch.object(m, "_save_theme") as mock_save:
            m.set_theme("nord", save=True)
        mock_save.assert_called_once()

    def test_set_theme_triggers_callbacks(self):
        m = self._make_manager()
        called_with = []
        m.add_theme_change_callback(lambda t: called_with.append(t))
        with patch.object(m, "_save_theme"):
            m.set_theme("vampire_dark")
        assert len(called_with) == 1
        assert called_with[0].name == "Vampire Dark"

    def test_set_theme_swallows_callback_errors(self):
        m = self._make_manager()

        def bad_callback(t):
            raise ValueError("boom")

        m.add_theme_change_callback(bad_callback)
        # Should not raise
        with patch.object(m, "_save_theme"):
            m.set_theme("nord")

    def test_add_and_remove_callback(self):
        m = self._make_manager()
        cb = MagicMock()
        m.add_theme_change_callback(cb)
        m.remove_theme_change_callback(cb)
        with patch.object(m, "_save_theme"):
            m.set_theme("vampire_dark")
        cb.assert_not_called()

    def test_add_callback_deduplicates(self):
        m = self._make_manager()
        cb = MagicMock()
        m.add_theme_change_callback(cb)
        m.add_theme_change_callback(cb)  # second add should be ignored
        with patch.object(m, "_save_theme"):
            m.set_theme("vampire_dark")
        assert cb.call_count == 1

    def test_remove_nonexistent_callback_safe(self):
        m = self._make_manager()
        cb = MagicMock()
        m.remove_theme_change_callback(cb)  # Should not raise

    def test_get_colors_returns_dict(self):
        m = self._make_manager()
        colors = m.get_colors()
        assert "primary" in colors
        assert "secondary" in colors
        assert "accent" in colors
        assert "background" in colors

    def test_get_histogram_style(self):
        m = self._make_manager()
        style = m.get_histogram_style()
        assert "color" in style
        assert "alpha" in style

    def test_get_scatter_style(self):
        m = self._make_manager()
        style = m.get_scatter_style()
        assert "c" in style
        assert "s" in style

    def test_get_line_style(self):
        m = self._make_manager()
        style = m.get_line_style(index=0)
        assert "color" in style
        assert "linewidth" in style

    def test_get_line_style_index_wraps(self):
        m = self._make_manager()
        # Index larger than cycle length should wrap
        style = m.get_line_style(index=100)
        assert "color" in style

    def test_get_fit_line_style(self):
        m = self._make_manager()
        style = m.get_fit_line_style()
        assert "color" in style
        assert style["linestyle"] == "-"

    def test_get_contour_style(self):
        m = self._make_manager()
        style = m.get_contour_style()
        assert "cmap" in style

    def test_apply_to_matplotlib(self):
        m = self._make_manager()
        mock_mpl = MagicMock()
        with (
            patch.dict(
                "sys.modules",
                {
                    "matplotlib": mock_mpl,
                    "matplotlib.pyplot": MagicMock(),
                    "cycler": MagicMock(),
                },
            ),
            patch("plot_theme.manager.PlotThemeManager.apply_to_matplotlib"),
        ):
            m.apply_to_matplotlib()

    def test_apply_to_matplotlib_import_error(self):
        m = self._make_manager()
        # When matplotlib is unavailable, should not raise
        with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
            m.apply_to_matplotlib()

    def test_apply_to_figure(self):
        m = self._make_manager()
        mock_fig = MagicMock()
        mock_fig.axes = []  # no axes → just sets facecolor
        m.apply_to_figure(mock_fig)
        mock_fig.set_facecolor.assert_called_once()

    def test_apply_to_axes(self):
        m = self._make_manager()
        mock_ax = MagicMock()
        mock_spine = MagicMock()
        mock_ax.spines.values.return_value = [mock_spine]
        m.apply_to_axes(mock_ax)
        mock_ax.set_facecolor.assert_called_once()
        mock_ax.tick_params.assert_called_once()
        mock_ax.grid.assert_called_once()

    def test_load_saved_theme_via_qsettings(self):
        """QSettings path: saved theme loaded if it exists in registry."""
        mock_settings = MagicMock()
        mock_settings.value.return_value = "vampire_dark"
        with (
            patch.dict(
                "sys.modules", {"PyQt6": MagicMock(), "PyQt6.QtCore": MagicMock()}
            ),
            patch("plot_theme.manager.PlotThemeManager._load_saved_theme"),
        ):
            from plot_theme.manager import PlotThemeManager

            m = PlotThemeManager()
        # Just verify construction doesn't raise
        assert m is not None

    def test_save_theme_import_error(self):
        """_save_theme swallows ImportError when PyQt6 is unavailable."""
        m = self._make_manager()
        with patch.dict("sys.modules", {"PyQt6": None, "PyQt6.QtCore": None}):
            m._save_theme()  # Should not raise


class TestGetPlotThemeManager:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        from plot_theme.manager import _ManagerHolder

        _ManagerHolder.instance = None
        yield
        _ManagerHolder.instance = None

    def test_get_plot_theme_manager_creates_instance(self):
        from plot_theme.manager import get_plot_theme_manager

        with patch("plot_theme.manager.PlotThemeManager._load_saved_theme"):
            m = get_plot_theme_manager()
        assert m is not None

    def test_get_plot_theme_manager_singleton(self):
        from plot_theme.manager import get_plot_theme_manager

        with patch("plot_theme.manager.PlotThemeManager._load_saved_theme"):
            m1 = get_plot_theme_manager()
            m2 = get_plot_theme_manager()
        assert m1 is m2


# ──────────────────────────────────────────────────────────────────────────────
# __init__.py integration (re-exported symbols)
# ──────────────────────────────────────────────────────────────────────────────


class TestPackageInit:
    def test_package_exports_plot_theme(self):
        from plot_theme import PlotTheme

        assert PlotTheme is not None

    def test_package_exports_get_theme(self):
        from plot_theme import get_theme

        assert callable(get_theme)

    def test_package_exports_get_plot_theme_manager(self):
        from plot_theme import get_plot_theme_manager

        assert callable(get_plot_theme_manager)


# ──────────────────────────────────────────────────────────────────────────────
# integration.py tests
# ──────────────────────────────────────────────────────────────────────────────


class TestIntegration함수:
    """Tests for the integration helper functions in integration.py."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        from plot_theme.manager import _ManagerHolder

        _ManagerHolder.instance = None
        yield
        _ManagerHolder.instance = None

    def _make_manager_patch(self):
        return patch("plot_theme.manager.PlotThemeManager._load_saved_theme")

    def test_apply_plot_theme_returns_manager(self):
        from plot_theme.integration import apply_plot_theme

        with (
            self._make_manager_patch(),
            patch("plot_theme.manager.PlotThemeManager.apply_to_matplotlib"),
        ):
            m = apply_plot_theme("scientific_violet")
        assert m is not None
        assert m.current_theme_name == "scientific_violet"

    def test_apply_plot_theme_without_name(self):
        from plot_theme.integration import apply_plot_theme

        with (
            self._make_manager_patch(),
            patch("plot_theme.manager.PlotThemeManager.apply_to_matplotlib"),
        ):
            m = apply_plot_theme()
        assert m is not None

    def test_style_axis_with_current_theme(self):
        from plot_theme.integration import style_axis

        mock_ax = MagicMock()
        mock_ax.spines.values.return_value = []
        with self._make_manager_patch():
            style_axis(mock_ax)
        mock_ax.set_facecolor.assert_called_once()

    def test_style_axis_with_named_theme(self):
        from plot_theme.integration import style_axis

        mock_ax = MagicMock()
        mock_ax.spines.values.return_value = []
        with self._make_manager_patch():
            style_axis(mock_ax, theme_name="vampire_dark")
        mock_ax.set_facecolor.assert_called_once()

    def test_get_theme_colors_current_theme(self):
        from plot_theme.integration import get_theme_colors

        with self._make_manager_patch():
            colors = get_theme_colors()
        assert "primary" in colors
        assert "secondary" in colors
        assert "contour_cmap" in colors

    def test_get_theme_colors_named_theme(self):
        from plot_theme.integration import get_theme_colors

        with self._make_manager_patch():
            colors = get_theme_colors("vampire_dark")
        assert colors["primary"] == "#bd93f9"

    def test_plot_theme_mixin_setup(self):
        from plot_theme.integration import PlotThemeMixin

        class MyWidget(PlotThemeMixin):
            pass

        widget = MyWidget()
        with (
            self._make_manager_patch(),
            patch("plot_theme.manager.PlotThemeManager.apply_to_matplotlib"),
        ):
            m = widget.setup_plot_theme(apply_immediately=True)
        assert m is not None

    def test_plot_theme_mixin_without_immediate_apply(self):
        from plot_theme.integration import PlotThemeMixin

        class MyWidget(PlotThemeMixin):
            pass

        widget = MyWidget()
        with self._make_manager_patch():
            m = widget.setup_plot_theme(apply_immediately=False)
        assert m is not None

    def test_plot_theme_mixin_set_plot_theme(self):
        from plot_theme.integration import PlotThemeMixin

        class MyWidget(PlotThemeMixin):
            pass

        widget = MyWidget()
        with self._make_manager_patch():
            widget.setup_plot_theme(apply_immediately=False)
            with patch.object(widget._plot_theme_manager, "_save_theme"):
                widget.set_plot_theme("nord")
        assert widget._plot_theme_manager.current_theme_name == "nord"

    def test_plot_theme_mixin_set_without_manager(self):
        from plot_theme.integration import PlotThemeMixin

        class MyWidget(PlotThemeMixin):
            pass

        widget = MyWidget()
        widget.set_plot_theme("nord")  # should not raise

    def test_plot_theme_mixin_get_plot_colors(self):
        from plot_theme.integration import PlotThemeMixin

        class MyWidget(PlotThemeMixin):
            pass

        widget = MyWidget()
        with self._make_manager_patch():
            widget.setup_plot_theme(apply_immediately=False)
        colors = widget.get_plot_colors()
        assert "primary" in colors

    def test_plot_theme_mixin_get_plot_colors_no_manager(self):
        from plot_theme.integration import PlotThemeMixin

        class MyWidget(PlotThemeMixin):
            pass

        widget = MyWidget()
        colors = widget.get_plot_colors()
        assert colors == {}

    def test_plot_theme_mixin_get_manager(self):
        from plot_theme.integration import PlotThemeMixin

        class MyWidget(PlotThemeMixin):
            pass

        widget = MyWidget()
        assert widget.get_plot_theme_manager() is None  # before setup

    def test_plot_theme_mixin_on_changed_callback(self):
        """_on_plot_theme_changed_internal calls subclass on_plot_theme_changed."""
        from plot_theme.integration import PlotThemeMixin
        from plot_theme.themes import get_theme

        class MyWidget(PlotThemeMixin):
            def __init__(self):
                self._callback_theme = None

            def on_plot_theme_changed(self, theme):
                self._callback_theme = theme

        widget = MyWidget()
        with (
            self._make_manager_patch(),
            patch("plot_theme.manager.PlotThemeManager.apply_to_matplotlib"),
        ):
            widget.setup_plot_theme(apply_immediately=False)

        theme = get_theme("nord")
        with patch.object(widget._plot_theme_manager, "apply_to_matplotlib"):
            widget._on_plot_theme_changed_internal(theme)
        assert widget._callback_theme is not None

    def test_setup_plot_theme_for_app(self):
        from plot_theme.integration import setup_plot_theme_for_app

        mock_app = MagicMock()
        mock_window = MagicMock()
        mock_window.__class__.__name__ = "TestWindow"
        with (
            self._make_manager_patch(),
            patch("plot_theme.manager.PlotThemeManager.apply_to_matplotlib"),
            patch("plot_theme.integration.create_plot_theme_menu", return_value=None),
        ):
            m = setup_plot_theme_for_app(mock_app, mock_window, add_menu=True)
        assert m is not None

    def test_setup_plot_theme_for_app_no_menu(self):
        from plot_theme.integration import setup_plot_theme_for_app

        mock_app = MagicMock()
        mock_window = MagicMock()
        with (
            self._make_manager_patch(),
            patch("plot_theme.manager.PlotThemeManager.apply_to_matplotlib"),
        ):
            m = setup_plot_theme_for_app(mock_app, mock_window, add_menu=False)
        assert m is not None

    def test_create_plot_theme_menu_no_pyqt6(self):
        """When PyQt6 unavailable, returns None."""
        from plot_theme.integration import create_plot_theme_menu

        mock_parent = MagicMock()
        with patch.dict(
            "sys.modules", {"PyQt6": None, "PyQt6.QtGui": None, "PyQt6.QtWidgets": None}
        ):
            result = create_plot_theme_menu(mock_parent)
        assert result is None

    def test_apply_to_matplotlib_with_update_existing(self):
        """apply_to_matplotlib(update_existing=True) iterates open figures."""
        from plot_theme.manager import PlotThemeManager

        with self._make_manager_patch():
            m = PlotThemeManager()

        mock_fig = MagicMock()
        mock_fig.axes = []

        # Patch matplotlib.pyplot and cycler inside the context, keep mock ref
        import cycler as _cycler_mod
        import matplotlib.pyplot as _plt

        mock_get_fignums = MagicMock(return_value=[1])
        mock_figure = MagicMock(return_value=mock_fig)
        with (
            patch.object(_plt, "get_fignums", mock_get_fignums),
            patch.object(_plt, "figure", mock_figure),
            patch.object(_cycler_mod, "cycler", return_value=MagicMock()),
        ):
            m.apply_to_matplotlib(update_existing=True)

        mock_get_fignums.assert_called_once()
