"""Tests for src.shared.python.plotting.config (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.plotting.config import (
    COLOR_CYCLE,
    COLORS,
    DEFAULT_CONFIG,
    ColorPalette,
    PlotConfig,
    resolve_figure,
)


class TestColors:
    def test_primary_key_present(self) -> None:
        assert "primary" in COLORS

    def test_background_key_present(self) -> None:
        assert "background" in COLORS

    def test_color_values_are_hex(self) -> None:
        for key, val in COLORS.items():
            assert val.startswith("#"), f"Color {key!r} = {val!r} is not a hex string"

    def test_color_cycle_nonempty(self) -> None:
        assert len(COLOR_CYCLE) > 0


class TestColorPalette:
    def test_config_default_construction(self) -> None:
        palette = ColorPalette()
        assert palette.primary == COLORS["primary"]

    def test_cycle_length_matches_color_cycle(self) -> None:
        palette = ColorPalette()
        assert len(palette.cycle) == len(COLOR_CYCLE)

    def test_get_color_index_zero(self) -> None:
        palette = ColorPalette()
        assert palette.get_color(0) == palette.cycle[0]

    def test_get_color_wraps_around(self) -> None:
        palette = ColorPalette()
        n = len(palette.cycle)
        assert palette.get_color(n) == palette.cycle[0]

    def test_config_frozen(self) -> None:
        palette = ColorPalette()
        with pytest.raises((AttributeError, TypeError)):
            palette.primary = "#000000"  # type: ignore[misc]

    def test_custom_primary(self) -> None:
        palette = ColorPalette(primary="#aabbcc")
        assert palette.primary == "#aabbcc"


class TestPlotConfig:
    def test_default_width(self) -> None:
        cfg = PlotConfig()
        assert cfg.width == pytest.approx(10.0)

    def test_default_height(self) -> None:
        cfg = PlotConfig()
        assert cfg.height == pytest.approx(6.0)

    def test_default_dpi(self) -> None:
        assert PlotConfig().dpi == 100

    def test_default_show_grid(self) -> None:
        assert PlotConfig().show_grid is True

    def test_colors_is_color_palette(self) -> None:
        assert isinstance(PlotConfig().colors, ColorPalette)

    def test_presentation_preset_larger(self) -> None:
        cfg = PlotConfig.presentation()
        assert cfg.width > PlotConfig().width

    def test_publication_preset_higher_dpi(self) -> None:
        cfg = PlotConfig.publication()
        assert cfg.dpi >= 300

    def test_dashboard_preset_smaller(self) -> None:
        cfg = PlotConfig.dashboard()
        assert cfg.width < PlotConfig().width

    def test_create_figure_returns_tuple(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        cfg = PlotConfig()
        fig, ax = cfg.create_figure()
        assert fig is not None
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_create_figure_nrows_ncols(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        cfg = PlotConfig()
        fig, axes = cfg.create_figure(nrows=2, ncols=2)
        assert axes.shape == (2, 2)
        import matplotlib.pyplot as plt

        plt.close("all")


class TestDefaultConfig:
    def test_default_config_is_plot_config(self) -> None:
        assert isinstance(DEFAULT_CONFIG, PlotConfig)


class TestResolveFigure:
    def test_none_ax_creates_figure(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        fig, ax, cfg = resolve_figure(None)
        assert fig is not None
        assert ax is not None
        assert cfg is DEFAULT_CONFIG
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_existing_ax_returned(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig0, ax0 = plt.subplots()
        fig_out, ax_out, cfg = resolve_figure(ax0)
        assert ax_out is ax0
        plt.close("all")

    def test_custom_config_used(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        custom = PlotConfig(width=5.0)
        fig, ax, cfg = resolve_figure(None, config=custom)
        assert cfg is custom
        import matplotlib.pyplot as plt

        plt.close("all")
