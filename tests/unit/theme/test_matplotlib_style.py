"""Tests for shared matplotlib theme styling helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_rgba

from src.shared.python.theme.colors import CHART_COLORS, Colors
from src.shared.python.theme.matplotlib_style import (
    GOLF_SUITE_STYLE,
    apply_golf_suite_style,
    create_styled_figure,
    get_chart_color,
    style_for_export,
)


def test_apply_golf_suite_style_updates_global_rc_params() -> None:
    with plt.rc_context():
        apply_golf_suite_style()

        assert plt.rcParams["figure.facecolor"] == Colors.BG_BASE
        assert plt.rcParams["axes.facecolor"] == Colors.BG_SURFACE
        assert plt.rcParams["axes.grid"] is True
        assert plt.rcParams["font.sans-serif"][0] == "Inter"


def test_apply_golf_suite_style_styles_existing_figure_axes() -> None:
    fig, ax = plt.subplots()
    try:
        apply_golf_suite_style(fig)

        assert fig.get_facecolor() == to_rgba(Colors.BG_BASE)
        assert ax.get_facecolor() == to_rgba(Colors.BG_SURFACE)
        assert ax.spines["top"].get_visible() is False
        assert ax.spines["right"].get_visible() is False
        assert ax.spines["bottom"].get_edgecolor() == to_rgba(Colors.BORDER_DEFAULT)
        assert ax.spines["left"].get_linewidth() == 1.0
    finally:
        plt.close(fig)


def test_get_chart_color_cycles_through_palette() -> None:
    assert get_chart_color(0) == CHART_COLORS[0]
    assert get_chart_color(len(CHART_COLORS)) == CHART_COLORS[0]
    assert get_chart_color(-1) == CHART_COLORS[-1]


def test_create_styled_figure_uses_default_figsize_and_style() -> None:
    with plt.rc_context():
        fig, ax = create_styled_figure()
        try:
            assert tuple(fig.get_size_inches()) == (10.0, 6.0)
            assert fig.get_facecolor() == to_rgba(Colors.BG_BASE)
            assert ax is not None
            assert plt.rcParams["axes.facecolor"] == GOLF_SUITE_STYLE["axes.facecolor"]
        finally:
            plt.close(fig)


def test_create_styled_figure_rejects_missing_row_count() -> None:
    with pytest.raises(ValueError, match="nrows"):
        create_styled_figure(nrows=None)  # type: ignore[arg-type]


def test_style_for_export_sets_dpi_and_tight_layout() -> None:
    fig, _ax = plt.subplots()
    try:
        style_for_export(fig, dpi=123)

        assert fig.dpi == 123
        assert fig.get_facecolor() == to_rgba(Colors.BG_BASE)
    finally:
        plt.close(fig)


def test_style_for_export_rejects_missing_figure() -> None:
    with pytest.raises(ValueError, match="fig"):
        style_for_export(None)  # type: ignore[arg-type]
