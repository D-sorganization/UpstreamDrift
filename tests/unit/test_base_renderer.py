"""Tests for the shared base renderer module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.shared.python.plotting.renderers.base import (
    DEFAULT_GRID_ALPHA,
    DEFAULT_GRID_STYLE,
    DEFAULT_LABEL_FONTSIZE,
    DEFAULT_LABEL_FONTWEIGHT,
    DEFAULT_LEGEND_ALPHA,
    DEFAULT_TITLE_FONTSIZE,
    BaseRenderer,
)


@pytest.fixture()
def mock_data_manager() -> MagicMock:
    """Create a mock DataManager."""
    return MagicMock()


@pytest.fixture()
def renderer(mock_data_manager: MagicMock) -> BaseRenderer:
    """Create a BaseRenderer with mock data manager."""
    return BaseRenderer(mock_data_manager)


class TestBaseRendererInit:
    """Test BaseRenderer initialization."""

    def test_stores_data_manager(
        self, renderer: BaseRenderer, mock_data_manager: MagicMock
    ) -> None:
        """Test that data manager is stored."""
        assert renderer.data is mock_data_manager

    def test_loads_colors(self, renderer: BaseRenderer) -> None:
        """Test that colors are loaded from config."""
        assert renderer.colors is not None


class TestBaseRendererClearFigure:
    """Test figure clearing."""

    def test_clears_figure(self, renderer: BaseRenderer) -> None:
        """Test that clear_figure delegates to figure.clear()."""
        mock_fig = MagicMock()
        renderer.clear_figure(mock_fig)
        mock_fig.clear.assert_called_once()


class TestFormatAxis:
    """Test the format_axis static method."""

    def test_sets_xlabel(self) -> None:
        """Test that xlabel is set with default formatting."""
        ax = MagicMock()
        ax.get_legend_handles_labels.return_value = ([], [])
        BaseRenderer.format_axis(ax, xlabel="Time (s)")
        ax.set_xlabel.assert_called_once_with(
            "Time (s)",
            fontsize=DEFAULT_LABEL_FONTSIZE,
            fontweight=DEFAULT_LABEL_FONTWEIGHT,
        )

    def test_sets_ylabel(self) -> None:
        """Test that ylabel is set with default formatting."""
        ax = MagicMock()
        ax.get_legend_handles_labels.return_value = ([], [])
        BaseRenderer.format_axis(ax, ylabel="Angle (deg)")
        ax.set_ylabel.assert_called_once_with(
            "Angle (deg)",
            fontsize=DEFAULT_LABEL_FONTSIZE,
            fontweight=DEFAULT_LABEL_FONTWEIGHT,
        )

    def test_sets_title(self) -> None:
        """Test that title is set with default formatting."""
        ax = MagicMock()
        ax.get_legend_handles_labels.return_value = ([], [])
        BaseRenderer.format_axis(ax, title="Joint Angles vs Time")
        ax.set_title.assert_called_once_with(
            "Joint Angles vs Time",
            fontsize=DEFAULT_TITLE_FONTSIZE,
            fontweight=DEFAULT_LABEL_FONTWEIGHT,
        )

    def test_sets_grid(self) -> None:
        """Test that grid is set with defaults."""
        ax = MagicMock()
        ax.get_legend_handles_labels.return_value = ([], [])
        BaseRenderer.format_axis(ax, grid=True)
        ax.grid.assert_called_once_with(
            True,
            alpha=DEFAULT_GRID_ALPHA,
            linestyle=DEFAULT_GRID_STYLE,
        )

    def test_no_grid_when_disabled(self) -> None:
        """Test that grid is not set when disabled."""
        ax = MagicMock()
        ax.get_legend_handles_labels.return_value = ([], [])
        BaseRenderer.format_axis(ax, grid=False)
        ax.grid.assert_not_called()

    def test_legend_when_handles_exist(self) -> None:
        """Test that legend is shown when there are legend handles."""
        ax = MagicMock()
        ax.get_legend_handles_labels.return_value = ([MagicMock()], ["label1"])
        BaseRenderer.format_axis(ax, legend=True)
        ax.legend.assert_called_once_with(
            loc="best",
            framealpha=DEFAULT_LEGEND_ALPHA,
        )

    def test_no_legend_when_no_handles(self) -> None:
        """Test that legend is skipped when there are no legend handles."""
        ax = MagicMock()
        ax.get_legend_handles_labels.return_value = ([], [])
        BaseRenderer.format_axis(ax, legend=True)
        ax.legend.assert_not_called()

    def test_no_legend_when_disabled(self) -> None:
        """Test that legend is skipped when disabled."""
        ax = MagicMock()
        ax.get_legend_handles_labels.return_value = ([MagicMock()], ["label1"])
        BaseRenderer.format_axis(ax, legend=False)
        ax.legend.assert_not_called()

    def test_empty_labels_skipped(self) -> None:
        """Test that empty labels are not set."""
        ax = MagicMock()
        ax.get_legend_handles_labels.return_value = ([], [])
        BaseRenderer.format_axis(ax)
        ax.set_xlabel.assert_not_called()
        ax.set_ylabel.assert_not_called()
        ax.set_title.assert_not_called()

    def test_custom_fontsize(self) -> None:
        """Test that custom fontsize overrides default."""
        ax = MagicMock()
        ax.get_legend_handles_labels.return_value = ([], [])
        BaseRenderer.format_axis(ax, xlabel="X", label_fontsize=16)
        ax.set_xlabel.assert_called_once_with(
            "X",
            fontsize=16,
            fontweight=DEFAULT_LABEL_FONTWEIGHT,
        )

    def test_full_formatting(self) -> None:
        """Test applying all formatting at once."""
        ax = MagicMock()
        ax.get_legend_handles_labels.return_value = ([MagicMock()], ["data"])
        BaseRenderer.format_axis(
            ax,
            xlabel="Time (s)",
            ylabel="Angle (deg)",
            title="Test Plot",
        )
        assert ax.set_xlabel.called
        assert ax.set_ylabel.called
        assert ax.set_title.called
        assert ax.grid.called
        assert ax.legend.called
