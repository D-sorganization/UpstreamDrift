"""Tests for static spine report figures (SPN-15a, #12771)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.shared.python.spine.test_viz_data import _series, _stack
from src.shared.python.spine import plots
from src.shared.python.spine.viz_data import load_heatmap

pytestmark = pytest.mark.unit

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_stack_figure_renders_and_writes_png(tmp_path) -> None:
    fig = plots.plot_angle_stack(_stack())
    out = plots.save_png(fig, tmp_path / "stack.png")
    assert out.read_bytes()[:8] == PNG_MAGIC
    texts = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "UNQUALIFIED" in texts
    assert fig.axes[0].get_ylabel() == "Contribution (deg)"
    handles, labels = fig.axes[0].get_legend_handles_labels()
    assert labels == ["L5_S1", "L4_L5", "L3_L4"]


def test_heatmap_figure_labels_units_status(tmp_path) -> None:
    data = load_heatmap(_series(), "tension")
    fig = plots.plot_load_heatmap(data)
    out = plots.save_png(fig, tmp_path / "heat.png")
    assert out.read_bytes()[:8] == PNG_MAGIC
    ax = fig.axes[0]
    assert "(N)" in fig.axes[1].get_ylabel()
    assert "UNQUALIFIED" in ax.get_title()
    assert "Tension" in ax.get_title()
    assert [t.get_text() for t in ax.get_yticklabels()] == ["L5_S1", "L4_L5", "L3_L4"]


def test_qualified_label_only_when_declared() -> None:
    data = load_heatmap(_series(), "compression", scientific_status="qualified")
    assert "UNQUALIFIED" not in plots.plot_load_heatmap(data).axes[0].get_title()


def test_plots_validate_types_and_paths(tmp_path) -> None:
    with pytest.raises(TypeError, match="StackData"):
        plots.plot_angle_stack(np.zeros(3))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="HeatmapData"):
        plots.plot_load_heatmap("x")  # type: ignore[arg-type]
    fig = plots.plot_angle_stack(_stack())
    with pytest.raises(ValueError, match=".png"):
        plots.save_png(fig, tmp_path / "x.jpg")
