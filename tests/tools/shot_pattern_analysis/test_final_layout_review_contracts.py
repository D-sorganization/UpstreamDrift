"""Independent text-size/collision checks; no simulations or saved-data mutations."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _plots():
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.text import Text

        from src.tools.shot_pattern_analysis import academic_plots
    except (ImportError, OSError) as exc:
        pytest.skip(f"Matplotlib unavailable: {exc}")
    return academic_plots, Figure, FigureCanvasAgg, Text


def _font_pixels(text: Any, figure: Any) -> float:
    # Per-line em size is independent of multiline block height and rotation.
    return float(text.get_fontsize()) * 1080.0 / (72.0 * figure.get_figwidth())


def test_multiline_and_rotated_bbox_height_cannot_prove_readable_font():
    _, figure_type, canvas_type, _ = _plots()
    figure = figure_type(figsize=(10.8, 13.5))
    canvas_type(figure)
    multiline = figure.text(0.1, 0.7, "Tiny Caption\n" * 8, fontsize=4)
    rotated = figure.text(
        0.5, 0.7, "A Very Long Tiny Axis Label", fontsize=4, rotation=90
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for text in (multiline, rotated):
        # Reproduces why the previous height-only assertion could pass tiny text.
        assert text.get_window_extent(renderer).height * 1080 / figure.bbox.width >= 11
        assert _font_pixels(text, figure) < 11


@pytest.mark.parametrize("panels", [2, 3])
def test_actual_social_fonts_and_axis_legend_clearance(
    tmp_path: Path, monkeypatch, panels
):
    plots, _, canvas_type, text_type = _plots()
    captured = []

    def save(figure, _stem, *, dpi):
        captured.append(figure)
        return {}

    monkeypatch.setattr(plots, "_save_formats", save)
    rng = np.random.default_rng(42)
    names = ["Straight", "Draw", "Fade"][:panels]
    points = {name: rng.normal([180, 0], [12, 6], (120, 2)) for name in names}
    plots.write_academic_dispersion_figure(
        points,
        tmp_path / "font_review",
        scenario="Driver · Lean 5° · Curve ×1 · Shaft Twist SD 1°",
        metadata={
            "social_club": "Driver",
            "Delivery Noise": "Path SD 0.5° • Speed SD 2 mph • Strike SD 0.25 in on Each Axis",
            "Population": "Individually Supported Endpoints by Physical Pattern",
            "valid_shots_by_pattern": dict.fromkeys(names, 120),
            "total_shots_by_pattern": dict.fromkeys(names, 150),
        },
        target_point_m=(180, 0),
        presentation="social",
        dpi=96,
    )
    figure = captured[0]
    canvas_type(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for text in figure.findobj(text_type):
        if text.get_visible() and text.get_text().strip():
            assert _font_pixels(text, figure) >= 11, text.get_text()
    for panel in range(panels):
        top, main, right = figure.axes[3 * panel : 3 * panel + 3]
        main_ticks = [
            t for t in main.get_xticklabels() if t.get_visible() and t.get_text()
        ]
        right_ticks = [
            t for t in right.get_xticklabels() if t.get_visible() and t.get_text()
        ]
        for left in main_ticks:
            for right_tick in right_ticks:
                assert not left.get_window_extent(renderer).overlaps(
                    right_tick.get_window_extent(renderer)
                )
        # Legend clearance must include the panel title, not just the axes rectangle.
        for legend in figure.legends:
            assert not legend.get_window_extent(renderer).overlaps(
                top.title.get_window_extent(renderer)
            )
        for axis in (top, main, right):
            for label in (axis.xaxis.label, axis.yaxis.label, axis.title):
                if not label.get_visible() or not label.get_text():
                    continue
                box = label.get_window_extent(renderer)
                footer_box = figure.texts[-1].get_window_extent(renderer)
                assert not box.overlaps(footer_box), (label.get_text(), box, footer_box)
                assert box.x0 >= 0 and box.y0 >= 0
                assert box.x1 <= figure.bbox.width and box.y1 <= figure.bbox.height
