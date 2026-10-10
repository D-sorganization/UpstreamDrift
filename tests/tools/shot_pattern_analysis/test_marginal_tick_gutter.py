"""Independent neighboring marginal-label readability regression."""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("presentation", ["academic", "social"])
def test_neighboring_marginal_tick_labels_have_readable_gutter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, presentation: str
) -> None:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        from src.tools.shot_pattern_analysis import academic_plots
    except (ImportError, OSError) as exc:
        pytest.skip(f"Matplotlib unavailable: {exc}")

    captured = []

    def capture(figure, _stem, *, dpi):
        captured.append(figure)
        return {}

    monkeypatch.setattr(academic_plots, "_save_formats", capture)
    rng = np.random.default_rng(20261009)
    points = {
        name: rng.normal([180, 0], [12, 6], (10_000, 2))
        for name in ("Straight", "Draw", "Fade")
    }
    academic_plots.write_academic_dispersion_figure(
        points,
        tmp_path / presentation,
        scenario="Marginal Tick Readability Review",
        metadata={"social_club": "Driver"},
        target_point_m=(180, 0),
        presentation=presentation,
        dpi=180,
    )
    estimate = academic_plots.shared_histogram_density(
        points, bins=64, target_point_m=(180, 0), region_masses=(0.50, 0.80, 0.95)
    )
    expected_marginals = {
        name: np.sum(item.density_per_m2, axis=0) * np.diff(estimate.x_edges_m)[0]
        for name, item in estimate.patterns.items()
    }
    expected_limit = max(float(np.max(v)) for v in expected_marginals.values()) * 1.05
    figure = captured[0]
    FigureCanvasAgg(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for panel in range(3):
        right = figure.axes[3 * panel + 2]
        np.testing.assert_allclose(
            right.get_xlim(), (0, expected_limit), rtol=0, atol=0
        )
        name = tuple(points)[panel]
        np.testing.assert_array_equal(
            right.patches[0].get_data().values, expected_marginals[name]
        )
        labels = [
            text
            for text in right.get_xticklabels()
            if text.get_visible() and text.get_text()
        ]
        labels.sort(key=lambda text: text.get_window_extent(renderer).x0)
        assert len(labels) >= 2, "Retain a quantitatively interpretable marginal axis"
        main = figure.axes[3 * panel + 1]
        for main_tick in main.get_xticklabels():
            if not main_tick.get_visible() or not main_tick.get_text():
                continue
            for marginal_tick in labels:
                assert not main_tick.get_window_extent(renderer).overlaps(
                    marginal_tick.get_window_extent(renderer)
                ), "A new origin tick must not collide with the main-axis ticks"
        for left, right_label in zip(labels, labels[1:], strict=False):
            gap_points = (
                (
                    right_label.get_window_extent(renderer).x0
                    - left.get_window_extent(renderer).x1
                )
                * 72
                / figure.dpi
            )
            assert gap_points >= 4.0, (
                presentation,
                panel,
                left.get_text(),
                right_label.get_text(),
                gap_points,
            )
