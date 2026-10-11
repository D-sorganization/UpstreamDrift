"""Heatmap decomposition retains normalization and missing-data behavior."""

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("metric", "values", "limits"),
    [
        ("conditional_sg", [-0.2, 0.1], (-0.2, 0.2)),
        ("conditional_sg", [0.0], (-1e-9, 1e-9)),
        ("supported_fraction", [0.8, 0.9], (0.0, 1.0)),
        ("target_hit_fraction", [0.5], (0.0, 1.0)),
        ("target_rmse_m", [4.0], (4.0, 4.2)),
        ("radial_p95_m", [0.0], (0.0, 1e-6)),
    ],
)
def test_metric_color_scale_preserves_limits(metric, values, limits):
    pytest.importorskip("matplotlib")
    from src.tools.shot_pattern_analysis.matched_report_plots import _metric_color_scale

    cmap, norm = _metric_color_scale(values, metric)
    assert (norm.vmin, norm.vmax) == pytest.approx(limits)
    assert tuple(cmap.get_bad()) == pytest.approx((229 / 255, 231 / 255, 235 / 255, 1))


def test_empty_metric_color_scale_remains_unavailable():
    pytest.importorskip("matplotlib")
    from src.tools.shot_pattern_analysis.matched_report_plots import _metric_color_scale

    _, norm = _metric_color_scale([], "conditional_sg")
    assert norm is None
