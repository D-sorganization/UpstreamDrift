"""Equal-aspect grip/hand matplotlib preview helper (#15141)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_preview_matplotlib import (
    apply_equal_axis_limits,
    axis_half_widths,
    equal_half_width_from_points,
)

pytestmark = pytest.mark.unit


def test_equal_half_width_uses_one_span_for_all_axes():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.02, 1.0],
            [-0.05, 0.01, 0.04],
        ]
    )
    _, half = equal_half_width_from_points(
        pts, margin_fraction=1.0, min_half_width_m=0.0
    )
    assert half == pytest.approx(0.5, rel=1e-6)


def test_apply_equal_axis_limits_yields_equal_spans():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    tube = np.linspace(0, 1, 10)
    pts = np.column_stack([0.05 * np.sin(tube), 0.05 * np.cos(tube), tube * 0.9])
    half = apply_equal_axis_limits(ax, pts)
    spans = axis_half_widths(ax)
    assert spans[0] == pytest.approx(half, rel=1e-6)
    assert spans[1] == pytest.approx(half, rel=1e-6)
    assert spans[2] == pytest.approx(half, rel=1e-6)
    plt.close(fig)


def test_unequal_per_axis_limits_fail_guard():
    """Regression: independent per-axis club limits must not be used for hands."""

    def legacy_club_limits(ax, points: np.ndarray) -> None:
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        pad = 0.06
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_zlim(lo[2] - pad, hi[2] + pad)
        ax.set_box_aspect((1, 1, 1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    mesh = Mesh(
        np.array([[0, 0, 0], [0.02, 0.02, 0.9], [-0.01, 0.01, 0.05]]),
        np.array([[0, 1, 2]], dtype=np.int64),
    )
    legacy_club_limits(ax, mesh.vertices)
    x_half, y_half, z_half = axis_half_widths(ax)
    assert x_half != pytest.approx(z_half, rel=0.05)
    plt.close(fig)
