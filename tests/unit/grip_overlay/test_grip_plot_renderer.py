"""Matplotlib renderer for the grip force and couple plots (GCV-10, #11716)."""

from __future__ import annotations

from matplotlib.figure import Figure
import numpy as np
import pytest

from src.shared.python.biomechanics.grip_extraction import net_only_analysis
from src.shared.python.biomechanics.grip_plot_model import build_grip_plot_series
from src.shared.python.biomechanics.grip_wrench import HandWrench, analyze_grip
from src.shared.python.plotting.renderers.grip_wrench import plot_grip_wrench

pytestmark = pytest.mark.unit


def _series(unavailable_at=None, n=5):
    analyses = []
    for k in range(n):
        if k == unavailable_at:
            analyses.append(
                net_only_analysis(
                    point_m=(0, 0, 0.9),
                    force_on_club_n=(5.0, 0, 0),
                    torque_on_club_nm=(0, 1.0, 0),
                    split_method="allocation",
                    reason="net only",
                )
            )
            continue
        analyses.append(
            analyze_grip(
                HandWrench("L", (0, 0, 1.0), (10.0 * k, 0, 0), (0, 0, 0)),
                HandWrench("R", (0, 0, 0.8), (-5.0 * k, 0, 0), (0, 0, 0)),
                split_method="efc_force",
            )
        )
    return build_grip_plot_series(
        [0.01 * k for k in range(n)], analyses, events={"impact": 0.03}
    )


def test_four_panels_with_titles_split_label_and_impact_marker():
    fig = Figure()
    plot_grip_wrench(fig, _series())
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == [
        "Hand Force Magnitude",
        "Net Force at Grip Midpoint",
        "Equivalent Couple at Midpoint (World)",
        "Contact Force Moment vs Applied Free Torque",
    ]
    assert "efc_force" in fig._suptitle.get_text()
    for ax in fig.axes:
        assert any(
            len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), [0.03, 0.03])
            for line in ax.get_lines()
        )
    assert all(ax.get_xlabel() in ("", "Time (s)") for ax in fig.axes)


def test_unavailable_samples_leave_gaps_not_zeros():
    fig = Figure()
    plot_grip_wrench(fig, _series(unavailable_at=2))
    left = next(line for line in fig.axes[0].get_lines() if line.get_label() == "Left")
    y = np.asarray(left.get_ydata(), dtype=float)
    assert np.isnan(y[2]) and not np.isnan(y[1]) and not np.isnan(y[3])


def test_club_frame_selects_local_couple():
    fig = Figure()
    plot_grip_wrench(fig, _series(), couple_frame="club")
    assert fig.axes[2].get_title().endswith("(Club)")
    with pytest.raises(ValueError):
        plot_grip_wrench(Figure(), _series(), couple_frame="bogus")


def test_unavailable_series_draws_the_reason():
    from src.shared.python.biomechanics.grip_extraction import unavailable_analysis

    p = build_grip_plot_series([0.0, 0.1], [unavailable_analysis("no welds")] * 2)
    fig = Figure()
    plot_grip_wrench(fig, p)
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any("unavailable" in t and "no welds" in t for t in texts)
