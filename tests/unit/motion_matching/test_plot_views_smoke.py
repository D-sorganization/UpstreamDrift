"""Smoke + error-path coverage for the plot_*.py view modules.

Each plot is rendered with the non-interactive ``Agg`` backend so the
tests run on a headless CI box.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from src.shared.python.motion_matching.cost import (  # noqa: E402
    CostBreakdown,
    SimOutput,
)
from src.shared.python.motion_matching.plot_error_timecourse import (  # noqa: E402
    plot_error_timecourse,
)
from src.shared.python.motion_matching.plot_fit_quality_card import (  # noqa: E402
    fit_quality_summary,
    plot_fit_quality_card,
)
from src.shared.python.motion_matching.plot_trajectory_overlay import (  # noqa: E402
    plot_trajectory_overlay,
)

from ._fixtures import make_target  # noqa: E402


def _matching_sim(target):
    n = target.time.shape[0]
    butt = target.butt.copy()
    clubhead = target.clubhead.copy()
    quat = target.club_quat.copy()
    return SimOutput(butt=butt, clubhead=clubhead, club_quat=quat, time=target.time)


def _breakdown() -> CostBreakdown:
    return CostBreakdown(
        position=1.0,
        orientation=0.5,
        impact_anchor=0.25,
        body_marker=0.0,
        regularizer=0.01,
        total=1.76,
    )


def test_plot_trajectory_overlay_smoke() -> None:
    """Pin: full-shape inputs render a 1x2 overlay figure."""
    t = make_target()
    fig = plot_trajectory_overlay(t, _matching_sim(t), title="x")
    assert fig is not None
    plt.close(fig)


def test_plot_trajectory_overlay_butt_mismatch() -> None:
    """Pin: butt shape mismatch is rejected up front."""
    t = make_target()
    sim = _matching_sim(t)
    bad = SimOutput(
        butt=sim.butt[:-1],
        clubhead=sim.clubhead,
        club_quat=sim.club_quat,
        time=sim.time,
    )
    with pytest.raises(ValueError, match="butt shape mismatch"):
        plot_trajectory_overlay(t, bad)


def test_plot_trajectory_overlay_clubhead_mismatch() -> None:
    """Pin: clubhead shape mismatch rejected."""
    t = make_target()
    sim = _matching_sim(t)
    bad = SimOutput(
        butt=sim.butt,
        clubhead=sim.clubhead[:-1],
        club_quat=sim.club_quat,
        time=sim.time,
    )
    with pytest.raises(ValueError, match="clubhead shape mismatch"):
        plot_trajectory_overlay(t, bad)


def test_plot_error_timecourse_smoke() -> None:
    """Pin: error timecourse plot renders without crashing."""
    t = make_target()
    fig = plot_error_timecourse(t, _matching_sim(t), title="err")
    plt.close(fig)


def test_plot_error_timecourse_shape_mismatch() -> None:
    """Pin: mismatched butt/clubhead shape is rejected."""
    t = make_target()
    sim = _matching_sim(t)
    bad = SimOutput(
        butt=sim.butt[:-1],
        clubhead=sim.clubhead,
        club_quat=sim.club_quat,
        time=sim.time,
    )
    with pytest.raises(ValueError, match="match target time length"):
        plot_error_timecourse(t, bad)


def test_plot_error_timecourse_quat_shape() -> None:
    """Pin: club_quat shape mismatch is rejected."""
    t = make_target()
    sim = _matching_sim(t)
    bad = SimOutput(
        butt=sim.butt,
        clubhead=sim.clubhead,
        club_quat=sim.club_quat[:, :3],
        time=sim.time,
    )
    with pytest.raises(ValueError, match=r"club_quat must have shape"):
        plot_error_timecourse(t, bad)


def test_fit_quality_summary_finite() -> None:
    """Pin: summary returns finite scalars for a perfect-match sim."""
    t = make_target()
    s = fit_quality_summary(t, _matching_sim(t), _breakdown())
    # Perfect overlap -> ~zero RMSEs.
    assert s.rmse_butt_m < 1e-9
    assert s.rmse_clubhead_m < 1e-9


def test_fit_quality_summary_shape_errors() -> None:
    """Pin: shape mismatches in summary raise ValueError."""
    t = make_target()
    sim = _matching_sim(t)
    bad = SimOutput(
        butt=sim.butt[:-1],
        clubhead=sim.clubhead,
        club_quat=sim.club_quat,
        time=sim.time,
    )
    with pytest.raises(ValueError, match="match target time length"):
        fit_quality_summary(t, bad, _breakdown())
    bad_q = SimOutput(
        butt=sim.butt,
        clubhead=sim.clubhead,
        club_quat=sim.club_quat[:, :3],
        time=sim.time,
    )
    with pytest.raises(ValueError, match=r"club_quat must have shape"):
        fit_quality_summary(t, bad_q, _breakdown())


def test_plot_fit_quality_card_smoke() -> None:
    """Pin: quality card renders for a valid (target, sim, breakdown)."""
    t = make_target()
    fig = plot_fit_quality_card(t, _matching_sim(t), _breakdown(), title="card")
    plt.close(fig)
