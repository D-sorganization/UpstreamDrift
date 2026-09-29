"""Unit tests for ``motion_matching.diagnostics.clubhead_trace``."""

from __future__ import annotations

import hashlib

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import numpy as np
import pytest
from src.shared.python.motion_matching.club_target import (
    ClubTarget,
    SourceProvenance,
)
from src.shared.python.motion_matching.diagnostics.clubhead_trace import (
    TraceCompareOptions,
    compare_clubhead_traces,
    plot_3d_overlay,
    plot_per_axis_timeseries,
    plot_setup_pose_skeletons,
    plot_speed_comparison,
)

pytestmark = pytest.mark.unit


def _provenance(name: str = "synthetic.bin") -> SourceProvenance:
    return SourceProvenance(
        filename=name,
        format="synthetic",
        subject_id="UNIT",
        trial_id="0",
        sha256=hashlib.sha256(b"").hexdigest(),
    )


def _swing_target(
    n: int = 301,
    *,
    pos_offset: np.ndarray | None = None,
    duration: float = 0.3,
    impact_at: float = 0.25,
    name: str = "synthetic.bin",
) -> ClubTarget:
    """A swing-shaped synthetic ClubTarget with a clear speed peak at impact_at."""
    time = np.linspace(0.0, duration, n)
    # phase(t) is monotonic and its derivative is a Gaussian centred at
    # impact_at, so |d/dt clubhead| has a single sharp maximum there.
    sigma = 0.04
    bell = np.exp(-0.5 * ((time - impact_at) / sigma) ** 2)
    phase = np.cumsum(bell) * (time[1] - time[0])
    radius = 1.0
    clubhead = np.column_stack(
        [
            radius * np.sin(phase),
            radius * (1.0 - np.cos(phase)),
            -0.5 * np.ones_like(time),
        ]
    )
    if pos_offset is not None:
        clubhead = clubhead + np.asarray(pos_offset, dtype=np.float64)
    butt = clubhead - np.array([0.0, 0.0, -1.1])  # butt above head
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    impact_idx = int(np.argmin(np.abs(time - impact_at))) + 1  # 1-based
    return ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=impact_idx,
        source=_provenance(name),
    )


def test_identical_traces_yield_zero_rmse() -> None:
    a = _swing_target()
    b = _swing_target()
    rep = compare_clubhead_traces(a, b)
    assert rep.total_rmse_mm == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(rep.rmse_per_axis_mm, 0.0, atol=1e-6)
    assert rep.max_error_mm == pytest.approx(0.0, abs=1e-6)


def test_known_offset_yields_correct_rmse() -> None:
    a = _swing_target()
    b = _swing_target(pos_offset=np.array([0.001, 0.0, 0.0]))  # 1 mm in X
    rep = compare_clubhead_traces(a, b, TraceCompareOptions(time_alignment="none"))
    assert rep.rmse_per_axis_mm[0] == pytest.approx(1.0, abs=1e-3)
    assert rep.rmse_per_axis_mm[1] == pytest.approx(0.0, abs=1e-3)
    assert rep.rmse_per_axis_mm[2] == pytest.approx(0.0, abs=1e-3)
    assert rep.total_rmse_mm == pytest.approx(1.0, abs=1e-3)


def test_per_axis_rmse_components_sum_correctly_to_total() -> None:
    a = _swing_target()
    b = _swing_target(pos_offset=np.array([0.002, 0.003, 0.006]))
    rep = compare_clubhead_traces(a, b, TraceCompareOptions(time_alignment="none"))
    expected = float(np.sqrt(np.sum(rep.rmse_per_axis_mm**2)))
    assert rep.total_rmse_mm == pytest.approx(expected, rel=1e-6)


def test_impact_alignment_centers_max_speed_at_zero() -> None:
    # measured impact at 0.25, simulated impact at 0.18 — alignment should
    # bring both peaks onto the same shifted timegrid near t=0.
    meas = _swing_target(impact_at=0.25)
    sim = _swing_target(impact_at=0.18)
    rep = compare_clubhead_traces(
        meas, sim, TraceCompareOptions(time_alignment="impact")
    )
    t_at_peak = rep.time[rep.impact_idx]
    assert abs(t_at_peak) < 0.02


def test_address_alignment_finds_first_motion() -> None:
    # Two traces where the second one is "delayed" by 50 ms before any motion
    # — address alignment should remove the delay.
    n = 301
    time = np.linspace(0.0, 0.3, n)
    quiet_then_move = np.zeros((n, 3))
    move_idx = int(0.10 * n / 0.3)
    quiet_then_move[move_idx:, 0] = np.linspace(0, 0.5, n - move_idx)
    butt = quiet_then_move - np.array([0.0, 0.0, -1.1])
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    a = ClubTarget(
        time=time,
        butt=butt,
        clubhead=quiet_then_move,
        club_quat=quat,
        impact_idx=n,
        source=_provenance("a.bin"),
    )
    b = ClubTarget(
        time=time,
        butt=butt.copy(),
        clubhead=quiet_then_move.copy(),
        club_quat=quat.copy(),
        impact_idx=n,
        source=_provenance("b.bin"),
    )
    rep = compare_clubhead_traces(a, b, TraceCompareOptions(time_alignment="address"))
    # Both first-motion frames are at the same time => offset == 0 and RMSE 0.
    assert rep.time_alignment_offset_s == pytest.approx(0.0, abs=1e-9)
    assert rep.total_rmse_mm == pytest.approx(0.0, abs=1e-6)


def test_setup_pose_skeleton_returns_figure_with_two_axes() -> None:
    a = _swing_target()
    b = _swing_target(pos_offset=np.array([0.01, 0.0, 0.0]))
    fig = plot_setup_pose_skeletons(a, b)
    assert len(fig.axes) == 2


def test_3d_overlay_returns_figure() -> None:
    a = _swing_target()
    b = _swing_target(pos_offset=np.array([0.01, 0.0, 0.0]))
    rep = compare_clubhead_traces(a, b)
    fig = plot_3d_overlay(rep)
    assert fig is not None
    assert len(fig.axes) >= 1


def test_per_axis_and_speed_plots_run() -> None:
    a = _swing_target()
    b = _swing_target(pos_offset=np.array([0.005, 0.0, 0.0]))
    rep = compare_clubhead_traces(a, b)
    fig1 = plot_per_axis_timeseries(rep)
    fig2 = plot_speed_comparison(rep)
    assert len(fig1.axes) == 3
    assert len(fig2.axes) == 1


def test_speed_units_mph() -> None:
    # Constant 1 m/s clubhead motion in +x ⇒ ~2.2369 mph everywhere.
    n = 51
    time = np.linspace(0.0, 0.05, n)
    clubhead = np.column_stack([time * 1.0, np.zeros(n), np.zeros(n)])
    butt = clubhead + np.array([0.0, 0.0, 1.1])
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    a = ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=n,
        source=_provenance(),
    )
    b = ClubTarget(
        time=time,
        butt=butt.copy(),
        clubhead=clubhead.copy(),
        club_quat=quat.copy(),
        impact_idx=n,
        source=_provenance(),
    )
    rep = compare_clubhead_traces(a, b, TraceCompareOptions(time_alignment="none"))
    # Interior samples should be ~2.2369 mph.
    interior = rep.measured_speed_mph[5:-5]
    assert np.allclose(interior, 2.236936, atol=1e-3)


def test_handles_different_timegrids_via_resampling() -> None:
    a = _swing_target(n=301)
    b = _swing_target(n=151)  # half the sample density
    rep = compare_clubhead_traces(a, b, TraceCompareOptions(time_alignment="none"))
    # Resampling onto a common grid should still yield a near-zero RMSE because
    # the underlying continuous trajectory is the same.
    assert rep.total_rmse_mm < 0.5  # < 0.5 mm
    assert rep.time.shape[0] >= 2


def test_input_validation_rejects_non_ClubTarget() -> None:
    a = _swing_target()
    with pytest.raises(TypeError):
        compare_clubhead_traces(a, "not a target")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compare_clubhead_traces({"wrong": "shape"}, a)  # type: ignore[arg-type]
