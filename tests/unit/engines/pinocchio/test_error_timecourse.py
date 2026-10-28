"""Unit tests for Pinocchio error timecourse visualization (UD-23, issue #13949).

Verifies that target.impact_idx is treated as 1-based per CLUB_IK_SPEC convention
when plotting the vertical impact line across all panels.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

_PIN_PY = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "engines"
    / "physics_engines"
    / "pinocchio"
    / "python"
)
if str(_PIN_PY) not in sys.path:
    sys.path.insert(0, str(_PIN_PY))

from motion_matching.leaderboard_writer import FitResult  # noqa: E402
from motion_matching.viz.error_timecourse import plot_error_timecourse  # noqa: E402


class _DummyTarget:
    """Minimal ClubTargetLike object for error_timecourse."""

    def __init__(self, n: int = 101, impact_idx: int | None = 51) -> None:
        self.time = np.linspace(0.0, 1.0, n)
        self.butt = np.zeros((n, 3))
        self.clubhead = np.zeros((n, 3))
        self.club_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
        if impact_idx is not None:
            self.impact_idx = impact_idx


def _make_dummy_result(n: int = 101) -> FitResult:
    return FitResult(
        trial_id="test_trial",
        solver="lm",
        butt_sim=np.zeros((n, 3)),
        clubhead_sim=np.zeros((n, 3)),
        club_quat_sim=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
        time=np.linspace(0.0, 1.0, n),
        joint_torques=np.zeros((n, 4)),
        clubhead_speed_mph=np.zeros(n),
        grip_rmse_mm=1.0,
        clubhead_rmse_mm=1.0,
        orientation_rmse_deg=1.0,
        total_work_J=100.0,
        wall_clock_s=1.0,
        n_iterations=10,
        commit="test",
    )


def _get_impact_vlines(fig: Any) -> list[Any]:
    vlines = []
    for ax in fig.axes:
        for line in ax.get_lines():
            if line.get_linestyle() == ":":
                vlines.append(line)
    return vlines


@pytest.mark.unit
def test_error_timecourse_impact_idx_is_one_based() -> None:
    """impact_idx=51 (1-based) in a 101-frame trajectory (indices 0..100) must map to t[50] = 0.50."""
    target = _DummyTarget(n=101, impact_idx=51)
    result = _make_dummy_result(n=101)

    fig = plot_error_timecourse(target, result)
    try:
        vlines = _get_impact_vlines(fig)
        assert len(vlines) == 4, (
            f"Expected 4 impact lines across 4 panels, got {len(vlines)}"
        )
        for line in vlines:
            xdata = line.get_xdata()
            assert xdata[0] == pytest.approx(0.50, abs=1e-5), (
                f"Expected impact line at t=0.50 (0-based index 50 for 1-based impact_idx=51), got {xdata[0]}"
            )
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


@pytest.mark.unit
def test_error_timecourse_impact_idx_first_frame() -> None:
    """impact_idx=1 (1-based) must map to 0-based index 0 (t[0] = 0.0)."""
    target = _DummyTarget(n=101, impact_idx=1)
    result = _make_dummy_result(n=101)

    fig = plot_error_timecourse(target, result)
    try:
        vlines = _get_impact_vlines(fig)
        assert len(vlines) == 4
        for line in vlines:
            xdata = line.get_xdata()
            assert xdata[0] == pytest.approx(0.0, abs=1e-5), (
                f"Expected impact line at t=0.0 (0-based index 0 for 1-based impact_idx=1), got {xdata[0]}"
            )
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


@pytest.mark.unit
def test_error_timecourse_impact_idx_last_frame() -> None:
    """impact_idx=101 (1-based) must map to 0-based index 100 (t[100] = 1.0)."""
    target = _DummyTarget(n=101, impact_idx=101)
    result = _make_dummy_result(n=101)

    fig = plot_error_timecourse(target, result)
    try:
        vlines = _get_impact_vlines(fig)
        assert len(vlines) == 4
        for line in vlines:
            xdata = line.get_xdata()
            assert xdata[0] == pytest.approx(1.0, abs=1e-5), (
                f"Expected impact line at t=1.0 (0-based index 100 for 1-based impact_idx=101), got {xdata[0]}"
            )
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


@pytest.mark.unit
def test_error_timecourse_impact_idx_missing_defaults_to_last() -> None:
    """When impact_idx is missing, fallback defaults to last frame t[-1] = 1.0."""
    target = _DummyTarget(n=101, impact_idx=None)
    result = _make_dummy_result(n=101)

    fig = plot_error_timecourse(target, result)
    try:
        vlines = _get_impact_vlines(fig)
        assert len(vlines) == 4
        for line in vlines:
            xdata = line.get_xdata()
            assert xdata[0] == pytest.approx(1.0, abs=1e-5)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
