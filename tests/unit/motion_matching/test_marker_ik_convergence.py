"""Tests for IK convergence tracking, termination reporting, and bounds (UpstreamDrift#14938)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.motion_matching.full_body_ik import (
    IKFrameReport,
    IKTermination,
    IKTrajectory,
    PoseFit,
    _run_lm_loop,
    _run_trf_loop,
    compute_marker_rms_trajectory,
    solve_full_body_ik_trajectory,
    solve_full_body_ik_trajectory_with_report,
)
from src.shared.python.motion_matching.marker_calibration import Array, Pose
from src.shared.python.motion_matching.tour_capture_contract import TourCapture
from src.shared.python.motion_matching.trajectory_ik_options import TrajectoryIkOptions

pytestmark = pytest.mark.unit

_OFF, _Q0, _FREE = {"m": ("b", (0.0, 0.0, 0.0))}, np.array([0.0]), np.ones(1, bool)
_B_LM = (np.array([-10.0]), np.array([10.0]))
_B_TRF = (np.array([-2.0]), np.array([2.0]))


def _cap(valid: tuple[bool, ...]) -> TourCapture:
    v = np.array([[x] for x in valid], dtype=bool)
    n = len(v)
    return TourCapture(
        np.arange(n, dtype=float), ("m",), np.full((n, 1, 3), 2.0), v, "h" * 64
    )


def _poses(q: Array) -> dict[str, Pose]:
    return {"b": (np.eye(3), np.array([q[0], 0.0, 0.0]))}


def test_marker_ik_convergence_and_reporting() -> None:
    all_terms = {"converged", "iteration_cap", "damping_diverged", "non_finite"}
    assert {e.value for e in IKTermination} == all_terms | {"no_markers", "at_bound"}
    rep = IKFrameReport(0, IKTermination.CONVERGED, 5, 0.001, 3, True)
    assert rep.converged and rep.termination == IKTermination.CONVERGED
    traj = IKTrajectory(q=np.array([[1.0]]), reports=[rep])
    assert traj.q[0, 0] == 1.0 and list(traj)[1] == [rep]
    fit = PoseFit(np.zeros(3), 0.01, {"m": 0.01}, 0.0, 0.0, 0.0, 10)
    assert fit.termination == IKTermination.CONVERGED and fit.converged

    # LM & TRF loop tests
    fl = lambda x: (x - 2.0, np.eye(1))  # noqa: E731
    fd = lambda x: (np.array([100.0 if abs(x[0]) > 0.01 else 1.0]), np.eye(1))  # noqa: E731
    fn = lambda x: (np.array([float("nan")]), np.eye(1))  # noqa: E731
    q, _, term = _run_lm_loop(fl, _Q0, _FREE, *_B_LM)
    assert term == IKTermination.CONVERGED and q[0] == pytest.approx(2.0, abs=1e-5)
    t_lc = _run_lm_loop(fl, _Q0, _FREE, *_B_LM, iterations=1)[2]
    t_ld = _run_lm_loop(fd, _Q0, _FREE, *_B_LM, damping=1e11)[2]
    assert (
        t_lc == IKTermination.ITERATION_CAP and t_ld == IKTermination.DAMPING_DIVERGED
    )
    assert _run_lm_loop(fn, _Q0, _FREE, *_B_LM)[2] == IKTermination.NON_FINITE

    fh = lambda x: (x - 0.5, np.eye(1))  # noqa: E731
    fs = lambda x: ((x - 5.0) ** 2, np.array([[2.0 * (x[0] - 5.0)]]))  # noqa: E731
    q_trf, _, t_trf = _run_trf_loop(fh, _Q0, _FREE, *_B_TRF, iterations=50)
    assert t_trf == IKTermination.CONVERGED and q_trf[0] == pytest.approx(0.5, abs=1e-5)
    t_tb = _run_trf_loop(
        fs, _Q0, _FREE, np.array([-1.0]), np.array([1.0]), iterations=50
    )[2]
    t_tc = _run_trf_loop(fh, _Q0, _FREE, *_B_TRF, iterations=1)[2]
    assert t_tb == IKTermination.AT_BOUND and t_tc == IKTermination.ITERATION_CAP

    # Empty marker RMS and Trajectory
    c0, c1, c2 = _cap((False,)), _cap((True,)), _cap((True, False))
    f_err, m_err, tot = compute_marker_rms_trajectory(_poses, _OFF, c0, _Q0[None])
    assert math.isnan(f_err[0]) and math.isnan(m_err["m"]) and math.isnan(tot)
    r0 = compute_marker_rms_trajectory(
        _poses, _OFF, c0, _Q0[None], empty_marker_rms=0.0
    )
    assert r0[0][0] == 0.0 and r0[1]["m"] == 0.0 and r0[2] == 0.0

    solve = solve_full_body_ik_trajectory_with_report
    t2 = solve(_poses, _OFF, c2, _Q0, reg_weight=0.0)
    assert t2.reports[0].converged and not t2.reports[1].converged
    assert (
        t2.reports[1].termination == IKTermination.NO_MARKERS
        and t2.q[1, 0] == t2.q[0, 0]
    )

    bnd = (np.array([0.0]), np.array([3.0]))
    opts = TrajectoryIkOptions(reg_weight=0.0, coordinate_bounds=bnd)
    r_b = solve(_poses, _OFF, c1, _Q0, reg_weight=0.0, coordinate_bounds=bnd)
    r_f = solve(_poses, _OFF, c1, _Q0, max_nfev=1, reg_weight=0.0)
    assert (
        r_b.reports[0].converged
        and solve(_poses, _OFF, c1, _Q0, options=opts).reports[0].converged
    )
    assert not r_f.reports[0].converged
    res = solve_full_body_ik_trajectory(_poses, _OFF, c1, _Q0, reg_weight=0.0)
    assert res.shape == (1, 1) and res[0, 0] == pytest.approx(2.0, abs=1e-5)
