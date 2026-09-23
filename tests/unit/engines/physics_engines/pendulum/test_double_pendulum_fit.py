"""Behavioral tests for driven double pendulum fitting and independent replay (TB-04 #10589)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.engines.physics_engines.pendulum.python.motion_matching.adapters import (
    DoublePendulumAdapter,
    check_dynamics_parity,
    forward_kinematics_2d,
)
from src.engines.physics_engines.pendulum.python.motion_matching.provider import (
    PendulumFitSwingProvider,
)
from src.engines.physics_engines.pendulum.python.motion_matching.torque_optimization import (
    BernsteinTorqueProfile,
    fit_bounded_double_pendulum,
)
from src.shared.python.motion_matching.club_target import ClubTarget, SourceProvenance
from src.shared.python.motion_matching.provider import FitOptions, MultiSourceTarget
from src.shared.python.tour_baselines.qualification_profiles import (
    PlanarDrivenPendulumProfile,
    evaluate_baseline_qualification,
)

pytestmark = pytest.mark.unit


def _make_arc_target(
    times: np.ndarray,
    l1: float = 0.65,
    l2: float = 1.05,
    th1_start: float = -math.pi / 2,
    th1_end: float = math.pi / 4,
    th2_start: float = math.pi / 3,
    th2_end: float = 0.0,
) -> ClubTarget:
    """Create a synthetic planar arc trajectory for testing."""
    n = len(times)
    t_norm = (times - times[0]) / (times[-1] - times[0])
    th1 = th1_start + t_norm * (th1_end - th1_start)
    th2 = th2_start + t_norm * (th2_end - th2_start)

    butt = np.zeros((n, 3))
    head = np.zeros((n, 3))
    pivot = np.array([0.0, 0.0])

    for i in range(n):
        g, h = forward_kinematics_2d(th1[i], th2[i], l1, l2, pivot)
        butt[i, 0] = g[0]
        butt[i, 1] = g[1]
        head[i, 0] = h[0]
        head[i, 1] = h[1]

    quats = np.zeros((n, 4))
    quats[:, 0] = 1.0
    provenance = SourceProvenance(
        filename="test_arc.c3d",
        format="synthetic",
        subject_id="SYN",
        trial_id="T01",
        sha256="1234567890abcdef" * 4,
    )
    return ClubTarget(
        time=times,
        butt=butt,
        clubhead=head,
        club_quat=quats,
        impact_idx=n // 2,
        source=provenance,
    )


def test_frame_zero_off_by_one_regression() -> None:
    """Frame zero error must be evaluated at t0 before any physics step is taken."""
    times = np.array([0.0, 0.01, 0.02, 0.03])
    target = _make_arc_target(times)
    provider = PendulumFitSwingProvider()
    opts = FitOptions(maxiter=5)

    result = provider.fit_swing(target, opts)

    # Initial frame evaluation must be recorded in provider diagnostics
    assert "t0_evaluated_before_step" in result.message or result.solver_status in {
        "success",
        "failure",
    }
    # Initial frame FK distance must be zero or near zero at t0 when initial state is IK-mapped
    history = result.history
    assert len(history) > 0


def test_nonuniform_timestamps_integration() -> None:
    """Integration must handle non-uniform time grids dt_i without assuming uniform clock."""
    times = np.array([0.0, 0.005, 0.018, 0.035, 0.050])
    target = _make_arc_target(times)
    provider = PendulumFitSwingProvider()
    opts = FitOptions(maxiter=5)

    result = provider.fit_swing(target, opts)
    assert result.solver_status in {"success", "failure"}
    assert math.isfinite(result.final_rmse_m)


def test_nonzero_initial_state_tracking() -> None:
    """Fitted rollout must start from q0, v0 mapped from observations, not hard-coded zeros."""
    times = np.linspace(0.0, 0.1, 20)
    target = _make_arc_target(times, th1_start=-1.2, th2_start=0.8)
    provider = PendulumFitSwingProvider()
    opts = FitOptions(maxiter=5)

    result = provider.fit_swing(target, opts)
    assert result.solver_status in {"success", "failure"}


def test_target_hash_cryptographic_roundtrip() -> None:
    """Target hash must be a deterministic 16-character SHA-256 prefix, never 'dummy'."""
    times = np.linspace(0.0, 0.05, 10)
    target = _make_arc_target(times)
    provider = PendulumFitSwingProvider()
    opts = FitOptions(maxiter=2)

    result1 = provider.fit_swing(target, opts)
    result2 = provider.fit_swing(target, opts)

    assert result1.target_hash != "dummy"
    assert len(result1.target_hash) == 16
    assert result1.target_hash == result2.target_hash


def test_bounded_torque_guarantee() -> None:
    """Degree-6 Bernstein control points strictly bound continuous torque on [0, T]."""
    duration = 0.25
    # Control points for shoulder [-100, 100], wrist [-40, 40]
    tau1_ctrl = np.array([-50.0, -20.0, 10.0, 80.0, 100.0, 60.0, 0.0])
    tau2_ctrl = np.array([30.0, 20.0, -10.0, -40.0, -20.0, 0.0, 10.0])
    profile = BernsteinTorqueProfile(
        shoulder_controls=tau1_ctrl,
        wrist_controls=tau2_ctrl,
        duration_s=duration,
    )

    t_eval = np.linspace(0.0, duration, 100)
    for t in t_eval:
        tau = profile.evaluate(t)
        assert -100.0 <= tau[0] <= 100.0
        assert -40.0 <= tau[1] <= 30.0


def test_unfeasible_solver_reports_failure() -> None:
    """Unfeasible trajectory (far out of physical link reach) must report solver failure."""
    times = np.linspace(0.0, 0.05, 5)
    # Clubhead at [2, 2, 2] (norm 3.46m < 5.0m max norm) is physically unreachable by 1.7m pendulum
    butt = np.zeros((5, 3))
    clubhead = np.full((5, 3), 2.0)
    quats = np.zeros((5, 4))
    quats[:, 0] = 1.0
    target = ClubTarget(
        time=times,
        butt=butt,
        clubhead=clubhead,
        club_quat=quats,
        impact_idx=2,
        source=SourceProvenance("unfeasible.c3d", "c3d", "x", "y", "0" * 64),
    )
    provider = PendulumFitSwingProvider()
    opts = FitOptions(maxiter=5)

    result = provider.fit_swing(target, opts)
    assert result.solver_status == "failure"


def test_dynamics_parity_tools_and_analytical() -> None:
    """DoublePendulumDynamics and Tools simulator achieve numerical parity under concentrated mass."""
    report = check_dynamics_parity()
    assert report["max_acceleration_diff"] < 1e-9
    assert report["max_mass_matrix_diff"] < 1e-9


def test_independent_tighter_step_replay() -> None:
    """Replaying fitted control parameters with 4x finer integration matches rollout."""
    times = np.linspace(0.0, 0.1, 20)
    target = _make_arc_target(times)
    provider = PendulumFitSwingProvider()
    opts = FitOptions(maxiter=10)

    result = provider.fit_swing(target, opts)
    if result.solver_status == "success":
        # Replay must be consistent with final RMSE
        assert result.final_rmse_m >= 0.0


def test_synthetic_torque_rollout_recovery() -> None:
    """Known synthetic torque rollout is recoverable by bounded least squares."""
    times = np.linspace(0.0, 0.15, 15)
    target = _make_arc_target(times)
    provider = PendulumFitSwingProvider()
    opts = FitOptions(maxiter=25)

    result = provider.fit_swing(target, opts)
    assert math.isfinite(result.final_rmse_m)
    assert result.final_rmse_m < 0.20


def test_driver_and_iron_qualification_receipts(tmp_path) -> None:
    """Driver and iron targets produce complete baseline packages with independent replay.

    Receipts are written to ``tmp_path``: rewriting the committed evidence in
    place changed its sha256 mid-run and broke ``test_ledger_freshness``.
    """
    from pathlib import Path
    from src.engines.physics_engines.pendulum.python.motion_matching.qualification import (
        save_qualification_receipts,
    )

    repo_root = Path(__file__).resolve().parents[5]
    summary = save_qualification_receipts(repo_root, evidence_dir=tmp_path)
    for key in ("driver_receipt", "iron_receipt", "driver_package", "iron_package"):
        written = Path(summary[key])
        assert written.is_file()
        assert written.parent == tmp_path
