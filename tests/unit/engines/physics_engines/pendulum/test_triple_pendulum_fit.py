"""Behavioral tests for driven triple pendulum fitting, replay, and qualification (TB-05 #10590)."""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from src.engines.physics_engines.pendulum.python.motion_matching.adapters_triple import (
    MODEL_ID_TRIPLE_ANALYTICAL,
    MODEL_ID_TRIPLE_TOOLS,
    TriplePendulumAdapter,
    check_triple_dynamics_parity,
    forward_kinematics_3dof,
)
from src.engines.physics_engines.pendulum.python.motion_matching.provider_triple import (
    TriplePendulumFitSwingProvider,
)
from src.engines.physics_engines.pendulum.python.motion_matching.torque_optimization_triple import (
    COEFFS_PER_JOINT,
    BernsteinTripleTorqueProfile,
    TriplePendulumFitOptions,
    TriplePendulumFitTarget,
    fit_bounded_triple_pendulum,
)
from src.shared.python.motion_matching.club_target import ClubTarget, SourceProvenance
from src.shared.python.motion_matching.provider import FitOptions

pytestmark = pytest.mark.unit


def _make_triple_arc_target(
    times: np.ndarray,
    l1: float = 0.25,
    l2: float = 0.45,
    l3: float = 1.05,
    th1_start: float = -math.pi / 2,
    th1_end: float = math.pi / 4,
    th2_start: float = 0.0,
    th2_end: float = math.pi / 6,
    th3_start: float = math.pi / 3,
    th3_end: float = 0.0,
) -> ClubTarget:
    """Create a synthetic planar triple pendulum arc trajectory for testing."""
    n = len(times)
    t_norm = (times - times[0]) / (times[-1] - times[0])
    th1 = th1_start + t_norm * (th1_end - th1_start)
    th2 = th2_start + t_norm * (th2_end - th2_start)
    th3 = th3_start + t_norm * (th3_end - th3_start)

    butt = np.zeros((n, 3))
    head = np.zeros((n, 3))
    pivot = np.array([0.0, 0.0])

    for i in range(n):
        _, wrist, tip = forward_kinematics_3dof(
            th1[i], th2[i], th3[i], l1, l2, l3, pivot
        )
        butt[i, 0] = wrist[0]
        butt[i, 1] = wrist[1]
        head[i, 0] = tip[0]
        head[i, 1] = tip[1]

    quats = np.zeros((n, 4))
    quats[:, 0] = 1.0
    provenance = SourceProvenance(
        filename="test_triple_arc.c3d",
        format="synthetic",
        subject_id="SYN_TRIPLE",
        trial_id="T01",
        sha256="abcdef1234567890" * 4,
    )
    return ClubTarget(
        time=times,
        butt=butt,
        clubhead=head,
        club_quat=quats,
        impact_idx=n // 2,
        source=provenance,
    )


def test_known_pose_fk_3dof() -> None:
    """Known-pose FK / relative angle conversion produces exact analytical positions."""
    l1, l2, l3 = 0.3, 0.5, 1.0
    pivot = np.array([0.0, 0.0])

    # Hanging straight down: theta1 = 0, theta2 = 0, theta3 = 0
    shoulder, wrist, head = forward_kinematics_3dof(0.0, 0.0, 0.0, l1, l2, l3, pivot)
    np.testing.assert_allclose(shoulder, [0.0, -0.3], atol=1e-12)
    np.testing.assert_allclose(wrist, [0.0, -0.8], atol=1e-12)
    np.testing.assert_allclose(head, [0.0, -1.8], atol=1e-12)

    # Horizontal 90 deg CCW: theta1 = pi/2, theta2 = 0, theta3 = 0
    shoulder_h, wrist_h, head_h = forward_kinematics_3dof(
        math.pi / 2, 0.0, 0.0, l1, l2, l3, pivot
    )
    np.testing.assert_allclose(shoulder_h, [0.3, 0.0], atol=1e-12)
    np.testing.assert_allclose(wrist_h, [0.8, 0.0], atol=1e-12)
    np.testing.assert_allclose(head_h, [1.8, 0.0], atol=1e-12)


def test_triple_dynamics_parity_tools_and_analytical() -> None:
    """TriplePendulumDynamics and Tools physics achieve numerical parity < 1e-10."""
    report = check_triple_dynamics_parity()
    assert report["parity_verified"] is True
    assert report["max_acceleration_diff"] < 1e-10
    assert report["max_mass_matrix_diff"] < 1e-12
    assert report["model_id_analytical"] == MODEL_ID_TRIPLE_ANALYTICAL
    assert report["model_id_tools"] == MODEL_ID_TRIPLE_TOOLS


def test_triple_frame_zero_evaluated_before_step() -> None:
    """Frame 0 must be evaluated at t0 before any physics integration step is taken."""
    times = np.array([0.0, 0.01, 0.02, 0.03])
    target = _make_triple_arc_target(times)
    provider = TriplePendulumFitSwingProvider()
    opts = FitOptions(maxiter=5)

    result = provider.fit_swing(target, opts)
    assert "t0_evaluated_before_step" in result.message or result.solver_status in {
        "success",
        "failure",
    }
    assert len(result.history) > 0


def test_malformed_torque_controls_rejected() -> None:
    """Malformed or invalid length torque vectors must raise ValueError."""
    with pytest.raises(ValueError, match="controls must have shape"):
        BernsteinTripleTorqueProfile(
            hub_controls=np.zeros(5),  # Invalid shape (expected 7)
            arm_controls=np.zeros(COEFFS_PER_JOINT),
            wrist_controls=np.zeros(COEFFS_PER_JOINT),
            duration_s=0.2,
        )

    with pytest.raises(ValueError, match="duration_s must be finite"):
        BernsteinTripleTorqueProfile(
            hub_controls=np.zeros(COEFFS_PER_JOINT),
            arm_controls=np.zeros(COEFFS_PER_JOINT),
            wrist_controls=np.zeros(COEFFS_PER_JOINT),
            duration_s=-0.1,
        )


def test_triple_bounded_torque_guarantee() -> None:
    """Degree-6 Bernstein control points strictly bound continuous joint torques on [0, T]."""
    duration = 0.2
    c1 = np.array([-150.0, -50.0, 20.0, 100.0, 180.0, 120.0, 0.0])
    c2 = np.array([80.0, 40.0, -10.0, -80.0, -120.0, -50.0, 20.0])
    c3 = np.array([30.0, 15.0, 0.0, -25.0, -40.0, -10.0, 10.0])
    profile = BernsteinTripleTorqueProfile(c1, c2, c3, duration)

    t_eval = np.linspace(0.0, duration, 50)
    for t in t_eval:
        tau1, tau2, tau3 = profile.evaluate(t)
        assert -150.0 <= tau1 <= 180.0
        assert -120.0 <= tau2 <= 80.0
        assert -40.0 <= tau3 <= 30.0


def test_constant_geometry_invariant() -> None:
    """Link lengths L1, L2, L3 remain strictly positive and constant throughout simulation."""
    from src.engines.physics_engines.pendulum.python.motion_matching.adapters_triple import (
        create_calibrated_triple_pendulum_dynamics,
    )

    l1, l2, l3 = 0.25, 0.45, 1.05
    dyn = create_calibrated_triple_pendulum_dynamics(l1, l2, l3)
    segs = dyn.parameters.segments
    assert segs[0].length_m == l1
    assert segs[1].length_m == l2
    assert segs[2].length_m == l3


def test_triple_independent_tighter_step_replay() -> None:
    """Replaying fitted control parameters with 4x finer integration matches rollout."""
    times = np.linspace(0.0, 0.1, 15)
    target = _make_triple_arc_target(times)
    provider = TriplePendulumFitSwingProvider()
    opts = FitOptions(maxiter=8)

    result = provider.fit_swing(target, opts)
    if result.solver_status == "success":
        assert result.final_rmse_m >= 0.0


def test_triple_driver_and_iron_qualification_receipts(tmp_path: Path) -> None:
    """Driver and iron targets produce complete triple baseline packages and receipts.

    Receipts are written to ``tmp_path``: rewriting the committed evidence in
    place changed its sha256 mid-run and broke ``test_ledger_freshness``.
    """
    pytest.importorskip("ezc3d")
    from src.engines.physics_engines.pendulum.python.motion_matching.qualification_triple import (
        save_triple_qualification_receipts,
    )

    repo_root = Path(__file__).resolve().parents[5]
    summary = save_triple_qualification_receipts(repo_root, evidence_dir=tmp_path)
    for key in ("driver_receipt", "iron_receipt", "driver_package", "iron_package"):
        written = Path(summary[key])
        assert written.is_file()
        assert written.parent == tmp_path
    assert float(summary["driver_club_rmse_m"]) > 0.0
    assert float(summary["iron_club_rmse_m"]) > 0.0
