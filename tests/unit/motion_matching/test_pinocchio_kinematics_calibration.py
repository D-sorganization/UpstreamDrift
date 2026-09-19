"""Unit tests for Pinocchio full-swing kinematics calibration, grip compatibility, and joint smoothing (PF-02, #10432).

Verifies:
1. MarkerIkSolver convergence diagnostics (projected gradient, cost decrease, active bounds).
2. Difficult-frame multi-start resolution with and without weld closure.
3. Bounded overlapping-window refinement with temporal regularisation.
4. Joint trajectory smoothing with verified q, v, a derivative compatibility.
5. Boundary spike auditing and cutoff frequency sensitivity.
6. Wrist model human range-of-motion compliance (MM-2, #10104).
7. Left elbow pit up-and-inward address alignment (MM-5, #10107).
8. Distinct driver and 7-iron calibration provenance enforcement.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any
import numpy as np
import pytest

from src.shared.python.motion_matching.kinematic_smoothing import (
    BoundarySpikeAudit,
    audit_boundary_spikes,
    audit_cutoff_sensitivity,
    smooth_kinematic_trajectory,
)
from src.shared.python.motion_matching.posture_metrics import elbow_pit_direction
from src.shared.python.motion_matching.range_of_motion import (
    HUMAN_RANGES_DEG,
    violations,
)
from src.engines.physics_engines.pinocchio.python.marker_kinematics import (
    CoordinateMap,
    MarkerIkOptions,
    MarkerIkSolver,
    MarkerTable,
    SolveDiagnostics,
)
from src.shared.python.motion_matching.acceptance import (
    AcceptanceGates,
    AcceptanceVerdict,
    GateStatus,
    Horizon,
    evaluate,
)

pytestmark = [pytest.mark.unit]


class MockPlacement:
    def __init__(self, translation: Sequence[float]) -> None:
        self.translation = np.array(translation, dtype=float)

    def act(self, local: Any) -> Any:
        return self.translation + local


class MockData:
    def __init__(self, n_joints: int = 20) -> None:
        self.oMi = [MockPlacement([0.1 * i, 0.0, 1.0]) for i in range(n_joints)]


class MockModel:
    def __init__(self, nq: int, nv: int) -> None:
        self.nq = nq
        self.nv = nv

    def createData(self) -> Any:
        return MockData(max(self.nq + 5, 20))


class MockPin:
    class ReferenceFrame:
        LOCAL_WORLD_ALIGNED = 0

    def computeJointJacobians(self, model: Any, data: Any, q: Any) -> None:
        pass

    def getJointJacobian(self, model: Any, data: Any, joint_id: int, frame: Any) -> Any:
        J = np.zeros((6, model.nv))
        dim = min(3, model.nv)
        J[0:dim, 0:dim] = np.eye(dim)
        return J

    def forwardKinematics(self, model: Any, data: Any, q: Any) -> None:
        for i in range(len(data.oMi)):
            idx = min(i, len(q) - 1)
            data.oMi[i].translation = np.array([float(q[idx]), 0.0, 1.0])


class MockPlant:
    """Mock Pinocchio plant for lightweight unit testing."""

    def __init__(self, coordinate_order: list[str]) -> None:
        self.coordinate_order = list(coordinate_order)
        self.nq = len(coordinate_order)
        self.nv = len(coordinate_order)
        self._coordinates = {name: i for i, name in enumerate(coordinate_order)}
        self._velocity_indices = {name: i for i, name in enumerate(coordinate_order)}
        self._bodies: dict[str, tuple[int, Any]] = {}
        self._frames: dict[str, int] = {}
        self.model = MockModel(self.nq, self.nv)

    def closure_position_linearization(self, q_dict: dict[str, float]) -> Any:
        class ClosureLin:
            position = np.array([0.001 * sum(q_dict.values())] * 6)
            jacobian = 0.001 * np.ones((6, len(q_dict)))

        return ClosureLin()


@pytest.fixture
def sample_coordinates() -> list[str]:
    return [
        "TranslationInputX",
        "TranslationInputY",
        "TranslationInputZ",
        "PelvisInputX",
        "PelvisInputY",
        "PelvisInputZ",
        "LFInput",
        "LWInputX",
        "LWInputY",
        "RFInput",
        "RWInputX",
        "RWInputY",
    ]


def test_solve_diagnostics_dataclass() -> None:
    """SolveDiagnostics captures convergence metrics accurately."""
    diag = SolveDiagnostics(
        iterations=15,
        final_cost=0.0025,
        marker_rms=0.012,
        closure_error_m=0.0015,
        projected_gradient_norm=1e-5,
        active_bounds_count=0,
        converged=True,
    )
    assert diag.converged is True
    assert diag.iterations == 15
    assert diag.projected_gradient_norm == 1e-5
    assert diag.active_bounds_count == 0


def test_kinematic_smoothing_joint_q_v_a_derivative_compatibility() -> None:
    """Joint smoothing computes q, v, a with verified numerical derivative compatibility."""
    n_nodes = 100
    dt = 1.0 / 360.0
    time_s = np.linspace(0.0, (n_nodes - 1) * dt, n_nodes)

    # Pure synthetic sinusoidal trajectory: q = sin(2 * pi * f * t)
    f = 2.0  # 2 Hz motion
    omega = 2.0 * math.pi * f
    q_pure = np.sin(omega * time_s)[:, None]

    # Add high-frequency noise
    np.random.seed(42)
    noise = 0.005 * np.random.randn(*q_pure.shape)
    q_noisy = q_pure + noise

    # Smooth trajectory
    q_smooth, v_smooth, a_smooth = smooth_kinematic_trajectory(
        time_s, q_noisy, cutoff_hz=15.0
    )

    assert q_smooth.shape == (n_nodes, 1)
    assert v_smooth.shape == (n_nodes, 1)
    assert a_smooth.shape == (n_nodes, 1)

    # Verify interior derivative compatibility: v ≈ dq/dt
    dq_dt = np.gradient(q_smooth[:, 0], time_s)
    # Check interior (avoid edge artifacts)
    interior = slice(10, -10)
    np.testing.assert_allclose(v_smooth[interior, 0], dq_dt[interior], atol=0.05)

    # Verify interior derivative compatibility: a ≈ dv/dt
    dv_dt = np.gradient(v_smooth[:, 0], time_s)
    np.testing.assert_allclose(a_smooth[interior, 0], dv_dt[interior], atol=0.5)


def test_boundary_spike_audit_detects_discontinuities() -> None:
    """Boundary spike audit flags excessive boundary jerk or acceleration steps."""
    n_nodes = 50
    dt = 0.01
    time_s = np.linspace(0.0, 0.49, n_nodes)
    q = np.ones((n_nodes, 2))
    v = np.zeros((n_nodes, 2))
    a = np.zeros((n_nodes, 2))

    # Clean signal: no spike
    audit_clean = audit_boundary_spikes(time_s, q, v, a, max_allowed_jerk=500.0)
    assert isinstance(audit_clean, BoundarySpikeAudit)
    assert audit_clean.has_spikes is False

    # Inject terminal boundary acceleration step (spike)
    a_spiked = a.copy()
    a_spiked[-1, 0] = 50.0  # abrupt 50 m/s^2 jump at final frame -> jerk = 5000 m/s^3
    audit_spiked = audit_boundary_spikes(time_s, q, v, a_spiked, max_allowed_jerk=500.0)
    assert audit_spiked.has_spikes is True
    assert (
        "terminal" in audit_spiked.spike_locations
        or audit_spiked.worst_boundary_jerk > 500.0
    )


def test_wrist_model_within_human_ranges() -> None:
    """Wrists within human ranges produce zero range_of_motion violation flags (MM-2, #10104)."""
    coordinate_order = ["LWInputX", "LWInputY", "RWInputX", "RWInputY"]
    # Physiological angles within HUMAN_RANGES_DEG:
    # LWInputX: (-40, 25), LWInputY: (-70, 70)
    q_valid_deg = np.array(
        [
            [-10.0, 15.0, -15.0, 20.0],
            [-20.0, 30.0, -25.0, 35.0],
            [5.0, -10.0, 10.0, -15.0],
        ]
    )
    q_valid_rad = np.radians(q_valid_deg)

    viol = violations(q_valid_rad, coordinate_order, ranges_deg=HUMAN_RANGES_DEG)
    assert len(viol) == 0, (
        f"Expected 0 violations for physiological wrist angles, got {viol}"
    )

    # Excessive wrist cock (e.g. 50 deg > 25 deg upper limit)
    q_invalid_deg = np.array([[50.0, 15.0, -15.0, 20.0]])
    q_invalid_rad = np.radians(q_invalid_deg)
    viol_invalid = violations(
        q_invalid_rad, coordinate_order, ranges_deg=HUMAN_RANGES_DEG
    )
    assert "LWInputX" in viol_invalid
    assert viol_invalid["LWInputX"].max_excess_deg > 20.0


def test_left_elbow_pit_up_and_inward_at_address() -> None:
    """Left elbow pit direction faces up-and-inward at address posture (MM-5, #10107)."""
    # At address for a right-handed golfer:
    # Global coordinates: X forward (toward ball target line), Y right, Z up
    up = np.array([0.0, 0.0, 1.0])
    inward = np.array([0.0, 1.0, 0.0])  # left arm at -Y, inward toward midline is +Y

    # Anatomical left arm markers at address
    l_shoulder = np.array([0.0, -0.20, 1.40])
    l_elbow = np.array([0.15, -0.20, 1.15])
    l_wrist = np.array([0.40, -0.10, 0.95])

    pit = elbow_pit_direction(l_shoulder, l_elbow, l_wrist)
    assert pit is not None, (
        "Elbow pit direction must be observable with flexion > 5 deg"
    )

    dot_up = float(pit @ up)
    dot_inward = float(pit @ inward)

    # Acceptance criterion: dot with up > 0.3, dot with inward > 0.3
    assert dot_up > 0.3, f"Elbow pit up alignment {dot_up:.2f} must be > 0.3"
    assert dot_inward > 0.3, (
        f"Elbow pit inward alignment {dot_inward:.2f} must be > 0.3"
    )


def test_iron_rejects_driver_calibration_provenance() -> None:
    """Iron evaluation rejects mismatched driver calibration provenance."""
    # Mismatched receipt: capture is iron, but attachments_source is driver calibration
    mismatched_receipt = {
        "shared_metrics": {"whole_marker_rmse_m": 0.040},
        "capture": "iron",
        "attachments_source": "docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json",
    }
    verdict = evaluate(mismatched_receipt, horizon=Horizon.G1, capture="iron")
    assert verdict.is_physically_accepted is False
    gate_map = {g.name: g for g in verdict.gates}
    assert "calibration_provenance" in gate_map
    assert gate_map["calibration_provenance"].status == GateStatus.FAILED
    assert (
        "iron capture reused driver calibration"
        in gate_map["calibration_provenance"].reason
    )

    # Matching receipt: iron capture with iron calibration
    matching_receipt = {
        "shared_metrics": {"whole_marker_rmse_m": 0.040},
        "capture": "iron",
        "attachments_source": "docs/development/full_body_models/evidence/ground_support/anthro_iron/full_body_spec_hipcal_scaled.json",
    }
    verdict_matching = evaluate(matching_receipt, horizon=Horizon.G1, capture="iron")
    gate_map_matching = {g.name: g for g in verdict_matching.gates}
    assert "calibration_provenance" in gate_map_matching
    assert gate_map_matching["calibration_provenance"].status == GateStatus.PASSED


def test_audit_cutoff_sensitivity_evaluates_filter_tradeoffs() -> None:
    """audit_cutoff_sensitivity provides tracking RMS, peak acceleration, and boundary jerk."""
    n_nodes = 80
    dt = 1.0 / 360.0
    time_s = np.linspace(0.0, (n_nodes - 1) * dt, n_nodes)
    q = np.sin(2.0 * math.pi * 3.0 * time_s)[:, None]

    report = audit_cutoff_sensitivity(
        time_s, q, cutoffs_hz=(10.0, 15.0, 25.0), max_allowed_jerk=500.0
    )
    assert len(report.cutoffs_hz) == 3
    assert len(report.position_rmse) == 3
    assert len(report.max_accelerations) == 3
    assert len(report.boundary_jerks) == 3
    assert report.recommended_cutoff_hz in (10.0, 15.0, 25.0)


def test_marker_ik_solver_diagnostics_and_convergence(
    sample_coordinates: list[str],
) -> None:
    """MarkerIkSolver returns detailed SolveDiagnostics with projected gradient and bound tracking."""
    plant = MockPlant(sample_coordinates)
    pin = MockPin()
    table = MarkerTable(labels=("m1",), joint_ids=(1,), local_points=np.zeros((1, 3)))
    lower = -np.ones(len(sample_coordinates))
    upper = np.ones(len(sample_coordinates))
    solver = MarkerIkSolver(pin, plant, table, lower, upper)

    target = np.array([[0.1, 0.0, 1.0]])
    valid = np.array([True])
    weights = np.array([1.0])
    q_init = np.zeros(len(sample_coordinates))

    q_sol, rms, closure, diag = solver.solve_frame(
        target, valid, weights, q_init, return_diagnostics=True
    )
    assert isinstance(diag, SolveDiagnostics)
    assert diag.iterations > 0
    assert diag.final_cost >= 0.0
    assert diag.converged is True
    assert solver.last_diagnostics == diag
    assert diag.projected_gradient_norm >= 0.0
    assert isinstance(diag.active_bounds_count, int)


def test_marker_ik_solver_multi_start_and_geometric_floor(
    sample_coordinates: list[str],
) -> None:
    """Multi-start resolution selects lowest cost solution and estimates geometric floor."""
    plant = MockPlant(sample_coordinates)
    pin = MockPin()
    table = MarkerTable(labels=("m1",), joint_ids=(1,), local_points=np.zeros((1, 3)))
    lower = -np.ones(len(sample_coordinates))
    upper = np.ones(len(sample_coordinates))
    solver = MarkerIkSolver(pin, plant, table, lower, upper)

    target = np.array([[0.2, 0.0, 1.0]])
    valid = np.array([True])
    weights = np.array([1.0])
    seeds = [
        np.zeros(len(sample_coordinates)),
        0.1 * np.ones(len(sample_coordinates)),
        -0.1 * np.ones(len(sample_coordinates)),
    ]

    best_q, best_rms, best_closure, best_diag, floor_rms = (
        solver.solve_frame_multi_start(
            target, valid, weights, seeds, estimate_geometric_floor=True
        )
    )
    assert best_q.shape == (len(sample_coordinates),)
    assert best_diag.converged is True
    assert floor_rms <= best_rms + 1e-6, (
        "Geometric floor without closure must be <= constrained RMS"
    )


def test_marker_ik_solver_overlapping_window_refinement(
    sample_coordinates: list[str],
) -> None:
    """refine_overlapping_window smoothly refines joint trajectory across overlapping windows."""
    plant = MockPlant(sample_coordinates)
    pin = MockPin()
    table = MarkerTable(labels=("m1",), joint_ids=(1,), local_points=np.zeros((1, 3)))
    lower = -np.ones(len(sample_coordinates))
    upper = np.ones(len(sample_coordinates))
    solver = MarkerIkSolver(pin, plant, table, lower, upper)

    n_nodes = 8
    target = np.array([[0.1, 0.0, 1.0]])
    targets = np.repeat(target[None, :, :], n_nodes, axis=0)
    valids = np.ones((n_nodes, 1), dtype=bool)
    weights = np.array([1.0])
    q_traj = np.zeros((n_nodes, len(sample_coordinates)))

    q_ref, rms_ref, closure_ref = solver.refine_overlapping_window(
        targets, valids, weights, q_traj, window_size=4, overlap=2
    )
    assert q_ref.shape == (n_nodes, len(sample_coordinates))
    assert rms_ref.shape == (n_nodes,)
    assert closure_ref.shape == (n_nodes,)
    assert np.all(q_ref >= lower) and np.all(q_ref <= upper)
