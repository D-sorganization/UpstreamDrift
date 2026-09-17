"""Unit and contract tests for FullBodyForwardModel and dynamics integration (#10130).

Validates:
1. ForwardModel protocol conformance.
2. Exact RPY body Jacobian and closed-form inverse.
3. Machine-precision bidirectional native <-> canonical 41-coordinate state and velocity round-trips.
4. Forward rollout without TourCapture marker data.
5. Deterministic repeat replay.
6. Gate G4 replay audit: single reset, root force detection, grip/contact errors.
7. Actionable error when engine backend is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.shadow_tracker.contracts import (
    ForwardModel,
    ModelCapabilities,
    ReplayAudit,
    RolloutRequest,
    RolloutResult,
)
from src.shared.python.shadow_tracker.forward_model import (
    FullBodyForwardModel,
    canonical_to_native_full_body,
    native_to_canonical_full_body,
    rpy_jacobian_body,
    rpy_jacobian_body_inv,
)
from src.shared.python.motion_matching.full_body_forward_dynamics import (
    ForwardRolloutResult,
    RolloutOptions,
    simulate_full_body_forward,
)

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO_ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


# ---------------------------------------------------------------------------
# Test Helpers & Minimal Mocks
# ---------------------------------------------------------------------------


class MockSkeletalModel:
    """Mock full-body skeletal dynamics model with 41 coordinates."""

    def __init__(self, ground_height: float = 0.0) -> None:
        self.coordinate_order: list[str] = [f"coord_{i}" for i in range(41)]
        self.ground_plane = _MockGround(ground_height)
        self.resets = 0

    def accelerations(
        self, q: dict[str, float], qd: dict[str, float], tau: dict[str, float]
    ) -> dict[str, float]:
        return {
            name: -0.1 * qd[name] + tau.get(name, 0.0) for name in self.coordinate_order
        }

    def closure_errors(self) -> tuple[np.ndarray, np.ndarray]:
        # 1mm translation error, 0.01 rad rotation error
        err_p = np.array([0.001, 0.0, 0.0, 0.01, 0.0, 0.0], dtype=np.float64)
        err_v = np.zeros(6, dtype=np.float64)
        return err_p, err_v

    def evaluate_contact_samples(
        self, q: dict[str, float], qd: dict[str, float]
    ) -> dict[str, Any]:
        return {
            "sphere_heel_r": _MockContactSample(
                normal_force_n=np.array([0.0, 0.0, 450.0]),
                friction_force_n=np.array([20.0, 10.0, 0.0]),
                penetration_m=0.002,
            )
        }


class _MockGround:
    def __init__(self, height_m: float = 0.0) -> None:
        self.normal = (0.0, 0.0, 1.0)
        self.height_m = height_m


class _MockContactSample:
    def __init__(
        self,
        normal_force_n: np.ndarray,
        friction_force_n: np.ndarray,
        penetration_m: float,
    ) -> None:
        self.normal_force_n = normal_force_n
        self.friction_force_n = friction_force_n
        self.penetration_m = penetration_m


# ---------------------------------------------------------------------------
# 1. Protocol Conformance
# ---------------------------------------------------------------------------


def test_forward_model_protocol_conformance() -> None:
    """FullBodyForwardModel must satisfy the runtime_checkable ForwardModel protocol."""
    mock_model = MockSkeletalModel()
    adapter = FullBodyForwardModel(model=mock_model)
    assert isinstance(adapter, ForwardModel)


# ---------------------------------------------------------------------------
# 2. RPY Body Jacobian Tests
# ---------------------------------------------------------------------------


def test_rpy_jacobian_matches_numerical_differentiation() -> None:
    """rpy_jacobian_body must match finite-difference R^T * dR/dt to machine precision."""
    from src.shared.python.pose_interchange.se3 import euler_xyz_deg_to_matrix

    angles = [
        np.array([0.1, 0.2, 0.3]),
        np.array([-0.3, 0.5, -0.2]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.8, -0.4, 0.6]),
    ]
    rates = [
        np.array([1.0, -0.5, 0.2]),
        np.array([-2.0, 1.5, -1.0]),
        np.array([0.5, 0.5, 0.5]),
    ]

    eps = 1e-7
    for rpy in angles:
        for drpy in rates:
            rpy_p = rpy + eps * drpy
            rpy_m = rpy - eps * drpy
            R_p = euler_xyz_deg_to_matrix(np.degrees(rpy_p))
            R_m = euler_xyz_deg_to_matrix(np.degrees(rpy_m))
            R_0 = euler_xyz_deg_to_matrix(np.degrees(rpy))

            dR_dt = (R_p - R_m) / (2.0 * eps)
            # S(omega_body) = R_0^T @ dR_dt
            S = R_0.T @ dR_dt
            omega_body_num = np.array([S[2, 1], S[0, 2], S[1, 0]])

            J_b = rpy_jacobian_body(rpy)
            omega_body_ana = J_b @ drpy

            np.testing.assert_allclose(omega_body_ana, omega_body_num, atol=1e-8)


def test_rpy_jacobian_inverse_exact() -> None:
    """rpy_jacobian_body_inv must be the exact algebraic inverse of rpy_jacobian_body."""
    rpy_samples = [
        np.array([0.2, 0.3, -0.4]),
        np.array([-0.5, -0.6, 0.7]),
        np.array([0.0, 0.0, 0.0]),
    ]
    for rpy in rpy_samples:
        J = rpy_jacobian_body(rpy)
        J_inv = rpy_jacobian_body_inv(rpy)
        product = J @ J_inv
        np.testing.assert_allclose(product, np.eye(3), atol=1e-12)


def test_rpy_jacobian_inverse_gimbal_lock() -> None:
    """rpy_jacobian_body_inv must raise ValueError when pitch is +/- pi/2."""
    gimbal_lock_rpy = np.array([0.1, np.pi / 2.0, 0.2])
    with pytest.raises(ValueError, match="Gimbal lock"):
        rpy_jacobian_body_inv(gimbal_lock_rpy)


# ---------------------------------------------------------------------------
# 3. Canonical <-> Native 41-Coordinate State Round-Trips
# ---------------------------------------------------------------------------


def test_native_to_canonical_and_inverse_roundtrip() -> None:
    """State and velocity must round-trip between 41-dim native and canonical-v2 to < 1e-12."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        q_native = rng.uniform(-1.0, 1.0, size=41)
        # Keep pitch away from gimbal lock (+/- pi/2)
        q_native[4] = np.clip(q_native[4], -1.2, 1.2)
        qd_native = rng.uniform(-5.0, 5.0, size=41)

        q_canon, v_canon = native_to_canonical_full_body(q_native, qd_native)
        assert len(q_canon) == 42  # 3 pos + 4 quat + 35 joints
        assert len(v_canon) == 41  # 3 lin + 3 ang + 35 joints

        # Verify unit quaternion norm
        quat = q_canon[3:7]
        assert pytest.approx(float(np.linalg.norm(quat)), rel=1e-12) == 1.0

        q_rec, qd_rec = canonical_to_native_full_body(q_canon, v_canon)
        np.testing.assert_allclose(q_rec, q_native, atol=1e-12)
        np.testing.assert_allclose(qd_rec, qd_native, atol=1e-12)


def test_state_mapping_rejects_invalid_dimensions() -> None:
    """State mapping functions must reject mismatched or non-finite vectors."""
    with pytest.raises(ValueError, match="shape"):
        native_to_canonical_full_body(np.zeros(40), np.zeros(41))
    with pytest.raises(ValueError, match="shape"):
        native_to_canonical_full_body(np.zeros(41), np.zeros(42))
    with pytest.raises(ValueError, match="finite"):
        q_nan = np.zeros(41)
        q_nan[0] = np.nan
        native_to_canonical_full_body(q_nan, np.zeros(41))


# ---------------------------------------------------------------------------
# 4. Decoupled Rollout Without TourCapture
# ---------------------------------------------------------------------------


def test_simulate_full_body_forward_without_tour_capture() -> None:
    """simulate_full_body_forward must succeed without capture or marker offsets."""
    model = MockSkeletalModel(ground_height=-0.85)
    initial_q = np.zeros(41, dtype=np.float64)
    initial_qd = np.zeros(41, dtype=np.float64)
    time_grid = np.linspace(0.0, 0.05, 6)
    theta = np.zeros((41, 7), dtype=np.float64)

    result = simulate_full_body_forward(
        model=model,
        ik_adapter=None,
        theta=theta,
        time_grid=time_grid,
        initial_state=(initial_q, initial_qd),
        marker_offsets=None,
        capture=None,
        options=RolloutOptions(substeps=2),
    )

    assert isinstance(result, ForwardRolloutResult)
    assert result.status == "success"
    assert result.q.shape == (6, 41)
    assert result.qd.shape == (6, 41)
    assert result.shared_metrics is None
    assert result.predicted_markers_m.shape == (6, 0, 3)
    assert result.max_closure_translation_m == pytest.approx(0.001, rel=1e-5)
    assert result.max_closure_rotation_rad == pytest.approx(0.01, rel=1e-5)
    assert result.contact_audit.max_normal_force_n == pytest.approx(450.0, rel=1e-5)


# ---------------------------------------------------------------------------
# 5. FullBodyForwardModel Rollout & Repeat Replay
# ---------------------------------------------------------------------------


def test_forward_model_capabilities() -> None:
    """capabilities() must declare supported 41 coordinates and valid conventions."""
    model = MockSkeletalModel()
    adapter = FullBodyForwardModel(model=model)
    caps = adapter.capabilities()
    assert isinstance(caps, ModelCapabilities)
    assert len(caps.supported_bodies) == 41
    assert caps.state_convention == "canonical_articulated_v1"
    assert "torque_polynomial_deg6" in caps.actuator_modes
    assert "hunt_crossley_regularized_coulomb" in caps.contact_modes
    assert caps.is_available is True


def test_forward_model_repeat_replay_deterministic() -> None:
    """Two independent rollouts with identical requests must produce identical output."""
    model = MockSkeletalModel()
    adapter = FullBodyForwardModel(model=model)

    q0_native = np.zeros(41, dtype=np.float64)
    qd0_native = np.zeros(41, dtype=np.float64)
    q0_canon, _ = native_to_canonical_full_body(q0_native, qd0_native)

    times = tuple(float(t) for t in np.linspace(0.0, 0.04, 5))
    controls = tuple(tuple(0.0 for _ in range(7)) for _ in range(41))

    req = RolloutRequest(
        initial_state=tuple(q0_canon),
        controls=controls,
        time_points_s=times,
    )

    res1 = adapter.rollout(req)
    res2 = adapter.rollout(req)

    assert isinstance(res1, RolloutResult)
    assert isinstance(res2, RolloutResult)
    assert res1.trajectory == res2.trajectory
    assert res1.realized_controls == res2.realized_controls
    assert (
        res1.audit.max_grip_translation_error_m
        == res2.audit.max_grip_translation_error_m
    )


# ---------------------------------------------------------------------------
# 6. Gate G4 Replay Audit & Detection
# ---------------------------------------------------------------------------


def test_forward_model_detects_undeclared_root_forces() -> None:
    """Rollout must fail physical acceptance if torques are applied to root DOFs 0..5."""
    model = MockSkeletalModel()
    adapter = FullBodyForwardModel(model=model)

    q0_native = np.zeros(41, dtype=np.float64)
    q0_canon, _ = native_to_canonical_full_body(q0_native, np.zeros(41))
    times = tuple(float(t) for t in np.linspace(0.0, 0.04, 5))

    # Apply torque to root coordinate 0 (TranslationInputX)
    bad_controls_list = [[0.0] * 7 for _ in range(41)]
    bad_controls_list[0][0] = 50.0  # 50 Nm undeclared root force!
    controls = tuple(tuple(row) for row in bad_controls_list)

    req = RolloutRequest(
        initial_state=tuple(q0_canon),
        controls=controls,
        time_points_s=times,
    )

    res = adapter.rollout(req)
    # Replay audit must flag as physically rejected
    assert res.audit.is_physically_accepted is False


def test_forward_model_requires_single_initial_reset() -> None:
    """Replay audit must enforce exactly 1 reset invariant for continuous replay."""
    model = MockSkeletalModel()
    adapter = FullBodyForwardModel(model=model)

    q0_canon, _ = native_to_canonical_full_body(np.zeros(41), np.zeros(41))
    times = tuple(float(t) for t in np.linspace(0.0, 0.04, 5))
    controls = tuple(tuple(0.0 for _ in range(7)) for _ in range(41))

    req = RolloutRequest(
        initial_state=tuple(q0_canon),
        controls=controls,
        time_points_s=times,
    )

    res = adapter.rollout(req)
    assert res.audit.reset_count == 1
    assert res.audit.coverage_start_s == 0.0
    assert res.audit.coverage_end_s == 0.04


def test_forward_model_missing_engine_fails_actionably() -> None:
    """Unavailable model backend must raise actionable error upon rollout."""
    adapter = FullBodyForwardModel(model=None)
    assert adapter.capabilities().is_available is False

    q0_canon, _ = native_to_canonical_full_body(np.zeros(41), np.zeros(41))
    req = RolloutRequest(
        initial_state=tuple(q0_canon),
        controls=tuple(tuple(0.0 for _ in range(7)) for _ in range(41)),
        time_points_s=(0.0, 0.01),
    )
    with pytest.raises(RuntimeError, match="No physical engine"):
        adapter.rollout(req)
