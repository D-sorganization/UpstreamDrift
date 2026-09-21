"""TDD unit tests for full-body forward dynamics matching and contact audit (#10069).

Validates:
1. Degree-6 polynomial torque evaluation for 41 full-body coordinates.
2. Zero torque on unactuated root degrees of freedom.
3. Forward numerical integration with contact forces and weld loop-closure.
4. Contact audit metrics (normal force, friction force, penetration depth, contact duty cycle).
5. Uninterrupted original-state acceptance with the five shared metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.engines.physics_engines.mujoco.python.full_body_ik import (
    MujocoFullBodyIK,
)
from src.shared.python.motion_matching.full_body_forward_dynamics import (
    ContactAuditResult,
    ForwardRolloutResult,
    evaluate_polynomial_torques,
    simulate_full_body_forward,
)
from src.shared.python.motion_matching.tour_capture_contract import (
    load_tour_capture,
)

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO_ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
C3D_PATH = REPO_ROOT / "data/C3D_TA_Driver.c3d"
OFFSETS_PATH = (
    REPO_ROOT
    / "docs/development/full_body_models/evidence/fb4_calibration/mujoco/calibrated_offsets.json"
)


def test_evaluate_polynomial_torques() -> None:
    """Polynomial torque evaluation across 41 coordinates."""
    n_coords = 41
    coeffs_per_joint = 7
    theta = np.zeros((n_coords, coeffs_per_joint), dtype=float)

    coord_names = [f"coord_{i}" for i in range(n_coords)]
    unactuated_indices = {0, 1, 2, 3, 4, 5}

    # Set constant torque 10.0 Nm on joint 6, and linear slope 5.0 on joint 7
    theta[6, 0] = 10.0
    theta[7, 1] = 5.0

    torques = evaluate_polynomial_torques(
        theta=theta,
        t=0.5,
        duration_s=1.0,
        coordinate_names=coord_names,
        unactuated_indices=unactuated_indices,
    )

    assert len(torques) == n_coords
    for i in range(6):
        assert torques[coord_names[i]] == 0.0
    assert torques[coord_names[6]] == 10.0
    assert torques[coord_names[7]] == 2.5  # 5.0 * (0.5 / 1.0)


def test_simulate_full_body_forward_short() -> None:
    """Short forward simulation from t=0 with zero torques."""
    if not SPEC_PATH.is_file() or not OFFSETS_PATH.is_file() or not C3D_PATH.is_file():
        pytest.skip("Required model or evidence files not found")

    spec_bytes = SPEC_PATH.read_bytes()
    model = NativeMujocoFullBodyModel(spec_bytes)
    ik_adapter = MujocoFullBodyIK(spec_bytes.decode("utf-8"))
    capture = load_tour_capture(C3D_PATH)
    offsets_data = json.loads(OFFSETS_PATH.read_text(encoding="utf-8"))
    marker_offsets = offsets_data["marker_offsets"]

    # Initial state at address (q0 from IK frame 0, qd0 = 0)
    ik_traj_path = (
        REPO_ROOT
        / "docs/development/full_body_models/evidence/fb4_calibration/mujoco/ik_trajectory.npz"
    )
    if not ik_traj_path.is_file():
        pytest.skip("IK trajectory not found")

    ik_traj = np.load(ik_traj_path)
    q0 = ik_traj["q"][0]
    qd0 = np.zeros_like(q0)

    # 10 frames (~0.025 s)
    time_grid = capture.time_s[:10]
    theta = np.zeros((41, 7), dtype=float)

    result = simulate_full_body_forward(
        model=model,
        ik_adapter=ik_adapter,
        theta=theta,
        time_grid=time_grid,
        initial_state=(q0, qd0),
        marker_offsets=marker_offsets,
        capture=capture,
    )

    assert isinstance(result, ForwardRolloutResult)
    assert result.status == "success"
    assert result.q.shape == (10, 41)
    assert result.qd.shape == (10, 41)
    assert np.isfinite(result.q).all()
    assert np.isfinite(result.qd).all()

    # Contact audit verification
    assert isinstance(result.contact_audit, ContactAuditResult)
    assert result.contact_audit.max_normal_force_n >= 0.0
    assert result.contact_audit.max_friction_force_n >= 0.0
    assert result.contact_audit.max_penetration_m >= 0.0

    # Five shared metrics verification
    assert result.shared_metrics.whole_marker_rmse_m > 0.0
    assert result.shared_metrics.early_marker_rmse_m > 0.0
    assert result.shared_metrics.terminal_marker_rmse_m > 0.0
    assert result.shared_metrics.club_marker_rmse_m > 0.0
    assert result.shared_metrics.pelvis_yaw_rmse_rad >= 0.0

    # Separated closure units verification
    assert hasattr(result, "max_closure_translation_m")
    assert hasattr(result, "max_closure_rotation_rad")
    assert result.max_closure_translation_m >= 0.0
    assert result.max_closure_rotation_rad is not None
    assert result.max_closure_rotation_rad >= 0.0
    assert result.max_closure_residual_m >= 0.0


def test_closure_units_separation_pure_rotation_small_translation() -> None:
    """Issue #10141: verify distinct displacement (m) and rotation (rad) closure metrics.

    A test scenario with small translation (~1e-5 m) and significant rotation (0.25 rad)
    must expose:
    1. max_closure_translation_m reflecting the ~1e-5 m displacement.
    2. max_closure_rotation_rad reflecting the 0.25 rad orientation error.
    3. max_closure_residual_m preserving the legacy mixed 6D norm.
    4. is_accepted() properly gating physical feasibility on separate thresholds.
    """
    from src.shared.python.motion_matching.full_body_forward_dynamics import (
        RolloutOptions,
    )
    from src.shared.python.motion_matching.tour_capture_contract import TourCapture

    coords = [f"coord_{i}" for i in range(41)]
    # err_p has translation 1e-5 m on X, rotation 0.25 rad on X
    err_vector = np.array([1e-5, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float64)

    class MockClosureModel:
        coordinate_order = coords

        def accelerations(self, q: dict, qd: dict, tau: dict) -> dict[str, float]:
            return dict.fromkeys(coords, 0.0)

        def closure_errors(self) -> tuple[np.ndarray, np.ndarray]:
            return err_vector.copy(), np.zeros(6, dtype=np.float64)

        def evaluate_contact_samples(self, q: dict, qd: dict) -> dict:
            return {}

    class MockIK:
        def pose_fn(self, q: np.ndarray) -> dict:
            return {"Head": (np.eye(3), np.zeros(3))}

    capture = TourCapture(
        time_s=np.array([0.0, 0.01, 0.02]),
        labels=("Head",),
        points_m=np.zeros((3, 1, 3)),
        valid=np.ones((3, 1), dtype=bool),
        source_sha256="test",
    )

    result = simulate_full_body_forward(
        model=MockClosureModel(),
        ik_adapter=MockIK(),
        theta=np.zeros((41, 7)),
        time_grid=capture.time_s,
        initial_state=(np.ones(41), np.zeros(41)),
        marker_offsets={"Head": {"body": "Head", "offset_m": np.zeros(3)}},
        capture=capture,
        options=RolloutOptions(substeps=1),
    )

    assert result.status == "success"
    # Pure rotation vs small translation separation
    assert result.max_closure_translation_m == pytest.approx(1e-5, rel=1e-6)
    assert result.max_closure_rotation_rad == pytest.approx(0.25, rel=1e-6)
    assert result.max_closure_residual_m == pytest.approx(
        np.linalg.norm(err_vector), rel=1e-6
    )

    # Acceptance gating:
    # 1. Translation 1e-5 <= 1e-3 is satisfied, but rotation 0.25 > 0.05 fails
    assert not result.is_accepted(max_translation_tol_m=1e-3, max_rotation_tol_rad=0.05)
    # 2. Both thresholds satisfied
    assert result.is_accepted(max_translation_tol_m=1e-3, max_rotation_tol_rad=0.30)


def test_rollout_acceptance_rejection_of_failed_or_zero_filled() -> None:
    """Issue #10141: failure and zero-filled output must not pass physical acceptance."""
    from src.shared.python.motion_matching.full_body_forward_dynamics import (
        _build_failed_rollout,
    )
    from src.shared.python.motion_matching.tour_capture_contract import TourCapture

    capture = TourCapture(
        time_s=np.array([0.0, 0.01]),
        labels=("Head",),
        points_m=np.zeros((2, 1, 3)),
        valid=np.ones((2, 1), dtype=bool),
        source_sha256="test",
    )
    failed = _build_failed_rollout(
        times=capture.time_s,
        n_coords=41,
        capture=capture,
        marker_offsets={"Head": {"body": "Head", "offset_m": np.zeros(3)}},
    )
    assert failed.status == "failed"
    # Failed rollout must reject even with infinite tolerances
    assert not failed.is_accepted(
        max_translation_tol_m=float("inf"), max_rotation_tol_rad=float("inf")
    )


def _make_valid_rollout(
    *,
    n_frames: int = 3,
    n_coords: int = 41,
    trans_err: float = 1e-4,
    rot_err: float | None = 0.01,
    mixed_err: float = 1e-4,
    status: str = "success",
) -> ForwardRolloutResult:
    """Helper to construct a valid baseline ForwardRolloutResult."""
    from src.shared.python.motion_matching.tour_capture_contract import TourCapture
    from src.shared.python.motion_matching.full_body_forward_dynamics import (
        ContactAuditResult,
        compute_shared_metrics,
    )

    times = np.linspace(0.0, 0.02, n_frames)
    q = np.ones((n_frames, n_coords), dtype=np.float64) * 0.1
    qd = np.zeros((n_frames, n_coords), dtype=np.float64)
    pred_markers = np.ones((n_frames, 1, 3), dtype=np.float64)
    capture = TourCapture(
        time_s=times,
        labels=("Head",),
        points_m=pred_markers.copy(),
        valid=np.ones((n_frames, 1), dtype=bool),
        source_sha256="test",
    )
    shared_metrics = compute_shared_metrics(
        capture=capture,
        predicted_points_m=pred_markers,
        tracked_labels=["Head"],
    )
    audit = ContactAuditResult(
        max_normal_force_n=10.0,
        max_friction_force_n=5.0,
        max_penetration_m=0.001,
        per_sphere_max_force_n={},
        per_sphere_contact_ratio={},
    )
    return ForwardRolloutResult(
        time_s=times,
        q=q,
        qd=qd,
        predicted_markers_m=pred_markers,
        shared_metrics=shared_metrics,
        contact_audit=audit,
        max_closure_translation_m=trans_err,
        max_closure_rotation_rad=rot_err,
        max_closure_residual_m=mixed_err,
        status=status,
    )


def test_rollout_acceptance_rejects_nan_in_states_and_times() -> None:
    """Issue #10166: is_accepted must fail closed on NaN state or time arrays."""
    base = _make_valid_rollout()
    assert base.is_accepted()
    assert hasattr(base, "is_closure_accepted")
    assert base.is_closure_accepted()

    # NaN in q
    q_nan = base.q.copy()
    q_nan[1, 5] = np.nan
    res_nan_q = ForwardRolloutResult(
        time_s=base.time_s,
        q=q_nan,
        qd=base.qd,
        predicted_markers_m=base.predicted_markers_m,
        shared_metrics=base.shared_metrics,
        contact_audit=base.contact_audit,
        max_closure_translation_m=base.max_closure_translation_m,
        max_closure_rotation_rad=base.max_closure_rotation_rad,
        max_closure_residual_m=base.max_closure_residual_m,
        status="success",
    )
    assert not res_nan_q.is_accepted()
    assert not res_nan_q.is_closure_accepted()

    # NaN in qd
    qd_nan = base.qd.copy()
    qd_nan[0, 0] = np.nan
    res_nan_qd = ForwardRolloutResult(
        time_s=base.time_s,
        q=base.q,
        qd=qd_nan,
        predicted_markers_m=base.predicted_markers_m,
        shared_metrics=base.shared_metrics,
        contact_audit=base.contact_audit,
        max_closure_translation_m=base.max_closure_translation_m,
        max_closure_rotation_rad=base.max_closure_rotation_rad,
        max_closure_residual_m=base.max_closure_residual_m,
        status="success",
    )
    assert not res_nan_qd.is_accepted()

    # NaN in predicted_markers_m
    m_nan = base.predicted_markers_m.copy()
    m_nan[2, 0, 1] = np.nan
    res_nan_m = ForwardRolloutResult(
        time_s=base.time_s,
        q=base.q,
        qd=base.qd,
        predicted_markers_m=m_nan,
        shared_metrics=base.shared_metrics,
        contact_audit=base.contact_audit,
        max_closure_translation_m=base.max_closure_translation_m,
        max_closure_rotation_rad=base.max_closure_rotation_rad,
        max_closure_residual_m=base.max_closure_residual_m,
        status="success",
    )
    assert not res_nan_m.is_accepted()

    # NaN in time_s
    t_nan = base.time_s.copy()
    t_nan[1] = np.nan
    res_nan_t = ForwardRolloutResult(
        time_s=t_nan,
        q=base.q,
        qd=base.qd,
        predicted_markers_m=base.predicted_markers_m,
        shared_metrics=base.shared_metrics,
        contact_audit=base.contact_audit,
        max_closure_translation_m=base.max_closure_translation_m,
        max_closure_rotation_rad=base.max_closure_rotation_rad,
        max_closure_residual_m=base.max_closure_residual_m,
        status="success",
    )
    assert not res_nan_t.is_accepted()


def test_rollout_acceptance_rejects_negative_and_infinite_closure_metrics() -> None:
    """Issue #10166: is_accepted must fail closed on negative or infinite closure metrics."""
    base = _make_valid_rollout()
    assert base.is_accepted()

    # Negative translation closure
    res_neg_trans = ForwardRolloutResult(
        time_s=base.time_s,
        q=base.q,
        qd=base.qd,
        predicted_markers_m=base.predicted_markers_m,
        shared_metrics=base.shared_metrics,
        contact_audit=base.contact_audit,
        max_closure_translation_m=-1e-4,
        max_closure_rotation_rad=base.max_closure_rotation_rad,
        max_closure_residual_m=base.max_closure_residual_m,
        status="success",
    )
    assert not res_neg_trans.is_accepted()

    # Negative rotation closure
    res_neg_rot = ForwardRolloutResult(
        time_s=base.time_s,
        q=base.q,
        qd=base.qd,
        predicted_markers_m=base.predicted_markers_m,
        shared_metrics=base.shared_metrics,
        contact_audit=base.contact_audit,
        max_closure_translation_m=base.max_closure_translation_m,
        max_closure_rotation_rad=-0.05,
        max_closure_residual_m=base.max_closure_residual_m,
        status="success",
    )
    assert not res_neg_rot.is_accepted()

    # Infinite closure error with infinite tolerance must reject!
    res_inf = ForwardRolloutResult(
        time_s=base.time_s,
        q=base.q,
        qd=base.qd,
        predicted_markers_m=base.predicted_markers_m,
        shared_metrics=base.shared_metrics,
        contact_audit=base.contact_audit,
        max_closure_translation_m=float("inf"),
        max_closure_rotation_rad=float("inf"),
        max_closure_residual_m=float("inf"),
        status="success",
    )
    assert not res_inf.is_accepted(
        max_translation_tol_m=float("inf"),
        max_rotation_tol_rad=float("inf"),
    )


def test_rollout_acceptance_requires_finite_positive_profile_limits() -> None:
    """Issue #10166: tolerances must be strictly positive and finite."""
    base = _make_valid_rollout()
    assert base.is_accepted(max_translation_tol_m=1e-3, max_rotation_tol_rad=0.05)

    # Infinite tolerance
    assert not base.is_accepted(max_translation_tol_m=float("inf"))
    assert not base.is_accepted(max_rotation_tol_rad=float("inf"))

    # Zero tolerance
    assert not base.is_accepted(max_translation_tol_m=0.0)
    assert not base.is_accepted(max_rotation_tol_rad=0.0)

    # Negative tolerance
    assert not base.is_accepted(max_translation_tol_m=-1e-3)
    assert not base.is_accepted(max_rotation_tol_rad=-0.05)

    # NaN tolerance
    assert not base.is_accepted(max_translation_tol_m=float("nan"))
    assert not base.is_accepted(max_rotation_tol_rad=float("nan"))


def test_rollout_acceptance_rejects_mismatched_shapes_and_non_monotonic_times() -> None:
    """Issue #10166: state shape mismatch or non-strictly-increasing time grid must reject."""
    base = _make_valid_rollout(n_frames=3)

    # q has fewer frames than time_s
    res_short_q = ForwardRolloutResult(
        time_s=base.time_s,
        q=base.q[:2],
        qd=base.qd,
        predicted_markers_m=base.predicted_markers_m,
        shared_metrics=base.shared_metrics,
        contact_audit=base.contact_audit,
        max_closure_translation_m=base.max_closure_translation_m,
        max_closure_rotation_rad=base.max_closure_rotation_rad,
        max_closure_residual_m=base.max_closure_residual_m,
        status="success",
    )
    assert not res_short_q.is_accepted()

    # Non-monotonic time_s
    res_non_mono = ForwardRolloutResult(
        time_s=np.array([0.0, 0.02, 0.01]),
        q=base.q,
        qd=base.qd,
        predicted_markers_m=base.predicted_markers_m,
        shared_metrics=base.shared_metrics,
        contact_audit=base.contact_audit,
        max_closure_translation_m=base.max_closure_translation_m,
        max_closure_rotation_rad=base.max_closure_rotation_rad,
        max_closure_residual_m=base.max_closure_residual_m,
        status="success",
    )
    assert not res_non_mono.is_accepted()


def test_closure_errors_missing_rotation_distinct_from_zero() -> None:
    """Issue #10166: 3-component residual must report rot_err=None, not 0.0."""
    from src.shared.python.motion_matching.full_body_forward_dynamics import (
        _extract_closure_errors,
    )

    class Mock3DModel:
        def closure_errors(self) -> tuple[np.ndarray, np.ndarray]:
            return np.array([0.005, 0.002, 0.001], dtype=np.float64), np.zeros(3)

    model_3d = Mock3DModel()
    trans_err, rot_err, mixed_err = _extract_closure_errors(model_3d)
    assert trans_err == pytest.approx(np.linalg.norm([0.005, 0.002, 0.001]))
    assert rot_err is None  # Must NOT be 0.0!
    assert mixed_err == pytest.approx(trans_err)

    # ForwardRolloutResult with None rotation must fail closed under is_accepted
    base = _make_valid_rollout(rot_err=None)
    assert not base.is_accepted()
    assert not base.is_closure_accepted()


def test_closure_errors_rejects_invalid_residual_shapes() -> None:
    """Issue #10166: exact declared closure capability/shape: reject residuals other than 3 or 6."""
    from src.shared.python.motion_matching.full_body_forward_dynamics import (
        _extract_closure_errors,
    )

    class MockInvalidModel:
        def __init__(self, size: int) -> None:
            self.size = size

        def closure_errors(self) -> tuple[np.ndarray, np.ndarray]:
            return np.ones(self.size, dtype=np.float64), np.zeros(self.size)

    for invalid_size in (1, 2, 4, 5, 7):
        with pytest.raises(ValueError, match="closure residual of size 3 or 6"):
            _extract_closure_errors(MockInvalidModel(invalid_size))


def test_nan_propagation_to_explicit_invalid_status_in_simulation() -> None:
    """Issue #10166: NaNs in accelerations or closure error must yield status='invalid'."""
    from src.shared.python.motion_matching.full_body_forward_dynamics import (
        RolloutOptions,
    )
    from src.shared.python.motion_matching.tour_capture_contract import TourCapture

    coords = [f"coord_{i}" for i in range(41)]

    class MockNaNClosureModel:
        coordinate_order = coords

        def accelerations(self, q: dict, qd: dict, tau: dict) -> dict[str, float]:
            return dict.fromkeys(coords, 0.0)

        def closure_errors(self) -> tuple[np.ndarray, np.ndarray]:
            return np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(6)

        def evaluate_contact_samples(self, q: dict, qd: dict) -> dict:
            return {}

    class MockIK:
        def pose_fn(self, q: np.ndarray) -> dict:
            return {"Head": (np.eye(3), np.zeros(3))}

    capture = TourCapture(
        time_s=np.array([0.0, 0.01, 0.02]),
        labels=("Head",),
        points_m=np.zeros((3, 1, 3)),
        valid=np.ones((3, 1), dtype=bool),
        source_sha256="test",
    )

    result = simulate_full_body_forward(
        model=MockNaNClosureModel(),
        ik_adapter=MockIK(),
        theta=np.zeros((41, 7)),
        time_grid=capture.time_s,
        initial_state=(np.ones(41), np.zeros(41)),
        marker_offsets={"Head": {"body": "Head", "offset_m": np.zeros(3)}},
        capture=capture,
        options=RolloutOptions(substeps=1),
    )

    assert result.status == "invalid"
    assert not result.is_accepted()


def test_simulate_euler_audits_terminal_state() -> None:
    """Issue #10166: audit terminal Euler state as well as intermediate states."""
    from src.shared.python.motion_matching.full_body_forward_dynamics import (
        RolloutOptions,
    )
    from src.shared.python.motion_matching.tour_capture_contract import TourCapture

    coords = [f"coord_{i}" for i in range(41)]
    call_states: list[dict[str, float]] = []

    class MockTerminalAuditModel:
        coordinate_order = coords

        def accelerations(self, q: dict, qd: dict, tau: dict) -> dict[str, float]:
            call_states.append(q.copy())
            # Inject spike on final state
            if len(call_states) >= 3:
                return dict.fromkeys(coords, 0.0)
            return dict.fromkeys(coords, 1.0)

        def closure_errors(self) -> tuple[np.ndarray, np.ndarray]:
            # If on terminal step, spike closure error
            if len(call_states) >= 3:
                return np.array([0.05, 0.0, 0.0, 0.1, 0.0, 0.0]), np.zeros(6)
            return np.array([1e-5, 0.0, 0.0, 0.001, 0.0, 0.0]), np.zeros(6)

        def evaluate_contact_samples(self, q: dict, qd: dict) -> dict:
            return {}

    class MockIK:
        def pose_fn(self, q: np.ndarray) -> dict:
            return {"Head": (np.eye(3), np.zeros(3))}

    capture = TourCapture(
        time_s=np.array([0.0, 0.01, 0.02]),
        labels=("Head",),
        points_m=np.zeros((3, 1, 3)),
        valid=np.ones((3, 1), dtype=bool),
        source_sha256="test",
    )

    result = simulate_full_body_forward(
        model=MockTerminalAuditModel(),
        ik_adapter=MockIK(),
        theta=np.zeros((41, 7)),
        time_grid=capture.time_s,
        initial_state=(np.ones(41), np.zeros(41)),
        marker_offsets={"Head": {"body": "Head", "offset_m": np.zeros(3)}},
        capture=capture,
        options=RolloutOptions(substeps=1),
    )

    assert result.status == "success"
    # Terminal closure spike of 0.05 m must be audited and captured in max_closure_translation_m
    assert result.max_closure_translation_m == pytest.approx(0.05, rel=1e-5)
    assert result.max_closure_rotation_rad == pytest.approx(0.1, rel=1e-5)
