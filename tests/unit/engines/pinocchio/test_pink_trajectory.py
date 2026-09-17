"""Unit tests for PinkTrajectoryService (Packet P2, #10277).

Verifies the ConstrainedIKBackend implementation, timing semantics,
cache refresh, dropout recovery, cancellation, and rate audits.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from src.engines.physics_engines.pinocchio.python.pink_trajectory import (
    PinkTrajectoryService,
)
from src.shared.python.motion_matching.constrained_ik import (
    ConstrainedIKBackend,
    IKOptions,
    IKTrajectoryRequest,
    IKTrajectoryResult,
)


def _make_spec() -> dict:
    coords = [f"coord_{i}" for i in range(41)]
    return {
        "schema_version": "full-body-v1",
        "coordinate_order": coords,
        "marker_attachments": {
            "HeadTop": {"body": "head", "offset": [0.0, 0.0, 0.1]},
            "WaistLeft": {"body": "pelvis", "offset": [-0.1, 0.0, 0.0]},
        },
        "closure": {
            "body_a": "hand_left",
            "placement_a": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "body_b": "hand_right",
            "placement_b": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        "bounds": {c: [-3.14, 3.14] for c in coords},
        "velocity_limits": dict.fromkeys(coords, 10.0),
    }


def _make_request(frames: int = 3, dt: float = 1.0 / 360.0) -> IKTrajectoryRequest:
    time_s = np.arange(frames, dtype=np.float64) * dt
    labels = ("HeadTop", "WaistLeft")
    targets = np.zeros((frames, 2, 3), dtype=np.float64)
    validity = np.ones((frames, 2), dtype=bool)
    q0 = np.zeros(41, dtype=np.float64)
    return IKTrajectoryRequest(
        initial_q=q0,
        time_s=time_s,
        marker_targets=targets,
        validity_mask=validity,
        labels=labels,
        model_name="test_spec",
    )


class _MockPlant:
    """Mock kinematic plant providing frame poses and weld closure residuals for tests."""

    def __init__(self, spec: dict[str, Any] | None = None) -> None:
        self.spec = spec or _make_spec()
        self.nq = 41
        self.nv = 41

    def frame_poses(self, coords: dict[str, float]) -> dict[str, np.ndarray]:
        # Return frame positions matching origin, plus coordinate offset if present
        z_offset = coords.get("coord_0", 0.0)
        return {
            "HeadTop": np.array([0.0, 0.0, 0.1 + z_offset]),
            "WaistLeft": np.array([-0.1, 0.0, 0.0]),
        }

    def closure_residuals(self, coords: dict[str, float]) -> tuple[np.ndarray, None]:
        # Zero translation and rotation closure residuals
        return np.zeros(6, dtype=np.float64), None


def _make_mock_service(spec: dict[str, Any] | None = None) -> PinkTrajectoryService:
    """Construct PinkTrajectoryService equipped with a mock plant and mock QP step."""
    s = spec or _make_spec()
    plant = _MockPlant(s)
    service = PinkTrajectoryService(s, model=plant)
    # Mock QP step to simulate successful integration step
    service._execute_qp_step = lambda q, bundle, dt, options: (q.copy(), True, None)  # type: ignore[method-assign]
    return service


def test_service_implements_protocol() -> None:
    spec = _make_spec()
    service = PinkTrajectoryService(spec)
    assert isinstance(service, ConstrainedIKBackend)
    assert service.backend_name == "pink_pinocchio"


def test_solve_trajectory_basic_execution() -> None:
    spec = _make_spec()
    service = _make_mock_service(spec)
    req = _make_request(frames=4)
    opts = IKOptions(step_mode="physical", max_iterations=5)

    result = service.solve_trajectory(req, opts)
    assert isinstance(result, IKTrajectoryResult)
    assert result.configurations.shape == (4, 41)
    assert result.frame_success.shape == (4,)
    assert len(result.frame_residuals) == 4
    assert len(result.rate_audits) == 4
    assert result.timing_ms.shape == (4,)
    assert result.total_time_ms > 0.0
    assert result.backend_name == "pink_pinocchio"


def test_cancellation_at_frame_boundary() -> None:
    spec = _make_spec()
    service = _make_mock_service(spec)

    # Cancel at frame 2
    call_count = 0

    def cancel_after_frame_1() -> bool:
        nonlocal call_count
        call_count += 1
        return call_count >= 2

    req = IKTrajectoryRequest(
        initial_q=np.zeros(41, dtype=np.float64),
        time_s=np.arange(5, dtype=np.float64) * (1.0 / 360.0),
        marker_targets=np.zeros((5, 2, 3), dtype=np.float64),
        validity_mask=np.ones((5, 2), dtype=bool),
        labels=("HeadTop", "WaistLeft"),
        model_name="test_spec",
        cancellation_token=cancel_after_frame_1,
    )
    opts = IKOptions()

    res = service.solve_trajectory(req, opts)
    assert res.cancelled
    assert not res.passed
    assert res.first_failed_frame is not None
    assert res.configurations.shape == (5, 41)  # Preserves complete time grid!


def test_timing_scale_velocity_audit() -> None:
    spec = _make_spec()
    service = _make_mock_service(spec)

    req1 = _make_request(frames=2, dt=1.0 / 360.0)
    req2 = _make_request(frames=2, dt=1.0 / 180.0)

    # In mock, let's verify rate audit dt_s matches request interval
    res1 = service.solve_trajectory(req1, IKOptions())
    res2 = service.solve_trajectory(req2, IKOptions())

    assert res1.rate_audits[1].dt_s == pytest.approx(1.0 / 360.0)
    assert res2.rate_audits[1].dt_s == pytest.approx(1.0 / 180.0)


def test_cache_refresh_matches_fresh_solver() -> None:
    spec = _make_spec()
    service1 = _make_mock_service(spec)
    service2 = _make_mock_service(spec)

    req = _make_request(frames=2)
    opts = IKOptions(step_mode="physical", max_iterations=2)

    # Warm service1 with distant solve
    distant_req = IKTrajectoryRequest(
        initial_q=np.ones(41, dtype=np.float64) * 0.5,
        time_s=np.array([0.0, 1.0 / 360.0], dtype=np.float64),
        marker_targets=np.ones((2, 2, 3), dtype=np.float64),
        validity_mask=np.ones((2, 2), dtype=bool),
        labels=("HeadTop", "WaistLeft"),
        model_name="test_spec",
    )
    service1.solve_trajectory(distant_req, opts)

    # Now solve req on service1 and on fresh service2
    res1 = service1.solve_trajectory(req, opts)
    res2 = service2.solve_trajectory(req, opts)

    np.testing.assert_allclose(res1.configurations, res2.configurations, atol=1e-10)


def test_deterministic_dropout_recovery() -> None:
    spec = _make_spec()
    service = _make_mock_service(spec)

    frames = 5
    # Provide targets matching mock plant frame_poses
    targets = np.zeros((frames, 2, 3), dtype=np.float64)
    targets[:, 0, :] = [0.0, 0.0, 0.1]
    targets[:, 1, :] = [-0.1, 0.0, 0.0]
    validity = np.ones((frames, 2), dtype=bool)
    # Frames 2 and 3 drop out
    validity[2, :] = False
    validity[3, :] = False

    req = IKTrajectoryRequest(
        initial_q=np.zeros(41, dtype=np.float64),
        time_s=np.arange(frames, dtype=np.float64) * (1.0 / 360.0),
        marker_targets=targets,
        validity_mask=validity,
        labels=("HeadTop", "WaistLeft"),
        model_name="test_spec",
    )
    res = service.solve_trajectory(req, IKOptions())
    assert np.isfinite(res.configurations).all()
    # Explicit insufficient data on dropped frames
    assert res.frame_success[0] and res.frame_success[1]
    assert not res.frame_success[2] and not res.frame_success[3]
    # Successfully recovers when markers return on frame 4
    assert res.frame_success[4]
    assert res.first_failed_frame == 2
    assert "Insufficient data: zero observed marker targets" in res.failure_reasons[0]


def test_post_smoothing_constraint_audit() -> None:
    spec = _make_spec()
    service = _make_mock_service(spec)
    req = _make_request(frames=3)

    # Given an arbitrary smoothed trajectory
    smoothed_q = np.zeros((3, 41), dtype=np.float64)
    smoothed_q[1, 0] = 0.05  # slight displacement

    audit_res = service.audit_trajectory(smoothed_q, req)
    assert isinstance(audit_res, IKTrajectoryResult)
    assert audit_res.configurations.shape == (3, 41)
    assert len(audit_res.frame_residuals) == 3
    assert len(audit_res.rate_audits) == 3


def test_missing_runtime_fails_closed() -> None:
    """Missing native Pink/Pinocchio stack or plant model must fail closed."""
    spec = _make_spec()
    # No plant passed and native stack unavailable in standard python env
    service = PinkTrajectoryService(spec)
    req = _make_request(frames=2)
    opts = IKOptions()

    res = service.solve_trajectory(req, opts)
    assert not res.passed
    assert not any(res.frame_success)
    assert res.first_failed_frame == 0
    assert any(
        "Native Pink/Pinocchio stack or plant model is unavailable" in r
        for r in res.failure_reasons
    )


def test_sentinel_exception_reaches_failure_reason() -> None:
    """Sentinel runtime exception in QP solve must be recorded in failure_reasons."""
    spec = _make_spec()
    plant = _MockPlant(spec)
    service = PinkTrajectoryService(spec, model=plant)

    def failing_qp_step(
        q: np.ndarray, bundle: Any, dt: float, options: IKOptions
    ) -> tuple[np.ndarray, bool, str | None]:
        return q.copy(), False, "PinkSolverError: QP problem infeasible at frame step"

    service._execute_qp_step = failing_qp_step  # type: ignore[method-assign]
    req = _make_request(frames=2)
    res = service.solve_trajectory(req, IKOptions())

    assert not res.passed
    assert not res.frame_success[0]
    assert res.first_failed_frame == 0
    assert any(
        "PinkSolverError: QP problem infeasible" in r for r in res.failure_reasons
    )


def test_unevaluated_constraints_fail_qualification() -> None:
    """Unevaluated weld errors must be NaN and cause frame_success to fail."""
    spec = _make_spec()

    # Construct service with mock plant that does NOT provide closure_residuals
    class IncompletePlant:
        def frame_poses(self, coords: dict[str, float]) -> dict[str, np.ndarray]:
            return {
                "HeadTop": np.array([0.0, 0.0, 0.1]),
                "WaistLeft": np.array([-0.1, 0.0, 0.0]),
            }

    service = PinkTrajectoryService(spec, model=IncompletePlant())
    service._execute_qp_step = lambda q, b, dt, opt: (q.copy(), True, None)  # type: ignore[assignment]

    req = _make_request(frames=2)
    res = service.solve_trajectory(req, IKOptions())

    assert not res.passed
    assert not res.frame_success[0]
    # Verify weld translation error is NaN
    assert np.isnan(res.frame_residuals[0].weld_translation_error_m)
    assert any(
        "Weld translation error nan exceeds tolerance" in r for r in res.failure_reasons
    )


def test_rate_limit_violation_fails_qualification() -> None:
    """Exceeding coordinate rate limits must fail overall result passed even if frames converge."""
    spec = _make_spec()
    # Velocity limit is 10.0 rad/s
    service = _make_mock_service(spec)
    req = _make_request(frames=2, dt=1.0 / 360.0)  # dt = 0.002777s

    # Simulate QP step producing a large coordinate jump: delta_q = 1.0 in dt = 1/360 => velocity = 360 rad/s
    def jump_qp_step(
        q: np.ndarray, bundle: Any, dt: float, options: IKOptions
    ) -> tuple[np.ndarray, bool, str | None]:
        q_jump = q.copy()
        q_jump[0] += 1.0  # 360 rad/s > 10 rad/s
        return q_jump, True, None

    service._execute_qp_step = jump_qp_step  # type: ignore[method-assign]
    res = service.solve_trajectory(req, IKOptions())

    assert not res.passed
    assert any("Joint velocity limits exceeded" in r for r in res.failure_reasons)
    assert len(res.rate_audits[1].exceeded_joints) > 0


def test_weld_tolerance_violation_fails_frame() -> None:
    """Weld error exceeding named tolerance must fail frame_success."""
    spec = _make_spec()

    class HighWeldResidualPlant:
        def frame_poses(self, coords: dict[str, float]) -> dict[str, np.ndarray]:
            return {
                "HeadTop": np.array([0.0, 0.0, 0.1]),
                "WaistLeft": np.array([-0.1, 0.0, 0.0]),
            }

        def closure_residuals(
            self, coords: dict[str, float]
        ) -> tuple[np.ndarray, None]:
            # 0.05m weld translation error > default 0.005m tolerance
            return np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64), None

    service = PinkTrajectoryService(spec, model=HighWeldResidualPlant())
    service._execute_qp_step = lambda q, b, dt, opt: (q.copy(), True, None)  # type: ignore[assignment]

    req = _make_request(frames=2)
    opts = IKOptions(weld_translation_tolerance_m=0.005)
    res = service.solve_trajectory(req, opts)

    assert not res.passed
    assert not res.frame_success[0]
    assert any(
        "Weld translation error" in r and "exceeds tolerance" in r
        for r in res.failure_reasons
    )


def test_displaced_target_produces_motion_and_residual_reduction() -> None:
    """A reachable displaced target produces nonzero motion and residual reduction."""
    spec = _make_spec()
    plant = _MockPlant(spec)
    service = PinkTrajectoryService(spec, model=plant)

    step_count = 0

    def moving_qp_step(
        q: np.ndarray, bundle: Any, dt: float, options: IKOptions
    ) -> tuple[np.ndarray, bool, str | None]:
        nonlocal step_count
        q_next = q.copy()
        if step_count > 0:
            q_next[0] = 0.02
        step_count += 1
        return q_next, True, None

    service._execute_qp_step = moving_qp_step  # type: ignore[method-assign]

    targets = np.zeros((2, 2, 3), dtype=np.float64)
    # Frame 0: nominal poses matching origin
    targets[0, 0] = np.array([0.0, 0.0, 0.1])
    targets[0, 1] = np.array([-0.1, 0.0, 0.0])
    # Frame 1: HeadTop displaced to 0.12, WaistLeft unchanged
    targets[1, 0] = np.array([0.0, 0.0, 0.12])
    targets[1, 1] = np.array([-0.1, 0.0, 0.0])

    req = IKTrajectoryRequest(
        initial_q=np.zeros(41, dtype=np.float64),
        time_s=np.array([0.0, 0.01], dtype=np.float64),
        marker_targets=targets,
        validity_mask=np.ones((2, 2), dtype=bool),
        labels=("HeadTop", "WaistLeft"),
        model_name="test_spec",
    )

    res = service.solve_trajectory(req, IKOptions())

    assert res.passed
    assert np.all(res.frame_success)
    displacement = float(np.linalg.norm(res.configurations[1] - res.configurations[0]))
    assert displacement > 0.0
    assert res.configurations[1, 0] == pytest.approx(0.02)
    assert res.frame_residuals[1].marker_errors_m["HeadTop"] == pytest.approx(
        0.0, abs=1e-6
    )


def test_infeasible_hard_constraint_fails_qualification() -> None:
    """Conflicting or infeasible hard constraints fail frame and qualification."""
    spec = _make_spec()
    plant = _MockPlant(spec)
    service = PinkTrajectoryService(spec, model=plant)

    def infeasible_step(
        q: np.ndarray, bundle: Any, dt: float, options: IKOptions
    ) -> tuple[np.ndarray, bool, str | None]:
        return q.copy(), False, "PinkSolverError: Infeasible hard equality constraints"

    service._execute_qp_step = infeasible_step  # type: ignore[method-assign]

    req = _make_request(frames=2)
    res = service.solve_trajectory(req, IKOptions())

    assert not res.passed
    assert not res.frame_success[0]
    assert res.first_failed_frame == 0
    assert any("Infeasible hard equality constraints" in r for r in res.failure_reasons)
