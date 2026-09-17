"""Unit tests for PinkTrajectoryService (Packet P2, #10277).

Verifies the ConstrainedIKBackend implementation, timing semantics,
cache refresh, dropout recovery, cancellation, and rate audits.
"""

from __future__ import annotations

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


def test_service_implements_protocol() -> None:
    spec = _make_spec()
    service = PinkTrajectoryService(spec)
    assert isinstance(service, ConstrainedIKBackend)
    assert service.backend_name == "pink_pinocchio"


def test_solve_trajectory_basic_execution() -> None:
    spec = _make_spec()
    service = PinkTrajectoryService(spec)
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
    service = PinkTrajectoryService(spec)

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
    service = PinkTrajectoryService(spec)

    req1 = _make_request(frames=2, dt=1.0 / 360.0)
    req2 = _make_request(frames=2, dt=1.0 / 180.0)

    # In mock, let's verify rate audit dt_s matches request interval
    res1 = service.solve_trajectory(req1, IKOptions())
    res2 = service.solve_trajectory(req2, IKOptions())

    assert res1.rate_audits[1].dt_s == pytest.approx(1.0 / 360.0)
    assert res2.rate_audits[1].dt_s == pytest.approx(1.0 / 180.0)


def test_cache_refresh_matches_fresh_solver() -> None:
    spec = _make_spec()
    service1 = PinkTrajectoryService(spec)
    service2 = PinkTrajectoryService(spec)

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
    service = PinkTrajectoryService(spec)

    frames = 5
    targets = np.zeros((frames, 2, 3), dtype=np.float64)
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
    service = PinkTrajectoryService(spec)
    req = _make_request(frames=3)

    # Given an arbitrary smoothed trajectory
    smoothed_q = np.zeros((3, 41), dtype=np.float64)
    smoothed_q[1, 0] = 0.05  # slight displacement

    audit_res = service.audit_trajectory(smoothed_q, req)
    assert isinstance(audit_res, IKTrajectoryResult)
    assert audit_res.configurations.shape == (3, 41)
    assert len(audit_res.frame_residuals) == 3
    assert len(audit_res.rate_audits) == 3
