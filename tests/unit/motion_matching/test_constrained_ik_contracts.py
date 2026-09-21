"""Unit tests for ConstrainedIK contracts and timing invariants (Packet P2, #10277).

Tests DbC preconditions, array shape validation, caller immutability,
timing scale independence, and structured failure invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from src.shared.python.motion_matching.constrained_ik import (
    FrameRateAudit,
    IKOptions,
    IKTrajectoryRequest,
    IKTrajectoryResult,
)
from src.shared.python.motion_matching.tour_capture_contract import TourCapture


def _make_valid_request_data() -> dict:
    time_s = np.array([0.0, 1.0 / 360.0, 2.0 / 360.0], dtype=np.float64)
    labels = ("HeadTop", "WaistLeft")
    marker_targets = np.zeros((3, 2, 3), dtype=np.float64)
    validity_mask = np.ones((3, 2), dtype=bool)
    initial_q = np.zeros(41, dtype=np.float64)
    return {
        "initial_q": initial_q,
        "time_s": time_s,
        "marker_targets": marker_targets,
        "validity_mask": validity_mask,
        "labels": labels,
        "model_name": "canonical_pinocchio_full_body",
    }


def test_ik_trajectory_request_valid() -> None:
    data = _make_valid_request_data()
    req = IKTrajectoryRequest(**data)
    assert req.num_frames == 3
    assert req.num_markers == 2
    assert req.labels == ("HeadTop", "WaistLeft")
    assert req.model_name == "canonical_pinocchio_full_body"
    assert req.initial_q.shape == (41,)


def test_ik_trajectory_request_immutability() -> None:
    data = _make_valid_request_data()
    req = IKTrajectoryRequest(**data)

    # Modifying caller's original array should not affect request
    data["initial_q"][0] = 999.0
    assert req.initial_q[0] == 0.0

    # Request arrays must be read-only
    with pytest.raises(ValueError):
        req.initial_q[0] = 1.0

    with pytest.raises(ValueError):
        req.marker_targets[0, 0, 0] = 1.0

    with pytest.raises(ValueError):
        req.validity_mask[0, 0] = False


def test_ik_trajectory_request_invalid_time() -> None:
    data = _make_valid_request_data()
    # Does not start at 0
    data["time_s"] = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    with pytest.raises(ValueError, match="Capture time must start at zero"):
        IKTrajectoryRequest(**data)

    # Non-monotonic time
    data["time_s"] = np.array([0.0, 0.2, 0.1], dtype=np.float64)
    with pytest.raises(
        ValueError, match="Capture time must start at zero and increase strictly"
    ):
        IKTrajectoryRequest(**data)

    # Non-finite time
    data["time_s"] = np.array([0.0, np.nan, 0.2], dtype=np.float64)
    with pytest.raises(
        ValueError, match="Capture time must start at zero and increase strictly"
    ):
        IKTrajectoryRequest(**data)


def test_ik_trajectory_request_dimension_mismatch() -> None:
    data = _make_valid_request_data()
    # Mismatch between time_s frames (3) and marker_targets (4)
    data["marker_targets"] = np.zeros((4, 2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="marker_targets frames"):
        IKTrajectoryRequest(**data)

    # Mismatch between marker_targets markers (2) and labels (3)
    data["marker_targets"] = np.zeros((3, 3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="marker_targets markers"):
        IKTrajectoryRequest(**data)

    # Mismatch between validity_mask shape and targets
    data = _make_valid_request_data()
    data["validity_mask"] = np.ones((3, 3), dtype=bool)
    with pytest.raises(ValueError, match="validity_mask shape"):
        IKTrajectoryRequest(**data)


def test_ik_trajectory_request_duplicate_labels() -> None:
    data = _make_valid_request_data()
    data["labels"] = ("HeadTop", "HeadTop")
    with pytest.raises(ValueError, match="labels must be unique"):
        IKTrajectoryRequest(**data)


def test_ik_trajectory_request_from_capture() -> None:
    time_s = np.array([0.0, 1.0 / 360.0], dtype=np.float64)
    labels = ("HeadTop", "WaistLeft")
    points_m = np.zeros((2, 2, 3), dtype=np.float64)
    valid = np.ones((2, 2), dtype=bool)
    capture = TourCapture(time_s=time_s, labels=labels, points_m=points_m, valid=valid)

    initial_q = np.zeros(41, dtype=np.float64)
    req = IKTrajectoryRequest.from_capture(
        capture, initial_q=initial_q, model_name="full_body_pinocchio"
    )
    assert req.num_frames == 2
    assert req.num_markers == 2
    assert req.labels == labels
    assert req.model_name == "full_body_pinocchio"


def test_ik_options_validation() -> None:
    opt = IKOptions(step_mode="physical", solver="quadprog", max_iterations=10)
    assert opt.step_mode == "physical"
    assert opt.max_iterations == 10

    with pytest.raises(ValueError, match="step_mode"):
        IKOptions(step_mode="invalid_mode")

    with pytest.raises(ValueError, match="max_iterations"):
        IKOptions(max_iterations=0)

    with pytest.raises(ValueError, match="damping"):
        IKOptions(damping=-1.0)

    with pytest.raises(ValueError, match="tolerance"):
        IKOptions(tolerance=-1e-4)


def test_frame_rate_audit_timing_scaling() -> None:
    dq = np.array([0.1, -0.2], dtype=np.float64)
    dt1 = 1.0 / 360.0
    dt2 = 1.0 / 180.0

    audit1 = FrameRateAudit.compute(
        frame_index=1,
        dt_s=dt1,
        delta_q=dq,
        velocity_limits=np.array([10.0, 10.0], dtype=np.float64),
        joint_names=("joint1", "joint2"),
    )
    audit2 = FrameRateAudit.compute(
        frame_index=1,
        dt_s=dt2,
        delta_q=dq,
        velocity_limits=np.array([10.0, 10.0], dtype=np.float64),
        joint_names=("joint1", "joint2"),
    )

    np.testing.assert_allclose(audit1.joint_velocities, 2.0 * audit2.joint_velocities)
    assert audit1.dt_s == dt1
    assert audit2.dt_s == dt2


def test_ik_trajectory_result_failure_and_cancellation_semantics() -> None:
    q = np.zeros((3, 41), dtype=np.float64)
    success = np.array([True, False, False], dtype=bool)
    timing = np.ones(3, dtype=np.float64)

    res = IKTrajectoryResult(
        configurations=q,
        frame_success=success,
        frame_residuals=(),
        rate_audits=(),
        timing_ms=timing,
        total_time_ms=3.0,
        backend_name="pink_pinocchio",
        passed=False,
        first_failed_frame=1,
        cancelled=False,
        failure_reasons=("Frame 1 QP infeasible",),
    )
    assert not res.passed
    assert res.first_failed_frame == 1
    assert not res.cancelled

    with pytest.raises(ValueError, match="passed cannot be True if any frame failed"):
        IKTrajectoryResult(
            configurations=q,
            frame_success=success,
            frame_residuals=(),
            rate_audits=(),
            timing_ms=timing,
            total_time_ms=3.0,
            backend_name="pink_pinocchio",
            passed=True,
            first_failed_frame=1,
            cancelled=False,
            failure_reasons=(),
        )

    with pytest.raises(ValueError, match="passed cannot be True if run was cancelled"):
        IKTrajectoryResult(
            configurations=q,
            frame_success=np.ones(3, dtype=bool),
            frame_residuals=(),
            rate_audits=(),
            timing_ms=timing,
            total_time_ms=3.0,
            backend_name="pink_pinocchio",
            passed=True,
            first_failed_frame=None,
            cancelled=True,
            failure_reasons=(),
        )
