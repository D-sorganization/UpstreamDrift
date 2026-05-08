"""Tests for the residual report produced by base solvers."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    JointStateFrame,
    JointTrajectory,
)
from src.shared.python.motion_pipeline.matching.base import (
    BaseMotionMatchingSolver,
    MotionMatchingResult,
)

from ._local_fixtures import make_pendulum_reference_trajectory, make_simple_rig


class _StubSolver(BaseMotionMatchingSolver):
    def match(self, reference, rig, request=None):  # type: ignore[no-untyped-def]
        return MotionMatchingResult(request_id="x", success=True)


def test_residual_report_has_required_fields() -> None:
    s = _StubSolver()
    ref = make_pendulum_reference_trajectory(num_frames=10)
    report = s._compute_residual_report(ref, ref)
    for key in ("mean_residual", "max_residual", "std_residual", "num_frames"):
        assert key in report


def test_residual_report_aggregate_non_negative() -> None:
    s = _StubSolver()
    ref = make_pendulum_reference_trajectory(num_frames=5)
    # Slightly perturbed copy
    perturbed_frames = [
        JointStateFrame(
            timestamp=f.timestamp,
            q=[v + 0.01 for v in f.q],
            frame_index=f.frame_index,
        )
        for f in ref.frames
    ]
    perturbed = JointTrajectory(
        id="pert", skeleton=ref.skeleton, frames=perturbed_frames
    )
    report = s._compute_residual_report(ref, perturbed)
    assert report["mean_residual"] >= 0.0
    assert report["max_residual"] >= report["mean_residual"]
    assert report["std_residual"] >= 0.0


def test_residual_report_num_frames_matches_reference() -> None:
    s = _StubSolver()
    ref = make_pendulum_reference_trajectory(num_frames=12)
    report = s._compute_residual_report(ref, ref)
    assert report["num_frames"] == 12


def test_residual_report_keys_are_strings() -> None:
    """Per-joint dict keys (when present) ⊆ rig joint_names."""
    rig = make_simple_rig(num_joints=2)
    joint_names = set(rig.joints.keys())
    # The current report doesn't have per-joint keys, but if any are added
    # they must be within rig.joints.
    s = _StubSolver()
    ref = make_pendulum_reference_trajectory(num_frames=5)
    report = s._compute_residual_report(ref, ref)
    extra_keys = set(report) - {
        "mean_residual",
        "max_residual",
        "std_residual",
        "num_frames",
    }
    for k in extra_keys:
        # If implementations add per-joint stats, they must be valid
        # joint names. Currently empty, so loop does not execute.
        assert k in joint_names
