"""Skeleton selection regressions for the motion-pipeline orchestrator."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointStateFrame,
    JointTrajectory,
    MotionMatchingResult,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.orchestrator import (
    AdapterOverride,
    InvalidInputError,
    MotionPipeline,
    PipelineConfig,
    StageResult,
)

pytestmark = pytest.mark.unit


def _skeleton() -> SkeletonRig:
    return SkeletonRig(
        id="test-rig",
        joints={
            "root": JointDef(
                name="root",
                parent=None,
                children=[],
                tpose_offset=[0.0, 0.0, 0.0],
                axes=["X"],
            )
        },
        root_joint="root",
    )


def _joint_trajectory(skeleton: SkeletonRig) -> JointTrajectory:
    return JointTrajectory(
        id="joint-traj",
        skeleton=skeleton,
        frames=[JointStateFrame(timestamp=0.0, q=[0.0])],
    )


def _pipeline() -> MotionPipeline:
    return MotionPipeline(
        PipelineConfig(
            adapter=AdapterOverride(format="passthrough"),
            ik_backend="geometric",
            matching_backend="mujoco",
        )
    )


def test_run_uses_payload_skeleton_without_calling_default() -> None:
    """Stage data with ``.skeleton`` must pass that rig into scaling directly."""
    pipeline = _pipeline()
    skeleton = _skeleton()
    payload = SimpleNamespace(skeleton=skeleton)
    seen: dict[str, Any] = {}

    pipeline._run_adapter = lambda source: StageResult(True, payload, {})  # type: ignore[method-assign]
    pipeline._run_preprocessing = lambda data: StageResult(True, data, {})  # type: ignore[method-assign]

    def fail_default() -> SkeletonRig:
        raise AssertionError("_get_default_skeleton must not be called")

    def run_scaling(data: Any, rig: SkeletonRig) -> StageResult:
        seen["data"] = data
        seen["skeleton"] = rig
        return StageResult(True, (data, rig), {})

    def run_ik(data: Any, rig: SkeletonRig) -> StageResult:
        return StageResult(True, _joint_trajectory(rig), {})

    def run_matching(trajectory: Any, rig: SkeletonRig) -> StageResult:
        return StageResult(
            True,
            MotionMatchingResult(
                request_id="request",
                success=True,
                matched_trajectory=trajectory,
            ),
            {},
        )

    pipeline._get_default_skeleton = fail_default  # type: ignore[method-assign]
    pipeline._run_scaling = run_scaling  # type: ignore[method-assign]
    pipeline._run_inverse_kinematics = run_ik  # type: ignore[method-assign]
    pipeline._run_motion_matching = run_matching  # type: ignore[method-assign]

    result = pipeline.run(payload)  # type: ignore[arg-type]

    assert result.success is True
    assert seen == {"data": payload, "skeleton": skeleton}


def test_run_without_payload_skeleton_raises_invalid_input() -> None:
    """Missing skeletons are caller input errors, not accidental runtime faults."""
    pipeline = _pipeline()
    payload = SimpleNamespace()

    pipeline._run_adapter = lambda source: StageResult(True, payload, {})  # type: ignore[method-assign]
    pipeline._run_preprocessing = lambda data: StageResult(True, data, {})  # type: ignore[method-assign]

    with pytest.raises(InvalidInputError, match="No skeleton provided"):
        pipeline.run(payload)  # type: ignore[arg-type]
