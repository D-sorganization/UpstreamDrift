"""Regression tests for the @invariant + HookPayload architectural bug fixes.

Guards two architectural bugs surfaced by Wave X (PR #4645):

* #4647 — ``contracts.py`` previously used the runtime assertion helper
  ``invariant()`` as a class/method decorator, which raised ``TypeError`` at
  import time on Python 3.13. Each broken site has been replaced with a
  proper Pydantic v2 ``@model_validator(mode="after")``.
* #4650 — ``orchestrator.HookPayload`` was defined as ``typing.Protocol`` and
  then instantiated in ``_fire_hooks``. It is now a Pydantic ``BaseModel``.

These tests exist to keep that fix in place — if anyone reverts to the broken
shape, these tests fail loudly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointLimit,
    JointStateFrame,
    Keypoint,
    KeypointFrame,
)
from src.shared.python.motion_pipeline.orchestrator import HookPayload, Stage


# =============================================================================
# #4647 — invariants enforced via Pydantic model_validator
# =============================================================================


class TestKeypointFrameDepthInvariant:
    """KeypointFrame must reject mixed 2D/3D keypoints."""

    def test_all_2d_accepted(self) -> None:
        frame = KeypointFrame(
            timestamp=0.0,
            keypoints=[Keypoint(x=1.0, y=2.0), Keypoint(x=3.0, y=4.0)],
            schema_name="custom",
        )
        assert len(frame.keypoints) == 2

    def test_all_3d_accepted(self) -> None:
        frame = KeypointFrame(
            timestamp=0.0,
            keypoints=[
                Keypoint(x=1.0, y=2.0, z=3.0),
                Keypoint(x=4.0, y=5.0, z=6.0),
            ],
            schema_name="custom",
        )
        assert all(kp.z is not None for kp in frame.keypoints)

    def test_mixed_depth_rejected(self) -> None:
        with pytest.raises(ValidationError, match="keypoints_have_consistent_depth"):
            KeypointFrame(
                timestamp=0.0,
                keypoints=[
                    Keypoint(x=1.0, y=2.0),  # 2D
                    Keypoint(x=3.0, y=4.0, z=5.0),  # 3D
                ],
                schema_name="custom",
            )


class TestJointDefAxesLimitsInvariant:
    """JointDef enforces axes/limits length match when limits are provided."""

    def test_no_limits_accepted(self) -> None:
        joint = JointDef(name="hip", axes=["X", "Y", "Z"])
        assert joint.limits == []

    def test_matching_axes_and_limits_accepted(self) -> None:
        joint = JointDef(
            name="knee",
            axes=["X"],
            limits=[JointLimit(lower=-1.0, upper=1.0)],
        )
        assert len(joint.axes) == len(joint.limits)

    def test_mismatched_axes_and_limits_rejected(self) -> None:
        with pytest.raises(ValidationError, match="axes_match_limits"):
            JointDef(
                name="bad",
                axes=["X", "Y", "Z"],
                limits=[JointLimit(lower=-1.0, upper=1.0)],  # only 1 limit, 3 axes
            )


class TestJointStateFrameDimensionsInvariant:
    """JointStateFrame enforces q/qdot/qddot length agreement."""

    def test_only_q_accepted(self) -> None:
        frame = JointStateFrame(timestamp=0.0, q=[0.1, 0.2, 0.3])
        assert frame.num_dofs == 3

    def test_matching_lengths_accepted(self) -> None:
        frame = JointStateFrame(
            timestamp=0.0,
            q=[0.0, 1.0],
            qdot=[0.0, 0.0],
            qddot=[0.0, 0.0],
        )
        assert frame.num_dofs == 2

    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValidationError, match="matching_dimensions"):
            JointStateFrame(
                timestamp=0.0,
                q=[0.0, 1.0, 2.0],
                qdot=[0.0, 0.0],  # length mismatch
            )


# =============================================================================
# #4650 — HookPayload is a Pydantic BaseModel
# =============================================================================


class TestHookPayloadConstruction:
    """HookPayload(stage=..., data=...) must construct + serialize cleanly."""

    def test_invariant_and_hookpayload_fixes_basic_construction(self) -> None:
        payload = HookPayload(stage=Stage.ADAPTER, data={"key": "value"})
        assert payload.stage is Stage.ADAPTER
        assert payload.data == {"key": "value"}
        assert payload.metadata == {}

    def test_full_construction_with_metadata(self) -> None:
        payload = HookPayload(
            stage=Stage.MOTION_MATCHING,
            data=None,
            metadata={"iterations": 42, "rmse": 0.001},
        )
        assert payload.metadata["iterations"] == 42

    def test_round_trip_through_model_dump_json(self) -> None:
        payload = HookPayload(stage=Stage.PREPROCESSING, data={"frames": 120})
        as_json = payload.model_dump_json()
        assert "preprocessing" in as_json
        rebuilt = HookPayload.model_validate_json(as_json)
        assert rebuilt.stage is Stage.PREPROCESSING
        assert rebuilt.data == {"frames": 120}

    @pytest.mark.parametrize("stage", list(Stage))
    def test_constructs_for_every_stage(self, stage: Stage) -> None:
        payload = HookPayload(stage=stage, data="anything")
        assert payload.stage is stage
