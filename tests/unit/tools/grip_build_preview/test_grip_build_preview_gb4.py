"""HRG-GB4 grip build preview (#15126)."""

from __future__ import annotations

import math
import os
from dataclasses import replace

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.shared.python.grip_build.builder import build_lead_fingers
from src.shared.python.grip_build.receipt import (
    BuildReceipt,
    BuildStageRecord,
    BuildStatus,
    GripBuildResult,
)
from src.shared.python.grip_spec.types import GripSpec, HandSize
from src.tools.grip_build_preview.frame_plan import sample_preview_frames
from src.tools.grip_build_preview.preview import (
    frame_content_hash,
    generate_preview_rgb_frames,
)
from src.tools.grip_build_preview.settings import DEFAULT_PREVIEW_SETTINGS
from src.tools.grip_build_preview.stage_timeline import build_stage_keyframes


def _neutral(hand_length_m: float = 0.2) -> GripSpec:
    """neutral."""
    return replace(GripSpec.neutral(), hand_size=HandSize(length_m=hand_length_m))


def _gapped_result(spec: GripSpec) -> GripBuildResult:
    """Return a build whose only recorded stage deterministically gaps."""
    stage = BuildStageRecord(
        stage_id="lead_index_mcp",
        finger="index",
        joint="mcp",
        flexion_rad=0.3,
        status=BuildStatus.GAP,
        min_clearance_m=0.012,
        target_tolerance_m=0.0005,
    )
    receipt = BuildReceipt(
        schema_version=1,
        builder_version="test-fixture",
        spec_sha256=spec.spec_sha256,
        curl_order=spec.curl_order,
        handedness=str(spec.handedness),
        bone_tolerance_m=0.0005,
        pad_tolerance_m=0.002,
        stages=(stage,),
        overall_status=BuildStatus.GAP,
    )
    return GripBuildResult(receipt=receipt, joint_angles_rad={}, status=BuildStatus.GAP)


@pytest.mark.unit
def test_preview_frames_follow_stage_order() -> None:
    """test preview frames follow stage order."""
    spec = _neutral()
    built = build_lead_fingers(spec)
    keyframes = build_stage_keyframes(spec, built)
    assert keyframes[0].stage_id == "flat_hand"
    stage_ids = [kf.stage_id for kf in keyframes[1:]]
    assert stage_ids == [s.stage_id for s in built.receipt.stages]
    plan = sample_preview_frames(keyframes, DEFAULT_PREVIEW_SETTINGS)
    labels = [frame.stage_id for frame in plan if frame.keyframe_index >= 0]
    assert labels[0] == "flat_hand"
    assert labels[-1] == built.receipt.stages[-1].stage_id
    assert len(plan) == (
        (len(keyframes) - 1) * DEFAULT_PREVIEW_SETTINGS.frames_per_segment + 1
    )


@pytest.mark.unit
def test_preview_final_frame_equals_built_pose() -> None:
    """test preview final frame equals built pose."""
    spec = _neutral()
    built = build_lead_fingers(spec)
    keyframes = build_stage_keyframes(spec, built)
    plan = sample_preview_frames(keyframes, DEFAULT_PREVIEW_SETTINGS)
    final = plan[-1]
    for finger, joints in built.joint_angles_rad.items():
        for joint, expected in joints.items():
            assert final.joint_angles_rad[finger][joint] == pytest.approx(expected)


@pytest.mark.unit
def test_preview_is_deterministic_headless() -> None:
    """test preview is deterministic headless."""
    spec = _neutral()
    built = build_lead_fingers(spec)
    frames_a = generate_preview_rgb_frames(spec, built, DEFAULT_PREVIEW_SETTINGS)
    frames_b = generate_preview_rgb_frames(spec, built, DEFAULT_PREVIEW_SETTINGS)
    assert len(frames_a) == len(frames_b) > 0
    hashes_a = [frame_content_hash(frame) for frame in frames_a]
    hashes_b = [frame_content_hash(frame) for frame in frames_b]
    assert hashes_a == hashes_b


@pytest.mark.unit
def test_preview_marks_gap_and_infeasible_status() -> None:
    """test preview marks gap and infeasible status."""
    spec = _neutral()
    built = _gapped_result(spec)
    keyframes = build_stage_keyframes(spec, built)
    plan = sample_preview_frames(keyframes, DEFAULT_PREVIEW_SETTINGS)
    non_ok = [frame for frame in plan if frame.status != BuildStatus.OK]
    assert non_ok, "expected at least one frame tagged with non-OK status"
    for frame in non_ok:
        assert frame.status in (
            BuildStatus.GAP,
            BuildStatus.JOINT_LIMIT,
            BuildStatus.INFEASIBLE,
        )
        assert frame.status_color_rgb is not None
    frames = generate_preview_rgb_frames(spec, built, DEFAULT_PREVIEW_SETTINGS)
    status_frame = next(i for i, f in enumerate(plan) if f.status != BuildStatus.OK)
    marker = frames[status_frame][0, 0]
    assert not np.array_equal(marker, frames[0][0, 0])


@pytest.mark.unit
def test_replay_matches_builder_per_stage_poses() -> None:
    """test replay matches builder per stage poses."""
    spec = _neutral()
    built = build_lead_fingers(spec)
    keyframes = build_stage_keyframes(spec, built)
    coupling = dict(spec.lead.joint_coupling_overrides)
    for keyframe in keyframes[1:]:
        record = keyframe.active_stage
        assert record is not None
        finger = record.finger
        joint = record.joint
        if joint == "dip":
            # DIP is derived from PIP via the per-finger Kapandji coupling ratio
            # (2/3 by default), so it never equals the raw DIP-stage flexion.
            # Assert the replayed DIP matches the coupling-derived value instead.
            ratio = coupling.get(finger, 2.0 / 3.0)
            expected_dip = min(
                keyframe.joint_angles_rad[finger]["pip"] * ratio,
                math.radians(80.0),
            )
            assert keyframe.joint_angles_rad[finger]["dip"] == pytest.approx(
                expected_dip, abs=1e-6
            )
            continue
        assert keyframe.joint_angles_rad[finger][joint] == pytest.approx(
            record.flexion_rad, abs=1e-6
        )
