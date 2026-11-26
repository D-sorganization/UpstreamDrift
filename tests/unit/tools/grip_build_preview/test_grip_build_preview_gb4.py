"""HRG-GB4 grip build preview (#15126)."""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.shared.python.grip_build.builder import build_lead_fingers
from src.shared.python.grip_build.receipt import BuildStatus
from src.shared.python.grip_spec.types import GripSpec, HandSize
from src.tools.grip_build_preview.frame_plan import sample_preview_frames
from src.tools.grip_build_preview.preview import (
    frame_content_hash,
    generate_preview_rgb_frames,
)
from src.tools.grip_build_preview.settings import DEFAULT_PREVIEW_SETTINGS
from src.tools.grip_build_preview.stage_timeline import build_stage_keyframes


def _neutral(hand_length_m: float = 0.2) -> GripSpec:
    return replace(GripSpec.neutral(), hand_size=HandSize(length_m=hand_length_m))


@pytest.mark.unit
def test_preview_frames_follow_stage_order() -> None:
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
    spec = _neutral()
    built = build_lead_fingers(spec)
    frames_a = generate_preview_rgb_frames(spec, built, DEFAULT_PREVIEW_SETTINGS)
    frames_b = generate_preview_rgb_frames(spec, built, DEFAULT_PREVIEW_SETTINGS)
    assert len(frames_a) == len(frames_b) > 0
    hashes_a = [frame_content_hash(frame) for frame in frames_a]
    hashes_b = [frame_content_hash(frame) for frame in frames_b]
    assert hashes_a == hashes_b
    digest = hashlib.sha256("".join(hashes_a).encode()).hexdigest()
    assert len(digest) == 64


@pytest.mark.unit
def test_preview_marks_gap_and_infeasible_status() -> None:
    spec = _neutral(hand_length_m=0.14)
    built = build_lead_fingers(spec)
    keyframes = build_stage_keyframes(spec, built)
    plan = sample_preview_frames(keyframes, DEFAULT_PREVIEW_SETTINGS)
    non_ok = [frame for frame in plan if frame.status != BuildStatus.OK]
    if built.status == BuildStatus.OK:
        pytest.skip("neutral large hand produced OK build; gap case not triggered")
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
    spec = _neutral()
    built = build_lead_fingers(spec)
    keyframes = build_stage_keyframes(spec, built)
    for keyframe in keyframes[1:]:
        record = keyframe.active_stage
        assert record is not None
        finger = record.finger
        joint = record.joint
        if joint == "dip":
            continue
        assert keyframe.joint_angles_rad[finger][joint] == pytest.approx(
            record.flexion_rad, abs=1e-6
        )
