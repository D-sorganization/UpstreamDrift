"""Edge-case coverage for OpenPose / MediaPipe providers.

Targets the missing branches reported by ``coverage.py``:
out-of-range frame index, hip-only-one-side, low-confidence variants.

Test-only; no production code changes (issue #4673).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.starting_pose_matcher.providers.mediapipe import (
    MediaPipeProvider,
    MediaPipeProviderError,
)
from src.tools.starting_pose_matcher.providers.openpose import (
    OpenPoseProvider,
    OpenPoseProviderError,
)


pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# OpenPose                                                                    #
# --------------------------------------------------------------------------- #


def _openpose_fixture(*, conf=0.9):
    """18 keypoints * (x, y, conf) values.  Low conf to stress the
    is_observed=False branches when desired."""
    flat = []
    for i in range(18):
        flat.extend([float(100 + i), float(100 + i), conf])
    return {"people": [{"pose_keypoints_2d": flat}]}


def test_openpose_loads_from_file(tmp_path: Path):
    import json

    fix = _openpose_fixture()
    p = tmp_path / "kp.json"
    p.write_text(json.dumps(fix))
    provider = OpenPoseProvider(json_path=str(p))
    assert len(provider.frames) == 1


def test_openpose_get_skeleton_out_of_range_raises():
    provider = OpenPoseProvider(json_data=_openpose_fixture())
    with pytest.raises(OpenPoseProviderError, match="out of range"):
        provider.get_skeleton(frame_index=99)


def test_openpose_confidence_map_out_of_range_raises():
    provider = OpenPoseProvider(json_data=_openpose_fixture())
    with pytest.raises(OpenPoseProviderError, match="out of range"):
        provider.get_confidence_map(frame_index=99)


def test_openpose_skips_person_with_no_keypoints():
    fix = {
        "people": [
            {"pose_keypoints_2d": []},
            {
                "pose_keypoints_2d": _openpose_fixture()["people"][0][
                    "pose_keypoints_2d"
                ]
            },
        ]
    }
    provider = OpenPoseProvider(json_data=fix)
    # Empty person was skipped; only the second is recorded.
    assert len(provider.frames) == 1


def test_openpose_only_left_hip_observed():
    """Right-hip absent (low conf) -> left-hip alone is used as the hip."""
    flat = []
    for i in range(18):
        # OpenPose index 8 = right_hip, 11 = left_hip
        if i == 8:
            flat.extend([0.0, 0.0, 0.0])  # right_hip is low conf
        else:
            flat.extend([float(100 + i), float(100 + i), 0.9])
    fix = {"people": [{"pose_keypoints_2d": flat}]}
    provider = OpenPoseProvider(json_data=fix)
    skel = provider.get_skeleton()
    assert "hip" in skel  # left-hip-only path


def test_openpose_only_right_hip_observed():
    """Left-hip absent -> right_hip falls through to the unprocessed-side
    branch (skipped because we 'continue' when both not present)."""
    flat = []
    for i in range(18):
        if i == 11:  # left_hip
            flat.extend([0.0, 0.0, 0.0])
        else:
            flat.extend([float(100 + i), float(100 + i), 0.9])
    fix = {"people": [{"pose_keypoints_2d": flat}]}
    provider = OpenPoseProvider(json_data=fix)
    skel = provider.get_skeleton()
    # Right-hip falls through and assigns hip from itself.
    assert "hip" in skel


# --------------------------------------------------------------------------- #
# MediaPipe                                                                   #
# --------------------------------------------------------------------------- #


class _MockLandmark:
    def __init__(self, x=0.0, y=0.0, z=0.0, visibility=0.9, presence=0.9):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence


def test_mediapipe_get_skeleton_out_of_range_raises():
    landmarks = [_MockLandmark() for _ in range(33)]
    p = MediaPipeProvider(landmarks_data=[landmarks])
    with pytest.raises(MediaPipeProviderError, match="out of range"):
        p.get_skeleton(frame_index=99)


def test_mediapipe_visibility_map_out_of_range_raises():
    landmarks = [_MockLandmark() for _ in range(33)]
    p = MediaPipeProvider(landmarks_data=[landmarks])
    with pytest.raises(MediaPipeProviderError, match="out of range"):
        p.get_visibility_map(frame_index=99)


def test_mediapipe_only_left_hip_observed():
    landmarks = [_MockLandmark() for _ in range(33)]
    # Index 24 = right_hip
    landmarks[24] = _MockLandmark(visibility=0.0, presence=0.0)
    p = MediaPipeProvider(landmarks_data=[landmarks])
    skel = p.get_skeleton()
    assert "hip" in skel


def test_mediapipe_only_right_hip_observed():
    landmarks = [_MockLandmark() for _ in range(33)]
    # Index 23 = left_hip
    landmarks[23] = _MockLandmark(visibility=0.0, presence=0.0)
    p = MediaPipeProvider(landmarks_data=[landmarks])
    skel = p.get_skeleton()
    assert "hip" in skel  # right-hip path


def test_mediapipe_skipped_landmark_index_outside_table():
    """Landmark indices above 32 don't appear in MEDIAPIPE_POSE_LANDMARKS;
    the parser silently skips them."""
    extras = [_MockLandmark() for _ in range(40)]  # 7 extras with no name
    p = MediaPipeProvider(landmarks_data=[extras])
    assert len(p.frames[0].landmarks) == 33  # only the named ones survive
