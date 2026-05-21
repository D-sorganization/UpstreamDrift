"""Tests for the observed-input extractors (mediapipe, openpose).

These are pure-Python landmark / keypoint mappers; no Qt or physics-engine
dependencies. They use numpy lazily.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.tools.starting_pose_matcher.skeleton_extractors import mediapipe as mp_mod
from src.tools.starting_pose_matcher.skeleton_extractors import openpose as op_mod

# ===========================================================================
# OpenPose
# ===========================================================================


def _op_kp(x=1.0, y=2.0, conf=0.9):
    return [x, y, conf]


def _make_op_data(keypoint_map):
    """keypoint_map: {idx: (x, y, conf)} -> openpose-style JSON."""
    arr = [0.0] * (3 * 18)
    for idx, (x, y, c) in keypoint_map.items():
        arr[idx * 3] = x
        arr[idx * 3 + 1] = y
        arr[idx * 3 + 2] = c
    return {"people": [{"pose_keypoints_2d": arr}]}


def test_openpose_provider_requires_input():
    with pytest.raises(op_mod.OpenPoseProviderError):
        op_mod.OpenPoseProvider()


def test_openpose_provider_with_json_data():
    data = _make_op_data({op_mod.OPENPOSE_COCO_INDICES["left_shoulder"]: (1, 2, 0.9)})
    p = op_mod.OpenPoseProvider(json_data=data)
    assert len(p.frames) == 1


def test_openpose_loads_from_file(tmp_path: Path):
    data = _make_op_data({op_mod.OPENPOSE_COCO_INDICES["left_shoulder"]: (1, 2, 0.9)})
    path = tmp_path / "op.json"
    path.write_text(json.dumps(data))
    p = op_mod.OpenPoseProvider(json_path=str(path))
    skel = p.get_skeleton()
    assert "ls" in skel


def test_openpose_get_skeleton_maps_observed_landmarks():
    indices = op_mod.OPENPOSE_COCO_INDICES
    data = _make_op_data(
        {
            indices["left_shoulder"]: (1, 1, 0.9),
            indices["right_shoulder"]: (-1, 1, 0.9),
            indices["left_wrist"]: (1, 3, 0.9),
            indices["right_wrist"]: (-1, 3, 0.9),
            indices["left_hip"]: (0.5, 0, 0.9),
            indices["right_hip"]: (-0.5, 0, 0.9),
            indices["neck"]: (0, 2, 0.9),
        }
    )
    p = op_mod.OpenPoseProvider(json_data=data, confidence_threshold=0.5)
    skel = p.get_skeleton()
    assert "ls" in skel and "rs" in skel
    # mp synthesised from lw/rw
    np.testing.assert_allclose(skel["mp"], (skel["lw"] + skel["rw"]) / 2)
    # torso synthesised from shoulders
    np.testing.assert_allclose(skel["torso"], (skel["ls"] + skel["rs"]) / 2)
    # hub from spine+torso
    assert "hub" in skel
    # hip averages left+right
    np.testing.assert_allclose(skel["hip"], [0.0, 0.0, 0.0], atol=1e-9)


def test_openpose_confidence_threshold_excludes_low_conf():
    indices = op_mod.OPENPOSE_COCO_INDICES
    data = _make_op_data({indices["left_shoulder"]: (1, 1, 0.1)})
    p = op_mod.OpenPoseProvider(json_data=data, confidence_threshold=0.5)
    assert "ls" not in p.get_skeleton()


def test_openpose_get_skeleton_only_left_hip_when_right_missing():
    indices = op_mod.OPENPOSE_COCO_INDICES
    data = _make_op_data({indices["left_hip"]: (0.5, 0, 0.9)})
    p = op_mod.OpenPoseProvider(json_data=data, confidence_threshold=0.5)
    skel = p.get_skeleton()
    np.testing.assert_allclose(skel["hip"], [0.5, 0, 0])


def test_openpose_frame_index_out_of_range():
    data = _make_op_data({})
    p = op_mod.OpenPoseProvider(json_data=data)
    with pytest.raises(op_mod.OpenPoseProviderError):
        p.get_skeleton(frame_index=99)
    with pytest.raises(op_mod.OpenPoseProviderError):
        p.get_confidence_map(frame_index=99)


def test_openpose_confidence_map_and_missing():
    indices = op_mod.OPENPOSE_COCO_INDICES
    data = _make_op_data(
        {
            indices["left_shoulder"]: (1, 1, 0.8),
            indices["right_shoulder"]: (-1, 1, 0.7),
        }
    )
    p = op_mod.OpenPoseProvider(json_data=data, confidence_threshold=0.5)
    cmap = p.get_confidence_map()
    assert "ls" in cmap
    missing = p.get_missing_keypoints()
    assert "ch" in missing  # never observable


def test_openpose_empty_people_yields_no_frames():
    p = op_mod.OpenPoseProvider(json_data={"people": []})
    assert p.frames == []


def test_openpose_skips_empty_keypoints():
    p = op_mod.OpenPoseProvider(
        json_data={
            "people": [{"pose_keypoints_2d": []}, {"pose_keypoints_2d": [0, 0, 0]}]
        }
    )
    # First person skipped (empty array)
    assert len(p.frames) == 1


def test_openpose_create_provider_factory():
    p = op_mod.create_provider(json_data={"people": []})
    assert isinstance(p, op_mod.OpenPoseProvider)


def test_openpose_reverse_mapping_includes_known_keypoints():
    assert "left_shoulder" in op_mod.MATCHER_TO_OPENPOSE["ls"]
    assert "left_hip" in op_mod.MATCHER_TO_OPENPOSE["hip"]


# ===========================================================================
# MediaPipe
# ===========================================================================


class _Landmark:
    def __init__(self, x=0.0, y=0.0, z=0.0, visibility=0.9, presence=0.9):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence


def _make_mp_frame(landmark_map):
    """landmark_map: {idx: _Landmark}. Returns full 33-landmark list."""
    out = []
    for i in range(33):
        out.append(landmark_map.get(i, _Landmark(visibility=0.0, presence=0.0)))
    return out


def test_mediapipe_requires_landmarks_data():
    with pytest.raises(mp_mod.MediaPipeProviderError):
        mp_mod.MediaPipeProvider()


def test_mediapipe_provider_parses_frames():
    indices = mp_mod.MEDIAPIPE_POSE_LANDMARKS
    frame = _make_mp_frame({indices["left_shoulder"]: _Landmark(1, 2, 3)})
    p = mp_mod.MediaPipeProvider(landmarks_data=[frame])
    assert len(p.frames) == 1
    assert "left_shoulder" in p.frames[0].landmarks


def test_mediapipe_get_skeleton_derives_synthetic_joints():
    indices = mp_mod.MEDIAPIPE_POSE_LANDMARKS
    frame = _make_mp_frame(
        {
            indices["left_shoulder"]: _Landmark(1, 0, 0),
            indices["right_shoulder"]: _Landmark(-1, 0, 0),
            indices["left_wrist"]: _Landmark(1, 0, 1),
            indices["right_wrist"]: _Landmark(-1, 0, 1),
            indices["left_hip"]: _Landmark(0.5, 0, -1),
            indices["right_hip"]: _Landmark(-0.5, 0, -1),
        }
    )
    p = mp_mod.MediaPipeProvider(landmarks_data=[frame])
    skel = p.get_skeleton()
    assert {"ls", "rs", "lw", "rw", "hip", "spine", "torso", "hub", "mp"}.issubset(skel)
    np.testing.assert_allclose(skel["mp"], (skel["lw"] + skel["rw"]) / 2)


def test_mediapipe_low_visibility_excluded():
    indices = mp_mod.MEDIAPIPE_POSE_LANDMARKS
    frame = _make_mp_frame(
        {indices["left_shoulder"]: _Landmark(1, 0, 0, visibility=0.1, presence=0.9)}
    )
    p = mp_mod.MediaPipeProvider(landmarks_data=[frame], visibility_threshold=0.5)
    assert "ls" not in p.get_skeleton()


def test_mediapipe_only_left_hip_no_right():
    indices = mp_mod.MEDIAPIPE_POSE_LANDMARKS
    frame = _make_mp_frame({indices["left_hip"]: _Landmark(0.5, 0, -1)})
    p = mp_mod.MediaPipeProvider(landmarks_data=[frame])
    skel = p.get_skeleton()
    np.testing.assert_allclose(skel["hip"], [0.5, 0, -1])


def test_mediapipe_frame_index_out_of_range():
    p = mp_mod.MediaPipeProvider(landmarks_data=[])
    with pytest.raises(mp_mod.MediaPipeProviderError):
        p.get_skeleton(frame_index=0)
    with pytest.raises(mp_mod.MediaPipeProviderError):
        p.get_visibility_map(frame_index=0)


def test_mediapipe_visibility_map_and_missing_landmarks():
    indices = mp_mod.MEDIAPIPE_POSE_LANDMARKS
    frame = _make_mp_frame({indices["left_shoulder"]: _Landmark(1, 0, 0)})
    p = mp_mod.MediaPipeProvider(landmarks_data=[frame])
    vmap = p.get_visibility_map()
    assert "ls" in vmap
    missing = p.get_missing_landmarks()
    assert "ch" in missing  # clubhead is never observed


def test_mediapipe_create_provider_factory():
    p = mp_mod.create_provider(landmarks_data=[])
    assert isinstance(p, mp_mod.MediaPipeProvider)


def test_mediapipe_reverse_mapping():
    assert "left_shoulder" in mp_mod.MATCHER_TO_MEDIAPIPE["ls"]
    assert set(mp_mod.MATCHER_TO_MEDIAPIPE["hip"]) == {"left_hip", "right_hip"}


# ===========================================================================
# Drake / MuJoCo / OpenSim / Pinocchio: "not installed" + vocab constants
# ===========================================================================


def _stub_import_failure(missing_pkg, real_import):
    def fake_import(name, *a, **kw):
        if name == missing_pkg or name.startswith(missing_pkg + "."):
            raise ImportError(f"no {missing_pkg}")
        return real_import(name, *a, **kw)

    return fake_import


def test_drake_provider_not_available_error():
    from src.tools.starting_pose_matcher.skeleton_extractors import drake

    real = __import__
    from unittest.mock import patch

    with (
        patch("builtins.__import__", side_effect=_stub_import_failure("pydrake", real)),
        pytest.raises(drake.DrakeNotAvailableError),
    ):
        drake.DrakeSkeletonProvider(model_xml="<urdf/>")


def test_drake_vocab_mapping_constants():
    from src.tools.starting_pose_matcher.skeleton_extractors import drake

    assert drake.DRAKE_TO_MATCHER_VOCAB["left_shoulder"] == "ls"
    assert drake.MATCHER_TO_DRAKE["ls"] == "left_shoulder"


def test_mujoco_provider_not_available_error():
    from src.tools.starting_pose_matcher.skeleton_extractors import mujoco as mj

    real = __import__
    from unittest.mock import patch

    with (
        patch("builtins.__import__", side_effect=_stub_import_failure("mujoco", real)),
        pytest.raises(mj.MuJoCoNotAvailableError),
    ):
        mj.MuJoCoSkeletonProvider(model_xml="<x/>")


def test_mujoco_vocab_constants():
    from src.tools.starting_pose_matcher.skeleton_extractors import mujoco as mj

    assert mj.MUJOCO_TO_MATCHER_VOCAB["clubhead"] == "ch"
    assert mj.MATCHER_TO_MUJOCO["ls"] == "left_shoulder"


def test_opensim_provider_not_available_error():
    from src.tools.starting_pose_matcher.skeleton_extractors import opensim as os_mod

    real = __import__
    from unittest.mock import patch

    with (
        patch("builtins.__import__", side_effect=_stub_import_failure("opensim", real)),
        pytest.raises(os_mod.OpenSimNotAvailableError),
    ):
        os_mod.OpenSimSkeletonProvider(model_xml="<x/>")


def test_opensim_vocab_constants():
    from src.tools.starting_pose_matcher.skeleton_extractors import opensim as os_mod

    assert os_mod.OPENSIM_TO_MATCHER_VOCAB["pelvis"] == "hip"


def test_pinocchio_provider_not_available_error():
    from src.tools.starting_pose_matcher.skeleton_extractors import pinocchio as pn

    real = __import__
    from unittest.mock import patch

    with (
        patch(
            "builtins.__import__", side_effect=_stub_import_failure("pinocchio", real)
        ),
        pytest.raises(pn.PinocchioNotAvailableError),
    ):
        pn.PinocchioSkeletonProvider(urdf_path="x")


def test_pinocchio_vocab_constants():
    from src.tools.starting_pose_matcher.skeleton_extractors import pinocchio as pn

    assert pn.PINOCCHIO_TO_MATCHER_VOCAB["left_elbow"] == "le"


# ===========================================================================
# Validation errors when input is missing (independent of optional dep)
# ===========================================================================


def test_drake_create_provider_requires_path_or_xml(monkeypatch):
    """If pydrake imports OK but no model is provided, raise ProviderError."""
    from src.tools.starting_pose_matcher.skeleton_extractors import drake

    # Build a fake pydrake module tree so import succeeds.
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock

    fake_pydrake = ModuleType("pydrake")
    fake_plant = ModuleType("pydrake.multibody")
    fake_plant_plant = ModuleType("pydrake.multibody.plant")
    fake_plant_plant.MultibodyPlant = MagicMock()
    fake_fw = ModuleType("pydrake.systems")
    fake_fw_fw = ModuleType("pydrake.systems.framework")
    fake_fw_fw.DiagramBuilder = MagicMock()

    monkeypatch.setitem(sys.modules, "pydrake", fake_pydrake)
    monkeypatch.setitem(sys.modules, "pydrake.multibody", fake_plant)
    monkeypatch.setitem(sys.modules, "pydrake.multibody.plant", fake_plant_plant)
    monkeypatch.setitem(sys.modules, "pydrake.systems", fake_fw)
    monkeypatch.setitem(sys.modules, "pydrake.systems.framework", fake_fw_fw)

    with pytest.raises(drake.DrakeProviderError):
        drake.DrakeSkeletonProvider()


def test_mujoco_create_provider_requires_path_or_xml(monkeypatch):
    from src.tools.starting_pose_matcher.skeleton_extractors import mujoco as mj
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock

    fake = ModuleType("mujoco")
    fake.MjModel = MagicMock()
    fake.MjData = MagicMock()
    fake.mjtObj = MagicMock()
    fake.mj_id2name = MagicMock()
    fake.mj_forward = MagicMock()
    monkeypatch.setitem(sys.modules, "mujoco", fake)

    with pytest.raises(mj.MuJoCoProviderError):
        mj.MuJoCoSkeletonProvider()


def test_opensim_create_provider_requires_path_or_xml(monkeypatch):
    from src.tools.starting_pose_matcher.skeleton_extractors import opensim as os_mod
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock

    fake = ModuleType("opensim")
    fake.Model = MagicMock()
    monkeypatch.setitem(sys.modules, "opensim", fake)
    with pytest.raises(os_mod.OpenSimProviderError):
        os_mod.OpenSimSkeletonProvider()


def test_pinocchio_create_provider_requires_urdf(monkeypatch):
    from src.tools.starting_pose_matcher.skeleton_extractors import pinocchio as pn
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock

    fake = ModuleType("pinocchio")
    fake.buildModelFromUrdf = MagicMock()
    fake.Data = MagicMock()
    fake.JointModelFreeFlyer = MagicMock()
    fake.neutral = MagicMock()
    fake.forwardKinematics = MagicMock()
    monkeypatch.setitem(sys.modules, "pinocchio", fake)
    with pytest.raises(pn.PinocchioProviderError):
        pn.PinocchioSkeletonProvider()


def test_create_provider_factories_callable():
    """Each create_provider is invokable (failures are domain errors, not type errors)."""
    from src.tools.starting_pose_matcher.skeleton_extractors import (
        drake,
        mediapipe,
        mujoco as mj,
        opensim as os_mod,
        openpose,
        pinocchio as pn,
    )

    # Just confirm callables exist:
    for fn in [
        drake.create_provider,
        mediapipe.create_provider,
        mj.create_provider,
        os_mod.create_provider,
        openpose.create_provider,
        pn.create_provider,
    ]:
        assert callable(fn)
