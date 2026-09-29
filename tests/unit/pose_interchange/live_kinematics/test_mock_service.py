"""Tests for :class:`MockKinematicsService` correctness against canonical FK."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.diagnostics.forward_kinematics import (
    forward_kinematics,
)
from src.shared.python.pose_interchange.canonical import (
    canonical_from_reference_setup,
    canonical_zero_pose,
)
from src.shared.python.pose_interchange.services._mock import (
    MockKinematicsService,
)

pytestmark = pytest.mark.unit


_EXPECTED_LANDMARKS = {
    "pelvis",
    "spine_top",
    "torso_top",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hand",
    "r_hand",
    "butt",
    "clubhead",
}


def test_mock_service_engine_name_propagates() -> None:
    svc = MockKinematicsService(engine_name="drake")
    assert svc.engine_name == "drake"


def test_mock_service_rejects_non_string_engine_name() -> None:
    with pytest.raises(TypeError, match="engine_name"):
        MockKinematicsService(engine_name=123)  # type: ignore[arg-type]


def test_mock_service_rejects_empty_engine_name() -> None:
    with pytest.raises(ValueError, match="engine_name"):
        MockKinematicsService(engine_name="")


def test_mock_service_load_accepts_path() -> None:
    svc = MockKinematicsService(engine_name="mujoco")
    svc.load(Path("ignored.xml"))


def test_mock_service_load_rejects_str() -> None:
    svc = MockKinematicsService(engine_name="mujoco")
    with pytest.raises(TypeError, match="model_path"):
        svc.load("ignored.xml")  # type: ignore[arg-type]


def test_mock_service_set_pose_rejects_non_canonical_pose() -> None:
    svc = MockKinematicsService(engine_name="mujoco")
    with pytest.raises(TypeError, match="CanonicalPose"):
        svc.set_pose({"not": "a pose"})  # type: ignore[arg-type]


def test_mock_service_returns_canonical_landmarks() -> None:
    """``get_link_transforms`` must produce exactly the canonical landmark set.

    Compares the translation columns to direct
    :func:`forward_kinematics` output exactly (no tolerance) — both
    code paths route through the same evaluator so the values must
    match bit-for-bit.
    """
    pose = canonical_from_reference_setup()
    svc = MockKinematicsService(engine_name="opensim")
    svc.set_pose(pose)
    transforms = svc.get_link_transforms()

    # Names match exactly.
    assert set(transforms) == _EXPECTED_LANDMARKS

    # Values match canonical FK exactly.
    expected = forward_kinematics(pose.angles_full_dict_deg())
    for name, transform in transforms.items():
        assert transform.shape == (4, 4)
        assert transform.dtype == np.float64
        # Rotation block is identity.
        np.testing.assert_array_equal(transform[:3, :3], np.eye(3))
        # Bottom row is [0, 0, 0, 1].
        np.testing.assert_array_equal(transform[3, :], np.array([0, 0, 0, 1.0]))
        # Translation column matches FK landmark exactly.
        np.testing.assert_array_equal(transform[:3, 3], expected.points[name])


def test_mock_service_default_pose_is_zero_pose_landmarks() -> None:
    """Without :meth:`set_pose`, the mock returns landmarks for the zero pose.

    This guarantees Pose Studio always renders *something* even before
    a pose has been pushed in.
    """
    svc = MockKinematicsService(engine_name="pinocchio")
    transforms = svc.get_link_transforms()
    expected = forward_kinematics(canonical_zero_pose().angles_full_dict_deg())
    for name, transform in transforms.items():
        np.testing.assert_array_equal(transform[:3, 3], expected.points[name])


def test_mock_service_reset_drops_pose() -> None:
    pose = canonical_from_reference_setup()
    svc = MockKinematicsService(engine_name="drake")
    svc.set_pose(pose)
    svc.reset()
    transforms = svc.get_link_transforms()
    # After reset the mock falls back to the zero pose, not the previous one.
    expected = forward_kinematics(canonical_zero_pose().angles_full_dict_deg())
    np.testing.assert_array_equal(
        transforms["pelvis"][:3, 3], expected.points["pelvis"]
    )


def test_mock_service_capabilities_are_all_false() -> None:
    svc = MockKinematicsService(engine_name="simscape")
    caps = svc.capabilities()
    assert caps.supports_dynamics_step is False
    assert caps.supports_collision_query is False
    assert caps.supports_realtime is False
