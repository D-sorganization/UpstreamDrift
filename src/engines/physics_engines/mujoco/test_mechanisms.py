"""Test suite for MuJoCo mechanism definitions.

Verifies that mechanism definitions (joints, actuators, constraints)
are structurally correct and physically plausible.
"""

from __future__ import annotations

import mujoco
import pytest

from src.engines.physics_engines.mujoco.head_models import (
    TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML,
)
from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


@pytest.fixture
def inclined_plane_model() -> mujoco.MjModel:
    """Fixture providing the 2-link inclined plane model."""
    try:
        model = mujoco.MjModel.from_xml_string(TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML)
        return model
    except Exception as e:
        pytest.fail(f"Failed to load inclined plane model: {e}")


def test_mechanism_hierarchy(inclined_plane_model: mujoco.MjModel) -> None:
    """Verify the kinematic chain hierarchy."""
    # Check body names exist
    body_names = [
        mujoco.mj_id2name(inclined_plane_model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(inclined_plane_model.nbody)
    ]

    assert "shoulder_base" in body_names
    assert "upper_arm" in body_names
    assert "wrist_body" in body_names
    assert "club" in body_names


def test_joint_definitions(inclined_plane_model: mujoco.MjModel) -> None:
    """Verify joint types and axes."""

    # Helper to get joint ID
    def get_joint_id(name: str) -> int:
        return int(
            mujoco.mj_name2id(inclined_plane_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        )

    # 1. Shoulder Hinge
    shoulder_id = get_joint_id("shoulder")
    assert shoulder_id != -1
    # Check type is hinge (mjtJoint.mjJNT_HINGE == 3)
    assert inclined_plane_model.jnt_type[shoulder_id] == mujoco.mjtJoint.mjJNT_HINGE
    # Check axis is Z (0, 0, 1)
    axis = inclined_plane_model.jnt_axis[shoulder_id]
    assert axis[0] == 0.0 and axis[1] == 0.0 and axis[2] == 1.0

    # 2. Wrist Universal Joint (modeled as 2 hinges)
    u1_id = get_joint_id("wrist_universal_1")
    u2_id = get_joint_id("wrist_universal_2")
    assert u1_id != -1 and u2_id != -1

    # Check orthogonality of axes (Y vs X)
    axis1 = inclined_plane_model.jnt_axis[u1_id]
    axis2 = inclined_plane_model.jnt_axis[u2_id]

    # Dot product should be 0
    dot_prod = sum(a * b for a, b in zip(axis1, axis2, strict=False))
    assert abs(dot_prod) < 1e-6


def test_actuator_coupling(inclined_plane_model: mujoco.MjModel) -> None:
    """Verify actuators are correctly coupled to joints."""
    # Iterate over actuators
    for i in range(inclined_plane_model.nu):
        # Get transmission type
        trntype = inclined_plane_model.actuator_trntype[i]
        assert trntype == mujoco.mjtTrn.mjTRN_JOINT

        # Get coupled joint ID
        joint_id = inclined_plane_model.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(
            inclined_plane_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )

        # Ensure we are actuating the expected joints
        assert joint_name in ["shoulder", "wrist_universal_1", "wrist_universal_2"]
