"""Unit tests for spatial joints module."""

import numpy as np

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.spatial_algebra.joints import (
    JOINT_AXIS_INDICES,
    S_PX,
    S_PY,
    S_PZ,
    S_RX,
    S_RY,
    S_RZ,
    jcalc,
)


def test_joints_exports():
    """Test that all required constants and functions are exported."""
    assert callable(jcalc)
    assert isinstance(JOINT_AXIS_INDICES, dict)
    assert isinstance(S_PX, np.ndarray)
    assert isinstance(S_PY, np.ndarray)
    assert isinstance(S_PZ, np.ndarray)
    assert isinstance(S_RX, np.ndarray)
    assert isinstance(S_RY, np.ndarray)
    assert isinstance(S_RZ, np.ndarray)
