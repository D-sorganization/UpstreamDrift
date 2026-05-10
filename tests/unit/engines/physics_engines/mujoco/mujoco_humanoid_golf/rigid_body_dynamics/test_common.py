"""Unit tests for rigid body dynamics common module."""

import numpy as np

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.rigid_body_dynamics.common import (
    DEFAULT_GRAVITY,
    NEG_DEFAULT_GRAVITY,
)
from src.shared.python.core import constants


def test_gravity_vectors():
    """Test gravity vectors."""
    assert isinstance(DEFAULT_GRAVITY, np.ndarray)
    assert isinstance(NEG_DEFAULT_GRAVITY, np.ndarray)

    assert DEFAULT_GRAVITY.shape == (6,)
    assert NEG_DEFAULT_GRAVITY.shape == (6,)

    assert not DEFAULT_GRAVITY.flags.writeable
    assert not NEG_DEFAULT_GRAVITY.flags.writeable

    expected = np.array([0, 0, 0, 0, 0, -constants.GRAVITY_M_S2])
    np.testing.assert_array_equal(DEFAULT_GRAVITY, expected)
    np.testing.assert_array_equal(NEG_DEFAULT_GRAVITY, -expected)
