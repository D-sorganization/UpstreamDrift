"""Unit tests for spatial transforms module."""

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.spatial_algebra.transforms import (
    inv_xtrans,
    xlt,
    xrot,
    xtrans,
)


def test_transforms_exports():
    """Test that all required functions are exported."""
    assert callable(inv_xtrans)
    assert callable(xlt)
    assert callable(xrot)
    assert callable(xtrans)
