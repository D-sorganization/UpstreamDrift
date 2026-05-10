"""Unit tests for spatial inertia module."""

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.spatial_algebra.inertia import (
    mci,
    transform_spatial_inertia,
)


def test_inertia_exports():
    """Test that all required functions are exported."""
    assert callable(mci)
    assert callable(transform_spatial_inertia)
