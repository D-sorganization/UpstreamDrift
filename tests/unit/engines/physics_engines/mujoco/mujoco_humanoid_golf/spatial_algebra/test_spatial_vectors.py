"""Unit tests for spatial vectors module."""

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.spatial_algebra.spatial_vectors import (
    crf,
    crm,
    cross_force,
    cross_force_fast,
    cross_motion,
    cross_motion_axis,
    cross_motion_fast,
    skew,
    spatial_cross,
)


def test_spatial_vectors_exports():
    """Test that all required functions are exported."""
    assert callable(crf)
    assert callable(crm)
    assert callable(cross_force)
    assert callable(cross_force_fast)
    assert callable(cross_motion)
    assert callable(cross_motion_axis)
    assert callable(cross_motion_fast)
    assert callable(skew)
    assert callable(spatial_cross)
