"""Tests for the MuJoCo tier metadata."""


def test_mujoco_tier():
    """Verify the tier configuration."""
    from src.engines.physics_engines.mujoco._tier import TIER

    assert TIER == "core"
