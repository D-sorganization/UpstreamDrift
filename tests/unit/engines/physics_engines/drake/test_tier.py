"""Tests for the Drake engine tier metadata."""

from src.engines.physics_engines.drake._tier import TIER


def test_drake_tier_is_extended() -> None:
    """Test that the Drake engine package is designated as the extended tier."""
    assert TIER == "extended"
