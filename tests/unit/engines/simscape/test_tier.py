"""Tests for the Simscape tier metadata."""


def test_simscape_tier():
    """Verify the tier configuration."""
    from src.engines.simscape._tier import TIER

    assert TIER == "experimental"
