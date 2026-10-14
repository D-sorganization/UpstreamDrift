"""Tests for the Simscape tier metadata."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_simscape_tier():
    """Verify the tier configuration."""
    from src.engines.simscape._tier import TIER

    assert TIER == "experimental"
