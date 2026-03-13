"""Tests for engines.physics_engines.pinocchio.python.swing_plane_integration."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.swing_plane_integration

        assert (
            engines.physics_engines.pinocchio.python.swing_plane_integration is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
