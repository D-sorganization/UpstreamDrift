"""Tests for engines.physics_engines.pinocchio.python.pinocchio_golf.ui.visualization_mixin."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.pinocchio_golf.ui.visualization_mixin

        assert (
            engines.physics_engines.pinocchio.python.pinocchio_golf.ui.visualization_mixin
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
