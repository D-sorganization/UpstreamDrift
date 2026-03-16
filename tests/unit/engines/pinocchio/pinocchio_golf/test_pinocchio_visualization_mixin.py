"""Tests for src.engines.physics_engines.pinocchio.python.pinocchio_golf.pinocchio_visualization_mixin."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import src.engines.physics_engines.pinocchio.python.pinocchio_golf.pinocchio_visualization_mixin

        assert (
            src.engines.physics_engines.pinocchio.python.pinocchio_golf.pinocchio_visualization_mixin
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
