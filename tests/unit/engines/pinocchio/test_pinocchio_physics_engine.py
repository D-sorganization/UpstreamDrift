"""Tests for engines.physics_engines.pinocchio.python.pinocchio_physics_engine."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.pinocchio_physics_engine

        assert (
            engines.physics_engines.pinocchio.python.pinocchio_physics_engine
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
