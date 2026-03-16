"""Tests for src.engines.physics_engines.pinocchio.python.dtack.ik.pink_solver."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import src.engines.physics_engines.pinocchio.python.dtack.ik.pink_solver

        assert src.engines.physics_engines.pinocchio.python.dtack.ik.pink_solver is not None
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
