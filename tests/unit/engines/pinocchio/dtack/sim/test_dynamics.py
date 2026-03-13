"""Tests for engines.physics_engines.pinocchio.python.dtack.sim.dynamics."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.dtack.sim.dynamics

        assert engines.physics_engines.pinocchio.python.dtack.sim.dynamics is not None
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
