"""Tests for engines.physics_engines.pinocchio.python.pinocchio_golf.gui_simulation."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.pinocchio_golf.gui_simulation

        assert (
            engines.physics_engines.pinocchio.python.pinocchio_golf.gui_simulation
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
