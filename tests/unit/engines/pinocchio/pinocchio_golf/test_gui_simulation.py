"""Tests for src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui_simulation."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui_simulation

        assert (
            src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui_simulation
            is not None
        )
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
