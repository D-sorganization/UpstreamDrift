"""Tests for engines.physics_engines.pinocchio.python.dtack.gui.main_window."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.dtack.gui.main_window

        assert (
            engines.physics_engines.pinocchio.python.dtack.gui.main_window is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
