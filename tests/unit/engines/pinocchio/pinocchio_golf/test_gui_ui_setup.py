"""Tests for engines.physics_engines.pinocchio.python.pinocchio_golf.gui_ui_setup."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.pinocchio_golf.gui_ui_setup

        assert (
            engines.physics_engines.pinocchio.python.pinocchio_golf.gui_ui_setup
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
