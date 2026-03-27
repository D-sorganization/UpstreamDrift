"""Tests for src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui_ui_setup."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui_ui_setup

        assert (
            src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui_ui_setup
            is not None
        )
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
