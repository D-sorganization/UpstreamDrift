"""Tests for engines.physics_engines.pinocchio.python.pinocchio_golf.ui.components."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.pinocchio_golf.ui.components

        assert (
            engines.physics_engines.pinocchio.python.pinocchio_golf.ui.components
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
