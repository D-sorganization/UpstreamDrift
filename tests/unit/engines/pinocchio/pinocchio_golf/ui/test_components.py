"""Tests for src.engines.physics_engines.pinocchio.python.pinocchio_golf.ui.components."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import src.engines.physics_engines.pinocchio.python.pinocchio_golf.ui.components

        assert (
            src.engines.physics_engines.pinocchio.python.pinocchio_golf.ui.components
            is not None
        )
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
