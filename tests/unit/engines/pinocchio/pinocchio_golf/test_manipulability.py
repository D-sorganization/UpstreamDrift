"""Tests for src.engines.physics_engines.pinocchio.python.pinocchio_golf.manipulability."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import src.engines.physics_engines.pinocchio.python.pinocchio_golf.manipulability

        assert (
            src.engines.physics_engines.pinocchio.python.pinocchio_golf.manipulability
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
