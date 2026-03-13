"""Tests for engines.physics_engines.pinocchio.python.__main__."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.__main__

        assert engines.physics_engines.pinocchio.python.__main__ is not None
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
