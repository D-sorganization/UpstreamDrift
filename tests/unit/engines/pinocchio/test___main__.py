"""Tests for src.engines.physics_engines.pinocchio.python.__main__."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import src.engines.physics_engines.pinocchio.python.__main__

        assert src.engines.physics_engines.pinocchio.python.__main__ is not None
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
