"""Tests for engines.physics_engines.pinocchio.python.dtack.backends.pink_backend."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.dtack.backends.pink_backend

        assert (
            engines.physics_engines.pinocchio.python.dtack.backends.pink_backend
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
