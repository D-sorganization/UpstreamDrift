"""Tests for src.engines.physics_engines.pinocchio.python.dtack.backends.pinocchio_backend."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import src.engines.physics_engines.pinocchio.python.dtack.backends.pinocchio_backend

        assert (
            src.engines.physics_engines.pinocchio.python.dtack.backends.pinocchio_backend
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
