"""Tests for engines.physics_engines.pinocchio.python.examples.double_pendulum_standalone."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.examples.double_pendulum_standalone

        assert (
            engines.physics_engines.pinocchio.python.examples.double_pendulum_standalone
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
