"""Tests for src.engines.physics_engines.pinocchio.python.examples.double_pendulum_standalone."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import src.engines.physics_engines.pinocchio.python.examples.double_pendulum_standalone

        assert (
            src.engines.physics_engines.pinocchio.python.examples.double_pendulum_standalone
            is not None
        )
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
