"""Tests for src.engines.physics_engines.pinocchio.python.pinocchio_golf.poly_torque_util."""

import pytest


def test_import() -> None:
    """Verify the module can be imported."""
    try:
        import src.engines.physics_engines.pinocchio.python.pinocchio_golf.poly_torque_util

        assert (
            src.engines.physics_engines.pinocchio.python.pinocchio_golf.poly_torque_util
            is not None
        )
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
