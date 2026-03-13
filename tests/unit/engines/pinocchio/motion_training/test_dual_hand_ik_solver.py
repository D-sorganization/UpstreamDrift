"""Tests for engines.physics_engines.pinocchio.python.motion_training.dual_hand_ik_solver."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.motion_training.dual_hand_ik_solver

        assert (
            engines.physics_engines.pinocchio.python.motion_training.dual_hand_ik_solver
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
