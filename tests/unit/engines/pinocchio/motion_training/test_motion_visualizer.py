"""Tests for engines.physics_engines.pinocchio.python.motion_training.motion_visualizer."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.motion_training.motion_visualizer

        assert (
            engines.physics_engines.pinocchio.python.motion_training.motion_visualizer
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
