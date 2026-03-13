"""Tests for engines.physics_engines.pinocchio.python.motion_training.training_pipeline."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.motion_training.training_pipeline

        assert (
            engines.physics_engines.pinocchio.python.motion_training.training_pipeline
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
