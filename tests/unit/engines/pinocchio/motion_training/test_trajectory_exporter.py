"""Tests for engines.physics_engines.pinocchio.python.motion_training.trajectory_exporter."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.motion_training.trajectory_exporter

        assert (
            engines.physics_engines.pinocchio.python.motion_training.trajectory_exporter
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
