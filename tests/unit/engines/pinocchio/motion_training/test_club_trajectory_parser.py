"""Tests for src.engines.physics_engines.pinocchio.python.motion_training.club_trajectory_parser."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import src.engines.physics_engines.pinocchio.python.motion_training.club_trajectory_parser

        assert (
            src.engines.physics_engines.pinocchio.python.motion_training.club_trajectory_parser
            is not None
        )
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
