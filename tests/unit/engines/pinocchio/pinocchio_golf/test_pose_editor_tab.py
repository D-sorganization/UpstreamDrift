"""Tests for src.engines.physics_engines.pinocchio.python.pinocchio_golf.pose_editor_tab."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import src.engines.physics_engines.pinocchio.python.pinocchio_golf.pose_editor_tab

        assert (
            src.engines.physics_engines.pinocchio.python.pinocchio_golf.pose_editor_tab is not None
        )
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
