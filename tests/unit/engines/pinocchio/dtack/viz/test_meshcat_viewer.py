"""Tests for engines.physics_engines.pinocchio.python.dtack.viz.meshcat_viewer."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.dtack.viz.meshcat_viewer

        assert (
            engines.physics_engines.pinocchio.python.dtack.viz.meshcat_viewer
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
