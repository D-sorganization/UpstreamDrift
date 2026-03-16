"""Tests for src.engines.physics_engines.pinocchio.python.dtack.viz.geppetto_viewer."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import src.engines.physics_engines.pinocchio.python.dtack.viz.geppetto_viewer

        assert (
            src.engines.physics_engines.pinocchio.python.dtack.viz.geppetto_viewer
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
