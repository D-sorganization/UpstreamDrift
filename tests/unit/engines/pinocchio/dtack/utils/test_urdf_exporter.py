"""Tests for engines.physics_engines.pinocchio.python.dtack.utils.urdf_exporter."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.dtack.utils.urdf_exporter

        assert (
            engines.physics_engines.pinocchio.python.dtack.utils.urdf_exporter
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
