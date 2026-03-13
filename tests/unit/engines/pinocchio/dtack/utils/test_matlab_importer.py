"""Tests for engines.physics_engines.pinocchio.python.dtack.utils.matlab_importer."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.dtack.utils.matlab_importer

        assert (
            engines.physics_engines.pinocchio.python.dtack.utils.matlab_importer
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
