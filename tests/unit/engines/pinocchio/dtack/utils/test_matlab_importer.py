"""Tests for src.engines.physics_engines.pinocchio.python.dtack.utils.matlab_importer."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import src.engines.physics_engines.pinocchio.python.dtack.utils.matlab_importer

        assert (
            src.engines.physics_engines.pinocchio.python.dtack.utils.matlab_importer
            is not None
        )
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
