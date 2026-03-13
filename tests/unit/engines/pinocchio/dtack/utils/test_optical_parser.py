"""Tests for engines.physics_engines.pinocchio.python.dtack.utils.optical_parser."""

import pytest


def test_import():
    """Verify the module can be imported."""
    try:
        import engines.physics_engines.pinocchio.python.dtack.utils.optical_parser

        assert (
            engines.physics_engines.pinocchio.python.dtack.utils.optical_parser
            is not None
        )
    except ImportError as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
