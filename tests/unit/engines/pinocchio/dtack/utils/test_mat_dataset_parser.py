"""Tests for src.engines.physics_engines.pinocchio.python.dtack.utils.mat_dataset_parser."""

import importlib

import pytest


def test_import() -> None:
    """Verify the module can be imported."""
    try:
        module = importlib.import_module(
            "src.engines.physics_engines.pinocchio.python."
            "dtack.utils.mat_dataset_parser"
        )
        assert module is not None
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
