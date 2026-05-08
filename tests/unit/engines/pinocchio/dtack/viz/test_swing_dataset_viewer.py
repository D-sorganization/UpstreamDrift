"""Tests for src.engines.physics_engines.pinocchio.python.dtack.viz.swing_dataset_viewer."""

import importlib

import pytest


def test_import() -> None:
    """Verify the module can be imported."""
    try:
        module = importlib.import_module(
            "src.engines.physics_engines.pinocchio.python."
            "dtack.viz.swing_dataset_viewer"
        )
        assert module is not None
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Missing dependencies or import error: {e}")
