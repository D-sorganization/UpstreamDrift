"""Regression tests for the motion_matching package root export contract."""

from __future__ import annotations

import importlib

import pytest


pytestmark = pytest.mark.unit


def test_callable_exports_survive_matching_submodule_imports() -> None:
    """Callable root exports must not be shadowed by same-named submodules."""
    import src.shared.python.motion_matching as motion_matching

    importlib.import_module("src.shared.python.motion_matching.compute_total_work")
    importlib.import_module("src.shared.python.motion_matching.load_body_target")

    from src.shared.python.motion_matching import compute_total_work, load_body_target

    assert callable(compute_total_work)
    assert callable(load_body_target)
    assert compute_total_work is motion_matching.compute_total_work
    assert load_body_target is motion_matching.load_body_target
