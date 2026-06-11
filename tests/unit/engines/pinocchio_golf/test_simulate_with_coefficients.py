"""Regression tests for the collapsed Pinocchio simulator facade."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.engines.physics_engines.pinocchio.python.motion_matching import simulate
from src.engines.physics_engines.pinocchio.python.motion_matching import synthesize
from src.engines.physics_engines.pinocchio.python.pinocchio_golf import (
    simulate_with_coefficients as facade,
)


pytestmark = [pytest.mark.unit]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _function_defs_under(root: Path, name: str) -> list[Path]:
    matches: list[Path] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(
            isinstance(node, ast.FunctionDef) and node.name == name
            for node in ast.walk(tree)
        ):
            matches.append(path)
    return matches


def test_pinocchio_golf_facade_delegates_to_canonical_simulator() -> None:
    """Legacy imports must resolve to the maintained motion-matching objects."""
    assert facade.SimOptions is simulate.SimOptions
    assert facade.SimOut is simulate.SimOut
    assert facade.evaluate_polynomial_torque is simulate.evaluate_polynomial_torque
    assert facade.simulate_with_coefficients is simulate.simulate_with_coefficients
    assert facade.synthesize_target_from_coefficients is (
        synthesize.synthesize_target_from_coefficients
    )


def test_pinocchio_forward_simulator_has_single_implementation() -> None:
    """Keep one maintained Pinocchio simulator implementation under the engine tree."""
    root = _repo_root() / "src" / "engines" / "physics_engines" / "pinocchio"
    defs = _function_defs_under(root, "simulate_with_coefficients")
    assert defs == [root / "python" / "motion_matching" / "simulate.py"]


def test_legacy_pinocchio_facade_does_not_reimplement_integration() -> None:
    """The compatibility module should stay a shim, not a second simulator."""
    source = Path(facade.__file__).read_text(encoding="utf-8")
    assert "pin.aba" not in source
    assert "def _rk4_step" not in source
    assert "def _semi_implicit_step" not in source
