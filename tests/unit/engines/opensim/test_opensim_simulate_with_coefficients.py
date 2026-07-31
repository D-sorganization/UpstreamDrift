"""Regression tests for the collapsed OpenSim simulator facade."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.motion_matching import simulate
from src.engines.physics_engines.opensim.python.motion_matching import synthesize
from src.engines.physics_engines.opensim.python.opensim_golf import fk
from src.engines.physics_engines.opensim.python.opensim_golf import (
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


def test_opensim_golf_facade_delegates_to_canonical_simulator() -> None:
    """Legacy imports must resolve to the maintained motion-matching objects."""
    assert facade.SimOptions is simulate.SimOptions
    assert facade.SimOut is simulate.SimOut
    assert facade.evaluate_polynomial_torque is simulate.evaluate_polynomial_torque
    assert facade.simulate_with_coefficients is simulate.simulate_with_coefficients
    assert facade.synthesize_target_from_coefficients is (
        synthesize.synthesize_target_from_coefficients
    )


def test_opensim_simulator_uses_canonical_fk_extractor() -> None:
    """The simulator must not carry an inline grip/clubhead frame resolver."""
    assert simulate.extract_full_pose is fk.extract_full_pose


def test_opensim_forward_simulator_has_single_implementation() -> None:
    """Keep one maintained OpenSim simulator implementation under the engine tree."""
    root = _repo_root() / "src" / "engines" / "physics_engines" / "opensim"
    defs = _function_defs_under(root, "simulate_with_coefficients")
    assert defs == [root / "python" / "motion_matching" / "simulate.py"]


def test_opensim_full_pose_has_single_implementation() -> None:
    """Frame resolution must remain centralized in opensim_golf.fk."""
    root = _repo_root() / "src" / "engines" / "physics_engines" / "opensim"
    defs = _function_defs_under(root, "extract_full_pose")
    assert defs == [root / "python" / "opensim_golf" / "fk.py"]


def test_legacy_opensim_facade_does_not_resolve_club_body_origin() -> None:
    """The compatibility module must not revive body-origin clubhead extraction."""
    source = Path(facade.__file__).read_text(encoding="utf-8")
    assert 'body_set.get("club")' not in source
    assert "getBodySet" not in source
