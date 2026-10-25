"""NSX-91 (#12720) acceptance guard: one paired Monte Carlo runner, no second loop.

Every strategy comparison that needs finite-amplitude truth (NSX-95 held-out
stability, noise-aware verification) must go through
``paired_monte_carlo.run_campaign``.  These AST checks make the single-runner
rule enforceable instead of conventional.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PKG = Path(__file__).resolve().parents[3] / "src/shared/python/loop_nullspace"
RUNNER = "paired_monte_carlo.py"
RUNNER_ENTRY_POINTS = {"run_campaign", "run_seed_set", "_run_trial"}
# Linear strategy-comparison modules must never draw random numbers themselves.
STRATEGY_MODULES = (
    "strategy_arms.py",
    "strategy_campaign.py",
    "strategy_evaluators.py",
    "strategy_layer_eval.py",
    "strategy_spec.py",
)
RNG_NAMES = {"default_rng", "SeedSequence", "RandomState", "Generator"}


def _tree(name: str) -> ast.Module:
    return ast.parse((PKG / name).read_text(encoding="utf-8"))


def _defined_functions(tree: ast.Module) -> set[str]:
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _rng_uses(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in RNG_NAMES:
            found.add(node.attr)
        elif isinstance(node, ast.Name) and node.id in RNG_NAMES:
            found.add(node.id)
    return found


def test_trial_loop_is_defined_only_in_the_paired_runner() -> None:
    assert _defined_functions(_tree(RUNNER)) >= RUNNER_ENTRY_POINTS
    for path in sorted(PKG.glob("*.py")):
        if path.name == RUNNER:
            continue
        clash = RUNNER_ENTRY_POINTS & _defined_functions(_tree(path.name))
        assert not clash, f"{path.name} re-defines the MC trial loop: {sorted(clash)}"


def test_strategy_comparison_modules_draw_no_random_numbers() -> None:
    for name in STRATEGY_MODULES:
        assert not _rng_uses(_tree(name)), f"{name} must not run its own Monte Carlo"


def test_heldout_stability_calls_the_shared_runner() -> None:
    tree = _tree("strategy_stability.py")
    calls = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "run_campaign" in calls


def test_the_guard_detects_a_planted_second_loop() -> None:
    planted = ast.parse(
        "import numpy as np\ndef run_campaign():\n    return np.random.default_rng(0)\n"
    )
    assert "run_campaign" in _defined_functions(planted)
    assert _rng_uses(planted) == {"default_rng"}
