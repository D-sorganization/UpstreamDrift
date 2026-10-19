"""Acceptance tests for issue #12847 architecture-budget refactors in scripts/."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci import check_architecture_budget as budget_mod

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PROXIMAL = ROOT / "scripts/research/proximal_distal_energy"
CONFIG = json.loads(
    (ROOT / "scripts/config/architecture_budget.json").read_text(encoding="utf-8")
)
MAX_FUNCTION_LINES = int(CONFIG["max_function_lines"])
MAX_PARAMETERS = int(CONFIG["max_parameters"])

ISSUE_12847_SYMBOLS: tuple[tuple[Path, str, str], ...] = (
    (
        PROXIMAL / "make_articulated_native_constraint_discrepancy_figure.py",
        "make_figure",
        "function-lines",
    ),
    (PROXIMAL / "run_two_hand_wscg_analysis.py", "_analyze_case", "function-lines"),
    (PROXIMAL / "run_two_hand_wscg_analysis.py", "build_outputs", "function-lines"),
    (
        PROXIMAL / "articulated_drift_contact_attribution.py",
        "_evaluate_controls",
        "parameters",
    ),
)


def _finding_for(
    path: Path, symbol: str, rule: str
) -> budget_mod.FunctionBudgetFinding | None:
    findings = budget_mod.analyze_python_file(
        path,
        max_function_lines=MAX_FUNCTION_LINES,
        max_parameters=MAX_PARAMETERS,
    )
    for finding in findings:
        if finding.symbol == symbol and finding.rule == rule:
            return finding
    return None


@pytest.mark.parametrize(
    ("path", "symbol", "rule"),
    ISSUE_12847_SYMBOLS,
    ids=[f"{symbol}:{rule}" for _, symbol, rule in ISSUE_12847_SYMBOLS],
)
def test_issue_12847_symbols_meet_architecture_budget(
    path: Path, symbol: str, rule: str
) -> None:
    finding = _finding_for(path, symbol, rule)
    assert finding is None, finding.format(ROOT) if finding else ""


def test_issue_12847_architecture_exceptions_removed() -> None:
    rows = CONFIG.get("exceptions", [])
    for path, symbol, rule in ISSUE_12847_SYMBOLS:
        rel = path.relative_to(ROOT).as_posix()
        matches = [
            row
            for row in rows
            if row.get("path") == rel
            and row.get("symbol") == symbol
            and row.get("rule") == rule
        ]
        assert not matches, f"stale exception still present for {rel} {symbol} {rule}"
