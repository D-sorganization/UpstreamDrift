"""Budget guard: WSCG ``_analyze_case`` and ``build_outputs`` fit the budget (#13852)."""

from __future__ import annotations

import ast
import json

import pytest

from tests.research.wscg_analysis_test_support import ANALYSIS_SCRIPT, REPO_ROOT

pytestmark = pytest.mark.unit

_BUDGET_PATH = REPO_ROOT / "scripts" / "config" / "architecture_budget.json"
_SCRIPT_KEY = ANALYSIS_SCRIPT.relative_to(REPO_ROOT).as_posix()
_BUDGETED_FUNCTIONS = ("_analyze_case", "build_outputs")


def _function_node(name: str) -> ast.FunctionDef:
    tree = ast.parse(ANALYSIS_SCRIPT.read_text(encoding="utf-8"))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one top-level {name!r}, found {len(matches)}")
    return matches[0]


@pytest.mark.parametrize("name", _BUDGETED_FUNCTIONS)
def test_function_within_function_line_budget(name: str) -> None:
    config = json.loads(_BUDGET_PATH.read_text(encoding="utf-8"))
    node = _function_node(name)
    assert node.end_lineno is not None
    lines = node.end_lineno - node.lineno + 1
    assert lines <= int(config["max_function_lines"]), lines


@pytest.mark.parametrize("name", _BUDGETED_FUNCTIONS)
def test_function_has_no_budget_exception(name: str) -> None:
    config = json.loads(_BUDGET_PATH.read_text(encoding="utf-8"))
    rows = [
        entry
        for entry in config["exceptions"]
        if entry["path"] == _SCRIPT_KEY and entry["symbol"] == name
    ]
    assert rows == []
