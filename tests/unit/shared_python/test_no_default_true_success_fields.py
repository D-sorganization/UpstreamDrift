"""Oracle 1 & 8 tests and AST scanner gate for default-false result contracts (#14934 Slice 4).

Paired with D-sorganization/Tools#6014.

Oracle 1 (Default-true removal):
Constructing any solver/calc result class with minimal required arguments
(or without explicitly asserting success) must NEVER report success/converged is True.

Oracle 8 (Old readers):
Existing readers and serializers (model_dump, .success attribute)
receive the exact boolean contract expected for both success and failure outcomes.

AST Gate:
Scans src/shared/python to prevent newly added result/contract classes from defaulting
success/converged/ok to True.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit
from src.shared.python.calc_backend.contracts.ode_solver import (
    ODESolverRequest,
    ODESolverResponse,
)
from src.shared.python.calc_backend.routers.ode_solver import solve_ode

# Existing legacy classes outside calc_backend/contracts allowlisted until migrated
LEGACY_ALLOWLIST = {
    ("src/shared/python/motion_matching/full_body_ik.py", "PoseFit", "converged"),
    ("src/shared/python/sg_optimizer/strategy_view.py", "StrategyView", "converged"),
}

FLAG_NAMES = {"success", "converged", "ok"}


def _is_default_true(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Constant) and node.value is True:
        return True
    if isinstance(node, ast.Call):
        for kw in node.keywords:
            if (
                kw.arg == "default"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
            ):
                return True
    return False


def _find_default_true_flags(root: Path, base_dir: Path) -> list[tuple[str, str, str]]:
    violations: list[tuple[str, str, str]] = []
    for py_path in root.rglob("*.py"):
        rel = py_path.relative_to(base_dir).as_posix()
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    if stmt.target.id in FLAG_NAMES and _is_default_true(stmt.value):
                        entry = (rel, node.name, stmt.target.id)
                        if entry not in LEGACY_ALLOWLIST:
                            violations.append(entry)
                elif isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id in FLAG_NAMES
                            and _is_default_true(stmt.value)
                        ):
                            entry = (rel, node.name, target.id)
                            if entry not in LEGACY_ALLOWLIST:
                                violations.append(entry)
    return violations


class TestASTScannerNoDefaultTrue:
    """AST scanner gate ensuring no default-true success flags exist in shared contracts."""

    def test_no_unallowlisted_default_true_flags_in_shared_python(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        shared_root = repo_root / "src" / "shared" / "python"
        violations = _find_default_true_flags(shared_root, repo_root)
        assert violations == [], (
            f"Found unallowlisted classes defaulting success/converged/ok to True: {violations}. "
            f"Result contracts must default to False so errors/partial construction paths never "
            f"silently report success (Oracle 1)."
        )


class TestOracle1DefaultFalse:
    """Oracle 1: constructing with minimal args must not report success=True."""

    def test_ode_solver_response_minimal_construction_defaults_to_false(self) -> None:
        resp = ODESolverResponse(
            times=[0.0, 1.0],
            solutions={"y": [1.0, 0.5]},
            variable_summaries=[],
        )
        assert resp.success is False, "ODESolverResponse.success must default to False"
        assert resp.message == "", "ODESolverResponse.message must default to empty"


class TestOracle8OldReaders:
    """Oracle 8: old readers receive expected boolean contracts and keys."""

    def test_ode_solver_reader_contract(self) -> None:
        req = ODESolverRequest(
            derivatives={"y": "-k*y"},
            parameters={"k": 0.1},
            initial_conditions={"y": 1.0},
            t_start=0.0,
            t_end=1.0,
            num_points=11,
        )
        resp = solve_ode(req)
        assert resp.success is True
        assert resp.message == "Solution computed successfully"
        dump = resp.model_dump()
        assert dump["success"] is True
        assert dump["message"] == "Solution computed successfully"

        parsed_true = ODESolverResponse.model_validate({**dump, "success": True})
        assert parsed_true.success is True
        parsed_false = ODESolverResponse.model_validate({**dump, "success": False})
        assert parsed_false.success is False
