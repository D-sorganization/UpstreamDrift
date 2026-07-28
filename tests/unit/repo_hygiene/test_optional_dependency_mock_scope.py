"""Optional-dependency mock isolation guard for issue #7307."""

from __future__ import annotations

import ast
import subprocess  # nosec B404 - fixed git invocation, no shell
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TESTS_ROOT = _REPO_ROOT / "tests"
_OPTIONAL_DEPENDENCY_MODULES = frozenset(
    {
        "cv2",
        "imageio",
        "mujoco",
        "mujoco.viewer",
        "opensim",
        "casadi",
        "pinocchio",
        "pinocchio.casadi",
        "pydrake",
        "pydrake.all",
    }
)
_PYTEST_COLLECTION_HOOKS = frozenset(
    {
        "pytest_configure",
        "pytest_sessionstart",
        "pytest_collection_modifyitems",
        "pytest_collect_file",
        "pytest_pycollect_makeitem",
        "pytest_generate_tests",
    }
)
_PREEXISTING_MODULE_SCOPE_PATCH_ALLOWLIST = frozenset(
    {
        ("tests/integration/test_physics_engines_strict.py", 62),
        ("tests/ui/test_golf_gui_tabs.py", 228),
        ("tests/integration/test_physics_interfaces.py", 20),
        ("tests/unit/engines/physics_engines/mujoco/test_docker_examples.py", 11),
        ("tests/unit/shared_python/test_plotting_coverage.py", 25),
        ("tests/unit/test_drake_wrapper.py", 88),
        ("tests/unit/test_golf_suite_launcher.py", 185),
        ("tests/unit/test_openpose_estimator.py", 17),
        ("tests/unit/test_opensim_physics_engine.py", 40),
        ("tests/unit/test_shot_tracer.py", 20),
        ("tests/unit/test_video_pose_pipeline.py", 30),
    }
)


def _child_statement_blocks(statement: ast.stmt) -> list[list[ast.stmt]]:
    child_blocks: list[list[ast.stmt]] = []
    if isinstance(statement, ast.If | ast.For | ast.While):
        child_blocks.extend([statement.body, statement.orelse])
    elif isinstance(statement, ast.With):
        child_blocks.append(statement.body)
    elif isinstance(statement, ast.Try):
        child_blocks.extend([statement.body, statement.orelse, statement.finalbody])
        child_blocks.extend(handler.body for handler in statement.handlers)
    elif isinstance(statement, ast.Match):
        child_blocks.extend(case.body for case in statement.cases)
    return child_blocks


def _iter_nested_statements(statements: Iterable[ast.stmt]) -> Iterator[ast.stmt]:
    for statement in statements:
        if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        yield statement

        for block in _child_statement_blocks(statement):
            yield from _iter_nested_statements(block)


def _iter_module_scope_nodes(statements: Iterable[ast.stmt]) -> Iterator[ast.stmt]:
    yield from _iter_nested_statements(statements)


def _iter_pytest_collection_hook_nodes(
    statements: Iterable[ast.stmt],
) -> Iterator[tuple[str, ast.stmt]]:
    for statement in statements:
        if not isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if statement.name not in _PYTEST_COLLECTION_HOOKS:
            continue
        for child in _iter_nested_statements(statement.body):
            yield statement.name, child


def _is_sys_modules(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "modules"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _is_sys_modules_reference(node: ast.AST) -> bool:
    return _is_sys_modules(node) or (
        isinstance(node, ast.Constant) and node.value == "sys.modules"
    )


def _sys_modules_literal_key(target: ast.AST) -> str | None:
    if not isinstance(target, ast.Subscript) or not _is_sys_modules(target.value):
        return None
    if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
        return target.slice.value
    return None


def _assigned_optional_dependency(statement: ast.stmt) -> str | None:
    targets: Iterable[ast.AST]
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, ast.AnnAssign | ast.AugAssign):
        targets = [statement.target]
    else:
        return None

    for target in targets:
        key = _sys_modules_literal_key(target)
        if key in _OPTIONAL_DEPENDENCY_MODULES:
            return key
    return None


def _calls_patch_dict_sys_modules(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.With):
        return False
    for item in statement.items:
        call = item.context_expr
        if not isinstance(call, ast.Call):
            continue
        if not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "dict"
            and call.args
        ):
            continue
        if _is_sys_modules_reference(call.args[0]):
            return True
    return False


def _offenders_for_tree(relative: str, tree: ast.Module) -> list[str]:
    offenders: list[str] = []
    for statement in _iter_module_scope_nodes(tree.body):
        dependency = _assigned_optional_dependency(statement)
        if dependency is not None:
            offenders.append(f"{relative}:{statement.lineno} assigns {dependency}")
        if _calls_patch_dict_sys_modules(statement):
            if (
                relative,
                statement.lineno,
            ) in _PREEXISTING_MODULE_SCOPE_PATCH_ALLOWLIST:
                continue
            offenders.append(
                f"{relative}:{statement.lineno} patches sys.modules at module scope"
            )

    for hook_name, statement in _iter_pytest_collection_hook_nodes(tree.body):
        dependency = _assigned_optional_dependency(statement)
        if dependency is not None:
            offenders.append(
                f"{relative}:{statement.lineno} assigns {dependency} in {hook_name}"
            )
        if _calls_patch_dict_sys_modules(statement):
            offenders.append(
                f"{relative}:{statement.lineno} patches sys.modules in {hook_name}"
            )
    return offenders


def _candidate_python_files() -> list[Path]:
    result = subprocess.run(  # nosec B603 - fixed args, no shell
        [
            "git",
            "grep",
            "-l",
            "-e",
            "sys.modules",
            "-e",
            "patch.dict",
            "--",
            "tests",
            "*.py",
        ],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode == 1:
        return []
    assert result.returncode == 0, result.stderr
    return [
        _REPO_ROOT / line.strip() for line in result.stdout.splitlines() if line.strip()
    ]


def test_no_module_scope_optional_dependency_sys_modules_mocks() -> None:
    offenders: list[str] = []
    for path in sorted(_candidate_python_files()):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except UnicodeDecodeError:
            continue

        relative = path.relative_to(_REPO_ROOT).as_posix()
        offenders.extend(_offenders_for_tree(relative, tree))

    assert not offenders, (
        "Optional dependency mocks must be installed by scoped fixtures or local "
        "helpers, not at module import/collection time:\n  " + "\n  ".join(offenders)
    )


def test_pytest_configure_optional_dependency_assignment_is_reported() -> None:
    tree = ast.parse(
        """
import sys
from unittest.mock import MagicMock

def pytest_configure(config):
    if "pydrake" not in sys.modules:
        sys.modules["pydrake"] = MagicMock()
"""
    )

    offenders = _offenders_for_tree("tests/unit/conftest.py", tree)

    assert offenders == ["tests/unit/conftest.py:7 assigns pydrake in pytest_configure"]


def test_pytest_collection_hook_patch_dict_is_reported() -> None:
    tree = ast.parse(
        """
from unittest.mock import MagicMock, patch

def pytest_sessionstart(session):
    with patch.dict("sys.modules", {"pinocchio": MagicMock()}):
        pass
"""
    )

    offenders = _offenders_for_tree("tests/unit/conftest.py", tree)

    assert offenders == [
        "tests/unit/conftest.py:5 patches sys.modules in pytest_sessionstart"
    ]


def test_pinocchio_and_casadi_are_optional_dependency_modules() -> None:
    assert {"pinocchio", "pinocchio.casadi", "casadi"} <= _OPTIONAL_DEPENDENCY_MODULES
