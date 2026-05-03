"""Validate pytest layout invariants for the repository.

Postconditions: the audit reports root-level tests, in-tree ``src`` tests, and
duplicate conftest fixture names in overlapping pytest scopes.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

LEGACY_SRC_TEST_DIRS = frozenset(
    {
        "src/engines/pendulum_models/javascript/tests",
        "src/engines/pendulum_models/matlab/tests",
        "src/engines/pendulum_models/python/double_pendulum_model/tests",
        "src/engines/physics_engines/drake/javascript/tests",
        "src/engines/physics_engines/drake/matlab/tests",
        "src/engines/physics_engines/drake/python/tests",
        "src/engines/physics_engines/mujoco/matlab/tests",
        "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/tests",
        "src/engines/physics_engines/mujoco/python/tests",
        "src/engines/physics_engines/opensim/python/tests",
        "src/engines/physics_engines/pinocchio/matlab/tests",
        "src/engines/physics_engines/pinocchio/python/motion_training/tests",
        "src/engines/physics_engines/pinocchio/python/tests",
        "src/engines/physics_engines/tests",
        "src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/tests",
        "src/engines/Simscape_Multibody_Models/2D_Golf_Model/python/tests",
        "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/tests",
        "src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/tests",
        "src/shared/python/chat/tests",
        "src/shared/python/dashboard/tests",
        "src/shared/python/data_processing/tests",
        "src/shared/python/humanoid_character_builder/tests",
        "src/shared/python/model_generation/tests",
        "src/shared/python/optimization/tests",
        "src/shared/python/plot_engine/tests",
        "src/shared/python/plot_theme/tests",
        "src/shared/python/spatial_algebra/tests",
        "src/shared/python/tests",
        "src/shared/python/upstream_drift_tools/tests",
    }
)

LEGACY_ROOT_TEST_FILES = frozenset(
    {
        "tests/test_architecture_dbc.py",
        "tests/test_ci_infrastructure.py",
        "tests/test_drag_drop_functionality.py",
        "tests/test_launcher_fixes.py",
        "tests/test_urdf_tools.py",
    }
)


@dataclass(frozen=True)
class LayoutFinding:
    """A single test-layout violation."""

    path: Path
    reason: str


@dataclass(frozen=True)
class FixtureDefinition:
    """A pytest fixture declared in a conftest file."""

    name: str
    conftest_path: Path


def _relative(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _fixture_name(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    for decorator in node.decorator_list:
        if _is_pytest_fixture(decorator):
            return node.name
    return None


def _is_pytest_fixture(decorator: ast.expr) -> bool:
    call = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(call, ast.Attribute):
        return call.attr == "fixture"
    return isinstance(call, ast.Name) and call.id == "fixture"


def _iter_fixture_definitions(conftest_path: Path) -> list[FixtureDefinition]:
    tree = ast.parse(conftest_path.read_text(encoding="utf-8"))
    fixtures = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = _fixture_name(node)
        if name is not None:
            fixtures.append(FixtureDefinition(name=name, conftest_path=conftest_path))
    return fixtures


def _scopes_overlap(parent: Path, child: Path) -> bool:
    parent_dir = parent.parent
    child_dir = child.parent
    return parent_dir == child_dir or parent_dir in child_dir.parents


def _audit_duplicate_fixtures(repo_root: Path) -> list[LayoutFinding]:
    conftests = [
        repo_root / "conftest.py",
        *sorted((repo_root / "tests").rglob("conftest.py")),
    ]
    existing_conftests = [path for path in conftests if path.exists()]
    fixtures_by_name: dict[str, list[FixtureDefinition]] = {}
    for conftest in existing_conftests:
        for fixture in _iter_fixture_definitions(conftest):
            fixtures_by_name.setdefault(fixture.name, []).append(fixture)

    findings = []
    for name, fixtures in fixtures_by_name.items():
        for index, fixture in enumerate(fixtures):
            earlier_fixtures = fixtures[:index]
            for earlier in earlier_fixtures:
                if not _scopes_overlap(earlier.conftest_path, fixture.conftest_path):
                    continue
                rel_path = _relative(fixture.conftest_path, repo_root)
                earlier_path = _relative(earlier.conftest_path, repo_root)
                findings.append(
                    LayoutFinding(
                        path=Path(rel_path),
                        reason=(
                            f"duplicate fixture {name!r} overlaps with {earlier_path}"
                        ),
                    )
                )
                break
    return findings


def audit_test_layout(repo_root: Path) -> list[LayoutFinding]:
    """Return all test-layout violations below ``repo_root``.

    Preconditions: ``repo_root`` must be an existing directory.
    Postconditions: returned findings use repository-relative paths.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be a pathlib.Path")
    if not repo_root.exists():
        raise ValueError(f"repo_root does not exist: {repo_root}")
    if not repo_root.is_dir():
        raise ValueError(f"repo_root must be a directory: {repo_root}")

    findings: list[LayoutFinding] = []
    tests_root = repo_root / "tests"
    if tests_root.exists():
        for test_file in sorted(tests_root.glob("test_*.py")):
            rel_path = _relative(test_file, repo_root)
            if rel_path in LEGACY_ROOT_TEST_FILES:
                continue
            findings.append(
                LayoutFinding(
                    path=Path(rel_path),
                    reason="root-level test file must live in a topic subdirectory",
                )
            )

    src_root = repo_root / "src"
    if src_root.exists():
        for path in sorted(src_root.rglob("tests")):
            rel_path = _relative(path, repo_root)
            if path.is_dir() and rel_path not in LEGACY_SRC_TEST_DIRS:
                findings.append(
                    LayoutFinding(
                        path=Path(rel_path),
                        reason="tests directory under src is not collected by root pytest",
                    )
                )

    findings.extend(_audit_duplicate_fixtures(repo_root))
    return findings


def main(argv: list[str] | None = None) -> int:
    """Run the test-layout audit and print findings for CI."""
    args = argv if argv is not None else sys.argv[1:]
    repo_root = Path(args[0]).resolve() if args else Path(__file__).resolve().parents[1]
    findings = audit_test_layout(repo_root)
    if not findings:
        return 0

    for finding in findings:
        print(f"{finding.path}: {finding.reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
