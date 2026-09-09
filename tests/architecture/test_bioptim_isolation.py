"""Epic #9762 Phase 0.3: ``bioptim`` may only be imported inside ``optimization/ocp``.

bioptim ships breaking API changes in every minor release and has a small
core team; confining every ``import bioptim`` to one subpackage keeps a
``casadi.Opti`` rewrite a two-day fallback instead of a fleet-wide hunt.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
ALLOWED_PREFIX = SRC_ROOT / "shared" / "python" / "optimization" / "ocp"
FORBIDDEN_ROOTS = ("bioptim", "biorbd", "biorbd_casadi", "bioviz", "pyorerun")
SKIP_DIRS = {"node_modules", "__pycache__", "vendor", "legacy"}


def _imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _violations() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for path in SRC_ROOT.rglob("*.py"):
        if SKIP_DIRS.intersection(path.parts):
            continue
        if ALLOWED_PREFIX in path.parents:
            continue
        for name in _imports(path):
            if name.split(".", 1)[0] in FORBIDDEN_ROOTS:
                found.append((str(path.relative_to(REPO_ROOT)), name))
    return sorted(found)


def test_bioptim_is_only_imported_inside_the_ocp_package() -> None:
    violations = _violations()
    assert not violations, (
        "bioptim/biorbd imported outside optimization/ocp:\n"
        + "\n".join(f"  {file} imports {module}" for file, module in violations)
    )


def test_ocp_package_imports_without_bioptim() -> None:
    """The package's public surface must not import bioptim eagerly."""
    import importlib
    import sys

    module = importlib.import_module("src.shared.python.optimization.ocp")
    assert hasattr(module, "bioptim_available")
    # Touching the lazy names must not have imported bioptim as a side effect
    # of merely importing the package.
    assert "bioptim" not in sys.modules or module.bioptim_available()
