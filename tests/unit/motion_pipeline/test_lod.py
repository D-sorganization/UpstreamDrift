"""Law-of-Demeter import-graph enforcement for the motion pipeline.

The Canonical Intermediate Representation (CIR) and the non-backend
submodules of ``src.shared.python.motion_pipeline`` must remain free of
imports that pull in heavyweight engine, API, learning, or app code. This
test walks the AST of each guarded file and asserts none of its
``import``/``from ... import`` statements reference forbidden top-level
prefixes.

Backend submodules (``ik/*_backend.py``, the four matching solver modules)
are explicitly exempt because they are *supposed* to bind to their backing
physics/optimization engines; the LoD invariant only applies to the CIR,
preprocessing, scaling, sources framework, and orchestrator.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PKG_ROOT = REPO_ROOT / "src" / "shared" / "python" / "motion_pipeline"
CONTRACTS_FILE = PKG_ROOT / "contracts.py"

# Top-level prefixes that no LoD-guarded motion-pipeline module may import.
# Both ``src.X`` and bare ``X`` forms are checked because UpstreamDrift's
# package import path varies (``src.engines.*`` vs ``engines.*`` depending on
# how PYTHONPATH is configured).
_FORBIDDEN_ROOTS = (
    "engines",
    "api",
    "learning",
    "apps",
    "tools",
    "deployment",
)
FORBIDDEN_PREFIXES = tuple(f"src.{r}." for r in _FORBIDDEN_ROOTS) + tuple(
    f"{r}." for r in _FORBIDDEN_ROOTS
)
# Exact-name guards so ``import src.engines`` (no dot) is also caught.
FORBIDDEN_EXACT = frozenset(
    {f"src.{r}" for r in _FORBIDDEN_ROOTS} | set(_FORBIDDEN_ROOTS)
)

# Files exempt from the broader LoD sweep because their entire purpose is
# to bind to a backing engine.
EXEMPT_RELATIVE_PATHS = frozenset(
    {
        Path("ik") / "drake_backend.py",
        Path("ik") / "mujoco_backend.py",
        Path("ik") / "opensim_backend.py",
        Path("ik") / "pinocchio_backend.py",
        Path("matching") / "cmc.py",
        Path("matching") / "rra.py",
        Path("matching") / "torque_mujoco.py",
        Path("matching") / "trajopt_drake.py",
    }
)


def _collect_imports(file_path: Path) -> list[tuple[int, str]]:
    """Return ``[(lineno, module_name), ...]`` for every Import/ImportFrom."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            # ``from . import x`` -> module is None; skip relative imports.
            if node.module is None or node.level:
                continue
            # Record both the module and the imported names to catch
            # forbidden roots that would otherwise be missed.
            # For ``from src import engines``, record both ``src`` and
            # ``src.engines`` so FORBIDDEN_EXACT/FORBIDDEN_PREFIXES match.
            out.append((node.lineno, node.module))
            for alias in node.names:
                full_name = f"{node.module}.{alias.name}"
                out.append((node.lineno, full_name))
    return out


def _is_forbidden(module_name: str) -> bool:
    return module_name in FORBIDDEN_EXACT or module_name.startswith(FORBIDDEN_PREFIXES)


def _format_violations(file_path: Path, viols: list[tuple[int, str]]) -> str:
    rel = file_path.relative_to(REPO_ROOT)
    lines = [f"LoD violation(s) in {rel}:"]
    for lineno, mod in viols:
        lines.append(f"  {rel}:{lineno}: imports forbidden module {mod!r}")
    return "\n".join(lines)


# =============================================================================
# Test 1: contracts.py is the LoD-critical module per issue #4560.
# =============================================================================


def test_contracts_imports_are_lod_clean() -> None:
    """``contracts.py`` must only import stdlib + pydantic + numpy + DbC."""
    assert CONTRACTS_FILE.is_file(), f"contracts.py missing at {CONTRACTS_FILE}"
    imports = _collect_imports(CONTRACTS_FILE)

    stdlib = sys.stdlib_module_names
    allowed_prefixes = (
        "pydantic",
        "numpy",
        "src.shared.python.contracts",
        "shared.python.contracts",
    )

    violations: list[tuple[int, str]] = []
    for lineno, mod in imports:
        top = mod.split(".")[0]
        if top in stdlib:
            continue
        if mod.startswith(allowed_prefixes) or mod in {
            "src.shared.python.contracts",
            "shared.python.contracts",
        }:
            continue
        if _is_forbidden(mod):
            violations.append((lineno, mod))
            continue
        # Anything else not on the allow-list is also a LoD violation for
        # contracts.py — keep the CIR module surgically minimal.
        violations.append((lineno, mod))

    assert not violations, _format_violations(CONTRACTS_FILE, violations)


# =============================================================================
# Test 2: every non-backend file in motion_pipeline must avoid forbidden roots.
# =============================================================================


def _guarded_files() -> list[Path]:
    files: list[Path] = []
    for p in sorted(PKG_ROOT.rglob("*.py")):
        rel = p.relative_to(PKG_ROOT)
        if rel in EXEMPT_RELATIVE_PATHS:
            continue
        if "__pycache__" in rel.parts:
            continue
        files.append(p)
    return files


_GUARDED = _guarded_files()


@pytest.mark.parametrize(
    "file_path",
    _GUARDED,
    ids=[str(p.relative_to(PKG_ROOT)).replace("\\", "/") for p in _GUARDED],
)
def test_motion_pipeline_module_no_forbidden_imports(file_path: Path) -> None:
    """Every CIR/preprocessing/scaling/orchestrator/sources file is LoD-clean."""
    imports = _collect_imports(file_path)
    violations = [(ln, m) for ln, m in imports if _is_forbidden(m)]
    assert not violations, _format_violations(file_path, violations)


# =============================================================================
# Test 3: backend exemptions still exist (regression guard against rename).
# =============================================================================


def test_exempt_backend_files_exist() -> None:
    """If a backend file is renamed, this test fails so we revisit the exemption."""
    missing = [rel for rel in EXEMPT_RELATIVE_PATHS if not (PKG_ROOT / rel).is_file()]
    assert not missing, (
        "Exempt backend files no longer exist (rename or removal?): "
        f"{[str(m) for m in missing]}. Update EXEMPT_RELATIVE_PATHS in test_lod.py."
    )
