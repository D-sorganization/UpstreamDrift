"""Package-wide Law-of-Demeter / import-graph audit for ``motion_pipeline``.

Closeout for epic #4558. Extends the per-module test introduced in
PR #4620 (``tests/unit/motion_pipeline/test_lod.py``) by walking *every*
``*.py`` file under ``src/shared/python/motion_pipeline/`` and asserting
that none of them import from forbidden top-level packages.

The motion pipeline subpackage MUST stay independent of:

- ``src.engines.*``       - physics-engine implementations
- ``src.api.*``           - higher-level API layer
- ``src.apps.*``          - GUI applications
- ``src.tools.*``         - tooling / scripts
- ``src.deployment.*``    - deployment shims
- ``src.learning.*``      - ML pipelines

Backend modules that intentionally bridge to a physics engine
(``ik/*_backend.py``, plus a small allow-list under ``matching/``) are
exempted by the epic's LoD policy.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.motion_pipeline]


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "shared"
    / "python"
    / "motion_pipeline"
)

# Files exempt by epic policy. Stored as POSIX relative paths.
EXEMPT_RELATIVE_POSIX = {
    "ik/mujoco_backend.py",
    "ik/drake_backend.py",
    "ik/pinocchio_backend.py",
    "ik/opensim_backend.py",
    "matching/cmc.py",
    "matching/rra.py",
    "matching/torque_mujoco.py",
    "matching/trajopt_drake.py",
    "matching/inverse_dyn_pinocchio.py",
}

FORBIDDEN_PREFIXES = (
    "src.engines.",
    "src.api.",
    "src.apps.",
    "src.tools.",
    "src.deployment.",
    "src.learning.",
)
FORBIDDEN_EXACT = {
    "src.engines",
    "src.api",
    "src.apps",
    "src.tools",
    "src.deployment",
    "src.learning",
}


def _collect_imports(source: str) -> list[tuple[int, str]]:
    """Return ``(lineno, dotted_name)`` for every import statement in ``source``."""
    tree = ast.parse(source)
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            out.append((node.lineno, node.module))
            for alias in node.names:
                out.append((node.lineno, f"{node.module}.{alias.name}"))
    return out


def _violates(name: str) -> bool:
    if name in FORBIDDEN_EXACT:
        return True
    return any(name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)


def _enumerate_package_files() -> list[Path]:
    if not PACKAGE_ROOT.exists():
        return []
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def test_motion_pipeline_package_root_exists() -> None:
    """Sanity check - the motion_pipeline package is importable from disk."""
    assert (
        PACKAGE_ROOT.is_dir()
    ), f"Expected motion_pipeline package at {PACKAGE_ROOT}, not found."


def test_motion_pipeline_no_forbidden_imports_anywhere() -> None:
    """No file under ``motion_pipeline/`` (except backend bridges) imports forbidden roots.

    This is an aggregated assertion - we collect every violation across
    the package so a single failure surfaces every offending file in one
    error message.
    """
    files = _enumerate_package_files()
    assert files, "Found no .py files under motion_pipeline - test setup is broken."

    violations: list[str] = []
    for file_path in files:
        rel = file_path.relative_to(PACKAGE_ROOT).as_posix()
        if rel in EXEMPT_RELATIVE_POSIX:
            continue
        try:
            source = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:  # pragma: no cover - I/O fence
            violations.append(f"{rel}: could not read ({exc})")
            continue
        try:
            imports = _collect_imports(source)
        except SyntaxError as exc:
            violations.append(f"{rel}: syntax error ({exc})")
            continue
        for lineno, name in imports:
            if _violates(name):
                violations.append(f"{rel}:{lineno} imports forbidden '{name}'")

    assert (
        not violations
    ), "motion_pipeline package contains forbidden imports:\n  - " + "\n  - ".join(
        violations
    )


def test_exempt_list_only_references_existing_files() -> None:
    """Catch typos in EXEMPT_RELATIVE_POSIX before they silently disable checks."""
    missing = [
        rel for rel in EXEMPT_RELATIVE_POSIX if not (PACKAGE_ROOT / rel).is_file()
    ]
    assert not missing, (
        "EXEMPT_RELATIVE_POSIX references files that do not exist: "
        f"{missing}. Update the exempt list."
    )
