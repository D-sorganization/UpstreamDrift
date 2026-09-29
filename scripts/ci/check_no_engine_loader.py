#!/usr/bin/env python3
"""Enforce the canonical-target-loader contract (closes issue #4254).

CROSS_ENGINE_PARITY_SPEC.md §2.1 mandates::

    Engine-specific loaders are forbidden. Use the canonical Python loader in
    ``shared/python/motion_matching/load_club_target.py``.

This guard scans every Python file under
``src/engines/physics_engines/{mujoco,drake,pinocchio,opensim}/`` and rejects:

* function or async-function definitions whose name starts with
  ``load_club_target`` or ``load_target`` (case-insensitive); and
* function definitions matching engine-specific Excel/loader patterns
  (``load_*_excel``, ``load_*_c3d``).

It also rejects the legacy import patterns called out in the issue --
``from {engine}_loaders import ...`` and ``import {engine}_loaders`` --
where ``{engine}`` is one of ``opensim``, ``drake``, ``mujoco``, ``pinocchio``.

Allow-listed sources:

* the canonical loader at
  ``src/shared/python/motion_matching/load_club_target.py`` and the
  ``src/shared/python/motion_matching/loaders/`` sub-package; and
* thin re-exports of the canonical loader -- a module qualifies if every
  ``load_club_target*`` / ``load_target*`` symbol it defines is bound via an
  ``ImportFrom`` whose module path resolves to
  ``src.shared.python.motion_matching.load_club_target`` (or its
  ``loaders.{excel,c3d}`` siblings).

Design-by-contract:

* The script REQUIRES execution from the repository root -- the working
  directory must contain ``pyproject.toml`` -- and emits a descriptive
  ``RuntimeError`` otherwise. This protects callers from silently scanning
  the wrong tree.
* The script raises ``ValueError`` if the repo root does not contain
  ``src/engines/physics_engines/`` (the contract being enforced is moot in
  that case and the failure should be loud).
* Exit code is ``0`` on pass, ``1`` on violation, ``2`` on contract failure.
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

# -- engine surfaces under contract ------------------------------------------

ENGINE_DIRS: tuple[str, ...] = (
    "src/engines/physics_engines/mujoco",
    "src/engines/physics_engines/drake",
    "src/engines/physics_engines/pinocchio",
    "src/engines/physics_engines/opensim",
)

# Banned legacy module names from the issue body.
BANNED_LEGACY_MODULES: frozenset[str] = frozenset(
    {
        "opensim_loaders",
        "drake_loaders",
        "mujoco_loaders",
        "pinocchio_loaders",
    }
)

# Canonical loader (allow-listed) -- the gate must NOT flag this file
# even though it lives in src/ (it does not live under physics_engines/, but
# we list it here so the allow-list logic is explicit and discoverable).
CANONICAL_LOADER_REL = "src/shared/python/motion_matching/load_club_target.py"
CANONICAL_LOADER_DOTTED = "src.shared.python.motion_matching.load_club_target"
CANONICAL_LOADER_PACKAGE_DOTTED = "src.shared.python.motion_matching.loaders"

# Function-name prefixes the gate rejects when defined in engine code.
RESERVED_NAME_PREFIXES: tuple[str, ...] = ("load_club_target", "load_target")

# Engine-specific *target* loader patterns the gate also rejects -- catches
# Wiffle-style ``load_swing_target_excel`` / ``load_club_swing_xlsx`` names.
# We require BOTH a target-domain noun AND an Excel/c3d/xlsx token to avoid
# tripping on generic motion-capture readers like ``load_c3d`` that return
# raw marker arrays rather than a ``ClubTarget``.
TARGET_DOMAIN_TOKENS: tuple[str, ...] = ("target", "swing", "club")
TABLE_FORMAT_TOKENS: tuple[str, ...] = ("excel", "_xlsx", "xlsx_", "_c3d", "c3d_")


# -- contract checks ---------------------------------------------------------


def _assert_repo_root(cwd: Path) -> None:
    """DbC: caller must invoke us from the repository root.

    Precondition: ``cwd / 'pyproject.toml'`` exists.
    """
    if not (cwd / "pyproject.toml").is_file():
        raise RuntimeError(
            f"check_no_engine_loader.py must run at the repository root "
            f"(directory containing pyproject.toml); got cwd={cwd!s}"
        )


def _assert_engine_tree_present(repo_root: Path) -> None:
    """DbC: the contract surface must exist."""
    physics_root = repo_root / "src" / "engines" / "physics_engines"
    if not physics_root.is_dir():
        raise ValueError(
            f"src/engines/physics_engines/ not found under {repo_root!s}; "
            "this guard cannot enforce the canonical-loader contract on a "
            "tree that does not contain the physics-engine surface."
        )


# -- file discovery ----------------------------------------------------------


def _iter_engine_python_files(repo_root: Path) -> Iterator[Path]:
    """Yield every ``*.py`` file under the four engine directories.

    Skips ``__pycache__`` and any vendored third-party tree we identify by
    the conventional ``vendor`` segment.
    """
    for rel in ENGINE_DIRS:
        engine_root = repo_root / rel
        if not engine_root.is_dir():
            continue
        for path in engine_root.rglob("*.py"):
            parts = set(path.parts)
            if "__pycache__" in parts or "vendor" in parts:
                continue
            yield path


# -- AST analysis ------------------------------------------------------------


def _is_reserved_loader_name(name: str) -> bool:
    """Return True if ``name`` matches a forbidden target-loader pattern."""
    lname = name.lower()
    if any(lname.startswith(prefix) for prefix in RESERVED_NAME_PREFIXES):
        return True
    if not lname.startswith("load_"):
        return False
    has_domain = any(tok in lname for tok in TARGET_DOMAIN_TOKENS)
    has_format = any(tok in lname for tok in TABLE_FORMAT_TOKENS)
    return has_domain and has_format


def _resolve_relative_import(module: str | None, level: int, file_dotted: str) -> str:
    """Resolve a ``from .x import y`` style import to its absolute dotted form.

    ``file_dotted`` is the dotted module path of the file containing the
    import (e.g. ``src.engines.physics_engines.drake.foo``). The returned
    string is the absolute dotted path of the imported module, or an empty
    string if the relative-import level is invalid for the file location.
    """
    if level == 0:
        return module or ""
    parts = file_dotted.split(".")
    if level > len(parts):
        return ""  # malformed; treat as unresolved
    base = parts[: len(parts) - level]
    if module:
        base.append(module)
    return ".".join(base)


def _file_dotted_path(repo_root: Path, path: Path) -> str:
    """Best-effort dotted module path for ``path`` relative to ``repo_root``.

    The repo uses ``src/`` as a source root, so we strip neither the ``src``
    prefix (kept for clarity) nor the ``.py`` suffix (stripped). Relative
    paths under ``repo_root`` only.
    """
    rel = path.relative_to(repo_root).with_suffix("")
    return ".".join(rel.parts)


def _imports_canonical_loader(node: ast.ImportFrom, file_dotted: str) -> bool:
    """Return True if ``node`` imports from the canonical-loader module.

    Accepts both the absolute dotted path and any equivalent relative form
    that resolves to the same target.
    """
    resolved = _resolve_relative_import(node.module, node.level, file_dotted)
    if not resolved:
        return False
    if resolved in (CANONICAL_LOADER_DOTTED, CANONICAL_LOADER_PACKAGE_DOTTED):
        return True
    return resolved.startswith(
        (CANONICAL_LOADER_DOTTED + ".", CANONICAL_LOADER_PACKAGE_DOTTED + ".")
    )


def _scan_imports(
    tree: ast.AST, rel: str, file_dotted: str
) -> tuple[set[str], list[str]]:
    """Return (aliases imported from canonical loader, legacy-import violations)."""
    canonical_aliases: set[str] = set()
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_name = (node.module or "").split(".")[0]
            if module_name in BANNED_LEGACY_MODULES:
                violations.append(
                    f"{rel}:{node.lineno}: forbidden import "
                    f"'from {node.module} import ...' (banned engine-loader "
                    "module; use src.shared.python.motion_matching.load_club_target)"
                )
                continue
            if _imports_canonical_loader(node, file_dotted):
                for alias in node.names:
                    canonical_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in BANNED_LEGACY_MODULES:
                    violations.append(
                        f"{rel}:{node.lineno}: forbidden import "
                        f"'import {alias.name}' (banned engine-loader module; "
                        "use src.shared.python.motion_matching.load_club_target)"
                    )
    return canonical_aliases, violations


def _scan_function_defs(
    tree: ast.AST, rel: str, canonical_aliases: set[str]
) -> list[str]:
    """Reject reserved function-def names not bound through the canonical loader."""
    violations: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _is_reserved_loader_name(node.name)
            and node.name not in canonical_aliases
        ):
            violations.append(
                f"{rel}:{node.lineno}: engine-specific loader "
                f"'def {node.name}(...)' is forbidden by "
                "CROSS_ENGINE_PARITY_SPEC.md §2.1; import "
                "src.shared.python.motion_matching.load_club_target instead"
            )
    return violations


def _scan_file(repo_root: Path, path: Path) -> list[str]:
    """Return a list of human-readable violations for ``path``.

    An empty list means the file is compliant (or thinly re-exports the
    canonical loader, in which case any defined ``load_*`` symbols are
    considered allow-listed).
    """
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - exotic encodings
        return [f"{path}: cannot decode as utf-8: {exc}"]
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: syntax error at line {exc.lineno}: {exc.msg}"]

    rel = path.relative_to(repo_root).as_posix()
    file_dotted = _file_dotted_path(repo_root, path)

    canonical_aliases, legacy_violations = _scan_imports(tree, rel, file_dotted)
    def_violations = _scan_function_defs(tree, rel, canonical_aliases)
    return [*legacy_violations, *def_violations]


# -- driver ------------------------------------------------------------------


def collect_violations(repo_root: Path, files: Iterable[Path]) -> list[str]:
    """Run the AST scan over ``files`` and aggregate violations."""
    violations: list[str] = []
    for path in sorted(files):
        violations.extend(_scan_file(repo_root, path))
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reject engine-specific target loaders per "
            "CROSS_ENGINE_PARITY_SPEC.md §2.1 (closes issue #4254)."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override the repository root (defaults to current working directory).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cwd = Path.cwd()
    repo_root = (args.repo_root or cwd).resolve()
    try:
        _assert_repo_root(repo_root)
        _assert_engine_tree_present(repo_root)
    except (RuntimeError, ValueError) as exc:
        logger.error("FAIL (contract): %s", exc)
        return 2

    files = list(_iter_engine_python_files(repo_root))
    violations = collect_violations(repo_root, files)

    if violations:
        logger.error("FAIL: engine-specific target loader(s) detected:\n")
        for line in violations:
            logger.error("  %s", line)
        logger.error(
            "\nFix: import the canonical loader instead --\n"
            "    from src.shared.python.motion_matching.load_club_target "
            "import load_club_target"
        )
        return 1

    logger.info(
        "OK: %d engine Python files scanned; canonical-target-loader contract upheld.",
        len(files),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
