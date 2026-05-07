#!/usr/bin/env python3
"""Audit engine fit drivers for forbidden engine-specific target loaders."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

CANONICAL_LOADER_MODULE = "src.shared.python.motion_matching.load_club_target"
FORBIDDEN_LOADER_MODULES = frozenset(
    {
        "drake_loaders",
        "mujoco_loaders",
        "opensim_loaders",
        "pinocchio_loaders",
    }
)


@dataclass(frozen=True)
class EngineLoaderFinding:
    """A single engine-loader policy violation."""

    path: Path
    reason: str


def _relative(path: Path, repo_root: Path) -> Path:
    return Path(path.relative_to(repo_root).as_posix())


def _is_fit_swing_file(path: Path, repo_root: Path) -> bool:
    rel_parts = path.relative_to(repo_root).parts
    if "tests" in rel_parts:
        return False
    return path.name == "fit_swing.py" or (
        path.name.startswith("fit_swing_") and path.suffix == ".py"
    )


def _iter_fit_swing_files(repo_root: Path) -> list[Path]:
    src_root = repo_root / "src"
    if not src_root.exists():
        return []
    return [
        path
        for path in sorted(src_root.rglob("fit_swing*.py"))
        if path.is_file() and _is_fit_swing_file(path, repo_root)
    ]


def _base_module_name(module_name: str) -> str:
    return module_name.split(".", 1)[0]


def _imported_forbidden_loaders(tree: ast.Module) -> set[str]:
    forbidden: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = _base_module_name(alias.name)
                if module_name in FORBIDDEN_LOADER_MODULES:
                    forbidden.add(module_name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            module_name = _base_module_name(node.module)
            if module_name in FORBIDDEN_LOADER_MODULES:
                forbidden.add(module_name)
    return forbidden


def audit_engine_loaders(repo_root: Path) -> list[EngineLoaderFinding]:
    """Return fit-driver loader-policy violations below ``repo_root``.

    Preconditions: ``repo_root`` must be an existing directory.
    Postconditions: findings use repository-relative POSIX paths.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be a pathlib.Path")
    if not repo_root.exists():
        raise ValueError(f"repo_root does not exist: {repo_root}")
    if not repo_root.is_dir():
        raise ValueError(f"repo_root must be a directory: {repo_root}")

    findings: list[EngineLoaderFinding] = []
    for fit_file in _iter_fit_swing_files(repo_root):
        tree = ast.parse(fit_file.read_text(encoding="utf-8"), filename=str(fit_file))
        for module_name in sorted(_imported_forbidden_loaders(tree)):
            findings.append(
                EngineLoaderFinding(
                    path=_relative(fit_file, repo_root),
                    reason=(
                        f"forbidden engine-specific loader import {module_name!r}; "
                        f"use {CANONICAL_LOADER_MODULE!r}"
                    ),
                )
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    """Run the engine-loader audit and print findings for CI."""
    args = argv if argv is not None else sys.argv[1:]
    repo_root = Path(args[0]).resolve() if args else Path(__file__).resolve().parents[1]
    findings = audit_engine_loaders(repo_root)
    if not findings:
        print("PASS")
        return 0

    for finding in findings:
        print(f"{finding.path}: {finding.reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
