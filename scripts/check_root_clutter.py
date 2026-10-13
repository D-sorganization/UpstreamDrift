#!/usr/bin/env python3
"""Fail if the repo root contains files or directories outside the allowlists."""

from __future__ import annotations

import sys
from pathlib import Path

ALLOWLIST: frozenset[str] = frozenset(
    {
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        # Fleet agent-handoff policy (Repository_Management#1390) puts the
        # current-state handoff document at the repo root.
        "AGENT_HANDOFF.md",
        "CLAUDE.md",
        "SECURITY.md",
        "SPEC.md",
        "AGENTS.md",
        "pyproject.toml",
        "Makefile",
        "Cargo.toml",
        "rust-toolchain.toml",
        "package.json",
        "package-lock.json",
        "environment.yml",
        "requirements.lock",
        "requirements-dev.lock",
        "requirements-tools.txt",  # Tools release-wheel pin (UD #9406)
        "alembic.ini",
        "Dockerfile",
        "Dockerfile.heavy_test",
        "Dockerfile.modular",
        "docker-compose.yml",
        "docker-compose.gpu.yml",
        "docker-compose.profiles.yml",
        "build_hooks.py",
        "conftest.py",
        "GEMINI.md",
        "launch_golf_suite.py",
        # Unified Launcher entry point (#6620). Substantive CLI not yet wired
        # into [project.scripts]; mirrors launch_golf_suite.py until promoted.
        "launch_upstream_drift.py",
        "launch.bat",
        "install.sh",
        "VERSION",
        "sidekick.spec",
    }
)

DIRECTORY_ALLOWLIST: frozenset[str] = frozenset(
    {
        "apps",
        "assets",
        "benchmarks",
        "build",
        "candidates",
        "changes",
        "data",
        "deploy",
        "dist",
        "docker",
        "docs",
        "evidence",
        "examples",
        "htmlcov",
        "installer",
        "knowledge",
        "logs",
        "manuals",
        "models",
        "notebooks",
        "output",
        "reports",
        "rust_core",
        "saved_states",
        "schemas",
        "scripts",
        "shared",
        "shared_scripts",
        "src",
        "stubs",
        "temp",
        "tests",
        "tools",
        "ui",
        "venv",
        "vendor",
        "__pycache__",
    }
)


def check_root_clutter(root: Path | None = None) -> list[str]:
    """Return a sorted list of disallowed files or directories in the root directory."""
    if root is None:
        root = Path(__file__).resolve().parent.parent
    bad: list[str] = []
    for entry in root.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            if entry.name not in DIRECTORY_ALLOWLIST:
                bad.append(f"{entry.name}/")
        elif entry.name not in ALLOWLIST:
            bad.append(entry.name)
    return sorted(bad)


def main(root: Path | None = None) -> int:
    bad = check_root_clutter(root)
    if bad:
        print("Disallowed files or directories at repo root:", file=sys.stderr)
        for name in bad:
            print(f"  {name}", file=sys.stderr)
        print(
            "\nMove or delete them. Edit ALLOWLIST or DIRECTORY_ALLOWLIST in this script if you "
            "are adding a new top-level entry (PR review required).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
