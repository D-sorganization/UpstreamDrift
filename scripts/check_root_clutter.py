#!/usr/bin/env python3
"""Fail if the repo root contains files outside the allowlist."""

from __future__ import annotations

import sys
from pathlib import Path

ALLOWLIST = frozenset(
    {
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
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
        "launch.bat",
        "install.sh",
        "VERSION",
        "sidekick.spec",
        "fix.py",
        "patch_local_server_main.py",
    }
)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    bad = []
    for entry in root.iterdir():
        if entry.is_dir() or entry.name.startswith("."):
            continue
        if entry.name not in ALLOWLIST:
            bad.append(entry.name)
    if bad:
        print("Disallowed files at repo root:", file=sys.stderr)
        for name in sorted(bad):
            print(f"  {name}", file=sys.stderr)
        print(
            "\nMove or delete them. Edit ALLOWLIST in this script if you "
            "are adding a new top-level file (PR review required).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
