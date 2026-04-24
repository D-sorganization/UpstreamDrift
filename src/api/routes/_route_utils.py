"""Shared utilities for API route modules."""

from __future__ import annotations

from pathlib import Path


def find_project_root() -> Path:
    """Find the project root directory by looking for known markers."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src" / "shared" / "urdf").exists():
            return parent
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()
