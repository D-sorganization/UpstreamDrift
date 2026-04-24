"""Shared API version resolution for server metadata."""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python <3.11 fallback
    import tomli as tomllib


@lru_cache(maxsize=1)
def get_app_version() -> str:
    """Return the canonical application version string."""
    for package_name in ("upstream-drift", "golf-modeling-suite"):
        try:
            return version(package_name)
        except PackageNotFoundError:
            continue

    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as fh:
            data = tomllib.load(fh)
        return str(data["project"]["version"])
    except (FileNotFoundError, KeyError, OSError, TypeError):
        return "0.0.0"
