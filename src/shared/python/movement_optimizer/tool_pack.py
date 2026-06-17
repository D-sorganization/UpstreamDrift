# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tool-pack discovery surface for the UpstreamDrift launcher integration.

Exposes the three callables required by the ``biomech.tool_pack`` entry-point
group so the UpstreamDrift launcher can discover this repo without importing
the optimizer's heavy modules:

* :func:`manifest` -- parsed ``tool_pack.yaml`` contents.
* :func:`list_exercises` -- enumerated ``supported_exercises`` IDs.
* :func:`run_headless` -- run a single optimization and write JSON to disk.

See the umbrella tracking issue
``D-sorganization/UpstreamDrift#5179`` and the coordination issue
``D-sorganization/Movement-Optimizer#456``.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

_MANIFEST_FILENAME: str = "tool_pack.yaml"


def _load_manifest_text() -> str:
    """Return the raw text of ``tool_pack.yaml``.

    Editable installs find the manifest at the repository root; the loader
    walks up from the package directory until it locates the file. When a
    future wheel ships the manifest inside the package via setuptools
    ``package-data``, ``importlib.resources`` resolves it directly.
    """
    pkg = resources.files("movement_optimizer")
    candidate = pkg / _MANIFEST_FILENAME
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    pkg_path = Path(str(pkg)).resolve()
    for parent in (pkg_path, *pkg_path.parents):
        repo_manifest = parent / _MANIFEST_FILENAME
        if repo_manifest.is_file():
            return repo_manifest.read_text(encoding="utf-8")
    raise FileNotFoundError(f"{_MANIFEST_FILENAME} not found alongside movement_optimizer.")


def manifest() -> dict[str, Any]:
    """Return the parsed ``tool_pack.yaml`` manifest as a dictionary."""
    data = yaml.safe_load(_load_manifest_text())
    if not isinstance(data, dict):
        raise ValueError(f"{_MANIFEST_FILENAME} must contain a YAML mapping")
    return data


def list_exercises() -> list[str]:
    """Return the ordered list of exercise IDs declared in ``supported_exercises``."""
    data = manifest()
    exercises = data.get("supported_exercises", [])
    if not isinstance(exercises, list):
        raise ValueError("manifest 'supported_exercises' must be a list")
    return [str(name) for name in exercises]


def run_headless(exercise: str, output: Path | str) -> int:
    """Run a single headless optimization for *exercise*, write JSON to *output*.

    Returns the underlying CLI exit code (0 on success).
    """
    from .cli import main as cli_main

    return cli_main(["--exercise", exercise, "--output", str(output)])


__all__ = ["list_exercises", "manifest", "run_headless"]
