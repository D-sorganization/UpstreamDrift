"""Paths a remote (REST or MCP) request may read and write (#15150 G8, D10).

Settings name ``input_roots_env``/``output_root_env`` (environment overrides)
and ``default_input_roots``/``default_output_root`` (repository-relative).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_roots(roots: Mapping[str, Any]) -> tuple[tuple[Path, ...], Path]:
    """Resolved input roots and output root (environment over settings)."""
    env_inputs = os.environ.get(roots["input_roots_env"], "")
    inputs = (
        env_inputs.split(os.pathsep) if env_inputs else roots["default_input_roots"]
    )
    output = os.environ.get(roots["output_root_env"]) or roots["default_output_root"]
    return (
        tuple((REPO_ROOT / p).resolve() for p in inputs if p),
        (REPO_ROOT / output).resolve(),
    )


def contained_below(value: Any, bases: Sequence[Path], where: str) -> Path:
    """``value`` resolved below one of ``bases`` (symlinks followed).

    Raises:
        TypeError: ``value`` is not a non-empty string.
        ValueError: the resolved path leaves every base.
    """
    if not isinstance(value, str) or not value:
        raise TypeError("path options take a non-empty string")
    for base in bases:
        resolved = (base / value).resolve()
        if resolved.is_relative_to(base):
            return resolved
    raise ValueError(f"path {value!r} is outside the allowed {where}")
