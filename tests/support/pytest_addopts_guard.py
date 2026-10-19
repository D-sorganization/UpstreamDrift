"""Helpers for guarding default addopts marker exclusions (#13000)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_LANES_CONFIG = REPO_ROOT / "scripts" / "config" / "pytest_addopts_exclusion_lanes.json"
_ADDOPTS_MARKER_EXPR = re.compile(r'-m\s+(["\'])(.*?)\1')


def load_exclusion_lane_config() -> dict[str, Any]:
    """Load the documented lane map for markers negated in default addopts."""
    with _LANES_CONFIG.open(encoding="utf-8") as handle:
        return json.load(handle)


def markers_negated_in_default_addopts(pyproject_text: str) -> list[str]:
    """Return marker names excluded by the root ``addopts`` ``-m`` expression."""
    match = re.search(
        r'addopts\s*=\s*".*-m\s+\\"([^\\"]+)\\"',
        pyproject_text,
    )
    if not match:
        raise ValueError("Could not parse addopts -m expression from pyproject.toml")
    expr = match.group(1)
    return re.findall(r"not\s+(\w+)", expr)


def pytest_marker_expressions(text: str) -> list[str]:
    """Extract pytest ``-m`` expressions from workflow or shell text."""
    return [expr for _, expr in _ADDOPTS_MARKER_EXPR.findall(text)]


def marker_selected_positively(marker: str, expr: str) -> bool:
    """Return True when ``expr`` selects ``marker`` without negating it."""
    if marker not in expr:
        return False
    return f"not {marker}" not in expr


def file_selects_marker_positively(path: Path, marker: str) -> bool:
    """Return True when a CI script/workflow positively selects ``marker``."""
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return any(
        marker_selected_positively(marker, expr)
        for expr in pytest_marker_expressions(text)
    )
