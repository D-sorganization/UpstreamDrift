# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Configuration guard: every ``pytest.mark.<name>`` literal must be registered.

Regression test for #7912. The repository runs with ``--strict-markers``, and
``docs/testing/testing-guide.md`` advertises marker-based selection, so an
unregistered marker silently degrades marker selection (and errors outright
whenever strict enforcement is active). This test walks the tracked test tree
and fails on any marker that is neither declared in ``pyproject.toml`` nor
provided by pytest itself or an installed plugin.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Markers supplied by pytest core or by plugins declared in ``pyproject.toml``
#: (``pytest-anyio``/``anyio``, ``pytest-timeout``). These are registered by the
#: plugin at runtime and must not be duplicated in our own ``markers`` list.
_PLUGIN_PROVIDED = frozenset(
    {
        "anyio",
        "filterwarnings",
        "no_cover",
        "parametrize",
        "skip",
        "skipif",
        "timeout",
        "tryfirst",
        "trylast",
        "usefixtures",
        "xfail",
    }
)

_SEARCH_GLOBS = (
    "tests/**/*.py",
    "src/**/tests/**/*.py",
)


def _markers_used(source: str) -> set[str]:
    """Return marker names reached via a real ``pytest.mark.<name>`` attribute access.

    The AST walk (rather than a regex over the raw text) keeps prose mentions
    such as ``pytest.mark.requires_<engine>`` inside docstrings out of the
    result set.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - defensive
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        parent = node.value
        if (
            isinstance(parent, ast.Attribute)
            and parent.attr == "mark"
            and isinstance(parent.value, ast.Name)
            and parent.value.id == "pytest"
        ):
            names.add(node.attr)
    return names


def _registered_markers() -> set[str]:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = config["tool"]["pytest"]["ini_options"]["markers"]
    return {entry.split(":", 1)[0].strip() for entry in declared}


def _iter_test_files() -> list[Path]:
    seen: set[Path] = set()
    for pattern in _SEARCH_GLOBS:
        for path in REPO_ROOT.glob(pattern):
            if not path.is_file():
                continue
            parts = set(path.parts)
            if parts & {".venv", "node_modules", "__pycache__", "vendor"}:
                continue
            seen.add(path)
    return sorted(seen)


def test_registered_marker_names_are_unique() -> None:
    """The ``markers`` list must not declare the same marker twice."""
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = [
        entry.split(":", 1)[0].strip()
        for entry in config["tool"]["pytest"]["ini_options"]["markers"]
    ]
    duplicates = sorted({name for name in declared if declared.count(name) > 1})
    assert (
        not duplicates
    ), f"duplicate marker declarations in pyproject.toml: {duplicates}"


def test_every_used_marker_is_registered() -> None:
    """No tracked test may use a marker absent from ``pyproject.toml``."""
    registered = _registered_markers() | _PLUGIN_PROVIDED

    offenders: dict[str, list[str]] = {}
    for path in _iter_test_files():
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # pragma: no cover - defensive
            continue
        for marker in _markers_used(source):
            if marker in registered:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            offenders.setdefault(marker, []).append(rel)

    assert not offenders, (
        "unregistered pytest markers found; add them to "
        "[tool.pytest.ini_options].markers in pyproject.toml: "
        + "; ".join(
            f"{marker} -> {sorted(files)[:5]}"
            for marker, files in sorted(offenders.items())
        )
    )
