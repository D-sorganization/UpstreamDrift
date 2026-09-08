"""Shared-contract conftest: resolve the real or vendored Tools checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _candidate_tools_roots(mode: str = "local") -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    # Match root conftest's explicit override and command-line mode. A sibling
    # checkout named by workspace setup must not relabel a vendored-mode run.
    explicit = os.environ.get("TOOLS_REPO_PATH")
    if explicit or mode == "vendored":
        selected = Path(explicit) if explicit else repo_root / "vendor" / "ud-tools"
        return [selected.resolve()] if selected.exists() else []
    candidates = [
        (
            Path(os.environ["TOOLS_REPO_ROOT"])
            if os.environ.get("TOOLS_REPO_ROOT")
            else None
        ),
        repo_root / "_tools_dep",
        repo_root / "vendor" / "ud-tools",
        repo_root.parent / "Tools",
    ]
    return [
        candidate.resolve()
        for candidate in candidates
        if candidate and candidate.exists()
    ]


def _tools_python_paths(tools_root: Path) -> list[Path]:
    return [
        path
        for path in (
            tools_root / "src" / "shared" / "python",
            tools_root / "src",
            tools_root / "src" / "python" / "src",
        )
        if path.exists()
    ]


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    require_real_tools = _is_truthy(os.environ.get("REQUIRE_REAL_TOOLS_REPO"))
    tools_roots = _candidate_tools_roots(config.getoption("--tools-mode"))

    if require_real_tools and not tools_roots:
        raise RuntimeError(
            "REQUIRE_REAL_TOOLS_REPO=1 but no Tools checkout was found. "
            "Expected _tools_dep, vendor/ud-tools, or ../Tools."
        )

    if not tools_roots:
        return

    os.environ["TOOLS_REPO_ROOT"] = str(tools_roots[0])

    for path in reversed(_tools_python_paths(tools_roots[0])):
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)
