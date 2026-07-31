"""Law-of-Demeter import-graph enforcement for source adapters.

Adapters must depend only on stdlib + a small allowlist of third-party
packages + the motion_pipeline contracts/sources base+registry. They MUST
NOT import from engines, API surface, apps, deployment, learning, or
tools.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCES_DIR = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "shared"
    / "python"
    / "motion_pipeline"
    / "sources"
)

FORBIDDEN_PREFIXES = (
    "src.engines",
    "src.api",
    "src.apps",
    "src.tools",
    "src.deployment",
    "src.learning",
)

ALLOWED_FIRST_PARTY = (
    "src.shared.python.contracts",
    "src.shared.python.motion_pipeline.contracts",
    "src.shared.python.motion_pipeline.sources",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def _adapter_files() -> list[Path]:
    return sorted(p for p in SOURCES_DIR.glob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("path", _adapter_files(), ids=lambda p: p.name)
def test_no_forbidden_imports(path: Path) -> None:
    mods = _imported_modules(path)
    for m in mods:
        for forbidden in FORBIDDEN_PREFIXES:
            assert not m.startswith(
                forbidden
            ), f"{path.name} imports {m!r}; forbidden prefix {forbidden!r}"


@pytest.mark.parametrize("path", _adapter_files(), ids=lambda p: p.name)
def test_first_party_imports_are_allowlisted(path: Path) -> None:
    mods = _imported_modules(path)
    for m in mods:
        if m.startswith("src."):
            assert any(
                m.startswith(p) for p in ALLOWED_FIRST_PARTY
            ), f"{path.name} imports {m!r}, which is not in the LoD allowlist."
