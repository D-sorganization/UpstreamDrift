"""Tile help pages link AffineDrift theory (Refs #15016)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.affine_drift_theory_links import (
    AFFINEDRIFT_HOST,
    background_section_present,
    help_markdown_paths,
    load_theory_registry,
)

pytestmark = pytest.mark.unit


def test_theory_registry_covers_every_help_page() -> None:
    registry = load_theory_registry()
    stems = {path.stem for path in help_markdown_paths()}
    assert stems == set(registry)


def test_every_help_page_links_affine_drift_theory() -> None:
    registry = load_theory_registry()
    missing: list[str] = []
    for path in help_markdown_paths():
        link = registry[path.stem]
        text = path.read_text(encoding="utf-8")
        if not background_section_present(text, link):
            missing.append(path.name)
    assert not missing, f"help pages missing AffineDrift Background: {missing}"


def test_readme_points_at_affine_drift_theory() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "README.md").read_text(encoding="utf-8")
    assert AFFINEDRIFT_HOST in text
    assert "Theory" in text
