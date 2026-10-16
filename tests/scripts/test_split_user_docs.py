"""Characterization tests for the split user manual and functional test plan (#12850)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 51200
SPLITS = (
    ROOT / "docs" / "user_guide" / "upstream_drift_user_manual.md",
    ROOT / "docs" / "testing" / "functional-test-plan.md",
)
LINK = re.compile(r"\]\(([^)#\s]+\.md)(?:#([^)\s]+))?\)")


def _parts(index: Path) -> list[Path]:
    return sorted(index.with_suffix("").glob("*.md"))


def _slugs(text: str) -> set[str]:
    seen: dict[str, int] = {}
    slugs: set[str] = set()
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        match = None if fenced else re.match(r"#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            title = re.sub(r"[`*]", "", match.group(1))
            slug = re.sub(r"[^\w\- ]", "", title.strip().lower()).replace(" ", "-")
            count = seen.get(slug, 0)
            seen[slug] = count + 1
            slugs.add(slug if count == 0 else f"{slug}-{count}")
    return slugs


@pytest.mark.unit
@pytest.mark.parametrize("index", SPLITS, ids=lambda p: p.stem)
def test_index_and_parts_fit_doc_size_budget(index: Path) -> None:
    parts = _parts(index)
    assert parts, f"{index.name} must have chapter files"
    for path in (index, *parts):
        assert path.stat().st_size < MAX_BYTES, path


@pytest.mark.unit
@pytest.mark.parametrize("index", SPLITS, ids=lambda p: p.stem)
def test_index_links_every_part_and_anchors_resolve(index: Path) -> None:
    text = index.read_text(encoding="utf-8")
    linked: set[str] = set()
    for target, anchor in LINK.findall(text):
        resolved = (index.parent / target).resolve()
        assert resolved.is_file(), target
        linked.add(resolved.name)
        if anchor:
            assert anchor in _slugs(resolved.read_text(encoding="utf-8")), (
                target,
                anchor,
            )
    assert {p.name for p in _parts(index)} <= linked


@pytest.mark.unit
def test_manual_help_content_includes_index_and_chapters() -> None:
    from src.shared.python.gui_pkg import help_system

    content = help_system.get_user_manual_content()
    assert "UpstreamDrift User Manual" in content
    assert "# Chapter 1: Introduction" in content
    assert "## Appendix F: Glossary" in content


@pytest.mark.unit
def test_manual_readers_scan_chapter_files() -> None:
    from scripts.ci import check_tutorial_imports
    from scripts import check_formatter_guidance

    chapters = _parts(SPLITS[0])
    assert set(chapters) <= set(check_tutorial_imports.DOC_TARGETS)
    assert check_formatter_guidance.manual_paths() == [SPLITS[0], *chapters]
