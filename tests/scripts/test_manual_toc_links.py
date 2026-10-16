"""Every in-manual link must resolve to a real heading (#12866).

The user manual index and its chapter files are checked for dead ``](#anchor)``
and ``](file.md#anchor)`` links. Slugs follow GitHub's heading-id rules. This
test deliberately does not import the TOC generator, so the generator cannot
certify its own output.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANUAL_INDEX = ROOT / "docs" / "user_guide" / "upstream_drift_user_manual.md"
CHAPTER_DIR = MANUAL_INDEX.with_suffix("")

SAME_FILE_LINK = re.compile(r"\]\(#([^)\s]+)\)")
CROSS_FILE_LINK = re.compile(r"\]\(([^)#\s]+\.md)#([^)\s]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _manual_files() -> list[Path]:
    return [MANUAL_INDEX, *sorted(CHAPTER_DIR.glob("*.md"))]


def _prose_lines(text: str) -> list[str]:
    """Return lines outside fenced code blocks, so code samples are not parsed."""
    prose: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            prose.append(line)
    return prose


def _github_slug(title: str) -> str:
    """GitHub heading id: strip markup, lowercase, drop punctuation but - and _, spaces to -."""
    plain = re.sub(r"[`*~]", "", title).strip().lower()
    kept = "".join(ch for ch in plain if ch.isalnum() or ch in "-_ ")
    return kept.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    """Return every anchor a file exposes, with GitHub's -1, -2 dedupe suffixes."""
    seen: dict[str, int] = {}
    anchors: set[str] = set()
    for line in _prose_lines(path.read_text(encoding="utf-8")):
        match = HEADING.match(line)
        if not match:
            continue
        slug = _github_slug(match.group(2))
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def _dead_links() -> list[str]:
    anchors_by_file: dict[Path, set[str]] = {}

    def anchors_of(path: Path) -> set[str]:
        if path not in anchors_by_file:
            anchors_by_file[path] = _anchors(path)
        return anchors_by_file[path]

    dead: list[str] = []
    for source in _manual_files():
        for line in _prose_lines(source.read_text(encoding="utf-8")):
            for anchor in SAME_FILE_LINK.findall(line):
                if anchor not in anchors_of(source):
                    dead.append(f"{source.name} -> #{anchor}")
            for target, anchor in CROSS_FILE_LINK.findall(line):
                resolved = (source.parent / target).resolve()
                if not resolved.is_file():
                    dead.append(f"{source.name} -> {target} (missing file)")
                elif anchor not in anchors_of(resolved):
                    dead.append(f"{source.name} -> {target}#{anchor}")
    return dead


@pytest.mark.unit
def test_slug_rules_match_github_headings() -> None:
    assert _github_slug("Chapter 1: Introduction") == "chapter-1-introduction"
    assert _github_slug("1.1 Project Overview") == "11-project-overview"
    assert _github_slug("A.1 pyproject.toml Options") == "a1-pyprojecttoml-options"
    assert _github_slug("Design by Contract (DbC)") == "design-by-contract-dbc"


@pytest.mark.unit
def test_manual_index_and_chapters_exist() -> None:
    assert MANUAL_INDEX.is_file()
    assert len(list(CHAPTER_DIR.glob("*.md"))) == 28


@pytest.mark.unit
def test_every_manual_link_resolves_to_a_heading() -> None:
    dead = _dead_links()
    assert not dead, f"{len(dead)} dead manual links:\n" + "\n".join(dead)
