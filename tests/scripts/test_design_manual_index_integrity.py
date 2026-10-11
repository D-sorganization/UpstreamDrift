"""Integrity of the design-manual chapter includes in ``index.qmd`` (#12217).

Every chapter file must be included exactly once, every include must resolve,
and no two chapter files may share a numeric filename prefix. Prefix collisions
that predate this check are grandfathered below; the allowlist may only shrink.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANUAL_ROOT = REPO_ROOT / "manuals" / "upstreamdrift"
CHAPTERS = MANUAL_ROOT / "chapters"
pytestmark = pytest.mark.unit

_INCLUDE = re.compile(r"\{\{<\s*include\s+chapters/(\S+?)\s*>\}\}")
_PREFIX = re.compile(r"^(\d+)-")

# Prefixes shared by two chapter files on main when #12217 landed. Renumbering
# them touches cross-links in in-flight manual PRs, so they are tracked for a
# follow-up instead of renamed here. Remove an entry once its collision is gone.
GRANDFATHERED_PREFIX_COLLISIONS = frozenset(
    {"13", "16", "17", "24", "25", "27", "28", "30"}
)


def _included_chapters() -> list[str]:
    return _INCLUDE.findall((MANUAL_ROOT / "index.qmd").read_text(encoding="utf-8"))


def _chapter_files() -> list[str]:
    return sorted(path.name for path in CHAPTERS.glob("*.qmd"))


def _prefix_collisions() -> dict[str, list[str]]:
    by_prefix: dict[str, list[str]] = defaultdict(list)
    for name in _chapter_files():
        match = _PREFIX.match(name)
        assert match is not None, f"chapter {name} lacks a numeric prefix"
        by_prefix[match.group(1)].append(name)
    return {prefix: names for prefix, names in by_prefix.items() if len(names) > 1}


def test_every_include_resolves_to_a_chapter_file() -> None:
    missing = sorted(set(_included_chapters()) - set(_chapter_files()))
    assert missing == [], f"index.qmd includes missing chapters: {missing}"


def test_no_chapter_is_included_twice() -> None:
    repeated = sorted(
        name for name, count in Counter(_included_chapters()).items() if count > 1
    )
    assert repeated == [], f"index.qmd includes chapters more than once: {repeated}"


def test_every_chapter_file_is_included() -> None:
    orphaned = sorted(set(_chapter_files()) - set(_included_chapters()))
    assert orphaned == [], f"chapters missing from index.qmd: {orphaned}"


def test_no_new_chapter_number_collisions() -> None:
    new = {
        prefix: names
        for prefix, names in _prefix_collisions().items()
        if prefix not in GRANDFATHERED_PREFIX_COLLISIONS
    }
    assert new == {}, f"chapter filename prefixes reused: {new}"


def test_grandfathered_collisions_are_still_present() -> None:
    resolved = sorted(GRANDFATHERED_PREFIX_COLLISIONS - set(_prefix_collisions()))
    assert resolved == [], (
        f"prefix collisions {resolved} are resolved; drop them from "
        "GRANDFATHERED_PREFIX_COLLISIONS"
    )
