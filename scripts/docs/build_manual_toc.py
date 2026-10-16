#!/usr/bin/env python3
"""Rebuild the user manual table of contents from the real chapter headings (#12866).

The index ``docs/user_guide/upstream_drift_user_manual.md`` holds a block between
``MARKER_START`` and ``MARKER_END``. ``--write`` regenerates that block from the
chapter files; ``--check`` exits non-zero when the block is stale.

Each chapter file contributes one entry per ``Chapter N`` or ``Appendix X``
heading (H1 or H2). An entry lists the headings exactly one level below it:
H2 sections under a chapter, H3 subsections under an appendix. H3 headings under
chapters are omitted because they would push the index past the 51,200-byte
documentation budget. Headings inside fenced code blocks are ignored.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "docs" / "user_guide" / "upstream_drift_user_manual.md"
CHAPTER_DIR = INDEX.with_suffix("")
MARKER_START = "<!-- manual-toc:start -->"
MARKER_END = "<!-- manual-toc:end -->"
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
ROOT_TITLE = re.compile(r"^(Chapter \d+|Appendix [A-Z])\b")
MAX_ROOT_LEVEL = 2


@dataclass(frozen=True)
class Heading:
    """A real heading: its level, its text, and the GitHub anchor it resolves to."""

    level: int
    text: str
    anchor: str


def github_slug(title: str) -> str:
    """Lowercase, drop punctuation except ``-`` and ``_``, turn spaces into ``-``."""
    plain = re.sub(r"[`*~]", "", title).strip().lower()
    kept = "".join(ch for ch in plain if ch.isalnum() or ch in "-_ ")
    return kept.replace(" ", "-")


def headings_of(path: Path) -> list[Heading]:
    """Return the headings of one file in order, deduplicating anchors as GitHub does."""
    found: list[Heading] = []
    seen: dict[str, int] = {}
    fenced = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        match = None if fenced else HEADING.match(line)
        if match is None:
            continue
        text = match.group(2)
        if "[" in text or "]" in text:
            raise ValueError(f"{path.name}: heading may not contain brackets: {text!r}")
        slug = github_slug(text)
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchor = slug if count == 0 else f"{slug}-{count}"
        found.append(Heading(len(match.group(1)), text, anchor))
    return found


def _children(heads: list[Heading], start: int) -> list[Heading]:
    """Return the headings one level below ``heads[start]`` before its section ends."""
    root = heads[start]
    children: list[Heading] = []
    for head in heads[start + 1 :]:
        if head.level <= root.level:
            break
        if head.level == root.level + 1:
            children.append(head)
    return children


def chapter_blocks(path: Path) -> list[str]:
    """Return one Markdown block per chapter or appendix heading in ``path``."""
    heads = headings_of(path)
    rel = f"{CHAPTER_DIR.name}/{path.name}"
    roots = [
        i
        for i, head in enumerate(heads)
        if head.level <= MAX_ROOT_LEVEL and ROOT_TITLE.match(head.text)
    ]
    if not roots:
        raise ValueError(f"{path.name}: no 'Chapter N' or 'Appendix X' heading")
    blocks: list[str] = []
    for i in roots:
        root = heads[i]
        title = f"- **[{root.text}]({rel}#{root.anchor})**"
        items = [
            f"  - [{child.text}]({rel}#{child.anchor})" for child in _children(heads, i)
        ]
        # Blank line after the chapter line matches prettier's loose-list layout.
        blocks.append("\n\n".join([title, "\n".join(items)]) if items else title)
    return blocks


def build_block() -> str:
    """Return the full TOC block, ending with a newline, from every chapter file."""
    blocks: list[str] = []
    for path in sorted(CHAPTER_DIR.glob("*.md")):
        blocks.extend(chapter_blocks(path))
    return "\n\n".join(blocks)


def splice(index_text: str, block: str) -> str:
    """Replace the text between the TOC markers with ``block``."""
    start = index_text.find(MARKER_START)
    end = index_text.find(MARKER_END)
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"{INDEX.name} must contain {MARKER_START} and {MARKER_END}")
    head = index_text[: start + len(MARKER_START)]
    # Blank lines around the block keep prettier from indenting the end marker.
    return f"{head}\n\n{block}\n\n{index_text[end:]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="rewrite the TOC block")
    mode.add_argument("--check", action="store_true", help="fail if the TOC is stale")
    args = parser.parse_args(argv)

    current = INDEX.read_text(encoding="utf-8")
    expected = splice(current, build_block())
    if expected == current:
        print(f"{INDEX.name}: manual TOC is up to date")
        return 0
    if args.check:
        print(
            f"{INDEX.name}: manual TOC is stale; run "
            "python scripts/docs/build_manual_toc.py --write",
            file=sys.stderr,
        )
        return 1
    INDEX.write_text(expected, encoding="utf-8", newline="\n")
    print(f"{INDEX.name}: manual TOC rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
