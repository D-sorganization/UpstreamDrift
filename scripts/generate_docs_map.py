#!/usr/bin/env python3
"""Generate the documentation map blocks in docs/index.md and docs/README.md.

Both files carry a hand-written narrative plus one machine-generated block
delimited by HTML comment markers. This script rewrites the generated blocks
from the real ``docs/`` tree so the navigation cannot drift away from the
filesystem the way the old hand-maintained structure diagram did (issue #8839).

Usage::

    python scripts/generate_docs_map.py           # rewrite the blocks
    python scripts/generate_docs_map.py --check   # fail if a block is stale

The catalog table in ``docs/index.md`` remains the owner/stability source of
truth and is validated separately by ``scripts/check_doc_catalog.py``. This
script reads the stability column from that table so the map groups directories
the same way the catalog classifies them.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
INDEX_PATH = DOCS_DIR / "index.md"
README_PATH = DOCS_DIR / "README.md"

INDEX_MARKER = "docs-map"
README_MARKER = "docs-structure"

CATALOG_ROW = re.compile(
    r"^\|\s*`(?P<directory>[^`]+)/`\s*\|\s*@[^|]+?\s*\|"
    r"\s*(?P<stability>stable|draft|archived)\s*\|"
)

#: Preferred landing files, in order, when a directory has an entry point.
ENTRY_CANDIDATES = ("README.md", "index.md", "INDEX.md", "overview.md")

STABILITY_ORDER = ("stable", "draft", "archived")
STABILITY_HEADINGS = {
    "stable": "Stable",
    "draft": "Draft",
    "archived": "Archived",
}


def begin_marker(name: str) -> str:
    """Return the opening HTML comment marker for a generated block."""
    return f"<!-- BEGIN GENERATED: {name} (scripts/generate_docs_map.py) -->"


def end_marker(name: str) -> str:
    """Return the closing HTML comment marker for a generated block."""
    return f"<!-- END GENERATED: {name} -->"


def read_stability() -> dict[str, str]:
    """Return ``{directory_name: stability}`` parsed from the index catalog."""
    stability: dict[str, str] = {}
    if not INDEX_PATH.exists():
        return stability
    for line in INDEX_PATH.read_text(encoding="utf-8").splitlines():
        match = CATALOG_ROW.match(line)
        if match is not None:
            stability[match.group("directory")] = match.group("stability")
    return stability


def entry_target(directory: Path) -> str:
    """Return the best link target for ``directory``, relative to ``docs/``.

    A directory with a conventional landing page links to that page. A
    directory holding exactly one Markdown file links straight to it, which is
    friendlier than a bare directory listing. Everything else links to the
    directory itself.
    """
    name = directory.name
    for candidate in ENTRY_CANDIDATES:
        if (directory / candidate).exists():
            return f"{name}/{candidate}"
    markdown = sorted(directory.rglob("*.md"))
    if len(markdown) == 1:
        return markdown[0].relative_to(DOCS_DIR).as_posix()
    return f"{name}/"


def page_count(directory: Path) -> int:
    """Return the number of Markdown pages under ``directory``."""
    return len(list(directory.rglob("*.md")))


def describe(directory: Path) -> str:
    """Return one map bullet for ``directory``."""
    count = page_count(directory)
    unit = "page" if count == 1 else "pages"
    suffix = f"{count} {unit}" if count else "no Markdown pages"
    return f"- [`{directory.name}/`]({entry_target(directory)}) - {suffix}"


def build_index_block() -> str:
    """Return the generated documentation-map block for docs/index.md."""
    stability = read_stability()
    directories = sorted(p for p in DOCS_DIR.iterdir() if p.is_dir())
    grouped: dict[str, list[Path]] = {key: [] for key in STABILITY_ORDER}
    unclassified: list[Path] = []
    for directory in directories:
        key = stability.get(directory.name)
        if key in grouped:
            grouped[key].append(directory)
        else:
            unclassified.append(directory)

    lines = [
        "Every top-level directory below is a live link. Grouping follows the",
        "stability column of the catalog table, so archived material is visibly",
        "separated from current guidance.",
    ]
    for key in STABILITY_ORDER:
        if not grouped[key]:
            continue
        lines.extend(["", f"### {STABILITY_HEADINGS[key]}", ""])
        lines.extend(describe(directory) for directory in grouped[key])
    if unclassified:
        lines.extend(["", "### Not yet catalogued", ""])
        lines.extend(describe(directory) for directory in unclassified)
    return "\n".join(lines)


def build_readme_block() -> str:
    """Return the generated structure block for docs/README.md."""
    directories = sorted(p for p in DOCS_DIR.iterdir() if p.is_dir())
    lines = [
        "```text",
        "docs/",
        "|-- README.md   <- you are here (task-oriented hub)",
        "|-- index.md    <- catalog: owner, stability, and full map",
    ]
    for directory in directories:
        count = page_count(directory)
        unit = "page" if count == 1 else "pages"
        detail = f"{count} {unit}" if count else "no Markdown pages"
        lines.append(f"|-- {directory.name}/".ljust(30) + f"# {detail}")
    lines.append("```")
    lines.extend(
        [
            "",
            f"That is {len(directories)} top-level directories. The tree above is",
            "generated by `scripts/generate_docs_map.py` from the real filesystem;",
            "do not edit it by hand. For owners, stability tags, and per-directory",
            "descriptions see [the documentation catalog](index.md).",
        ]
    )
    return "\n".join(lines)


def replace_block(text: str, name: str, body: str) -> str:
    """Return ``text`` with the ``name`` generated block replaced by ``body``."""
    begin = begin_marker(name)
    end = end_marker(name)
    if begin not in text or end not in text:
        raise SystemExit(f"missing generated-block markers for '{name}'")
    head, _, rest = text.partition(begin)
    _, _, tail = rest.partition(end)
    return f"{head}{begin}\n\n{body}\n\n{end}{tail}"


def apply(check_only: bool) -> int:
    """Rewrite (or verify) both generated blocks. Return a process exit code."""
    targets = (
        (INDEX_PATH, INDEX_MARKER, build_index_block()),
        (README_PATH, README_MARKER, build_readme_block()),
    )
    stale: list[str] = []
    for path, name, body in targets:
        original = path.read_text(encoding="utf-8")
        updated = replace_block(original, name, body)
        if updated == original:
            continue
        if check_only:
            stale.append(path.relative_to(ROOT).as_posix())
            continue
        path.write_text(updated, encoding="utf-8", newline="\n")
        print(f"regenerated {name} in {path.relative_to(ROOT).as_posix()}")
    if stale:
        sys.stderr.write(
            "documentation map is stale; run scripts/generate_docs_map.py:\n- "
            + "\n- ".join(stale)
            + "\n"
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and regenerate or verify the documentation map."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero when a generated block is out of date",
    )
    args = parser.parse_args(argv)
    return apply(check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
