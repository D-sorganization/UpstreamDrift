"""Verify docs/ top-level cleanliness per issue #7063.

Acceptance criteria:
- No loose *.md files at the docs/ root (except README.md and index.md).
- docs/examples/index.rst references only existing example pages.
- docs/sphinx/conf.py does not contain the 'TRACKED_TASK' placeholder
  extension that would cause sphinx to fail on import.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _ROOT / "docs"

# Only these files are permitted to live flat at the docs/ root.
_ALLOWED_ROOT_MD = {"README.md", "index.md"}


def test_no_loose_markdown_at_docs_root() -> None:
    """All *.md files except README.md and index.md must live in a subdir."""
    loose = sorted(p.name for p in _DOCS.glob("*.md") if p.name not in _ALLOWED_ROOT_MD)
    assert not loose, (
        f"Loose .md files found in docs/ root; move each to a subdir: {loose}"
    )


def test_examples_index_references_existing_pages() -> None:
    """The examples subtree is allowed when it has real runnable pages."""
    index = _DOCS / "examples" / "index.rst"
    if not index.exists():
        return

    missing: list[str] = []
    in_toctree = False
    for raw_line in index.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped == ".. toctree::":
            in_toctree = True
            continue
        if not in_toctree:
            continue
        if not raw_line.startswith((" ", "\t")):
            if stripped:
                in_toctree = False
            continue
        if not stripped or stripped.startswith(":"):
            continue
        entry = stripped.split()[0]
        entry_path = _DOCS / "examples" / entry
        candidates = (
            entry_path.with_suffix(".rst"),
            entry_path.with_suffix(".md"),
            entry_path / "index.rst",
            entry_path / "index.md",
        )
        if not any(candidate.exists() for candidate in candidates):
            missing.append(entry)

    assert not missing, f"docs/examples/index.rst has dangling entries: {missing}"


def test_sphinx_conf_no_placeholder_extension() -> None:
    """docs/sphinx/conf.py must not contain the 'TRACKED_TASK' placeholder
    that breaks sphinx-build at import time."""
    conf = _DOCS / "sphinx" / "conf.py"
    if conf.exists():
        text = conf.read_text(encoding="utf-8")
        assert "TRACKED_TASK" not in text, (
            "docs/sphinx/conf.py contains 'TRACKED_TASK' placeholder — "
            "replace with a real extension or remove the entry."
        )
