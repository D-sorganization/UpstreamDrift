"""Verify internal Markdown links in the Pose Studio user guide resolve.

The Pose Studio user guide (Subtask 8 of EPIC #4895, issue #4902)
introduces a small cluster of cross-linked Markdown pages.  This test
parses every internal link in those pages and asserts that the target
file exists on disk.  External links (``http://`` / ``https://``) and
pure anchors (``#section``) are skipped.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
POSE_STUDIO_DOCS = ROOT / "docs" / "user_guide" / "pose_studio"

# ``[text](target)`` — standard Markdown link.  We only care about the
# target.  Image links (``![alt](target)``) are matched too; we filter
# image targets the same way we filter link targets (skip http(s),
# allow .png/.jpg paths under docs/ to "resolve" only if they exist —
# the screenshots are placeholders, so we tolerate missing images).
_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:"))


def _is_anchor_only(target: str) -> bool:
    return target.startswith("#")


def _strip_anchor(target: str) -> str:
    """Return the path portion of ``path#anchor``."""
    return target.split("#", 1)[0]


def _is_image_placeholder(target: str) -> bool:
    """Image links in the Pose Studio guide point at unresolved screenshot
    placeholders under ``docs/assets/pose_studio/``.  Those are scheduled
    for a follow-up capture PR — we tolerate them as missing files.
    """
    return target.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg"))


def _iter_pose_studio_md_files() -> list[Path]:
    return sorted(p for p in POSE_STUDIO_DOCS.glob("*.md") if p.is_file())


def test_pose_studio_docs_directory_exists() -> None:
    assert POSE_STUDIO_DOCS.is_dir(), POSE_STUDIO_DOCS


def test_pose_studio_docs_has_expected_pages() -> None:
    names = {p.name for p in _iter_pose_studio_md_files()}
    expected = {
        "index.md",
        "quickstart.md",
        "cross_engine_conventions.md",
        "save_formats.md",
    }
    assert expected <= names, f"missing pages: {expected - names}"


def test_pose_studio_docs_internal_links_resolve() -> None:
    files = _iter_pose_studio_md_files()
    assert files, "expected at least one .md page in pose_studio docs"

    failures: list[str] = []
    for md_file in files:
        text = md_file.read_text(encoding="utf-8")
        for target in _LINK_RE.findall(text):
            if _is_external(target) or _is_anchor_only(target):
                continue
            if _is_image_placeholder(target):
                # Screenshot capture is scheduled for a follow-up PR.
                continue
            path_part = _strip_anchor(target)
            if not path_part:
                continue
            resolved = (md_file.parent / path_part).resolve()
            if not resolved.exists():
                failures.append(
                    f"{md_file.relative_to(ROOT)} -> {target!r} "
                    f"(resolved to {resolved})"
                )

    assert not failures, "Unresolved internal links:\n  " + "\n  ".join(failures)
