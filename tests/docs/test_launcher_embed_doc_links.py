"""Verify internal Markdown links in the launcher-embed docs resolve.

Subtask 7 of EPIC #4993 (issue #5000) introduces a small cluster of
cross-linked Markdown pages under
``docs/user_guide/launcher/`` and the launcher-embed pages under
``docs/development/`` (``embedding_a_tool.md`` and
``realtime_ipc.md``). This test parses every internal link in those
pages and asserts that the target file exists on disk.

External links (``http://`` / ``https://`` / ``mailto:``) and pure
anchors (``#section``) are skipped. Image links pointing at the
screenshot placeholder directory ``docs/assets/launcher/`` are
tolerated as missing (screenshots are scheduled for a follow-up
capture PR).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_USER_GUIDE = ROOT / "docs" / "user_guide" / "launcher"
DEVELOPMENT_DOCS = ROOT / "docs" / "development"

# The launcher-embed-specific developer pages (other files under
# ``docs/development/`` are out of scope for this subtask's link
# validation).
_DEV_PAGES = ("embedding_a_tool.md", "realtime_ipc.md")

# ``[text](target)`` — standard Markdown link. Image links
# (``![alt](target)``) are matched too.
_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:"))


def _is_anchor_only(target: str) -> bool:
    return target.startswith("#")


def _strip_anchor(target: str) -> str:
    """Return the path portion of ``path#anchor``."""
    return target.split("#", 1)[0]


def _is_image_placeholder(target: str) -> bool:
    """Image links in the launcher user guide point at unresolved
    screenshot placeholders under ``docs/assets/launcher/``. Those
    are scheduled for a follow-up capture PR — we tolerate them as
    missing files, exactly like the Pose Studio link test does.
    """
    return target.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg"))


def _iter_user_guide_md_files() -> list[Path]:
    if not LAUNCHER_USER_GUIDE.is_dir():
        return []
    return sorted(p for p in LAUNCHER_USER_GUIDE.glob("*.md") if p.is_file())


def _iter_dev_md_files() -> list[Path]:
    files: list[Path] = []
    for name in _DEV_PAGES:
        path = DEVELOPMENT_DOCS / name
        if path.is_file():
            files.append(path)
    return sorted(files)


def _iter_all_target_md_files() -> list[Path]:
    return _iter_user_guide_md_files() + _iter_dev_md_files()


def test_launcher_user_guide_directory_exists() -> None:
    assert LAUNCHER_USER_GUIDE.is_dir(), LAUNCHER_USER_GUIDE


def test_launcher_user_guide_has_expected_pages() -> None:
    names = {p.name for p in _iter_user_guide_md_files()}
    expected = {"embedded_view.md"}
    assert expected <= names, f"missing pages: {expected - names}"


def test_development_docs_have_expected_pages() -> None:
    for name in _DEV_PAGES:
        path = DEVELOPMENT_DOCS / name
        assert path.is_file(), f"missing developer-guide page: {path}"


def test_launcher_embed_internal_links_resolve() -> None:
    files = _iter_all_target_md_files()
    assert files, "expected at least one launcher-embed .md page"

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
