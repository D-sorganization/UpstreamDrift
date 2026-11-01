"""Root REVIEW.md exists, is allowlisted and cites only real paths (#14180)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "REVIEW.md"
# A backticked token with a slash or a file suffix is treated as a repo path.
PATH = re.compile(
    r"`([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+/?|[A-Za-z0-9_-]+\.(?:md|py|yaml|json))`"
)


def test_review_md_exists_at_repo_root() -> None:
    assert REVIEW.is_file()


def test_review_md_is_in_both_root_allowlists() -> None:
    clutter = (ROOT / "scripts" / "check_root_clutter.py").read_text(encoding="utf-8")
    entries = json.loads(
        (ROOT / "scripts" / "config" / "root_allowlist.json").read_text(
            encoding="utf-8"
        )
    )["entries"]
    assert '"REVIEW.md"' in clutter
    assert "REVIEW.md" in entries


def test_review_md_cites_only_existing_paths() -> None:
    cited = PATH.findall(REVIEW.read_text(encoding="utf-8"))
    assert cited, "REVIEW.md should cite the files its rules point at"
    missing = [p for p in cited if not (ROOT / p.rstrip("/")).exists()]
    assert missing == []
