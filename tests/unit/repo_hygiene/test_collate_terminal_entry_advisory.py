"""A live ``dl_state`` against a shipped entry is advisory at collate time.

RM#2044 / RM#2093: by the time a fragment is collated its pull request has
merged, so a stale ``in_review`` against an already shipped ``DL-#N`` entry must
not abort the collate (that stranded hundreds of fragments). The entry stays
terminal, the PR and SPEC row are still recorded and the stale state is dropped.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pytest

from shared_scripts import changes_fragment as cf

pytestmark = pytest.mark.unit

_TODAY = date(2026, 10, 14)
_SHA = "a" * 40
_ISSUE = 4242
_PR = 9001
_SPEC = (
    "# Spec\n\n## Change log\n\n"
    "| Date | Change | Summary |\n| --- | --- | --- |\n"
    "| 2026-01-01 | #1 | first |\n"
)


def _repo_files(root: Path) -> tuple[str, str]:
    """Return (spec, devlog) text with a shipped ``DL-#4242`` entry."""
    devlog = cf.schema.development_log.resolve_canonical_devlog_path(root)
    text = devlog.read_text(encoding="utf-8")
    entry = (
        f"### DL-#{_ISSUE} · Already shipped feature\n\n"
        "- **State:** shipped\n- **Owner:** codex\n"
        f"- **Issue:** #{_ISSUE}\n- **PR:** #5\n"
        "- **Started:** 2026-10-01\n- **Paths:** `src/x.py`\n"
        "- **Summary:** Shipped earlier.\n"
        "- **Last verified:** 2026-10-02 (tests passed).\n\n"
    )
    marker = "## Active\n\n"
    assert marker in text
    return _SPEC, text.replace(marker, marker + entry, 1)


def _fragment() -> cf.Fragment:
    return cf.Fragment(
        issue=_ISSUE,
        summary="Late fragment from a merged PR",
        dl_state="in_review",
        next_step="Await review.",
        branch="fix/late-4242",
    )


def test_shipped_entry_stays_shipped_and_spec_row_written(
    caplog: pytest.LogCaptureFixture,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    spec, devlog = _repo_files(repo_root)
    fragment = _fragment()
    with caplog.at_level(logging.WARNING):
        new_devlog = cf.apply_devlog_entry(
            devlog, fragment, pr=_PR, today=_TODAY, sha=_SHA, source="changes/x.md"
        )
    new_spec = cf.apply_spec_row(spec, pr=_PR, summary=fragment.summary, today=_TODAY)

    entry = next(
        e
        for e in cf.schema.development_log.parse_entries(new_devlog)
        if e.entry_id == f"DL-#{_ISSUE}"
    )
    assert entry.state == "shipped"
    assert f"#{_PR}" in entry.fields["PR"]
    assert f"| #{_PR} | {fragment.summary} |" in new_spec
    assert "never moves" in caplog.text
