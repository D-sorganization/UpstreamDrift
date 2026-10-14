"""SPEC.md stays a live specification; history lives in the change-log archive.

Issue #9415 (owner ruling 2026-10-12): the Section 12 change-log rows and the
per-PR sections that had accreted before Section 1 and after Section 12 moved
verbatim into ``docs/spec/CHANGELOG_ARCHIVE.md``. These tests pin that

* ``SPEC.md`` stays under the renewed cap (the global 50 KiB budget is not
  loosened),
* nothing was lost in the move (byte-exact SHA-256 parity of every moved block
  against the pre-move ``SPEC.md``), and
* the gates read the archive for history and ``SPEC.md`` for the live spec.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC = REPO_ROOT / "SPEC.md"
ARCHIVE = REPO_ROOT / "docs" / "spec" / "CHANGELOG_ARCHIVE.md"
BUDGET = REPO_ROOT / "scripts" / "config" / "doc_size_budget.json"

#: Cap for the live SPEC.md (owner ruling 2026-10-12: "about 300 KB", raised to 350 KB for collated fragments).
SPEC_CAP_BYTES = 350_000

# Pre-move parity baseline, taken from SPEC.md at origin/main 8e9b4b974b.
BASELINE = {
    "leading-sections": (
        "b35435a91584325a08081e14ec5304bf09ad7acd917eef30caf3dc1fae035573"
    ),
    "change-log-rows": (
        "7cc52b3fc2a0906aa021e0566a43e14e73d423f802a3fdaa248ee9e58cd00785"
    ),
    "trailing-sections": (
        "774a51dae716da2540f049f7659714577e7e656b1641a4aa38428e007bf78914"
    ),
}
BASELINE_ROW_COUNT = 1113
_ROW = re.compile(r"^\| \d{4}-\d{2}-\d{2} \| (?:#\d+|n/a) \|", re.MULTILINE)


def _read(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def _block(archive_text: str, name: str) -> str:
    begin = f"<!-- archive:{name}:begin -->\n"
    end = f"<!-- archive:{name}:end -->\n"
    assert archive_text.count(begin) == 1, f"{name}: begin marker missing/duplicated"
    assert archive_text.count(end) == 1, f"{name}: end marker missing/duplicated"
    start = archive_text.index(begin) + len(begin)
    return archive_text[start : archive_text.index(end)]


def test_spec_is_under_the_cap() -> None:
    assert SPEC.stat().st_size <= SPEC_CAP_BYTES


@pytest.mark.parametrize("name", sorted(BASELINE))
def test_moved_block_is_byte_identical_to_the_pre_move_spec(name: str) -> None:
    assert ARCHIVE.is_file(), "docs/spec/CHANGELOG_ARCHIVE.md must exist"
    body = _block(_read(ARCHIVE), name)
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == BASELINE[name]


def test_no_change_log_row_was_lost() -> None:
    body = _block(_read(ARCHIVE), "change-log-rows")
    assert len(_ROW.findall(body)) == BASELINE_ROW_COUNT


def test_archive_is_not_the_live_spec() -> None:
    """History sections must not linger in SPEC.md next to the live spec."""
    text = _read(SPEC)
    assert "## 3D Vector Distances Note" not in text
    assert "## Native Wrist Kinematic Compatibility" not in text
    assert "docs/spec/CHANGELOG_ARCHIVE.md" in text, "SPEC.md must point at it"


def test_spec_keeps_its_live_sections_and_a_change_log_table() -> None:
    text = _read(SPEC)
    for heading in (
        "## 1. Identity",
        "## 11. Roadmap & Open Issues",
        "## 12. Change Log",
        "SPEC MAINTENANCE RULES:",
    ):
        assert heading in text
    assert "| Date | PR | Changes |" in text


def test_budget_keeps_the_global_limit_and_dates_the_spec_exception() -> None:
    config = json.loads(_read(BUDGET))
    assert config["max_bytes"] == 51200, "the global 50 KiB budget must not loosen"
    exceptions = {item["path"]: item for item in config["exceptions"]}
    spec = exceptions["SPEC.md"]
    assert date.fromisoformat(spec["expires_on"]) > date(2026, 10, 31)
    assert "300" in spec["reason"] and "9415" in spec["reason"]
    assert "docs/spec/CHANGELOG_ARCHIVE.md" not in exceptions, (
        "the archive is exempt as a verbatim archive, not by dated exception"
    )


def test_collate_still_writes_new_rows_into_the_live_table() -> None:
    """New rows land in SPEC.md section 12; the archive is never touched."""
    from shared_scripts import changes_fragment_collate as collate

    text = _read(SPEC)
    updated = collate.apply_spec_row(
        text, pr=99999, summary="Live row.", today=date(2026, 10, 13)
    )
    rows = [line for line in updated.splitlines() if _ROW.match(line)]
    assert "| 2026-10-13 | #99999 | Live row. |" in rows
