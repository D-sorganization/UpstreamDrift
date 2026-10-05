"""Tests for per-PR change fragments (RM-5 / Repository_Management#1894 / #1976).

A fragment ``changes/<issue>-<slug>.md`` replaces the direct edits every pull
request used to make to ``SPEC.md``, ``DEVELOPMENT_LOG.md`` and ``HANDOFF.md``.
Each pull request writes its own file, so fragments cannot conflict; the
post-merge ``collate`` step folds them into the shared files.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

from shared_scripts import (
    changes_fragment,
    changes_fragment_collate,
    changes_fragment_schema,
    development_log,
    spec_changelog,
)

pytestmark = [pytest.mark.unit]

TODAY = date(2026, 10, 4)
SHA = "0123456789abcdef0123456789abcdef01234567"

SPEC_FIXTURE = """# Spec

## 12. Change Log

| Date       | PR    | Changes  |
| ---------- | ----- | -------- |
| 2026-09-03 | #1520 | old row  |

## 13. Appendix

Tail text.
"""

DEVLOG_FIXTURE = """# Development Log — Test

- **Portfolio:** infra
- **WIP limit:** 12

## States

`proposed` → `in_progress` → `in_review` → `shipped`.

## Active

### DL-#77 · Existing Feature

- **State:** in_progress
- **Owner:** claude
- **Issue:** #77
- **Branch:** `feat/77`
- **PR:** not created
- **Paths:** `src/x.py`
- **Started:** 2026-10-01
- **Last verified:** 2026-10-01 (`abcdef12`)
- **Summary:** Existing work.
- **Next step:** Open the PR.

### DL-#20 · Proposed Thing

- **State:** proposed
- **Owner:** claude
- **Issue:** #20
- **PR:** not created
- **Paths:** `src/p.py`
- **Started:** 2026-10-01
- **Last verified:** 2026-10-01 (`abcdef12`)
- **Summary:** Proposed work.
- **Next step:** Start it.

## Shipped (Last 90 Days)

### DL-#10 · Old Thing

- **State:** shipped
- **Owner:** claude
- **Issue:** #10
- **PR:** #11
- **Paths:** `src/old.py`
- **Started:** 2026-09-01
- **Last verified:** 2026-09-02 (`abcdef12`)
- **Summary:** Old thing.

## Archive
"""


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "SPEC.md").write_text(SPEC_FIXTURE, encoding="utf-8")
    devlog = tmp_path / "docs" / "development" / "DEVELOPMENT_LOG.md"
    devlog.parent.mkdir(parents=True)
    devlog.write_text(DEVLOG_FIXTURE, encoding="utf-8")
    return tmp_path


def _write_fragment(root: Path, name: str, text: str) -> Path:
    path = root / "changes" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _devlog(root: Path) -> str:
    return (root / "docs" / "development" / "DEVELOPMENT_LOG.md").read_text(
        encoding="utf-8"
    )


def _spec(root: Path) -> str:
    return (root / "SPEC.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_valid_fragment_parses(tmp_path: Path) -> None:
    path = _write_fragment(
        tmp_path,
        "77-thing.md",
        "---\nissue: 77\nsummary: Did the thing\ndl_state: in_review\n"
        'next_step: "Merge the PR."\nbranch: feat/77\n---\n\nHandoff notes.\n',
    )
    fragment = changes_fragment.load_fragment(path)
    assert fragment.issue == 77
    assert fragment.summary == "Did the thing"
    assert fragment.dl_state == "in_review"
    assert fragment.next_step == "Merge the PR."
    assert fragment.handoff == "Handoff notes."
    assert changes_fragment.validate_fragment_file(path) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("No frontmatter\n", "missing YAML frontmatter"),
        ("---\nissue: abc\nsummary: x\n---\n", "issue"),
        ("---\nissue: 0\nsummary: x\n---\n", "issue"),
        ("---\nsummary: x\n---\n", "issue"),
        ("---\nissue: 77\n---\n", "summary"),
        ("---\nissue: 77\nsummary: \n---\n", "summary"),
        ("---\nissue: 77\nsummary: a | b\n---\n", "|"),
        ("---\nissue: 77\nsummary: x\ndl_state: done\n---\n", "dl_state"),
        ("---\nissue: 77\nsummary: x\ndl_state: in_progress\n---\n", "next_step"),
        ("---\nissue: 77\nsummary: x\ncolour: red\n---\n", "unknown key"),
        ("---\nissue: 77\nsummary: <one line>\n---\n", "placeholder"),
        ("---\nissue: 77\nsummary: a \\| b\n---\n", "|"),
        (
            "---\nissue: 77\nsummary: x\ndl_state: in_review\nnext_step: y\n---\n",
            "branch",
        ),
        (
            "---\nissue: 77\nsummary: x\ndl_state: parked\nnext_step: y\n---\n",
            "parked",
        ),
        ("---\nissue: 77\nsummary: x\n", "closing"),
    ],
)
def test_invalid_fragments_are_rejected(
    tmp_path: Path, text: str, expected: str
) -> None:
    path = _write_fragment(tmp_path, "77-bad.md", text)
    findings = changes_fragment.validate_fragment_file(path)
    assert findings, f"expected a finding containing {expected!r}"
    assert any(expected in finding for finding in findings), findings


def test_filename_must_lead_with_the_fragment_issue(tmp_path: Path) -> None:
    path = _write_fragment(tmp_path, "78-other.md", "---\nissue: 77\nsummary: x\n---\n")
    findings = changes_fragment.validate_fragment_file(path)
    assert any("file name" in finding for finding in findings), findings


def test_secret_in_fragment_is_rejected(tmp_path: Path) -> None:
    token = "ghp_" + "a" * 36
    path = _write_fragment(
        tmp_path, "77-s.md", f"---\nissue: 77\nsummary: x\n---\n{token}\n"
    )
    assert any(
        "secret" in f.lower() for f in changes_fragment.validate_fragment_file(path)
    )


def test_is_fragment_path() -> None:
    assert changes_fragment.is_fragment_path("changes/12-x.md")
    assert changes_fragment.is_fragment_path("changes\\12.md")
    assert not changes_fragment.is_fragment_path("changes/README.md")
    assert not changes_fragment.is_fragment_path("changes/sub/12.md")
    assert not changes_fragment.is_fragment_path("docs/changes/12.md")
    assert not changes_fragment.is_fragment_path("changes/12.txt")


# ---------------------------------------------------------------------------
# new
# ---------------------------------------------------------------------------


def test_new_writes_a_valid_fragment(tmp_path: Path) -> None:
    path = changes_fragment.new_fragment(
        tmp_path,
        issue=1894,
        summary="Per-PR change fragments",
        dl_state="in_review",
        next_step="Merge the PR.",
        branch="feat/rm-5",
    )
    assert path == tmp_path / "changes" / "1894-per-pr-change-fragments.md"
    assert changes_fragment.validate_fragment_file(path) == []
    assert changes_fragment.load_fragment(path).next_step == "Merge the PR."


def test_new_refuses_to_overwrite(tmp_path: Path) -> None:
    changes_fragment.new_fragment(tmp_path, issue=5, summary="Same")
    with pytest.raises(FileExistsError):
        changes_fragment.new_fragment(tmp_path, issue=5, summary="Same")


def test_new_rejects_invalid_input(tmp_path: Path) -> None:
    with pytest.raises(changes_fragment.FragmentError):
        changes_fragment.new_fragment(tmp_path, issue=5, summary="x", dl_state="bogus")
    assert not (tmp_path / "changes").exists() or not any(
        (tmp_path / "changes").iterdir()
    )


def test_cli_new_and_validate(tmp_path: Path) -> None:
    rc = changes_fragment.main(
        ["new", "--repo-root", str(tmp_path), "--issue", "9", "--summary", "Nine"]
    )
    assert rc == 0
    created = tmp_path / "changes" / "9-nine.md"
    assert created.is_file()
    assert changes_fragment.main(["validate", str(created)]) == 0
    bad = _write_fragment(tmp_path, "9-bad.md", "nope\n")
    assert changes_fragment.main(["validate", str(bad)]) == 1


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------


def test_collate_creates_spec_row_and_new_devlog_entry(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    frag = _write_fragment(
        root, "88-new.md", "---\nissue: 88\nsummary: Brand new feature\n---\n"
    )

    changes_fragment.collate(root, [frag], pr=900, today=TODAY, sha=SHA)

    rows = spec_changelog.parse_changelog(_spec(root)).rows
    assert rows[0] == spec_changelog.Row("2026-10-04", "#900", "Brand new feature")
    assert "Tail text." in _spec(root)
    entries = {e.entry_id: e for e in development_log.parse_entries(_devlog(root))}
    entry = entries["DL-#88"]
    assert entry.state == "shipped"
    assert "#900" in entry.fields["PR"]
    assert "#88" in entry.fields["Issue"]
    assert "01234567" in entry.fields["Last verified"]
    assert not frag.exists()
    # Shipped entries land in the Shipped section, above the older entry.
    log = _devlog(root)
    assert log.index("## Shipped") < log.index("DL-#88") < log.index("DL-#10")
    assert development_log.validate_devlog_content(log, Path("DL.md")) == []
    assert spec_changelog.validate(spec_changelog.parse_changelog(_spec(root))) == []


def test_collate_updates_existing_entry_in_place(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    frag = _write_fragment(
        root,
        "77-review.md",
        "---\nissue: 77\nsummary: Existing work lands\ndl_state: in_review\n"
        "next_step: Watch the merge queue.\nbranch: feat/77\n---\n",
    )

    changes_fragment.collate(root, [frag], pr=901, today=TODAY, sha=SHA)

    log = _devlog(root)
    entries = development_log.parse_entries(log)
    assert [e.entry_id for e in entries].count("DL-#77") == 1
    entry = next(e for e in entries if e.entry_id == "DL-#77")
    assert entry.state == "in_review"
    assert entry.fields["PR"] == "#901"
    assert entry.fields["Next step"] == "Watch the merge queue."
    assert entry.fields["Last verified"].startswith("2026-10-04")
    assert entry.fields["Owner"] == "claude"
    assert log.index("DL-#77") < log.index("## Shipped")
    assert development_log.validate_devlog_content(log, Path("DL.md")) == []


def test_collate_moves_entry_to_shipped_when_it_ships(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    frag = _write_fragment(
        root, "77-done.md", "---\nissue: 77\nsummary: Done\ndl_state: shipped\n---\n"
    )

    changes_fragment.collate(root, [frag], pr=902, today=TODAY, sha=SHA)

    log = _devlog(root)
    assert log.index("## Shipped") < log.index("DL-#77")
    entry = next(
        e for e in development_log.parse_entries(log) if e.entry_id == "DL-#77"
    )
    assert entry.state == "shipped"
    assert "#902" in entry.fields["Next step"]
    assert development_log.validate_devlog_content(log, Path("DL.md")) == []


def test_collate_is_idempotent(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    text = "---\nissue: 88\nsummary: Brand new feature\n---\n"
    frag = _write_fragment(root, "88-new.md", text)
    changes_fragment.collate(root, [frag], pr=900, today=TODAY, sha=SHA)
    spec_once, log_once = _spec(root), _devlog(root)

    # A re-run (e.g. the workflow retried after a failed push) changes nothing.
    frag = _write_fragment(root, "88-new.md", text)
    changes_fragment.collate(root, [frag], pr=900, today=TODAY, sha=SHA)

    assert _spec(root) == spec_once
    assert _devlog(root) == log_once


def test_concurrent_fragments_never_collide(tmp_path: Path) -> None:
    """Two PRs merged back to back: both rows and both entries are applied."""
    root = _repo(tmp_path)
    first = _write_fragment(root, "88-a.md", "---\nissue: 88\nsummary: A\n---\n")
    second = _write_fragment(root, "89-b.md", "---\nissue: 89\nsummary: B\n---\n")

    changes_fragment.collate(root, [first], pr=903, today=TODAY, sha=SHA)
    changes_fragment.collate(root, [second], pr=904, today=TODAY, sha=SHA)

    keys = [r.key for r in spec_changelog.parse_changelog(_spec(root)).rows]
    assert keys[:2] == ["#904", "#903"]  # newest-first
    ids = {e.entry_id for e in development_log.parse_entries(_devlog(root))}
    assert {"DL-#88", "DL-#89"} <= ids


def test_two_fragments_in_one_pr_share_one_row(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    first = _write_fragment(root, "88-a.md", "---\nissue: 88\nsummary: A\n---\n")
    second = _write_fragment(root, "89-b.md", "---\nissue: 89\nsummary: B\n---\n")

    changes_fragment.collate(root, [first, second], pr=905, today=TODAY, sha=SHA)

    changelog = spec_changelog.parse_changelog(_spec(root))
    rows = [r for r in changelog.rows if r.key == "#905"]
    assert rows == [spec_changelog.Row("2026-10-04", "#905", "A; B")]
    assert spec_changelog.validate(changelog) == []


def test_collate_without_spec_or_devlog_only_deletes(tmp_path: Path) -> None:
    frag = _write_fragment(tmp_path, "5-x.md", "---\nissue: 5\nsummary: X\n---\n")
    changes_fragment.collate(tmp_path, [frag], pr=7, today=TODAY, sha=SHA)
    assert not frag.exists()
    assert not (tmp_path / "SPEC.md").exists()


def test_collate_refuses_invalid_fragment_and_changes_nothing(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    good = _write_fragment(root, "88-a.md", "---\nissue: 88\nsummary: A\n---\n")
    bad = _write_fragment(root, "89-b.md", "---\nissue: 89\n---\n")
    with pytest.raises(changes_fragment.FragmentError):
        changes_fragment.collate(root, [good, bad], pr=906, today=TODAY, sha=SHA)
    assert _spec(root) == SPEC_FIXTURE
    assert good.exists() and bad.exists()


def test_collate_rejects_bad_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        changes_fragment.collate(tmp_path, [], pr=0, today=TODAY, sha=SHA)
    with pytest.raises(ValueError):
        changes_fragment.collate(tmp_path, [], pr=1, today=TODAY, sha="nothex")


def test_cli_collate_defaults_to_every_fragment(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write_fragment(root, "88-a.md", "---\nissue: 88\nsummary: A\n---\n")
    rc = changes_fragment.main(
        [
            "collate",
            "--repo-root",
            str(root),
            "--pr",
            "907",
            "--sha",
            SHA,
            "--date",
            "2026-10-04",
        ]
    )
    assert rc == 0
    assert "| 2026-10-04 | #907 | A |" in _spec(root)
    assert not any((root / "changes").glob("*.md"))


# ---------------------------------------------------------------------------
# Review hardening: atomicity, state machine, round trip
# ---------------------------------------------------------------------------


def test_collate_keeps_fragments_when_spec_is_unparsable(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "SPEC.md").write_text("# Spec without a change log\n", encoding="utf-8")
    frag = _write_fragment(root, "88-a.md", "---\nissue: 88\nsummary: A\n---\n")
    with pytest.raises(changes_fragment.FragmentError, match="SPEC.md"):
        changes_fragment.collate(root, [frag], pr=910, today=TODAY, sha=SHA)
    assert frag.exists()
    assert _devlog(root) == DEVLOG_FIXTURE


@pytest.mark.parametrize("state", ["in_progress", "in_review", "proposed"])
def test_shipped_entry_never_returns_to_a_live_state(
    tmp_path: Path, state: str
) -> None:
    root = _repo(tmp_path)
    frag = _write_fragment(
        root,
        "10-again.md",
        f"---\nissue: 10\nsummary: Again\ndl_state: {state}\n"
        "next_step: Do it.\nbranch: feat/10\n---\n",
    )
    with pytest.raises(changes_fragment.FragmentError, match="new"):
        changes_fragment.collate(root, [frag], pr=911, today=TODAY, sha=SHA)
    assert frag.exists()
    assert _spec(root) == SPEC_FIXTURE
    assert _devlog(root) == DEVLOG_FIXTURE


def test_advancing_a_proposed_entry_sets_its_branch(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    frag = _write_fragment(
        root,
        "20-go.md",
        "---\nissue: 20\nsummary: Go\ndl_state: in_review\n"
        "next_step: Merge.\nbranch: feat/20\n---\n",
    )
    changes_fragment.collate(root, [frag], pr=912, today=TODAY, sha=SHA)
    log = _devlog(root)
    entry = next(
        e for e in development_log.parse_entries(log) if e.entry_id == "DL-#20"
    )
    assert entry.fields["Branch"] == "feat/20"
    assert development_log.validate_devlog_content(log, Path("DL.md")) == []


def test_facade_keeps_its_public_names() -> None:
    for name in (
        "FragmentError",
        "Fragment",
        "is_fragment_path",
        "validate_fragment_file",
        "load_fragment",
        "find_fragments",
        "new_fragment",
        "collate",
        "main",
    ):
        assert hasattr(changes_fragment, name), name


def test_fragment_modules_stay_under_the_size_limit() -> None:
    scripts = Path(changes_fragment.__file__).parent
    for path in scripts.glob("changes_fragment*.py"):
        lines = len(path.read_text(encoding="utf-8").splitlines())
        assert lines <= 400, f"{path.name} has {lines} lines"


_STATES = [None, "proposed", "in_progress", "in_review", "shipped", "abandoned"]
_TARGETS = [88, 77, 20, 10]  # new, in_progress, proposed, shipped


@pytest.mark.parametrize("state", _STATES)
@pytest.mark.parametrize("issue", _TARGETS)
@pytest.mark.parametrize("extras", [False, True])
def test_round_trip_valid_fragment_collates_to_valid_docs(
    tmp_path: Path, state: str | None, issue: int, extras: bool
) -> None:
    """Property: a fragment that validates either collates into SPEC.md and a
    development log that pass their validators, or is refused (shipped entries
    never change state) with every file untouched."""
    root = _repo(tmp_path)
    optional: dict[str, str] = {}
    if state is not None:
        optional["dl_state"] = state
    if state in development_log.ACTIVE_STATES:
        optional["next_step"] = "Do the next thing."
    if state in development_log.BRANCH_STATES or extras:
        optional["branch"] = f"feat/{issue}"
    if extras:
        optional.update(title="A Title", owner="codex", paths="`src/a.py`")
    frag = changes_fragment.new_fragment(
        root, issue=issue, summary="Round trip: a, b; c", **optional
    )
    assert changes_fragment.validate_fragment_file(frag) == []

    try:
        changes_fragment.collate(root, [frag], pr=920, today=TODAY, sha=SHA)
    except changes_fragment.FragmentError:
        assert issue == 10 and state not in (None, "shipped")
        assert frag.exists()
        assert _spec(root) == SPEC_FIXTURE and _devlog(root) == DEVLOG_FIXTURE
        return

    changelog = spec_changelog.parse_changelog(_spec(root))
    assert spec_changelog.validate(changelog) == []
    assert changelog.rows[0].key == "#920"
    assert development_log.validate_devlog_content(_devlog(root), Path("DL.md")) == []
    assert not frag.exists()


def test_collate_matches_existing_entry_with_em_dash_heading(tmp_path: Path) -> None:
    """An entry headed ``DL-#77 — Title`` is updated in place, never duplicated."""
    root = _repo(tmp_path)
    devlog = root / "docs" / "development" / "DEVELOPMENT_LOG.md"
    devlog.write_text(
        DEVLOG_FIXTURE.replace("### DL-#77 · ", "### DL-#77 \u2014 "), encoding="utf-8"
    )
    frag = _write_fragment(root, "77-x.md", "---\nissue: 77\nsummary: Touch it\n---\n")

    changes_fragment.collate(root, [frag], pr=950, today=TODAY, sha=SHA)

    text = devlog.read_text(encoding="utf-8")
    assert text.count("DL-#77") == 1
