"""Contract test for issue #14733: OpenSim net-moment gap proposal write-up.

Refs #13560, #14711, #14905. Verifies that the proposal document accurately
captures the conditioning math, Finding A (recorder inconsistency), recorder fix
options, and governance constraints, and that durable change-history evidence
remains validated both before and after normal change-fragment collation.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from shared_scripts import changes_fragment, development_log, spec_changelog

ROOT = Path(__file__).resolve().parents[2]
PROPOSAL_ISSUE = 14733
PROPOSAL_PATH = ROOT / "docs" / "proposals" / "14733-opensim-net-moment-gap-proposal.md"
FRAGMENT_REL_PATH = Path("changes") / "14733-opensim-net-moment-gap-proposal.md"
FRAGMENT_PATH = ROOT / FRAGMENT_REL_PATH
EXPECTED_SUMMARY_MARKERS = (
    "Proposal (write-up only)",
    "#13560",
    "OpenSim net-moment gap",
)
_PR_REF_RE = re.compile(r"#\d+")
_ARCHIVED_ROWS_BLOCK_RE = re.compile(
    r"^<!-- archive:(?P<name>change-log-rows[\w-]*):begin -->\n"
    r"(?P<body>.*?)^<!-- archive:(?P=name):end -->$",
    re.MULTILINE | re.DOTALL,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_proposal_document_exists_and_contains_governance_guards() -> None:
    """Proposal must exist and state write-up/governance constraints explicitly."""
    assert PROPOSAL_PATH.is_file(), f"Missing proposal document at {PROPOSAL_PATH}"
    text = PROPOSAL_PATH.read_text(encoding="utf-8")

    assert "#13560" in text
    assert "#14711" in text
    assert "Proposal / write-up only" in text
    assert "net-wrench gate stays in force exactly as it is" in text
    assert "per-hand values remain diagnostics only" in text


def test_proposal_contains_conditioning_math_and_cancellation_analysis() -> None:
    """Section 1 must detail the conditioning math and cancellation ratios."""
    text = PROPOSAL_PATH.read_text(encoding="utf-8")

    assert "1. conditioning math" in text.lower()
    assert "cancellation ratio" in text.lower()
    # Overlap swing cancellation ratio of ~57 and error threshold ~0.049%
    assert "0.049%" in text or "0.05%" in text
    assert "overlap-swing" in text
    assert "statically indeterminate" in text


def test_proposal_contains_finding_a_and_self_consistency_residuals() -> None:
    """Section 2 must document Finding A with reference and OpenSim residuals."""
    text = PROPOSAL_PATH.read_text(encoding="utf-8")

    assert "finding a" in text.lower()
    assert "0.0044" in text or "0.0046" in text
    assert "0.019" in text or "0.037" in text
    assert "central difference" in text.lower()
    assert "ClubDynamics.from_spec" in text


def test_proposal_specifies_options_and_risks() -> None:
    """Proposal must detail Option B (recorder fix) and risks/limitations."""
    text = PROPOSAL_PATH.read_text(encoding="utf-8")

    assert "option b" in text.lower()
    assert "recorder fix" in text.lower()
    assert "PairTraceRecorder" in text or "ContactGripSimulator" in text
    assert "dynamic-consistency check" in text.lower()
    assert "additional diagnostic only" in text.lower()
    assert "dieter" in text.lower()


def _validate_summary_text(summary: str, context: str) -> None:
    """Assert summary carries the required #14733 proposal markers."""
    for marker in EXPECTED_SUMMARY_MARKERS:
        assert marker in summary, f"Missing {marker!r} in {context}: {summary!r}"


def _validate_pending_fragment(fragment_path: Path) -> None:
    """Validate an uncollated changes/ fragment for #14733."""
    findings = changes_fragment.validate_fragment_file(fragment_path)
    assert not findings, f"Invalid #14733 fragment at {fragment_path}: {findings}"
    fragment = changes_fragment.load_fragment(fragment_path)
    assert fragment.issue == PROPOSAL_ISSUE, (
        f"Expected fragment issue {PROPOSAL_ISSUE}, got {fragment.issue}"
    )
    _validate_summary_text(fragment.summary, f"fragment {fragment_path}")


def _collect_devlog_entries(repo_root: Path) -> list[development_log.Entry]:
    """Return parsed entries across canonical and archived development logs."""
    entries: list[development_log.Entry] = []
    live_path = development_log.resolve_canonical_devlog_path(repo_root)
    if live_path.is_file():
        entries.extend(
            development_log.parse_entries(live_path.read_text(encoding="utf-8"))
        )
    dev_dir = repo_root / "docs" / "development"
    if dev_dir.is_dir():
        for archive_path in sorted(dev_dir.glob("DEVELOPMENT_LOG_ARCHIVE*.md")):
            if archive_path.is_file():
                entries.extend(
                    development_log.parse_entries(
                        archive_path.read_text(encoding="utf-8")
                    )
                )
    return entries


def _validate_collated_devlog_entry(repo_root: Path) -> set[str]:
    """Validate collated DL-#14733 entry and return its linked PR keys."""
    target_id = f"DL-#{PROPOSAL_ISSUE}"
    matches = [
        entry
        for entry in _collect_devlog_entries(repo_root)
        if entry.entry_id == target_id
    ]
    assert len(matches) == 1, (
        f"Expected exactly one {target_id} development-log entry after collation, "
        f"found {len(matches)}"
    )
    entry = matches[0]
    assert entry.fields.get("Issue") == f"#{PROPOSAL_ISSUE}"
    assert entry.state in development_log.VALID_STATES and entry.state != "abandoned"
    _validate_summary_text(entry.fields.get("Summary", ""), f"{target_id} Summary")
    last_verified = entry.fields.get("Last verified", "")
    assert FRAGMENT_REL_PATH.as_posix() in last_verified, (
        f"{target_id} Last verified must record collation provenance: {last_verified!r}"
    )
    pr_keys = set(_PR_REF_RE.findall(entry.fields.get("PR", "")))
    assert pr_keys, f"{target_id} PR field must name a pull request"
    return pr_keys


def _collect_spec_changelog_rows(repo_root: Path) -> list[spec_changelog.Row]:
    """Return parsed change-log rows across SPEC.md and CHANGELOG_ARCHIVE.md."""
    rows: list[spec_changelog.Row] = []
    spec_path = repo_root / "SPEC.md"
    if spec_path.is_file():
        rows.extend(
            spec_changelog.parse_changelog(spec_path.read_text(encoding="utf-8")).rows
        )
    archive_path = repo_root / "docs" / "spec" / "CHANGELOG_ARCHIVE.md"
    if archive_path.is_file():
        archive_text = archive_path.read_text(encoding="utf-8")
        blocks = [m["body"] for m in _ARCHIVED_ROWS_BLOCK_RE.finditer(archive_text)]
        tables = (
            [f"## Change Log\n\n{block}" for block in blocks]
            if blocks
            else [archive_text]
        )
        for table in tables:
            try:
                rows.extend(spec_changelog.parse_changelog(table).rows)
            except spec_changelog.SpecChangelogError:
                continue
    return rows


def _validate_collated_spec_row(repo_root: Path, pr_keys: set[str]) -> None:
    """Assert SPEC change log (or archive) carries the collated #14733 PR row."""
    matching_rows = [
        row
        for row in _collect_spec_changelog_rows(repo_root)
        if row.key in pr_keys and "OpenSim net-moment gap" in row.summary
    ]
    assert matching_rows, (
        f"Missing collated SPEC change-log row for {sorted(pr_keys)} after "
        f"{FRAGMENT_REL_PATH} removal"
    )
    for row in matching_rows:
        _validate_summary_text(row.summary, f"SPEC change-log row {row.key}")


def assert_proposal_change_evidence(repo_root: Path) -> None:
    """Validate #14733 durable change-history evidence in ``repo_root``.

    Preconditions:
        ``repo_root`` is an existing ``Path`` directory.
    Postconditions:
        Either the pending ``changes/14733-opensim-net-moment-gap-proposal.md``
        fragment validates cleanly, or (post-collation) both ``DL-#14733`` and
        its matching PR-keyed ``SPEC.md`` change-log row exist and carry the
        #14733 proposal summary.
    """
    if not isinstance(repo_root, Path):
        raise TypeError(f"repo_root must be a pathlib.Path, got {type(repo_root)!r}")
    if not repo_root.is_dir():
        raise ValueError(f"repo_root must be an existing directory: {repo_root}")

    fragment_path = repo_root / FRAGMENT_REL_PATH
    if fragment_path.exists():
        assert fragment_path.is_file(), f"Expected file at {fragment_path}"
        _validate_pending_fragment(fragment_path)
        return

    pr_keys = _validate_collated_devlog_entry(repo_root)
    _validate_collated_spec_row(repo_root, pr_keys)


def _seed_minimal_collation_repo(repo_root: Path) -> Path:
    """Create a minimal inert repo layout with the #14733 fragment."""
    fragment_path = repo_root / FRAGMENT_REL_PATH
    fragment_path.parent.mkdir(parents=True, exist_ok=True)
    fragment_path.write_text(
        "---\n"
        "issue: 14733\n"
        'summary: "Proposal (write-up only): #13560 OpenSim net-moment gap - '
        'recorder inconsistency finding, conditioning math, options"\n'
        "---\n",
        encoding="utf-8",
    )
    (repo_root / "SPEC.md").write_text(
        "## 12. Change Log\n\n| Date | PR | Changes |\n| --- | --- | --- |\n",
        encoding="utf-8",
    )
    devlog_path = repo_root / "docs" / "development" / "DEVELOPMENT_LOG.md"
    devlog_path.parent.mkdir(parents=True, exist_ok=True)
    devlog_path.write_text("# Development Log\n\n## Active\n\n", encoding="utf-8")
    return fragment_path


def test_change_fragment_exists_and_validates() -> None:
    """Live repository must carry valid #14733 change-history evidence."""
    assert_proposal_change_evidence(ROOT)


def test_change_history_evidence_survives_normal_collation(tmp_path: Path) -> None:
    """Normal change-fragment collation must preserve validated #14733 evidence."""
    fragment_path = _seed_minimal_collation_repo(tmp_path)
    assert_proposal_change_evidence(tmp_path)

    deleted = changes_fragment.collate(
        tmp_path,
        [fragment_path],
        pr=14888,
        today=date(2026, 11, 8),
        sha="36155e93d2000000000000000000000000000000",
    )
    assert deleted == [fragment_path]
    assert not fragment_path.exists()

    assert_proposal_change_evidence(tmp_path)


def test_change_history_evidence_survives_devlog_and_spec_archival(
    tmp_path: Path,
) -> None:
    """Archived DL-#14733 and archived SPEC rows must still satisfy evidence."""
    fragment_path = _seed_minimal_collation_repo(tmp_path)
    changes_fragment.collate(
        tmp_path,
        [fragment_path],
        pr=14888,
        today=date(2026, 11, 8),
        sha="36155e93d2000000000000000000000000000000",
    )

    devlog_path = tmp_path / "docs" / "development" / "DEVELOPMENT_LOG.md"
    collated_devlog = devlog_path.read_text(encoding="utf-8")
    entry_start = collated_devlog.index("### DL-#14733")
    archive_devlog = (
        tmp_path / "docs" / "development" / "DEVELOPMENT_LOG_ARCHIVE_2026.md"
    )
    archive_devlog.write_text(
        "# Development Log Archive — 2026\n\n" + collated_devlog[entry_start:],
        encoding="utf-8",
    )
    devlog_path.write_text("# Development Log\n\n## Active\n\n", encoding="utf-8")

    spec_path = tmp_path / "SPEC.md"
    spec_rows_table = spec_path.read_text(encoding="utf-8").split(
        "## 12. Change Log\n\n"
    )[1]
    spec_archive = tmp_path / "docs" / "spec" / "CHANGELOG_ARCHIVE.md"
    spec_archive.parent.mkdir(parents=True, exist_ok=True)
    spec_archive.write_text(
        "# SPEC.md Change-Log Archive\n\n"
        "<!-- archive:change-log-rows-wave4:begin -->\n"
        f"{spec_rows_table}"
        "<!-- archive:change-log-rows-wave4:end -->\n",
        encoding="utf-8",
    )
    spec_path.write_text(
        "## 12. Change Log\n\n| Date | PR | Changes |\n| --- | --- | --- |\n",
        encoding="utf-8",
    )

    assert_proposal_change_evidence(tmp_path)


def test_change_history_evidence_fails_closed_on_missing_or_partial_records(
    tmp_path: Path,
) -> None:
    """Missing, uncollated, partial, or invalid #14733 evidence must fail closed."""
    with pytest.raises(TypeError, match="pathlib.Path"):
        assert_proposal_change_evidence("not-a-path")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="existing directory"):
        assert_proposal_change_evidence(tmp_path / "nonexistent")

    fragment_path = _seed_minimal_collation_repo(tmp_path)
    fragment_path.unlink()
    with pytest.raises(AssertionError, match="DL-#14733"):
        assert_proposal_change_evidence(tmp_path)

    fragment_path = _seed_minimal_collation_repo(tmp_path)
    changes_fragment.collate(
        tmp_path,
        [fragment_path],
        pr=14888,
        today=date(2026, 11, 8),
        sha="36155e93d2000000000000000000000000000000",
    )
    spec_without_row = (
        "## 12. Change Log\n\n| Date | PR | Changes |\n| --- | --- | --- |\n"
    )
    saved_spec = (tmp_path / "SPEC.md").read_text(encoding="utf-8")
    (tmp_path / "SPEC.md").write_text(spec_without_row, encoding="utf-8")
    with pytest.raises(AssertionError, match="Missing collated SPEC change-log row"):
        assert_proposal_change_evidence(tmp_path)

    (tmp_path / "SPEC.md").write_text(saved_spec, encoding="utf-8")
    (tmp_path / "docs" / "development" / "DEVELOPMENT_LOG.md").write_text(
        "# Development Log\n\n## Active\n\n", encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="DL-#14733"):
        assert_proposal_change_evidence(tmp_path)

    fragment_path.write_text(
        '---\nissue: 99999\nsummary: "Wrong issue"\n---\n',
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="14733"):
        assert_proposal_change_evidence(tmp_path)
