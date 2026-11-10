"""Committed evidence validator (#14935)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.shared.python.motion_matching.evidence_validator import (
    validate_candidate_package,
    validate_committed_evidence,
    validate_receipt_paths,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_seeded_candidate_flags_accepted_gate_mismatch() -> None:
    path = REPO_ROOT / "candidates" / "candidate_ms_080s_seeded_package.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    issues = validate_candidate_package(path.relative_to(REPO_ROOT).as_posix(), data)
    assert issues
    assert issues[0].rule == "accepted_gates_consistency"


def test_receipt_host_paths_flagged() -> None:
    path = (
        REPO_ROOT / "evidence" / "matched" / "driver_g1_crocoddyl_rk45" / "receipt.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    issues = validate_receipt_paths(path.relative_to(REPO_ROOT).as_posix(), data)
    assert any(issue.rule == "host_absolute_provenance" for issue in issues)


def test_committed_scan_passes_with_waivers() -> None:
    report = validate_committed_evidence(REPO_ROOT, today=date(2026, 11, 10))
    assert report.waived
    assert not report.failed


def test_committed_scan_fails_without_waivers(tmp_path: Path) -> None:
    empty_waivers = tmp_path / "waivers.json"
    empty_waivers.write_text(json.dumps({"waivers": []}), encoding="utf-8")
    report = validate_committed_evidence(
        REPO_ROOT,
        waivers_path=empty_waivers,
        today=date(2026, 11, 10),
    )
    assert report.failed
    rules = {issue.rule for issue in report.issues}
    assert "accepted_gates_consistency" in rules
    assert "host_absolute_provenance" in rules


def test_loads_committed_ledger_and_reports_unclassified() -> None:
    from src.shared.python.motion_matching.evidence_validator import (
        ledger_unclassified_count,
    )

    ledger_path = REPO_ROOT / "reports" / "matched_swing_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    rows = ledger.get("rows", [])
    assert len(rows) > 0
    count = ledger_unclassified_count(rows)
    assert count > 0, "fixture expects historical unclassified rows in committed ledger"
