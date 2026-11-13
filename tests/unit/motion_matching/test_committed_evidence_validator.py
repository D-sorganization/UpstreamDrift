"""Committed evidence validator (#14935)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.shared.python.motion_matching.evidence_validator import (
    load_waivers,
    validate_candidate_package,
    validate_committed_evidence,
    validate_receipt_paths,
    validate_receipt_taxonomy,
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


def test_committed_scan_flags_missing_taxonomy_fields(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    evidence_dir = repo / "evidence" / "matched" / "driver_sample"
    evidence_dir.mkdir(parents=True)
    (evidence_dir / "receipt.json").write_text(
        json.dumps({"engine": "mujoco", "lane": "matched"}),
        encoding="utf-8",
    )
    empty_waivers = tmp_path / "waivers.json"
    empty_waivers.write_text(json.dumps({"waivers": []}), encoding="utf-8")
    report = validate_committed_evidence(
        repo,
        waivers_path=empty_waivers,
        today=date(2026, 11, 10),
    )
    rules = {issue.rule for issue in report.issues}
    assert "missing_taxonomy" in rules
    assert any("capture" in issue.message for issue in report.issues)


def test_committed_scan_flags_taxonomy_metadata_conflict(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    evidence_dir = repo / "evidence" / "tour_baselines" / "evidence" / "driver_sample"
    evidence_dir.mkdir(parents=True)
    (evidence_dir / "receipt.json").write_text(
        json.dumps(
            {
                "engine": "tools",
                "lane": "native",
                "capture": "driver",
            }
        ),
        encoding="utf-8",
    )
    empty_waivers = tmp_path / "waivers.json"
    empty_waivers.write_text(json.dumps({"waivers": []}), encoding="utf-8")
    report = validate_committed_evidence(
        repo,
        waivers_path=empty_waivers,
        today=date(2026, 11, 10),
    )
    rules = {issue.rule for issue in report.issues}
    assert "explicit_taxonomy" in rules


def test_validate_receipt_taxonomy_allows_matched_bucket_sub_lane() -> None:
    issues = validate_receipt_taxonomy(
        "evidence/matched/driver_x/receipt.json",
        {
            "engine": "pinocchio",
            "lane": "crocoddyl_native_fit",
            "capture": "driver",
        },
    )
    assert not any(issue.rule == "explicit_taxonomy" for issue in issues)


def test_validate_receipt_taxonomy_unit_lane_mismatch_outside_matched() -> None:
    issues = validate_receipt_taxonomy(
        "evidence/tour_baselines/evidence/driver_run/receipt.json",
        {"engine": "tools", "lane": "wrong_lane", "capture": "driver"},
    )
    assert any(issue.rule == "explicit_taxonomy" for issue in issues)


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


def test_accepted_package_rejects_malformed_gates_passed() -> None:
    data = {
        "accepted": True,
        "gates": {"a": True, "b": False},
        "gates_passed": "unknown",
    }
    issues = validate_candidate_package("candidates/test.json", data)
    assert any("not a valid N/M count" in issue.message for issue in issues)


def test_receipt_flags_forward_slash_windows_and_wsl_paths() -> None:
    for sample in ("C:/Users/x/out.json", "/mnt/c/Users/x/out.json"):
        issues = validate_receipt_paths(
            "evidence/matched/sample/receipt.json",
            {"warm_start_source": sample},
        )
        assert issues
        assert issues[0].rule == "host_absolute_provenance"


def test_malformed_waiver_entries_are_ignored(tmp_path: Path) -> None:
    waivers_path = tmp_path / "waivers.json"
    waivers_path.write_text(
        json.dumps(
            {
                "waivers": [
                    {
                        "id": "no-expiry",
                        "rule": "accepted_gates_consistency",
                        "path": "candidates/candidate_ms_080s_seeded_package.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_waivers(waivers_path) == []
    report = validate_committed_evidence(
        REPO_ROOT,
        waivers_path=waivers_path,
        today=date(2026, 11, 10),
    )
    assert report.failed


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
