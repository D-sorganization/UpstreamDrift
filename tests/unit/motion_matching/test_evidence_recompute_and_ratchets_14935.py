"""Red-first tests for #14935 evidence recomputation, provenance regex, and ratchets."""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from src.shared.python.motion_matching.evidence_validator import (
    validate_candidate_gate_metrics,
    validate_candidate_package,
    validate_committed_evidence,
    validate_ledger_ratchets,
    validate_receipt_paths,
)
from src.shared.python.motion_matching.ledger import scan_candidate_packages

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_false_accepted_with_all_gates_passing_and_bad_count_flagged() -> None:
    """Bidirectional consistency flags accepted=False when every gate passed."""
    all_passed = {
        "accepted": False,
        "gates": {"early_retention_pass": True, "pelvis_yaw_pass": True},
        "gates_passed": "2/2",
    }
    issues = validate_candidate_package("candidates/fake_all_pass.json", all_passed)
    assert any(issue.rule == "accepted_gates_consistency" for issue in issues)

    bad_count = {
        "accepted": False,
        "gates": {"early_retention_pass": True, "pelvis_yaw_pass": False},
        "gates_passed": "not-a-count",
    }
    count_issues = validate_candidate_package(
        "candidates/fake_bad_count.json", bad_count
    )
    assert any(issue.rule == "accepted_gates_consistency" for issue in count_issues)


def test_stored_gate_true_with_metric_above_threshold_is_flagged() -> None:
    """Metrics exceeding AcceptanceGates thresholds falsify stored True gates."""
    dishonest = {
        "duration_s": 0.8,
        "rmse_m": 0.045,
        "early_rmse_m": 0.050,
        "terminal_rmse_m": 0.120,
        "clubhead_terminal_rmse_m": 0.150,
        "pelvis_yaw_error_pct": 9.5,
        "accepted": True,
        "gates": {
            "whole_window_pass": True,
            "early_retention_pass": True,
            "terminal_rmse_pass": True,
            "clubhead_terminal_pass": True,
            "pelvis_yaw_pass": True,
        },
        "gates_passed": "5/5",
    }
    metric_issues = validate_candidate_gate_metrics(
        "candidates/dishonest.json", dishonest
    )
    rules = {issue.rule for issue in metric_issues}
    assert "gate_metric_mismatch" in rules
    messages = "\n".join(issue.message for issue in metric_issues)
    for gate_name in (
        "whole_window_pass",
        "early_retention_pass",
        "terminal_rmse_pass",
        "clubhead_terminal_pass",
        "pelvis_yaw_pass",
    ):
        assert gate_name in messages

    for pkg in sorted((REPO_ROOT / "candidates").glob("*.json")):
        payload = json.loads(pkg.read_text(encoding="utf-8"))
        committed_issues = validate_candidate_gate_metrics(
            pkg.relative_to(REPO_ROOT).as_posix(), payload
        )
        assert committed_issues == [], f"{pkg.name}: {committed_issues}"


def test_unanchored_and_uri_host_paths_are_flagged() -> None:
    """Unanchored /home, /Users, /mnt, ~/, file://, and drive paths must fail."""
    bad_samples = (
        "file:///home/owner/fits/run.npz",
        "~/fits/candidate.npz",
        "source: /Users/dev/out.json",
        "uri:D:\\runs\\stage.json",
        "loaded from /mnt/c/data/trial.json",
        "candidate:/home/owner/fits/w030r/candidate.npz",
    )
    for sample in bad_samples:
        issues = validate_receipt_paths(
            "evidence/matched/sample/receipt.json",
            {"warm_start_source": sample},
        )
        assert any(i.rule == "host_absolute_provenance" for i in issues), sample

    clean = validate_receipt_paths(
        "evidence/matched/sample/receipt.json",
        {
            "warm_start_source": "candidates/candidate_ms_080s_package.json",
            "warm_start_sha256": "a" * 64,
            "docs_url": "https://github.com/D-sorganization/UpstreamDrift",
        },
    )
    assert clean == []


def test_scan_candidate_packages_flips_verdict_on_gate_or_metric_mutation(
    tmp_path: Path,
) -> None:
    """Mutating a copied candidate's gates or metrics flips its derived verdict."""
    cand_dir = tmp_path / "candidates"
    cand_dir.mkdir()
    src = REPO_ROOT / "candidates" / "candidate_ms_080s_targeted_package.json"
    copied = cand_dir / src.name
    shutil.copy2(src, copied)

    initial_rows = scan_candidate_packages(tmp_path)
    assert len(initial_rows) == 1
    assert initial_rows[0]["verdict"] == "REJECTED"

    passing = json.loads(copied.read_text(encoding="utf-8"))
    passing.update(
        {
            "rmse_m": 0.015,
            "early_rmse_m": 0.008,
            "terminal_rmse_m": 0.020,
            "clubhead_terminal_rmse_m": 0.030,
            "pelvis_yaw_error_pct": 1.2,
            "accepted": True,
            "gates": dict.fromkeys(passing["gates"], True),
            "gates_passed": "5/5",
        }
    )
    copied.write_text(json.dumps(passing), encoding="utf-8")
    qualified_rows = scan_candidate_packages(tmp_path)
    assert qualified_rows[0]["verdict"] == "QUALIFIED"

    tampered = dict(passing, early_rmse_m=0.050)
    copied.write_text(json.dumps(tampered), encoding="utf-8")
    rejected_rows = scan_candidate_packages(tmp_path)
    assert rejected_rows[0]["verdict"] == "REJECTED"


def test_ledger_ratchets_and_committed_scan_bounds(tmp_path: Path) -> None:
    """Committed ledger stays within <=37 unclassified and <=122 missing verdicts."""
    ledger_payload = json.loads(
        (REPO_ROOT / "reports" / "matched_swing_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    rows = [r for r in ledger_payload["rows"] if isinstance(r, dict)]
    assert validate_ledger_ratchets(rows) == []

    extra_unclassified = [
        *rows,
        {"receipt_path": "x.json", "lane": "unclassified", "acceptance": None},
    ]
    breach = validate_ledger_ratchets(extra_unclassified)
    breach_rules = {issue.rule for issue in breach}
    assert "ledger_unclassified_ratchet" in breach_rules
    assert "ledger_unverified_ratchet" in breach_rules

    waivers_path = REPO_ROOT / "scripts" / "config" / "evidence_validation_waivers.json"
    report = validate_committed_evidence(
        REPO_ROOT, waivers_path=waivers_path, today=date(2026, 11, 10)
    )
    assert not report.failed
