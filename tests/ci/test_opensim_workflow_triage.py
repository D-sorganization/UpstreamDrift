"""Contract tests for OpenSim workflow failure triage (UD-CI-002t, #13582)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.ci.opensim_workflow_triage import (
    FailureCategory,
    classify_opensim_job_log,
    extract_failure_signatures,
    failure_rate,
    percentile,
    summarize_workflow_runs,
    top_failure_signatures,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = REPO_ROOT / "docs/operations/evidence/opensim-ci-triage-20261021.json"


SQUAT_AND_NATIVE_SNIPPET = """
____________ test_native_constant_interval_matches_fresh_raw_replay ____________
tests/opensim/test_native_muscle_effort_native.py:282: in test_native_constant_interval_matches_fresh_raw_replay
E   AssertionError: assert (False)
_____________________ test_squat_joint_reactions_computed ______________________
tests/unit/lifting/test_opensim_lift_joint_reactions.py:191: in test_squat_joint_reactions_computed
    assert sol["acceleration_error_max"] < 5e-6
E   AssertionError: assert 1.1454e-05 < 5e-06
= 2 failed, 258 passed, 16 skipped in 148.13s =
"""

ORPHAN_SNIPPET = "Cleaning up orphan processes"

DISK_SNIPPET = "OSError: [Errno 28] No space left on device"


class TestClassifyOpensimJobLog:
    def test_dual_test_regression_is_test_defect(self) -> None:
        result = classify_opensim_job_log(SQUAT_AND_NATIVE_SNIPPET)
        assert result.category == FailureCategory.TEST_DEFECT
        assert "test_squat_joint_reactions_computed" in result.signature
        assert (
            "test_native_constant_interval_matches_fresh_raw_replay" in result.signature
        )

    def test_orphan_cleanup_is_infra_runner(self) -> None:
        result = classify_opensim_job_log(ORPHAN_SNIPPET)
        assert result.category == FailureCategory.INFRA_RUNNER

    def test_disk_full_is_infra_disk(self) -> None:
        result = classify_opensim_job_log(DISK_SNIPPET)
        assert result.category == FailureCategory.INFRA_DISK

    def test_empty_log_is_infra_runner(self) -> None:
        result = classify_opensim_job_log("")
        assert result.category == FailureCategory.INFRA_RUNNER
        assert "log_unavailable" in result.signature


class TestSummarizeWorkflowRuns:
    def test_failure_rate_excludes_cancelled(self) -> None:
        runs = [
            {"conclusion": "success"},
            {"conclusion": "failure"},
            {"conclusion": "cancelled"},
        ]
        assert failure_rate(runs) == 0.5

    def test_flake_requires_same_sha_mixed_outcomes(self) -> None:
        runs = [
            {"status": "completed", "head_sha": "abc", "conclusion": "success"},
            {"status": "completed", "head_sha": "abc", "conclusion": "failure"},
            {"status": "completed", "head_sha": "def", "conclusion": "failure"},
        ]
        summary = summarize_workflow_runs(runs, window_days=14)
        assert summary["flake_sha_count"] == 1

    def test_percentile_p90(self) -> None:
        values = [float(i) for i in range(1, 11)]
        assert percentile(values, 0.9) == 9.1


class TestCommittedBaselineSnapshot:
    def test_baseline_file_exists_and_matches_schema(self) -> None:
        assert BASELINE.is_file(), (
            "regenerate with scripts/ci/opensim_workflow_triage.py"
        )
        payload = json.loads(BASELINE.read_text(encoding="utf-8"))
        assert payload["workflow"] == "ci-requires-opensim.yml"
        assert payload["window_days"] == 14
        assert "failure_rate_non_cancelled" in payload["rates"]
        assert payload["success_duration_seconds"]["p90"] > 0
        categories = payload["failure_categories"]
        assert set(categories) <= {
            "infra_disk",
            "infra_runner",
            "infra_timeout",
            "test_defect",
            "flake",
            "env_misconfig",
            "unknown",
        }
        top = payload["top_failure_signatures"]
        assert len(top) >= 3
        for row in top[:3]:
            assert row["owner"]
            assert row["decision"] in {"fix", "quarantine", "monitor", "fix_follow_up"}

    def test_top_signature_classifier_agrees_with_fixture(self) -> None:
        payload = json.loads(BASELINE.read_text(encoding="utf-8"))
        primary = payload["top_failure_signatures"][0]
        classified = classify_opensim_job_log(SQUAT_AND_NATIVE_SNIPPET)
        assert primary["category"] == classified.category.value
        assert (
            classified.signature in primary["signature"]
            or primary["signature"] in classified.signature
        )


class TestSignatureHelpers:
    def test_extract_failure_signatures_sorted(self) -> None:
        sigs = extract_failure_signatures(SQUAT_AND_NATIVE_SNIPPET)
        assert sigs == sorted(sigs)
        assert len(sigs) == 2

    def test_top_failure_signatures_orders_by_count(self) -> None:
        rows = top_failure_signatures(
            [
                ("test_defect", "a", 101, 1),
                ("infra_runner", "b", 5, 2),
                ("test_defect", "c", 50, 3),
            ],
            limit=3,
        )
        assert rows[0]["count"] == 101
        assert rows[0]["example_run_id"] == 1
