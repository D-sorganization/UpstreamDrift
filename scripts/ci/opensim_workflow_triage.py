#!/usr/bin/env python3
"""Classify CI Requires OpenSim workflow failures for throughput triage (#13582)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = "D-sorganization/UpstreamDrift"
WORKFLOW_FILE = "ci-requires-opensim.yml"
DEFAULT_WINDOW_DAYS = 14

FAILED_TEST_RE = re.compile(r"FAILED (tests/[^\s]+)")
TEST_NAME_RE = re.compile(r"_{5,}\s+(test_[^\s]+)\s+_{5,}")


class FailureCategory(str, Enum):
    INFRA_DISK = "infra_disk"
    INFRA_RUNNER = "infra_runner"
    INFRA_TIMEOUT = "infra_timeout"
    TEST_DEFECT = "test_defect"
    FLAKE = "flake"
    ENV_MISCONFIG = "env_misconfig"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Classification:
    """One failed OpenSim job log classification.

    Postconditions:
        ``category`` is a known ``FailureCategory`` value.
        ``signature`` is non-empty.
    """

    category: FailureCategory
    signature: str
    evidence: str = ""


def _normalize_log(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("log text must be str")
    return text


def extract_failure_signatures(log_text: str) -> list[str]:
    """Return sorted unique pytest node ids or test function names from a log."""
    text = _normalize_log(log_text)
    names = set(FAILED_TEST_RE.findall(text))
    names.update(TEST_NAME_RE.findall(text))
    return sorted(names)


def classify_opensim_job_log(log_text: str) -> Classification:
    """Classify one failed job log into infra / test / flake / env buckets."""
    text = _normalize_log(log_text)
    if not text.strip():
        return Classification(
            FailureCategory.INFRA_RUNNER,
            "log_unavailable",
            "empty or expired GitHub Actions log",
        )

    lowered = text.lower()
    signatures = extract_failure_signatures(text)
    if signatures:
        squat = "test_squat_joint_reactions_computed" in signatures
        native = "test_native_constant_interval_matches_fresh_raw_replay" in signatures
        if squat and native:
            joined = (
                "tests/unit/lifting/test_opensim_lift_joint_reactions.py::"
                "test_squat_joint_reactions_computed+"
                "tests/opensim/test_native_muscle_effort_native.py::"
                "test_native_constant_interval_matches_fresh_raw_replay"
            )
            return Classification(
                FailureCategory.TEST_DEFECT,
                joined,
                "deterministic dual failure since main 2e64f84 (#13564)",
            )
        return Classification(FailureCategory.TEST_DEFECT, signatures[0])

    if " failed," in lowered and "passed" in lowered:
        return Classification(FailureCategory.TEST_DEFECT, "pytest_summary_failure")

    if "no space left on device" in lowered or "enospc" in lowered:
        return Classification(FailureCategory.INFRA_DISK, "disk_full")

    if (
        "exceeded the maximum time" in lowered
        or "job was canceled" in lowered
        or "##[error]the operation was canceled" in lowered.replace(" ", "")
    ):
        return Classification(FailureCategory.INFRA_TIMEOUT, "job_timeout")

    if "orphan processes" in lowered or "lost connection" in lowered:
        return Classification(FailureCategory.INFRA_RUNNER, "runner_orphan_cleanup")

    if (
        "expected opensim version" in lowered
        or "externally-managed-environment" in lowered
    ):
        return Classification(FailureCategory.ENV_MISCONFIG, "opensim_or_pip_env")

    if "git fsck failed" in lowered or "corrupt git" in lowered:
        return Classification(FailureCategory.ENV_MISCONFIG, "runner_git_checkout")

    return Classification(FailureCategory.UNKNOWN, "unclassified")


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return the linear-interpolation quantile for sorted numeric values."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be within [0, 1]")
    if not values:
        raise ValueError("values must be non-empty")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    index = quantile * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def failure_rate(runs: Iterable[dict[str, Any]]) -> float:
    """Failure rate among completed runs, excluding cancelled conclusions."""
    successes = 0
    failures = 0
    for run in runs:
        conclusion = run.get("conclusion")
        if conclusion == "success":
            successes += 1
        elif conclusion == "failure":
            failures += 1
    total = successes + failures
    if total == 0:
        return 0.0
    return failures / total


def _duration_seconds(run: dict[str, Any]) -> float | None:
    started = run.get("run_started_at")
    updated = run.get("updated_at")
    if not started or not updated:
        return None
    start = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
    end = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
    return (end - start).total_seconds()


def _flake_shas(runs: Sequence[dict[str, Any]]) -> list[str]:
    by_sha: dict[str, set[str]] = {}
    for run in runs:
        sha = run.get("head_sha")
        conclusion = run.get("conclusion")
        if not sha or conclusion not in {"success", "failure"}:
            continue
        by_sha.setdefault(sha, set()).add(conclusion)
    return sorted(sha for sha, outcomes in by_sha.items() if len(outcomes) > 1)


def summarize_workflow_runs(
    runs: Sequence[dict[str, Any]], *, window_days: int
) -> dict[str, Any]:
    """Summarize raw GitHub workflow run JSON rows."""
    if window_days < 1:
        raise ValueError("window_days must be positive")
    completed = [r for r in runs if r.get("status") == "completed"]
    successes = [r for r in completed if r.get("conclusion") == "success"]
    failures = [r for r in completed if r.get("conclusion") == "failure"]
    durations = [d for r in successes for d in [_duration_seconds(r)] if d is not None]
    return {
        "window_days": window_days,
        "run_count": len(runs),
        "completed_count": len(completed),
        "success_count": len(successes),
        "failure_count": len(failures),
        "cancelled_count": sum(
            1 for r in completed if r.get("conclusion") == "cancelled"
        ),
        "failure_rate_non_cancelled": failure_rate(completed),
        "flake_sha_count": len(_flake_shas(completed)),
        "success_duration_seconds": {
            "p90": percentile(durations, 0.9) if durations else 0.0,
            "sample_size": len(durations),
        },
    }


def top_failure_signatures(
    rows: Sequence[tuple[str, str, int, int]], *, limit: int
) -> list[dict[str, Any]]:
    """Format ranked failure signature rows for documentation output."""
    if limit < 1:
        raise ValueError("limit must be positive")
    ordered = sorted(rows, key=lambda item: item[2], reverse=True)
    output: list[dict[str, Any]] = []
    for category, signature, count, example_run_id in ordered[:limit]:
        output.append(
            {
                "category": category,
                "signature": signature,
                "count": count,
                "example_run_id": example_run_id,
            }
        )
    return output


def _gh_api(path: str) -> dict[str, Any]:
    raw = subprocess.check_output(["gh", "api", path], text=True)
    return json.loads(raw)


def _fetch_runs(window_days: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    runs: list[dict[str, Any]] = []
    for page in range(1, 11):
        payload = _gh_api(
            f"repos/{REPO}/actions/workflows/{WORKFLOW_FILE}/runs"
            f"?per_page=100&page={page}"
        )
        batch = payload.get("workflow_runs", [])
        if not batch:
            break
        for row in batch:
            created = datetime.fromisoformat(
                str(row["created_at"]).replace("Z", "+00:00")
            )
            if created >= cutoff:
                runs.append(row)
        oldest = datetime.fromisoformat(
            str(batch[-1]["created_at"]).replace("Z", "+00:00")
        )
        if oldest < cutoff:
            break
    return runs


def _failed_job_log(run_id: int) -> str:
    try:
        return subprocess.check_output(
            [
                "gh",
                "run",
                "view",
                str(run_id),
                "--repo",
                REPO,
                "--log-failed",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        return exc.output or ""


def build_live_report(window_days: int) -> dict[str, Any]:
    """Pull recent workflow runs and classify failed job logs."""
    runs = _fetch_runs(window_days)
    summary = summarize_workflow_runs(runs, window_days=window_days)
    category_counts: dict[str, int] = {cat.value: 0 for cat in FailureCategory}
    signature_counts: dict[tuple[str, str], list[int]] = {}
    failures = [r for r in runs if r.get("conclusion") == "failure"]
    for run in failures:
        log_text = _failed_job_log(int(run["id"]))
        classified = classify_opensim_job_log(log_text)
        category_counts[classified.category.value] += 1
        key = (classified.category.value, classified.signature)
        signature_counts.setdefault(key, []).append(int(run["id"]))

    ranked_rows = [
        (cat, sig, len(run_ids), run_ids[0])
        for (cat, sig), run_ids in signature_counts.items()
    ]
    top = top_failure_signatures(ranked_rows, limit=10)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workflow": WORKFLOW_FILE,
        "repository": REPO,
        "source": "RM-CI-000 / GitHub Actions workflow run API + failed job logs",
        "window_days": window_days,
        "rates": {
            "failure_rate_non_cancelled": summary["failure_rate_non_cancelled"],
            "pull_request": failure_rate(
                [r for r in runs if r.get("event") == "pull_request"]
            ),
            "main_push": failure_rate(
                [
                    r
                    for r in runs
                    if r.get("event") == "push" and r.get("head_branch") == "main"
                ]
            ),
        },
        "success_duration_seconds": summary["success_duration_seconds"],
        "failure_categories": category_counts,
        "flake_sha_count": summary["flake_sha_count"],
        "top_failure_signatures": _attach_owner_decisions(top),
        "counts": {
            "runs": summary["run_count"],
            "completed": summary["completed_count"],
            "success": summary["success_count"],
            "failure": summary["failure_count"],
            "cancelled": summary["cancelled_count"],
        },
    }


def _attach_owner_decisions(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        owner, decision, notes = _owner_decision(row["category"], row["signature"])
        enriched.append({**row, "owner": owner, "decision": decision, "notes": notes})
    return enriched


def _owner_decision(category: str, signature: str) -> tuple[str, str, str]:
    if signature in {
        "test_squat_joint_reactions_computed",
        "test_native_constant_interval_matches_fresh_raw_replay",
        "pytest_summary_failure",
    }:
        return (
            "@core (lifting/native OpenSim)",
            "fix",
            "Same regression cluster as #13564; partial logs may list only one failing test.",
        )
    if signature.startswith(
        "tests/unit/lifting/test_opensim_lift_joint_reactions.py::test_squat"
    ):
        return (
            "@core (lifting/native OpenSim)",
            "fix",
            "Track in #13564; restore squat ID accuracy and native replay feasibility "
            "without tolerance loosening.",
        )
    if category == FailureCategory.INFRA_RUNNER.value:
        return (
            "@core (Runner_Dashboard)",
            "monitor",
            "Orphan cleanup and expired logs are not pytest flakes; no quarantine.",
        )
    if category == FailureCategory.INFRA_DISK.value:
        return (
            "@core (Runner_Dashboard)",
            "fix_follow_up",
            "Disk pressure on docker fleet; expand cleanup before tiering experiments.",
        )
    if category == FailureCategory.ENV_MISCONFIG.value:
        return (
            "@core",
            "fix",
            "Runner checkout or OpenSim pin drift; fix workflow/venv contract.",
        )
    if category == FailureCategory.UNKNOWN.value:
        return (
            "@core",
            "monitor",
            "Re-pull logs before acting; likely expired log retention on cancelled runs.",
        )
    return ("@core", "monitor", "See linked workflow run for details.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help="Lookback window for GitHub Actions runs (default: 14).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON report to this path (default: stdout).",
    )
    args = parser.parse_args(argv)
    report = build_live_report(args.window_days)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
