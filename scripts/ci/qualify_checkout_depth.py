"""Qualify CI job-specific checkout depth and prevent sparse-state regressions.

Part of #14896, parent #14893.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Sequence
import yaml

DEFAULT_JUSTIFIED_DEEP_JOBS = frozenset(
    {
        "spec-check",
        "changed-paths",
        "pr-size",
        "seam-drift-gate",
        "release",
        "docs-governance-gates",
        "publication-quality",
    }
)


@dataclass(frozen=True)
class CheckoutStepInfo:
    """Extracted checkout configuration from a workflow step."""

    job_name: str
    step_name: str
    fetch_depth: int | None
    path: str | None
    submodules: str | bool | None


@dataclass(frozen=True)
class QualificationReport:
    """Compliance assessment for a single job checkout."""

    job_name: str
    fetch_depth: int | None
    is_compliant: bool
    reason: str | None = None


def inventory_workflow_checkouts(workflow_path: Path | str) -> list[CheckoutStepInfo]:
    """Parse a GitHub Actions workflow and inventory all actions/checkout steps."""
    path = Path(workflow_path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    jobs = data.get("jobs", {})
    checkouts: list[CheckoutStepInfo] = []

    for job_name, job_spec in jobs.items():
        if not isinstance(job_spec, dict):
            continue
        for step in job_spec.get("steps", []):
            if not isinstance(step, dict):
                continue
            uses = step.get("uses", "")
            if "actions/checkout" in uses:
                with_params = step.get("with", {}) or {}
                raw_depth = with_params.get("fetch-depth")
                depth: int | None = (
                    int(raw_depth)
                    if raw_depth is not None and str(raw_depth).isdigit()
                    else None
                )
                checkouts.append(
                    CheckoutStepInfo(
                        job_name=job_name,
                        step_name=step.get("name", "checkout"),
                        fetch_depth=depth,
                        path=with_params.get("path"),
                        submodules=with_params.get("submodules"),
                    )
                )
    return checkouts


def qualify_checkout_depth(
    workflow_path: Path | str,
    justified_deep_jobs: set[str] | frozenset[str] | None = None,
) -> list[QualificationReport]:
    """Assess whether checkout depths conform to transfer and provenance budgets."""
    justified = (
        justified_deep_jobs
        if justified_deep_jobs is not None
        else DEFAULT_JUSTIFIED_DEEP_JOBS
    )
    checkouts = inventory_workflow_checkouts(workflow_path)
    reports: list[QualificationReport] = []

    for c in checkouts:
        if c.fetch_depth == 0 and c.job_name not in justified:
            reports.append(
                QualificationReport(
                    job_name=c.job_name,
                    fetch_depth=c.fetch_depth,
                    is_compliant=False,
                    reason=f"unjustified fetch-depth: 0 (job '{c.job_name}' is not in approved deep history list)",
                )
            )
        else:
            reports.append(
                QualificationReport(
                    job_name=c.job_name,
                    fetch_depth=c.fetch_depth,
                    is_compliant=True,
                )
            )
    return reports


def check_sparse_contamination(repo_root: Path | str) -> bool:
    """Return True if the git workspace carries leaked or corrupt sparse configuration."""
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "config", "--type=bool", "core.sparseCheckout"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip().lower() == "true":
        # Check if patterns exist
        list_proc = subprocess.run(
            ["git", "-C", str(repo_root), "sparse-checkout", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        patterns = (
            [p for p in list_proc.stdout.splitlines() if p.strip()]
            if list_proc.returncode == 0
            else []
        )
        if not patterns:
            return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for qualification checks."""
    parser = argparse.ArgumentParser(
        description="Qualify CI checkout depth and sparse contamination."
    )
    parser.add_argument("--workflow", help="Path to workflow YAML file to inspect")
    parser.add_argument(
        "--root", default=".", help="Root of git workspace to inspect for sparse state"
    )
    parser.add_argument(
        "--strict", action="store_true", help="Fail closed on non-compliant depths"
    )
    args = parser.parse_args(argv)

    if args.workflow:
        reports = qualify_checkout_depth(args.workflow)
        non_compliant = [r for r in reports if not r.is_compliant]
        for r in reports:
            status = "PASS" if r.is_compliant else "FAIL"
            print(f"[{status}] Job: {r.job_name} (depth: {r.fetch_depth})")
            if r.reason:
                print(f"  Reason: {r.reason}")
        if non_compliant and args.strict:
            return 1
        return 0

    if check_sparse_contamination(args.root):
        print(
            f"::error::Workspace at {args.root} has leaked or empty sparse-checkout configuration"
        )
        return 1

    print(f"Workspace at {args.root} is free of sparse-checkout contamination.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
