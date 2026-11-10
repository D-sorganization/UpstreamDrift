"""Meta-tests for CI coverage truth-in-summary contract (issue #14946).

Slice 1 requirements:
- No '--cov-fail-under' flags remain in .github/workflows/ci-standard.yml.
- The dead targeted-coverage branch in 'core-tests' is eliminated.
- The CI quality-gate aggregator explicitly reports 'coverage: not measured'
  to stdout and $GITHUB_STEP_SUMMARY.
- The 'core-tests' step explicitly reports 'coverage: not measured' to
  stdout and $GITHUB_STEP_SUMMARY when coverage is not measured (--no-cov).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CI_STANDARD_PATH = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"


def _load_ci_standard() -> dict[str, Any]:
    """Parse .github/workflows/ci-standard.yml into a Python dictionary."""
    assert CI_STANDARD_PATH.is_file(), f"Missing workflow file: {CI_STANDARD_PATH}"
    content = yaml.safe_load(CI_STANDARD_PATH.read_text(encoding="utf-8"))
    assert isinstance(content, dict), "ci-standard.yml did not parse as a dictionary"
    return content


def _find_job_step(
    workflow: dict[str, Any],
    job_id: str,
    step_id: str | None = None,
    step_name: str | None = None,
) -> dict[str, Any]:
    """Locate a step in a specified job by step_id or step_name."""
    jobs = workflow.get("jobs", {})
    assert job_id in jobs, f"Job '{job_id}' not found in workflow"
    job = jobs[job_id]
    steps = job.get("steps", [])
    for step in steps:
        if step_id and step.get("id") == step_id:
            return step
        if step_name and step.get("name") == step_name:
            return step
    raise AssertionError(
        f"Step with id='{step_id}', name='{step_name}' not found in job '{job_id}'"
    )


def test_no_cov_fail_under_flag_in_ci_standard() -> None:
    """Ensure no unreachable or dead '--cov-fail-under' flag remains in ci-standard.yml."""
    raw_text = CI_STANDARD_PATH.read_text(encoding="utf-8")
    assert "--cov-fail-under" not in raw_text, (
        "Found '--cov-fail-under' in .github/workflows/ci-standard.yml; "
        "Slice 1 requires removing unreachable coverage floor flags."
    )


def test_no_dead_targeted_coverage_branch_in_core_tests() -> None:
    """Ensure the dead targeted coverage arm in core-tests is removed."""
    workflow = _load_ci_standard()
    step = _find_job_step(workflow, "tests", step_id="core-tests")
    script = step.get("run", "")
    assert "Running PR-scoped core tests with targeted coverage:" not in script, (
        "Found dead targeted coverage branch in 'core-tests' step script"
    )


def test_quality_gate_aggregator_reports_coverage_not_measured() -> None:
    """The quality-gate aggregator must explicitly report 'coverage: not measured'."""
    workflow = _load_ci_standard()
    step = _find_job_step(
        workflow, "quality-gate", step_name="Aggregate quality gate results"
    )
    script = step.get("run", "")

    # Must print to stdout and GITHUB_STEP_SUMMARY
    assert "coverage: not measured" in script, (
        "quality-gate aggregator must explicitly report 'coverage: not measured'"
    )
    assert (
        'echo "coverage: not measured"' in script or "coverage: not measured" in script
    ), "quality-gate aggregator must print 'coverage: not measured' to stdout"
    assert "GITHUB_STEP_SUMMARY" in script and "coverage: not measured" in script, (
        "quality-gate aggregator must write 'coverage: not measured' to $GITHUB_STEP_SUMMARY"
    )


def test_core_tests_reports_coverage_not_measured() -> None:
    """When core tests run without coverage (--no-cov), explicitly report 'coverage: not measured'."""
    workflow = _load_ci_standard()
    step = _find_job_step(workflow, "tests", step_id="core-tests")
    script = step.get("run", "")

    assert "coverage: not measured" in script, (
        "core-tests step must explicitly include 'coverage: not measured'"
    )
