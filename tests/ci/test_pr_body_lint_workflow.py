"""Contracts for the report-only PR body lint reusable caller (RM #2427)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pr-body-lint.yml"
PINNED_SHA = "3b5e9f54febbd730bc2a650dc9193d26922ff49f"  # pragma: allowlist secret


def test_pr_body_lint_is_a_report_only_pull_request_target_caller() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    on = workflow.get("on", workflow.get(True, {}))
    assert isinstance(on, dict)
    target = on["pull_request_target"]
    assert set(target["types"]) == {"opened", "edited"}
    assert workflow["permissions"] == {
        "contents": "read",
        "issues": "read",
        "pull-requests": "read",
    }
    job = workflow["jobs"]["lint"]
    assert "runs-on" not in job
    uses = str(job["uses"])
    assert PINNED_SHA in uses
    assert job["with"]["report_only"] == "true"
    assert "checkout" not in WORKFLOW.read_text(encoding="utf-8").lower()
