"""Anti-phantom-merge never uses PR-head code and re-evaluates on every event.

The workflow runs on ``pull_request`` and, for label events,
``pull_request_target`` (base-repo credentials). It must therefore check out
trusted base code only and read the PR's files and commits from the REST API,
and the checks step must not be limited to one event (RM#1989).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github/workflows/anti-phantom-merge.yml"
)
HEAD_REF = re.compile(
    r"pull_request\.head\.(sha|ref)|github\.head_ref|refs/pull/|gh\s+pr\s+checkout"
)


def _steps() -> list[dict[str, Any]]:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return data["jobs"]["guard"]["steps"]


def _checks_step() -> dict[str, Any]:
    return next(s for s in _steps() if s.get("name") == "Run phantom guard checks")


def test_no_step_checks_out_or_fetches_pr_head() -> None:
    offenders = []
    for step in _steps():
        text = str(step.get("with", "")) + str(step.get("run", ""))
        if HEAD_REF.search(text):
            offenders.append(step.get("name"))
    assert offenders == []


def test_checkout_uses_base_sha_only() -> None:
    checkouts = [s for s in _steps() if "actions/checkout" in s.get("uses", "")]
    assert checkouts, "trusted base checkout is required for the Rule 3 helper"
    for step in checkouts:
        assert step["with"]["ref"] == "${{ github.event.pull_request.base.sha }}"
        assert "if" not in step, "checkout must be unconditional"


def test_checks_step_runs_on_every_event() -> None:
    condition = str(_checks_step().get("if", ""))
    assert "event_name" not in condition
    assert "override" in condition


def test_changed_files_and_commits_come_from_the_rest_api() -> None:
    run = _checks_step()["run"]
    assert 'pulls/$PR_NUMBER/files" --paginate' in run
    assert 'pulls/$PR_NUMBER/commits" --paginate' in run
    assert "git diff" not in run
    assert "git log" not in run
    assert "git merge-base" not in run


def test_run_blocks_do_not_interpolate_untrusted_expressions() -> None:
    for step in _steps():
        assert "${{" not in str(step.get("run", "")), step.get("name")


def test_concurrency_group_separates_event_types() -> None:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert "github.event_name" in data["concurrency"]["group"]
