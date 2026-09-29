"""Tests for workflow_run trust-boundary guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_workflow_run_trust_boundary import (
    default_workflows,
    find_violations,
)

pytestmark = pytest.mark.unit


def test_rejects_workflow_run_checkout_and_install(tmp_path: Path) -> None:
    workflow = tmp_path / "autofix.yml"
    workflow.write_text(
        """
name: unsafe
on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]
jobs:
  autofix:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10
      - run: |
          pip install -e ".[dev]"
          ruff check --fix src
          git push origin "$BRANCH"
""",
        encoding="utf-8",
    )

    violations = find_violations(workflow)

    assert [violation.pattern for violation in violations] == [
        "actions/checkout",
        "pip install",
        "ruff check --fix",
        "git push",
    ]


def test_allows_dispatch_only_autofix_job(tmp_path: Path) -> None:
    workflow = tmp_path / "autofix.yml"
    workflow.write_text(
        """
name: safe
on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]
  workflow_dispatch:
jobs:
  metadata:
    if: github.event_name == 'workflow_run'
    runs-on: ubuntu-latest
    steps:
      - run: gh pr list --json number
  autofix:
    if: github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10
      - run: |
          pip install ruff
          ruff check --fix src
          git push origin "$BRANCH"
""",
        encoding="utf-8",
    )

    assert find_violations(workflow) == []


def test_current_jules_workflow_preserves_workflow_run_boundary() -> None:
    """No repository workflow lets a ``workflow_run`` job execute PR code.

    The guard used to scan a single hardcoded ``Jules-PR-AutoFix.yml``; that
    workflow was retired with the rest of the unowned Jules suite (#1483), so
    the contract is now asserted over every workflow in the repository. The
    test name is unchanged because the unit-gate quarantine ledger is
    removal-only and keys on it.
    """
    violations = [
        violation
        for workflow in default_workflows()
        for violation in find_violations(workflow)
    ]

    assert violations == []
