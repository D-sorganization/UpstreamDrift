from __future__ import annotations

from pathlib import Path

from scripts import check_local_only_workflows as guard


def _write_workflow(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_guard_ignores_hosted_runner_mentions_outside_runs_on(tmp_path: Path) -> None:
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    _write_workflow(
        workflow,
        """
name: CI
jobs:
  quality:
    runs-on: d-sorg-fleet-docker
    steps:
      - run: echo "ubuntu-latest appears only in a diagnostic string"
""",
    )

    assert guard._workflow_failures(workflow) == []


def test_guard_rejects_hosted_runs_on(tmp_path: Path) -> None:
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    _write_workflow(
        workflow,
        """
name: CI
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - run: echo "real hosted routing"
""",
    )

    failures = guard._workflow_failures(workflow)
    assert failures == [
        f"{workflow}:quality: runs-on 'ubuntu-latest' is a hosted runner"
    ]
