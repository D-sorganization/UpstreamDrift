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


def test_guard_rejects_hosted_matrix_runs_on(tmp_path: Path) -> None:
    workflow = tmp_path / ".github" / "workflows" / "matrix.yml"
    _write_workflow(
        workflow,
        """
name: Matrix
jobs:
  quality:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
    steps: []
""",
    )

    failures = guard._workflow_failures(workflow)
    assert len(failures) == 2
    assert "ubuntu-latest" in failures[0]
    assert "windows-latest" in failures[1]


def test_guard_allows_fleet_and_self_hosted_runners(tmp_path: Path) -> None:
    workflow = tmp_path / ".github" / "workflows" / "fleet.yml"
    _write_workflow(
        workflow,
        """
name: Fleet
jobs:
  direct:
    runs-on: d-sorg-fleet-docker
    steps: []
  picked:
    runs-on: ${{ needs.pick-runner.outputs.runner }}
    steps: []
  self_hosted:
    runs-on: self-hosted
    steps: []
""",
    )

    assert guard._workflow_failures(workflow) == []


def test_guard_allows_canary_workflow_file(tmp_path: Path) -> None:
    workflow = tmp_path / ".github" / "workflows" / "local-only-runner-guard.yml"
    _write_workflow(
        workflow,
        """
name: Local-Only Workflow Runner Guard
jobs:
  guard:
    runs-on: ubuntu-latest
    steps: []
""",
    )

    assert guard._workflow_failures(workflow) == []


def test_guard_allows_canary_job_name(tmp_path: Path) -> None:
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    _write_workflow(
        workflow,
        """
name: CI
jobs:
  guard:
    name: Reject hosted runner routing
    runs-on: ubuntu-latest
    steps: []
""",
    )

    assert guard._workflow_failures(workflow) == []
