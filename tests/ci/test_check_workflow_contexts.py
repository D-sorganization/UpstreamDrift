from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci.check_workflow_contexts import find_violations, main

pytestmark = pytest.mark.unit


def test_rejects_runner_context_in_job_level_env(tmp_path: Path) -> None:
    workflow = tmp_path / "bad.yml"
    workflow.write_text(
        """
name: bad
on: [push]
jobs:
  rust:
    runs-on: d-sorg-fleet-docker
    env:
      CARGO_HOME: ${{ runner.temp }}/cargo-home
    steps:
      - run: cargo test
""".lstrip(),
        encoding="utf-8",
    )

    violations = find_violations([workflow])

    assert len(violations) == 1
    assert violations[0].job_id == "rust"
    assert violations[0].env_name == "CARGO_HOME"


def test_allows_runner_context_in_step_env(tmp_path: Path) -> None:
    workflow = tmp_path / "good.yml"
    workflow.write_text(
        """
name: good
on: [push]
jobs:
  metrics:
    runs-on: d-sorg-fleet-docker
    steps:
      - name: Setup cache
        env:
          RUNNER_TOOL_CACHE: ${{ runner.temp }}/hostedtoolcache
        run: mkdir -p "$RUNNER_TOOL_CACHE"
""".lstrip(),
        encoding="utf-8",
    )

    assert find_violations([workflow]) == []


def test_cli_returns_nonzero_for_invalid_job_env(tmp_path: Path, capsys) -> None:
    workflow = tmp_path / "bad.yml"
    workflow.write_text(
        """
name: bad
on: [push]
jobs:
  wheel:
    runs-on: d-sorg-fleet-docker
    env:
      CARGO_HOME: ${{ runner.temp }}/cargo-home
    steps:
      - run: cargo build
""".lstrip(),
        encoding="utf-8",
    )

    assert main([str(workflow)]) == 1
    assert (
        "job env CARGO_HOME uses unavailable runner context" in capsys.readouterr().out
    )
