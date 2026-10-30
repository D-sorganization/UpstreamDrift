"""Unit tests for headless simulation JSON batch CLI (Epic #14085 / #14090).

Verifies that:
- Falsifying test: CLI and programmatic API invocations produce equivalent run records
  for the identical fixture case.
- CLI properly resolves case JSON via positional file, --case option, and stdin (-).
- Nonzero exit codes are returned on failure (1) and invalid inputs (2).
- Output format can be JSON or human-readable summary.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.sim_run.cli import execute_cli_run, main
from src.sim_run.models import RunRecord, RunRequest
from src.sim_run.store import SimulationRunStore
from src.sim_run.worker import SimulationWorkerPool

pytestmark = pytest.mark.unit


@pytest.fixture
def fixture_case_dict() -> dict[str, Any]:
    return {
        "engine_type": "golf_swing_pendulum",
        "duration": 1.0,
        "timestep": 0.01,
        "initial_state": {"q": [0.0, 0.0]},
        "timeout_seconds": 15.0,
        "tags": {"batch": "ci_test"},
    }


def test_falsifying_cli_and_api_produce_equivalent_run_records(
    tmp_path: Path,
    fixture_case_dict: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Falsifying test: CLI and API produce equivalent run records for the same fixture case."""
    api_results_root = tmp_path / "api_results"
    api_store = SimulationRunStore(results_root=api_results_root)
    api_request = RunRequest.model_validate(fixture_case_dict)
    api_initial = api_store.create_run(request=api_request, run_id="run_falsify_api")
    api_pool = SimulationWorkerPool(store=api_store)
    api_completed = api_pool.execute_run(api_initial.run_id)

    cli_results_root = tmp_path / "cli_results"
    case_file = tmp_path / "case.json"
    case_file.write_text(json.dumps(fixture_case_dict), encoding="utf-8")

    cli_code = main(
        [
            "run",
            str(case_file),
            "--results-root",
            str(cli_results_root),
            "--run-id",
            "run_falsify_cli",
        ]
    )
    captured = capsys.readouterr()
    assert cli_code == 0

    cli_store = SimulationRunStore(results_root=cli_results_root)
    cli_completed = cli_store.get_run("run_falsify_cli")

    assert cli_completed.status == api_completed.status == "succeeded"
    assert cli_completed.request.duration == api_completed.request.duration
    assert cli_completed.request.timestep == api_completed.request.timestep
    assert cli_completed.request.engine_type == api_completed.request.engine_type
    assert cli_completed.request.tags == api_completed.request.tags

    assert cli_completed.result is not None and api_completed.result is not None
    assert (
        cli_completed.result.validation_status
        == api_completed.result.validation_status
        == "valid"
    )
    assert cli_completed.result.success == api_completed.result.success is True
    assert cli_completed.result.data == api_completed.result.data

    parsed_stdout = json.loads(captured.out)
    assert parsed_stdout["run_id"] == "run_falsify_cli"
    assert parsed_stdout["status"] == "succeeded"


def test_cli_named_case_option_and_summary_format(
    tmp_path: Path,
    fixture_case_dict: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_root = tmp_path / "summary_results"
    case_file = tmp_path / "case_opt.json"
    case_file.write_text(json.dumps(fixture_case_dict), encoding="utf-8")

    code = main(
        [
            "run",
            "--case",
            str(case_file),
            "--results-root",
            str(results_root),
            "--run-id",
            "run_summary_01",
            "--format",
            "summary",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "Run ID: run_summary_01" in captured.out
    assert "Status: succeeded" in captured.out


def test_cli_stdin_support(
    tmp_path: Path,
    fixture_case_dict: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_root = tmp_path / "stdin_results"
    stdin_data = io.StringIO(json.dumps(fixture_case_dict))
    monkeypatch.setattr("sys.stdin", stdin_data)

    code = main(
        [
            "run",
            "-",
            "--results-root",
            str(results_root),
            "--run-id",
            "run_stdin_01",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    parsed = json.loads(captured.out)
    assert parsed["run_id"] == "run_stdin_01"
    assert parsed["status"] == "succeeded"


def test_cli_invalid_case_json_exit_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")

    code = main(["run", str(bad_file)])
    captured = capsys.readouterr()
    assert code == 2
    assert "ud-sim run error" in captured.err


def test_cli_missing_file_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["run", "non_existent_file_12345.json"])
    captured = capsys.readouterr()
    assert code == 2
    assert "ud-sim run error" in captured.err


def test_cli_no_args_shows_help(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([])
    captured = capsys.readouterr()
    assert code == 2
    assert "usage:" in captured.err


def test_cli_failed_run_exit_1(
    tmp_path: Path,
    fixture_case_dict: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    case_file = tmp_path / "case.json"
    case_file.write_text(json.dumps(fixture_case_dict), encoding="utf-8")

    def _mock_execute(*args: Any, **kwargs: Any) -> RunRecord:
        store = SimulationRunStore(results_root=tmp_path / "failed_results")
        req = RunRequest.model_validate(fixture_case_dict)
        rec = store.create_run(req, run_id="run_mock_failed")
        return store.transition(rec.run_id, "failed")

    with patch("src.sim_run.cli.execute_cli_run", side_effect=_mock_execute):
        code = main(
            [
                "run",
                str(case_file),
                "--results-root",
                str(tmp_path / "failed_results"),
            ]
        )
    assert code == 1
