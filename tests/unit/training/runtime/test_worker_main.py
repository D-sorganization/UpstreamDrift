"""Tests for :mod:`training.runtime.worker_main`.

These tests verify the wire protocol in isolation: they spawn the
worker via ``subprocess.run`` (no :class:`SubprocessDriver` in the
loop) and assert the parent observes well-formed framed JSON events.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from training import (
    TrainingConfig,
    TrainingFramework,
    training_config_to_dict,
)
from training.runtime.wire_protocol import (
    COMMAND_RUN,
    EVENT_METRIC,
    EVENT_RESULT,
    EVENT_STATUS,
    encode_command,
)

pytestmark = pytest.mark.unit


_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC_PY = _REPO_ROOT / "src" / "shared" / "python"
_FIXTURE_DIR = Path(__file__).resolve().parent


def _spawn_env() -> dict[str, str]:
    """Return an env dict that lets the worker import training + fixture."""

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [str(_SRC_PY), str(_FIXTURE_DIR)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _run_worker(
    input_text: str, timeout: float = 10.0
) -> subprocess.CompletedProcess[str]:
    """Spawn the worker module with ``input_text`` on stdin."""

    return subprocess.run(  # noqa: S603 - test-only spawn of our own module
        [sys.executable, "-m", "training.runtime.worker_main"],
        input=input_text,
        capture_output=True,
        text=True,
        env=_spawn_env(),
        timeout=timeout,
        check=False,
    )


def _parse_stdout(stdout: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in stdout.splitlines() if line.strip()]


def _config_with_entry(
    entry_point: str,
    *,
    hyperparameters: dict[str, object] | None = None,
) -> dict[str, object]:
    cfg = TrainingConfig(
        framework=TrainingFramework.PYTORCH,
        entry_point=entry_point,
        output_dir=Path("/tmp/worker_main_test"),
        hyperparameters=hyperparameters or {},
    )
    return training_config_to_dict(cfg)


class TestWorkerHappyPath:
    def test_emits_status_metrics_and_result(self) -> None:
        config = _config_with_entry(
            "_fixture_runner:emit_metrics",
            hyperparameters={"num_metrics": 4},
        )
        input_line = encode_command(COMMAND_RUN, {"config": config})
        completed = _run_worker(input_line)
        assert completed.returncode == 0, completed.stderr

        events = _parse_stdout(completed.stdout)
        assert events[0]["event"] == EVENT_STATUS
        assert events[0]["status"] == "running"

        metric_events = [e for e in events if e["event"] == EVENT_METRIC]
        assert len(metric_events) == 4
        # Steps arrive in order.
        steps = [e["metric"]["step"] for e in metric_events]
        assert steps == [0, 1, 2, 3]

        # Exactly one result event, last.
        result_events = [e for e in events if e["event"] == EVENT_RESULT]
        assert len(result_events) == 1
        assert events[-1]["event"] == EVENT_RESULT
        assert result_events[0]["result"]["status"] == "completed"


class TestWorkerErrorPaths:
    def test_entry_point_exception_becomes_failed_result(self) -> None:
        config = _config_with_entry("_fixture_runner:raise_immediately")
        input_line = encode_command(COMMAND_RUN, {"config": config})
        completed = _run_worker(input_line)
        # The worker reports the failure cleanly — exit code is still 0
        # because the result event WAS delivered.
        assert completed.returncode == 0
        events = _parse_stdout(completed.stdout)
        result = next(e for e in events if e["event"] == EVENT_RESULT)
        assert result["result"]["status"] == "failed"
        assert "simulated training failure" in result["result"]["error"]

    def test_missing_entry_point_module_becomes_failed_result(self) -> None:
        config = _config_with_entry("not_a_real_module_xyz:fn")
        input_line = encode_command(COMMAND_RUN, {"config": config})
        completed = _run_worker(input_line)
        assert completed.returncode == 0
        events = _parse_stdout(completed.stdout)
        result = next(e for e in events if e["event"] == EVENT_RESULT)
        assert result["result"]["status"] == "failed"
        assert "cannot resolve entry_point" in result["result"]["error"]

    def test_malformed_first_command_becomes_failed_result(self) -> None:
        completed = _run_worker("this is not json\n")
        assert completed.returncode == 1
        events = _parse_stdout(completed.stdout)
        result = next(e for e in events if e["event"] == EVENT_RESULT)
        assert result["result"]["status"] == "failed"
        assert "malformed first command" in result["result"]["error"]

    def test_wrong_first_command_becomes_failed_result(self) -> None:
        # Sending CANCEL first is a protocol error.
        completed = _run_worker(encode_command("cancel"))
        assert completed.returncode == 1
        events = _parse_stdout(completed.stdout)
        result = next(e for e in events if e["event"] == EVENT_RESULT)
        assert result["result"]["status"] == "failed"
        assert "first command must be" in result["result"]["error"]

    def test_empty_stdin_becomes_failed_result(self) -> None:
        completed = _run_worker("")
        assert completed.returncode == 1
        events = _parse_stdout(completed.stdout)
        result = next(e for e in events if e["event"] == EVENT_RESULT)
        assert result["result"]["status"] == "failed"
        assert "stdin closed" in result["result"]["error"]

    def test_run_command_missing_config_becomes_failed_result(self) -> None:
        completed = _run_worker(encode_command(COMMAND_RUN))
        assert completed.returncode == 1
        events = _parse_stdout(completed.stdout)
        result = next(e for e in events if e["event"] == EVENT_RESULT)
        assert result["result"]["status"] == "failed"
        assert "missing 'config'" in result["result"]["error"]

    def test_stderr_does_not_pollute_event_stream(self) -> None:
        config = _config_with_entry(
            "_fixture_runner:emit_metrics_with_stderr",
            hyperparameters={"num_metrics": 1},
        )
        input_line = encode_command(COMMAND_RUN, {"config": config})
        completed = _run_worker(input_line)
        assert completed.returncode == 0
        # stderr has the noise; stdout has clean JSON only.
        assert "noise on stderr" in completed.stderr
        events = _parse_stdout(completed.stdout)
        assert all("event" in e for e in events)
