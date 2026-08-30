"""RED workflow contracts for articulated authority and rolling-native CI (#9236)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.research.proximal_distal_energy import (
    run_articulated_manufactured_solution as runner,
)

yaml = pytest.importorskip("yaml")
pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github/workflows/ci-optional-stack.yml"
AUTHORITY_JOB = "articulated-manufactured-authority"
ROLLING_JOB = "articulated-manufactured-rolling"
AUTHORITY_LOCK = runner.AUTHORITY_LOCK.relative_to(ROOT).as_posix()


def _workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _job(name: str) -> dict[str, Any]:
    job = _workflow()["jobs"].get(name)
    assert isinstance(job, dict), f"missing distinct {name} job"
    return job


def _run_text(job: dict[str, Any]) -> str:
    commands = [step.get("run", "") for step in job["steps"]]
    return "\n".join(command for command in commands if isinstance(command, str))


def _setup_python_version(job: dict[str, Any]) -> str:
    setup = next(
        step
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/setup-python@")
    )
    return str(setup["with"]["python-version"])


def _assert_fail_closed(job: dict[str, Any]) -> None:
    assert job.get("continue-on-error") is not True
    assert all(step.get("continue-on-error") is not True for step in job["steps"])


def test_authority_job_is_distinct_hash_locked_python311_and_no_deps() -> None:
    """Committed-byte authority must run in one exact dependency environment."""

    job = _job(AUTHORITY_JOB)
    commands = _run_text(job)

    _assert_fail_closed(job)
    assert _setup_python_version(job) == runner.AUTHORITY_PYTHON_VERSION
    assert AUTHORITY_LOCK in commands
    assert "--require-hashes" in commands
    assert "--no-deps" in commands
    assert "--profile authority" in commands
    assert "test_articulated_manufactured_solution" in commands
    assert "killswitch" in commands
    assert "compare-committed" in commands


def test_rolling_job_is_non_vacuous_and_explicitly_non_authoritative() -> None:
    """Rolling native compatibility must execute and cannot publish evidence."""

    job = _job(ROLLING_JOB)
    commands = _run_text(job)

    _assert_fail_closed(job)
    assert ROLLING_JOB != AUTHORITY_JOB
    assert "pip" in commands and "pin" in commands and "mujoco" in commands
    assert "--profile rolling" in commands
    assert "non-authoritative" in commands.lower()
    assert "RUNNER_TEMP" in commands
    assert "test_articulated_manufactured_solution" in commands
    assert "killswitch" in commands
    assert "passed=" in commands
    assert 'if [[ "$passed" -eq 0 ]]' in commands
    assert "exit 1" in commands
    assert AUTHORITY_LOCK not in commands


def test_authority_and_rolling_jobs_run_the_new_contract_suite() -> None:
    """Both lanes must exercise profile separation and semantic comparison."""

    contract = "test_articulated_manufactured_hybrid_authority_red.py"
    authority_commands = _run_text(_job(AUTHORITY_JOB))
    rolling_commands = _run_text(_job(ROLLING_JOB))

    assert contract in authority_commands
    assert contract in rolling_commands
    assert "same_environment_two_process" in authority_commands
    assert "rolling_native_output" in rolling_commands
