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
ROLLING_IMPORT_REQUIREMENTS = (
    '"defusedxml>=0.7,<1"',
    '"pydantic>=2.12,<3"',
    '"PyYAML>=6.0,<7"',
)


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


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    step = next(step for step in job["steps"] if step.get("name") == name)
    assert isinstance(step, dict)
    return step


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
    assert "articulated_manufactured_solution.json" in commands


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


def test_rolling_runtime_closes_governed_test_import_dependencies() -> None:
    """The rolling lane must collect the same governed native test modules."""

    commands = _run_text(_job(ROLLING_JOB))

    for requirement in ROLLING_IMPORT_REQUIREMENTS:
        assert requirement in commands


def test_authority_job_uploads_deterministic_candidate_on_publication_red() -> None:
    """A committed-byte mismatch must fail while preserving reviewable bytes."""

    job = _job(AUTHORITY_JOB)
    steps = job["steps"]
    generate_name = "Generate and Validate Deterministic Authority Candidates"
    compare_name = "Compare Candidate With Committed Publication Authority"
    upload_name = "Upload Independently Reviewable Authority Candidate"
    generate = _step(job, generate_name)
    compare = _step(job, compare_name)
    upload = _step(job, upload_name)
    generate_commands = str(generate["run"])
    compare_commands = str(compare["run"])
    upload_with = upload["with"]

    assert steps.index(generate) < steps.index(compare) < steps.index(upload)
    assert generate_commands.count("--profile authority") == 2
    assert generate_commands.count("--validate-generated") == 2
    assert "articulated-authority-first.json" in generate_commands
    assert "articulated-authority-second.json" in generate_commands
    assert "cmp --silent" in generate_commands
    assert "sha256sum" in generate_commands
    assert "articulated-authority-first.json" in compare_commands
    assert "articulated_manufactured_solution.json" in compare_commands
    assert "exit 1" in compare_commands
    assert str(upload["uses"]).startswith("actions/upload-artifact@")
    assert "always()" in str(upload["if"])
    assert "github.event.pull_request.head.sha" in str(upload_with["name"])
    assert str(upload_with["path"]).endswith("articulated-authority-first.json")
    assert upload_with["if-no-files-found"] == "error"
    assert upload_with.get("overwrite") is False
    assert "git commit" not in _run_text(job)
    assert "git push" not in _run_text(job)


def test_authority_and_rolling_jobs_run_the_new_contract_suite() -> None:
    """Both lanes must exercise profile separation and semantic comparison."""

    contract = "test_articulated_manufactured_hybrid_authority_red.py"
    authority_commands = _run_text(_job(AUTHORITY_JOB))
    rolling_commands = _run_text(_job(ROLLING_JOB))

    assert contract in authority_commands
    assert contract in rolling_commands
    assert "same_environment_two_process" in authority_commands
    assert "rolling_native_output" in rolling_commands
