"""Focused unit tests for the GS3DX MATLAB batch-runner decision contract.

Verifies Design-by-Contract (DbC), Law of Demeter (LoD), and truthful exit
determination across natural exits, watchdog forced kills, script-level failures,
and done-marker sanitization.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import pytest

pytestmark = pytest.mark.unit

_DECISION_PATH = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "matlab"
    / "exploratory_gs3dx"
    / "tools"
    / "runner_decision.py"
)

if not _DECISION_PATH.is_file():
    runner_decision = None
else:
    spec = importlib.util.spec_from_file_location(
        "runner_decision", str(_DECISION_PATH)
    )
    assert spec is not None and spec.loader is not None
    runner_decision = importlib.util.module_from_spec(spec)
    sys.modules["runner_decision"] = runner_decision
    spec.loader.exec_module(runner_decision)


def _get_runner_module():
    if runner_decision is None:
        pytest.fail(f"runner_decision module not found at {_DECISION_PATH}")
    return runner_decision


def test_natural_zero_success() -> None:
    """A natural 0 exit with exact done marker and status success yields exit 0."""
    rd = _get_runner_module()
    log_content = "Running simulation...\nSTATUS success\nGS3DX_BATCH_DONE\n"
    outcome = rd.decide_run_outcome(
        actual_process_exit_code=0,
        termination_reason="natural_exit",
        log_content=log_content,
        stderr_content="",
        command_identity="run('probe.m')",
        owned_pids=[1001],
    )
    assert outcome.final_exit_code == 0
    assert outcome.done_marker is True
    assert outcome.script_status == "success"
    assert outcome.actual_process_exit_code == 0
    assert outcome.termination_reason == "natural_exit"

    receipt = outcome.to_receipt()
    assert receipt["actual_process_exit_code"] == 0
    assert receipt["final_exit_code"] == 0
    assert receipt["done_marker"] is True
    assert receipt["script_status"] == "success"
    assert receipt["owned_pids"] == [1001]
    assert receipt["command_identity"] == "run('probe.m')"


def test_natural_zero_without_explicit_status_legacy_pass() -> None:
    """Absent status supports legacy natural exit when process exits 0 with done marker."""
    rd = _get_runner_module()
    log_content = "Simulation finished cleanly.\nGS3DX_BATCH_DONE\n"
    outcome = rd.decide_run_outcome(
        actual_process_exit_code=0,
        termination_reason="natural_exit",
        log_content=log_content,
        stderr_content="",
        command_identity="run('clean.m')",
        owned_pids=[1002],
    )
    assert outcome.final_exit_code == 0
    assert outcome.done_marker is True
    assert outcome.script_status is None
    assert outcome.actual_process_exit_code == 0


def test_natural_nonzero_exit() -> None:
    """A process exiting with natural code 42 preserves that truthful code."""
    rd = _get_runner_module()
    log_content = "Simulink error: model failed to converge.\n"
    outcome = rd.decide_run_outcome(
        actual_process_exit_code=42,
        termination_reason="natural_exit",
        log_content=log_content,
        stderr_content="Error: solver aborted",
        command_identity="run('fail.m')",
        owned_pids=[1003],
    )
    assert outcome.final_exit_code == 42
    assert outcome.actual_process_exit_code == 42
    assert outcome.done_marker is False


def test_windows_negative_large_exit_code_preserved_and_mapped_sanely() -> None:
    """Negative/large Windows exit codes (e.g. 0xC0000005 access violation) preserve receipt code."""
    rd = _get_runner_module()
    # STATUS_ACCESS_VIOLATION as signed 32-bit int: -1073741819 (0xC0000005)
    access_violation_code = -1073741819
    outcome = rd.decide_run_outcome(
        actual_process_exit_code=access_violation_code,
        termination_reason="natural_exit",
        log_content="Access violation detected\n",
        stderr_content="Faulting module: simscape.dll",
        command_identity="run('crash.m')",
        owned_pids=[1010],
    )
    # Receipt must preserve the actual uncorrupted integer
    assert outcome.actual_process_exit_code == access_violation_code
    assert outcome.to_receipt()["actual_process_exit_code"] == access_violation_code
    # Runner exit code must be mapped sanely to 1..255 (non-zero) without throwing postcondition error
    assert 1 <= outcome.final_exit_code <= 255


def test_missing_or_null_exit_fails_closed() -> None:
    """A natural exit with null/non-integer exit code fails closed with non-zero."""
    rd = _get_runner_module()
    log_content = "Process vanished mysteriously.\n"
    outcome = rd.decide_run_outcome(
        actual_process_exit_code=None,
        termination_reason="unverified_exit",
        log_content=log_content,
        stderr_content="",
        command_identity="run('vanish.m')",
        owned_pids=[1004],
    )
    assert outcome.final_exit_code != 0
    assert outcome.final_exit_code == rd.EXIT_CODE_SHUTDOWN_UNVERIFIED
    assert outcome.actual_process_exit_code is None
    assert outcome.to_receipt()["actual_process_exit_code"] is None


def test_marker_then_status_failure_rejects_zero() -> None:
    """CRITICAL: GS3DX_BATCH_DONE present but STATUS failed must NOT exit 0."""
    rd = _get_runner_module()
    log_content = (
        "ROW 0 0 0 0 0\n"
        "ERROR Ankle boundary exceeded\n"
        "STATUS failed\n"
        "GS3DX_BATCH_DONE\n"
    )
    outcome = rd.decide_run_outcome(
        actual_process_exit_code=0,
        termination_reason="natural_exit",
        log_content=log_content,
        stderr_content="",
        command_identity="run('shifted_leg.m')",
        owned_pids=[1005],
    )
    assert outcome.final_exit_code != 0
    assert outcome.final_exit_code == rd.EXIT_CODE_SCRIPT_FAILED
    assert outcome.actual_process_exit_code == 0
    assert outcome.done_marker is True
    assert outcome.script_status == "failed"


@pytest.mark.parametrize("status_val", ["pending", "unknown", "aborted", "partial"])
def test_unknown_explicit_status_fails_closed(status_val: str) -> None:
    """Unknown explicit STATUS values fail closed with exit code 1."""
    rd = _get_runner_module()
    log_content = f"Computation paused.\nSTATUS {status_val}\nGS3DX_BATCH_DONE\n"
    outcome = rd.decide_run_outcome(
        actual_process_exit_code=0,
        termination_reason="natural_exit",
        log_content=log_content,
        stderr_content="",
        command_identity="run('unknown_status.m')",
        owned_pids=[1011],
    )
    assert outcome.final_exit_code == rd.EXIT_CODE_SCRIPT_FAILED
    assert outcome.script_status == status_val


def test_marker_then_hung_exit_returns_125_never_synthesizes_zero() -> None:
    """CRITICAL: Watchdog-killed process post-marker must yield distinct 125, never 0."""
    rd = _get_runner_module()
    log_content = "STATUS success\nGS3DX_BATCH_DONE\n"
    outcome = rd.decide_run_outcome(
        actual_process_exit_code=None,
        termination_reason="hung_after_done",
        log_content=log_content,
        stderr_content="",
        command_identity="run('probe.m')",
        owned_pids=[1006],
    )
    assert outcome.final_exit_code == rd.EXIT_CODE_SHUTDOWN_UNVERIFIED
    assert outcome.final_exit_code == 125
    assert outcome.actual_process_exit_code is None
    assert outcome.termination_reason == "hung_after_done"
    assert outcome.done_marker is True
    assert outcome.to_receipt()["actual_process_exit_code"] is None


def test_timeout_returns_124() -> None:
    """Overall watchdog timeout must yield 124."""
    rd = _get_runner_module()
    log_content = "Simulating frame 5...\n"
    outcome = rd.decide_run_outcome(
        actual_process_exit_code=None,
        termination_reason="timeout",
        log_content=log_content,
        stderr_content="",
        command_identity="run('infinite_loop.m')",
        owned_pids=[1007],
    )
    assert outcome.final_exit_code == rd.EXIT_CODE_TIMEOUT
    assert outcome.final_exit_code == 124
    assert outcome.actual_process_exit_code is None
    assert outcome.termination_reason == "timeout"
    assert outcome.done_marker is False


@pytest.mark.parametrize(
    "text,expected",
    [
        ("GS3DX_BATCH_DONE\n", True),
        ("  GS3DX_BATCH_DONE  \r\n", True),
        ("% fprintf('GS3DX_BATCH_DONE\\n');\n", False),
        ("echo GS3DX_BATCH_DONE\n", False),
        ("GS3DX_BATCH_DONE_APPENDED\n", False),
        ("PREFIX_GS3DX_BATCH_DONE\n", False),
        ("Starting GS3DX_BATCH_DONE now\n", False),
        ("", False),
    ],
)
def test_done_marker_sanitization(text: str, expected: bool) -> None:
    """Exact complete line requirement prevents comments/echoes from falsely triggering."""
    rd = _get_runner_module()
    assert rd.has_exact_done_marker(text) == expected


@pytest.mark.parametrize(
    "text,expected_status",
    [
        ("STATUS success\n", "success"),
        ("STATUS failed\n", "failed"),
        ("STATUS error\n", "error"),
        ("STATUS   passed  \r\n", "passed"),
        ("No status here\n", None),
        ("STATUS_REPORT: all good\n", None),
    ],
)
def test_parse_script_status(text: str, expected_status: str | None) -> None:
    """Extracts explicit STATUS markers accurately."""
    rd = _get_runner_module()
    assert rd.parse_script_status(text) == expected_status


def test_stderr_warning_does_not_fail_successful_run() -> None:
    """MATLAB stderr warnings (fonts, Java, license) do not override success."""
    rd = _get_runner_module()
    log_content = "STATUS success\nGS3DX_BATCH_DONE\n"
    stderr = "Warning: Unable to load Java font rendering subsystem.\n"
    outcome = rd.decide_run_outcome(
        actual_process_exit_code=0,
        termination_reason="natural_exit",
        log_content=log_content,
        stderr_content=stderr,
        command_identity="run('probe.m')",
        owned_pids=[1008],
    )
    assert outcome.final_exit_code == 0
    assert outcome.script_status == "success"


def test_dbc_rejects_booleans_as_integers() -> None:
    """DbC preconditions strictly reject bool values masquerading as integers."""
    rd = _get_runner_module()
    with pytest.raises(rd.PreconditionError):
        rd.decide_run_outcome(
            actual_process_exit_code=True,  # type: ignore[arg-type]
            termination_reason="natural_exit",
            log_content="",
            stderr_content="",
            command_identity="run('test.m')",
            owned_pids=[1001],
        )

    with pytest.raises(rd.PreconditionError):
        rd.decide_run_outcome(
            actual_process_exit_code=False,  # type: ignore[arg-type]
            termination_reason="natural_exit",
            log_content="",
            stderr_content="",
            command_identity="run('test.m')",
            owned_pids=[1001],
        )

    with pytest.raises(rd.PreconditionError):
        rd.decide_run_outcome(
            actual_process_exit_code=0,
            termination_reason="natural_exit",
            log_content="",
            stderr_content="",
            command_identity="run('test.m')",
            owned_pids=[True],  # type: ignore[list-item]
        )


def test_dbc_preconditions() -> None:
    """DbC preconditions reject invalid inputs fail-closed."""
    rd = _get_runner_module()
    with pytest.raises(rd.PreconditionError):
        # Empty owned_pids
        rd.decide_run_outcome(
            actual_process_exit_code=0,
            termination_reason="natural_exit",
            log_content="",
            stderr_content="",
            command_identity="run('test.m')",
            owned_pids=[],
        )

    with pytest.raises(rd.PreconditionError):
        # Empty command_identity
        rd.decide_run_outcome(
            actual_process_exit_code=0,
            termination_reason="natural_exit",
            log_content="",
            stderr_content="",
            command_identity="",
            owned_pids=[1001],
        )

    with pytest.raises(rd.PreconditionError):
        # Invalid termination reason
        rd.decide_run_outcome(
            actual_process_exit_code=0,
            termination_reason="invalid_reason",  # type: ignore[arg-type]
            log_content="",
            stderr_content="",
            command_identity="run('test.m')",
            owned_pids=[1001],
        )
