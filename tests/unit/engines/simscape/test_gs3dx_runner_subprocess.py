"""Subprocess integration tests for run_matlab_locked.ps1 with controlled fixtures.

Validates:
- Natural zero exit with STATUS success yields exit code 0.
- Defect reproduction: GS3DX_BATCH_DONE with STATUS failed rejects 0 and exits 1.
- Defect reproduction: Watchdog kill post-marker returns 125 (shutdown unverified), NEVER synthesized 0.
- Timeout watchdog returns 124.
- Natural non-zero exit preserves exit code.
- Null / absent status supports legacy natural exit.
- Paths with spaces handled cleanly across scripts, logs, and locks.
- Missing decision helper fails closed with 125.
- Comment/echo containing GS3DX_BATCH_DONE does not falsely trigger completion.
- Real runner blocked on live foreign lock preserves ownership without deleting it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        sys.platform != "win32" or shutil.which("powershell") is None,
        reason="Windows PowerShell is required for subprocess runner tests.",
    ),
]

_SCRIPT_ROOT = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "matlab"
    / "exploratory_gs3dx"
    / "tools"
)
_RUNNER_PS1 = _SCRIPT_ROOT / "run_matlab_locked.ps1"


@pytest.fixture
def fake_matlab(tmp_path: Path) -> Path:
    """Creates a fake matlab runner (.bat) that dispatches behavior by script name."""
    worker_py = tmp_path / "fake_worker.py"
    worker_py.write_text(
        r"""import sys, time
script_name = ""
for arg in sys.argv[1:]:
    if "run(" in arg:
        script_name = arg.split("run('")[1].split("')")[0]

if script_name.endswith("success.m"):
    sys.stdout.write("Running simulation...\nSTATUS success\nGS3DX_BATCH_DONE\n")
    sys.stdout.flush()
    sys.exit(0)

elif script_name.endswith("null_status.m"):
    sys.stdout.write("Simulation completed with absent status.\nGS3DX_BATCH_DONE\n")
    sys.stdout.flush()
    sys.exit(0)

elif script_name.endswith("shifted_leg_fail.m"):
    # Simulates the P1 bug: script logs STATUS failed, then GS3DX_BATCH_DONE, and exits 0
    sys.stdout.write("ROW 0 0 0 0 0\nERROR Ankle boundary\nSTATUS failed\nGS3DX_BATCH_DONE\n")
    sys.stdout.flush()
    sys.exit(0)

elif script_name.endswith("hung_exit.m"):
    # Prints GS3DX_BATCH_DONE then hangs
    sys.stdout.write("STATUS success\nGS3DX_BATCH_DONE\n")
    sys.stdout.flush()
    time.sleep(30)
    sys.exit(0)

elif script_name.endswith("timeout.m"):
    # Hangs without printing done
    sys.stdout.write("Starting...\n")
    sys.stdout.flush()
    time.sleep(30)
    sys.exit(0)

elif script_name.endswith("nonzero.m"):
    sys.stdout.write("Error occurred\n")
    sys.stdout.flush()
    sys.exit(42)

elif script_name.endswith("comment_marker.m"):
    # Prints comment with marker, then hangs
    sys.stdout.write("% fprintf('GS3DX_BATCH_DONE\\n');\n")
    sys.stdout.flush()
    time.sleep(30)
    sys.exit(0)

else:
    sys.stdout.write(f"Unknown script: {script_name}\n")
    sys.exit(1)
""",
        encoding="utf-8",
    )

    fake_bat = tmp_path / "fake_matlab.bat"
    py_exe = sys.executable
    fake_bat.write_text(f'@echo off\n"{py_exe}" "{worker_py}" %*\n', encoding="utf-8")
    return fake_bat


def _run_ps1(
    script_path: Path,
    log_path: Path,
    fake_matlab: Path,
    tmp_path: Path,
    timeout_min: int = 1,
    done_grace_sec: int = 1,
    lock_dir: Path | None = None,
    queue_dir: Path | None = None,
    runner_script: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    lock = lock_dir or (tmp_path / "test.lock")
    queue = queue_dir or (tmp_path / "test.queue")
    runner = runner_script or _RUNNER_PS1
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(runner),
        "-Script",
        str(script_path),
        "-Log",
        str(log_path),
        "-Matlab",
        str(fake_matlab),
        "-TimeoutMin",
        str(timeout_min),
        "-DoneGraceSec",
        str(done_grace_sec),
        "-LockDir",
        str(lock),
        "-QueueDir",
        str(queue),
        "-PollIntervalSec",
        "0.2",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


def test_subprocess_natural_success(fake_matlab: Path, tmp_path: Path) -> None:
    """Verifies that a successful script exits 0 and records receipt."""
    script = tmp_path / "success.m"
    script.write_text("% success", encoding="utf-8")
    log = tmp_path / "success.log"

    res = _run_ps1(script, log, fake_matlab, tmp_path)
    assert res.returncode == 0, (
        f"Expected 0, got {res.returncode}. Stderr: {res.stderr}"
    )
    assert log.is_file()
    content = log.read_text(encoding="utf-8")
    assert "EXIT 0" in content
    assert "RECEIPT: " in content

    receipt_file = tmp_path / "success.log.receipt.json"
    assert receipt_file.is_file()
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert receipt["actual_process_exit_code"] == 0
    assert receipt["final_exit_code"] == 0
    assert receipt["done_marker"] is True
    assert receipt["script_status"] == "success"


def test_subprocess_null_status_legacy_success(
    fake_matlab: Path, tmp_path: Path
) -> None:
    """Absent status line succeeds naturally when exit code is 0 and done marker present."""
    script = tmp_path / "null_status.m"
    script.write_text("% null status", encoding="utf-8")
    log = tmp_path / "null_status.log"

    res = _run_ps1(script, log, fake_matlab, tmp_path)
    assert res.returncode == 0, (
        f"Expected 0, got {res.returncode}. Stderr: {res.stderr}"
    )
    content = log.read_text(encoding="utf-8")
    assert "EXIT 0" in content

    receipt_file = tmp_path / "null_status.log.receipt.json"
    assert receipt_file.is_file()
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert receipt["actual_process_exit_code"] == 0
    assert receipt["final_exit_code"] == 0
    assert receipt["script_status"] is None


def test_subprocess_defect_shifted_leg_marker_then_status_failure(
    fake_matlab: Path, tmp_path: Path
) -> None:
    """CRITICAL: GS3DX_BATCH_DONE with STATUS failed must exit 1, NOT 0."""
    script = tmp_path / "shifted_leg_fail.m"
    script.write_text("% shifted leg fail", encoding="utf-8")
    log = tmp_path / "shifted_leg.log"

    res = _run_ps1(script, log, fake_matlab, tmp_path)
    assert res.returncode == 1, (
        f"Expected returncode 1, got {res.returncode}. stderr: {res.stderr}"
    )
    content = log.read_text(encoding="utf-8")
    assert "EXIT 1" in content
    assert "EXIT 0" not in content.splitlines()[-1]

    receipt_file = tmp_path / "shifted_leg.log.receipt.json"
    assert receipt_file.is_file()
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert receipt["final_exit_code"] == 1
    assert receipt["script_status"] == "failed"
    assert receipt["done_marker"] is True


def test_subprocess_defect_marker_then_hung_exit_returns_125_never_zero(
    fake_matlab: Path, tmp_path: Path
) -> None:
    """CRITICAL: Watchdog-killed exit post marker must return 125, NEVER synthesize 0."""
    script = tmp_path / "hung_exit.m"
    script.write_text("% hung", encoding="utf-8")
    log = tmp_path / "hung.log"

    t0 = time.perf_counter()
    res = _run_ps1(script, log, fake_matlab, tmp_path, done_grace_sec=1)
    duration = time.perf_counter() - t0

    assert res.returncode == 125, (
        f"Expected returncode 125, got {res.returncode}. stderr: {res.stderr}"
    )
    content = log.read_text(encoding="utf-8")
    assert "EXIT 125" in content
    assert "EXIT 0" not in content

    receipt_file = tmp_path / "hung.log.receipt.json"
    assert receipt_file.is_file()
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert receipt["actual_process_exit_code"] is None
    assert receipt["final_exit_code"] == 125
    assert receipt["termination_reason"] == "hung_after_done"
    assert receipt["done_marker"] is True


def test_subprocess_natural_nonzero_preserves_code(
    fake_matlab: Path, tmp_path: Path
) -> None:
    """Natural non-zero exit code 42 is preserved and propagated."""
    script = tmp_path / "nonzero.m"
    script.write_text("% nonzero", encoding="utf-8")
    log = tmp_path / "nonzero.log"

    res = _run_ps1(script, log, fake_matlab, tmp_path)
    assert res.returncode == 42, (
        f"Expected 42, got {res.returncode}. Stderr: {res.stderr}"
    )
    content = log.read_text(encoding="utf-8")
    assert "EXIT 42" in content


def test_subprocess_timeout_returns_124(fake_matlab: Path, tmp_path: Path) -> None:
    """Timeout watchdog kills hanging process and yields 124."""
    script = tmp_path / "timeout.m"
    script.write_text("% timeout", encoding="utf-8")
    log = tmp_path / "timeout.log"

    res = _run_ps1(script, log, fake_matlab, tmp_path, timeout_min=0)
    assert res.returncode == 124, (
        f"Expected 124, got {res.returncode}. Stderr: {res.stderr}"
    )
    content = log.read_text(encoding="utf-8")
    assert "EXIT 124" in content


def test_subprocess_paths_with_spaces(fake_matlab: Path, tmp_path: Path) -> None:
    """Handles spaces in script path, log path, and lock path cleanly."""
    space_dir = tmp_path / "path with spaces"
    space_dir.mkdir(parents=True, exist_ok=True)
    script = space_dir / "space script.m"
    script.write_text("% space success", encoding="utf-8")
    # Point script name ending in success.m so fake_matlab recognizes it
    script = space_dir / "space_success.m"
    script.write_text("% space success", encoding="utf-8")
    log = space_dir / "space log.log"
    lock = space_dir / "space lock"
    queue = space_dir / "space queue"

    res = _run_ps1(
        script,
        log,
        fake_matlab,
        tmp_path,
        lock_dir=lock,
        queue_dir=queue,
    )
    assert res.returncode == 0, (
        f"Expected 0 with spaces, got {res.returncode}. Stderr: {res.stderr}"
    )
    assert log.is_file()
    assert "EXIT 0" in log.read_text(encoding="utf-8")


def test_subprocess_missing_helper_fails_closed(
    fake_matlab: Path, tmp_path: Path
) -> None:
    """When decision helper is absent, runner fails closed with returncode 125."""
    isolated_tool_dir = tmp_path / "isolated_tools"
    isolated_tool_dir.mkdir(parents=True, exist_ok=True)
    isolated_ps1 = isolated_tool_dir / "run_matlab_locked.ps1"
    shutil.copy2(_RUNNER_PS1, isolated_ps1)
    # Ensure runner_decision.py does NOT exist in isolated_tool_dir

    script = tmp_path / "success.m"
    script.write_text("% success", encoding="utf-8")
    log = tmp_path / "missing_helper.log"

    res = _run_ps1(
        script,
        log,
        fake_matlab,
        tmp_path,
        runner_script=isolated_ps1,
    )
    assert res.returncode == 125, (
        f"Expected 125 on missing helper, got {res.returncode}. Stderr: {res.stderr}"
    )
    content = log.read_text(encoding="utf-8")
    assert "EXIT 125" in content
    assert "decision helper or runtime unavailable" in content


def test_subprocess_comment_echo_marker_not_falsely_treated_as_done(
    fake_matlab: Path, tmp_path: Path
) -> None:
    """Comment containing GS3DX_BATCH_DONE must NOT trigger done grace watchdog."""
    script = tmp_path / "comment_marker.m"
    script.write_text("% comment", encoding="utf-8")
    log = tmp_path / "comment.log"

    # With timeout_min=0, if comment triggered done, it would be killed as hung_after_done (125).
    # Since comment is ignored, it times out immediately (124).
    res = _run_ps1(script, log, fake_matlab, tmp_path, timeout_min=0, done_grace_sec=10)
    assert res.returncode == 124
    content = log.read_text(encoding="utf-8")
    assert "EXIT 124" in content
    assert "EXIT 125" not in content


def test_subprocess_real_runner_blocked_on_live_foreign_lock(
    fake_matlab: Path, tmp_path: Path
) -> None:
    """Real runner blocked on a live foreign lock preserves ownership without deleting it."""
    lock_dir = tmp_path / "foreign.lock"
    lock_dir.mkdir(parents=True, exist_ok=True)
    owner_file = lock_dir / "owner.txt"

    # Use current pytest process PID as the live foreign owner
    live_pid = os.getpid()
    foreign_owner_line = f"{live_pid} C:\\foreign\\job.m 2026-10-01T19:00:00Z"
    owner_file.write_text(f"{foreign_owner_line}\n", encoding="utf-8")

    queue_dir = tmp_path / "foreign.queue"
    queue_dir.mkdir(parents=True, exist_ok=True)

    script = tmp_path / "success.m"
    script.write_text("% blocked test", encoding="utf-8")
    log = tmp_path / "blocked.log"

    # Launch a second runner as background process
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(_RUNNER_PS1),
        "-Script",
        str(script),
        "-Log",
        str(log),
        "-Matlab",
        str(fake_matlab),
        "-TimeoutMin",
        "1",
        "-DoneGraceSec",
        "1",
        "-LockDir",
        str(lock_dir),
        "-QueueDir",
        str(queue_dir),
        "-PollIntervalSec",
        "0.2",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    try:
        # Give runner time to poll and see the lock is held
        time.sleep(1.0)

        # Assert foreign lock was NOT stolen, deleted, or overwritten!
        assert lock_dir.is_dir(), (
            "Lock directory must not be deleted while foreign owner is live."
        )
        assert owner_file.is_file(), "Owner file must still exist."
        current_owner = owner_file.read_text(encoding="utf-8").strip()
        assert current_owner == foreign_owner_line, (
            f"Lock owner was modified! Got: {current_owner}"
        )

    finally:
        # Terminate ONLY test runner and its children
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True
        )
        proc.wait(timeout=5)

    # Double check after runner teardown that foreign lock is still intact
    assert lock_dir.is_dir()
    assert owner_file.read_text(encoding="utf-8").strip() == foreign_owner_line
