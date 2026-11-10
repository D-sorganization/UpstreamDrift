"""Red-first lifecycle tests for ProcessManager (#14949)."""

from __future__ import annotations

import textwrap
import time
from pathlib import Path

import pytest

from src.launchers.launcher_process_manager import ProcessManager
from src.launchers.managed_process import ProcessState

pytestmark = pytest.mark.unit


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "src" / "lifecycle_fixtures").mkdir(parents=True)
    return root


def _fixture_script(repo_root: Path, name: str, body: str) -> Path:
    script = repo_root / "src" / "lifecycle_fixtures" / name
    script.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return script


@pytest.fixture
def manager(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> ProcessManager:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    return ProcessManager(repo_root=repo_root, use_separate_terminals=False)


def test_stdout_does_not_pollute_last_stderr_tail(
    manager: ProcessManager, repo_root: Path
) -> None:
    script = _fixture_script(
        repo_root,
        "mixed_streams.py",
        """
        import sys
        print("stdout-only")
        sys.stdout.flush()
        sys.stderr.write("stderr-only\\n")
        sys.stderr.flush()
        """,
    )

    process = manager.launch_script(
        name="mixed-streams",
        script_path=script,
        cwd=repo_root,
    )
    assert process is not None

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        rows = manager.list_processes()
        if rows and rows[0]["exit_code"] == 0:
            break
        time.sleep(0.05)

    managed = manager.list_processes()[0]
    tail = managed["last_stderr_tail"]
    assert "stderr-only" in tail
    assert "stdout-only" not in tail


def test_fast_failing_child_is_crashed_not_running(
    manager: ProcessManager, repo_root: Path
) -> None:
    script = _fixture_script(
        repo_root,
        "fail_fast.py",
        """
        import sys
        sys.stderr.write("boom-fast-fail\\n")
        sys.stderr.flush()
        sys.exit(1)
        """,
    )

    process = manager.launch_script(
        name="fail-fast",
        script_path=script,
        cwd=repo_root,
    )
    assert process is not None

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        rows = manager.list_processes()
        if rows and rows[0]["exit_code"] == 1:
            break
        time.sleep(0.05)

    rows = manager.list_processes()
    assert len(rows) == 1
    managed = rows[0]
    assert managed["state"] == ProcessState.CRASHED.value
    assert managed["exit_code"] == 1
    assert "boom-fast-fail" in managed["last_stderr_tail"]


def test_second_launch_same_key_retains_first(
    manager: ProcessManager, repo_root: Path
) -> None:
    script = _fixture_script(
        repo_root,
        "sleep.py",
        """
        import time
        time.sleep(30)
        """,
    )

    first = manager.launch_script(name="dupe", script_path=script, cwd=repo_root)
    second = manager.launch_script(name="dupe", script_path=script, cwd=repo_root)
    assert first is not None
    assert second is not None
    assert first.pid != second.pid

    keys = {m["key"] for m in manager.list_processes()}
    slots = {m["slot_name"] for m in manager.list_processes()}
    assert keys == {"dupe"}
    assert "dupe" in slots
    assert any(s.startswith("dupe (#") for s in slots)
    assert len(manager.running_processes) == 2

    manager.cleanup_processes()
    assert len(manager.running_processes) == 0


def test_cleanup_processes_terminates_children(
    manager: ProcessManager, repo_root: Path
) -> None:
    script = _fixture_script(
        repo_root,
        "sleep_long.py",
        """
        import time
        time.sleep(120)
        """,
    )

    process = manager.launch_script(
        name="cleanup-me",
        script_path=script,
        cwd=repo_root,
    )
    assert process is not None
    assert process.poll() is None

    manager.cleanup_processes()
    deadline = time.monotonic() + 5.0
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert process.poll() is not None
    assert len(manager.running_processes) == 0
