from __future__ import annotations

import subprocess

import pytest
from scripts.install_nightly_system_deps import run_with_lock_retries


def _completed_process(
    command: list[str],
    returncode: int,
    *,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)


def test_run_with_lock_retries_retries_until_dpkg_lock_clears() -> None:
    command = ["sudo", "apt-get", "install", "-y", "libegl1", "xvfb"]
    calls: list[list[str]] = []
    sleeps: list[float] = []
    results = [
        _completed_process(
            command,
            100,
            stderr="E: Could not get lock /var/lib/dpkg/lock-frontend. It is held by process 123",
        ),
        _completed_process(command, 0),
    ]

    def fake_run(current_command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(current_command)
        return results.pop(0)

    run_with_lock_retries(
        command, attempts=3, delay_seconds=12.0, run=fake_run, sleep=sleeps.append
    )

    assert calls == [command, command]
    assert sleeps == [12.0]


def test_run_with_lock_retries_raises_for_non_lock_failures() -> None:
    command = ["sudo", "apt-get", "update"]
    sleeps: list[float] = []

    def fake_run(current_command: list[str]) -> subprocess.CompletedProcess[str]:
        return _completed_process(current_command, 100, stderr="E: package not found")

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_with_lock_retries(
            command, attempts=3, delay_seconds=12.0, run=fake_run, sleep=sleeps.append
        )

    assert exc_info.value.returncode == 100
    assert exc_info.value.cmd == command
    assert sleeps == []
