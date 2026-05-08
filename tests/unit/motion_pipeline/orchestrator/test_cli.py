"""Tests for the orchestrator CLI entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_MODULE = "src.shared.python.motion_pipeline.orchestrator"


def _run(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", _MODULE, *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=60,
    )


def test_cli_help_lists_commands(tmp_path: Path) -> None:
    result = _run("--help")
    # help exit code is 0 in argparse default
    assert result.returncode == 0
    assert (
        "Motion Capture Pipeline" in result.stdout or "usage" in result.stdout.lower()
    )


def test_cli_no_command_prints_help_and_exits_nonzero() -> None:
    result = _run()
    # No subcommand -> help printed and exit code 1
    assert result.returncode != 0


def test_cli_run_missing_source_exits_nonzero(tmp_path: Path) -> None:
    fake = tmp_path / "no_such.c3d"
    result = _run("run", str(fake), "--engine", "mujoco")
    assert result.returncode != 0
    # Actionable error message points at the missing file
    combined = (result.stdout + result.stderr).lower()
    assert "not found" in combined or "no such" in combined or "error" in combined


def test_cli_run_help_lists_engine_choices() -> None:
    result = _run("run", "--help")
    assert result.returncode == 0
    out = result.stdout.lower()
    assert "engine" in out
    # All four backend names should be advertised
    for engine in ("mujoco", "drake", "pinocchio", "opensim"):
        assert engine in out
