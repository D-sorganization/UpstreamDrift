"""Tests for the leaderboard CLI module.

Verifies that the leaderboard subcommand works end-to-end:
- Arguments are parsed correctly
- Results directory is scanned
- Output file is created

Issue #4248: Cross-engine leaderboard infrastructure
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Root of the UpstreamDrift repository (two levels above tests/unit/).
_REPO_ROOT = str(Path(__file__).resolve().parents[3])

# Subprocess environment with PYTHONPATH set so src.shared is importable
# regardless of the working directory used by each test.
_SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": _REPO_ROOT}


def _write_fit_result(trial_dir: Path, engine: str, data: dict | None = None) -> Path:
    """Write a FitResult JSON to a trial directory."""
    if data is None:
        data = {
            "engine": engine,
            "solver": "test_solver",
            "grip_rmse_mm": 2.5,
            "clubhead_rmse_mm": 3.0,
            "total_work_J": 100.0,
            "wall_clock_s": 10.0,
            "commit": "abc1234567890",
            "run_at": "2026-05-05T12:00:00Z",
        }
    trial_dir.mkdir(parents=True, exist_ok=True)
    path = trial_dir / f"{engine}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.unit
def test_leaderboard_cli_generates_report(tmp_path: Path) -> None:
    """The CLI generates a leaderboard report from a results directory."""
    results_dir = tmp_path / "results"
    trial_dir = results_dir / "test_trial"

    # Write sample results from multiple engines
    _write_fit_result(trial_dir, "mujoco")
    _write_fit_result(
        trial_dir,
        "pinocchio",
        {
            "engine": "pinocchio",
            "solver": "test_solver",
            "grip_rmse_mm": 1.5,
            "clubhead_rmse_mm": 2.0,
            "total_work_J": 100.0,
            "wall_clock_s": 5.0,
            "commit": "abc1234567890",
            "run_at": "2026-05-05T12:00:00Z",
        },
    )
    _write_fit_result(
        trial_dir,
        "drake",
        {
            "engine": "drake",
            "solver": "test_solver",
            "grip_rmse_mm": 3.0,
            "clubhead_rmse_mm": 3.5,
            "total_work_J": 100.0,
            "wall_clock_s": 15.0,
            "commit": "abc1234567890",
            "run_at": "2026-05-05T12:00:00Z",
        },
    )

    output_file = tmp_path / "leaderboard_output.md"

    # Run the CLI command
    cmd = [
        sys.executable,
        "-m",
        "src.shared.python.motion_matching",
        "leaderboard",
        "--results-dir",
        str(results_dir),
        "--output",
        str(output_file),
    ]
    result = subprocess.run(cmd, cwd=_REPO_ROOT, env=_SUBPROCESS_ENV)

    # Verify success
    assert result.returncode == 0
    assert output_file.exists()

    # Check the content has the expected structure
    content = output_file.read_text(encoding="utf-8")
    assert "Cross-engine leaderboard" in content
    assert "test_trial" in content
    assert "mujoco" in content
    assert "pinocchio" in content
    assert "drake" in content
    # Results should be sorted by grip_rmse_mm ascending
    lines = content.split("\n")
    pinocchio_line = next((ln for ln in lines if "pinocchio" in ln), None)
    mujoco_line = next((ln for ln in lines if "mujoco" in ln), None)
    if pinocchio_line and mujoco_line:
        # pinocchio should come first (lower rmse = 1.5 vs 2.5)
        assert lines.index(pinocchio_line) < lines.index(mujoco_line)


@pytest.mark.unit
def test_leaderboard_cli_default_output_path(tmp_path: Path) -> None:
    """The CLI writes to LEADERBOARD.md by default."""
    results_dir = tmp_path / "results"
    trial_dir = results_dir / "test_trial"
    _write_fit_result(trial_dir, "simscape")

    # Run without --output flag
    cmd = [
        sys.executable,
        "-m",
        "src.shared.python.motion_matching",
        "leaderboard",
        "--results-dir",
        str(results_dir),
    ]

    # Change to tmp_path so the default output lands there
    result = subprocess.run(
        cmd,
        cwd=str(tmp_path),
        env=_SUBPROCESS_ENV,
    )

    # Default should create LEADERBOARD.md in cwd
    assert result.returncode == 0
    default_file = tmp_path / "LEADERBOARD.md"
    assert default_file.exists()


@pytest.mark.unit
def test_leaderboard_cli_nonexistent_dir_fails(tmp_path: Path) -> None:
    """The CLI fails gracefully if results directory doesn't exist."""
    nonexistent = tmp_path / "nonexistent"

    cmd = [
        sys.executable,
        "-m",
        "src.shared.python.motion_matching",
        "leaderboard",
        "--results-dir",
        str(nonexistent),
        "--output",
        str(tmp_path / "output.md"),
    ]
    result = subprocess.run(cmd, cwd=_REPO_ROOT, env=_SUBPROCESS_ENV)

    # Should fail with exit code 1
    assert result.returncode == 1
