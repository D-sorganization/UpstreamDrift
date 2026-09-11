"""Runner options must fail before opening a licensed MATLAB session."""

from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
RUNNER = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/reproduction/first_prefix_fit.py"
)


def test_help_exposes_difference_step_without_matlab() -> None:
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--finite-difference-step" in result.stdout


@pytest.mark.parametrize("step", ["0", "-1", "nan", "inf"])
def test_invalid_step_is_rejected_before_loading_capture_or_matlab(
    tmp_path: Path, step: str
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--repo",
            str(ROOT),
            "--run-dir",
            str(tmp_path),
            "--initial-state",
            str(tmp_path / "missing.json"),
            "--finite-difference-step",
            step,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "finite-difference-step must be finite and positive" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
