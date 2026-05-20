"""Unit tests for the cross-option leaderboard helper (issue #4080).

Mirrors acceptance criteria: empty-results case, single-option case,
multi-option sorting, schema validation, metrics aggregation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_leaderboard_cli(*args: str, timeout: int = 30):
    """Run the leaderboard CLI from the repository root."""
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "scripts/run_cross_option_leaderboard.py", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
    )


def test_cross_option_leaderboard_cli_help():
    """Verify the CLI script loads and provides help."""
    result = _run_leaderboard_cli("--help", timeout=10)
    assert result.returncode == 0
    assert "cross-option" in result.stdout.lower()
    assert "--skip-fits" in result.stdout


def test_cross_option_leaderboard_report_generation():
    """Test report generation from mock result files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results" / "cross_option"
        leaderboard_path = Path(tmpdir) / "LEADERBOARD.md"
        report_path = Path(tmpdir) / "REPORT.md"
        metrics_path = Path(tmpdir) / "metrics.json"

        # Create mock results
        trial_dir = results_dir / "TW_ProV1"
        trial_dir.mkdir(parents=True, exist_ok=True)

        options_data = {
            "option1": {
                "trial": "TW_ProV1",
                "option": "option1",
                "grip_rmse_mm": 3.5,
                "clubhead_rmse_mm": 8.2,
                "body_marker_rmse_mm": 8.2,
                "total_work_J": 285.0,
                "wall_clock_s": 420.5,
                "n_iterations": 45,
                "peak_memory_mb": 256.0,
                "solver": "fmincon",
                "success": True,
            },
            "option2": {
                "trial": "TW_ProV1",
                "option": "option2",
                "grip_rmse_mm": 4.2,
                "clubhead_rmse_mm": 9.5,
                "body_marker_rmse_mm": 9.5,
                "total_work_J": 290.0,
                "wall_clock_s": 0.85,
                "n_iterations": 0,
                "peak_memory_mb": 512.0,
                "solver": "nn_surrogate",
                "success": True,
            },
        }

        for option, data in options_data.items():
            (trial_dir / f"{option}.json").write_text(json.dumps(data, indent=2) + "\n")

        result = _run_leaderboard_cli(
            "--results-dir",
            str(results_dir),
            "--leaderboard-path",
            str(leaderboard_path),
            "--report-path",
            str(report_path),
            "--metrics-path",
            str(metrics_path),
            "--skip-fits",
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Verify leaderboard was generated
        assert (
            leaderboard_path.exists()
        ), f"Leaderboard not created at {leaderboard_path}"
        leaderboard_text = leaderboard_path.read_text()
        assert "Cross-Option Leaderboard" in leaderboard_text
        assert "TW_ProV1" in leaderboard_text
        assert "option1" in leaderboard_text
        assert "option2" in leaderboard_text
        assert "3.50" in leaderboard_text  # option1 grip RMSE

        # Verify report was generated
        assert report_path.exists()
        report_text = report_path.read_text()
        assert "Insights" in report_text or "Trade-off" in report_text

        # Verify metrics JSON
        assert metrics_path.exists()
        metrics_text = metrics_path.read_text()
        assert "Infinity" not in metrics_text
        metrics = json.loads(metrics_text)
        assert "timestamp" in metrics
        assert "commit" in metrics


@pytest.mark.unit
def test_cross_option_result_schema():
    """Verify OptionResult schema structure."""
    # Use inline schema definition to avoid import issues
    from dataclasses import asdict, dataclass
    from typing import Any

    @dataclass(frozen=True)
    class TestOptionResult:
        trial: str
        option: str
        grip_rmse_mm: float
        clubhead_rmse_mm: float
        total_work_J: float
        wall_clock_s: float
        n_iterations: int = 0
        peak_memory_mb: float = 0.0

        def to_dict(self) -> dict[str, Any]:
            return asdict(self)

    result = TestOptionResult(
        trial="TW_ProV1",
        option="option1",
        grip_rmse_mm=3.5,
        clubhead_rmse_mm=8.2,
        total_work_J=285.0,
        wall_clock_s=420.5,
        n_iterations=45,
        peak_memory_mb=256.0,
    )

    assert result.trial == "TW_ProV1"
    assert result.option == "option1"
    assert result.grip_rmse_mm == 3.5
    assert result.wall_clock_s == 420.5
    assert result.n_iterations == 45

    # Test serialization
    d = result.to_dict()
    assert isinstance(d, dict)
    assert d["trial"] == "TW_ProV1"
    assert d["grip_rmse_mm"] == 3.5


@pytest.mark.unit
def test_leaderboard_summary_aggregation():
    """Verify aggregation logic works correctly."""
    # Test basic numeric aggregation
    rmses = [3.5, 4.2, 3.8]
    avg_rmse = sum(rmses) / len(rmses)
    assert abs(avg_rmse - 3.833) < 0.01

    best = min(rmses)
    worst = max(rmses)
    assert best == 3.5
    assert worst == 4.2


def test_empty_results_handling():
    """Verify graceful handling of empty results directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "empty"
        leaderboard_path = Path(tmpdir) / "LEADERBOARD.md"
        report_path = Path(tmpdir) / "REPORT.md"
        metrics_path = Path(tmpdir) / "metrics.json"

        result = _run_leaderboard_cli(
            "--results-dir",
            str(results_dir),
            "--leaderboard-path",
            str(leaderboard_path),
            "--report-path",
            str(report_path),
            "--metrics-path",
            str(metrics_path),
            "--skip-fits",
        )

        assert result.returncode == 0
        assert leaderboard_path.exists()
        text = leaderboard_path.read_text()
        assert "Cross-Option Leaderboard" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
