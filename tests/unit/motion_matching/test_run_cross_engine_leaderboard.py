"""Tests for the cross-engine leaderboard runner script."""

from __future__ import annotations

from pathlib import Path

from scripts import run_cross_engine_leaderboard as runner


def test_skip_fits_generates_report_without_optional_loader_imports(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    leaderboard_path = results_dir / "CROSS_ENGINE_LEADERBOARD.md"

    status = runner.main(
        [
            "--skip-fits",
            "--results-dir",
            str(results_dir),
            "--leaderboard-path",
            str(leaderboard_path),
        ]
    )

    assert status == 0
    assert leaderboard_path.exists()
