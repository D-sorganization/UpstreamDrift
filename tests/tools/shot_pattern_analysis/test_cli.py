"""Headless CLI export smoke test for shot-pattern analysis."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_headless_cli_writes_complete_analysis_bundle(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "QT_QPA_PLATFORM": "offscreen",
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tools.shot_pattern_analysis",
            "--output",
            str(tmp_path),
            "--shots",
            "20",
            "--seed",
            "20261008",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    expected = {
        "shots.csv",
        "summary.json",
        "overhead_flight.png",
        "dispersion.png",
        "receipt.json",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    with (tmp_path / "shots.csv").open(newline="", encoding="utf-8") as stream:
        shots = list(csv.DictReader(stream))
    assert len(shots) == 60
    assert {row["pattern"] for row in shots} == {"Straight", "Draw", "Fade"}

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    patterns = summary["patterns"]
    assert set(patterns) == {"Straight", "Draw", "Fade"}
    assert all(patterns[name]["n"] == 20 for name in patterns)
