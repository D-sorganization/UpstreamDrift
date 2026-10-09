"""Export contracts for shareable shot pattern results."""

from __future__ import annotations

import csv
import json
import math

import pytest

from src.tools.shot_pattern_analysis.core import AnalysisConfig, run_analysis
from src.tools.shot_pattern_analysis.physics import ShotOutcome
from src.tools.shot_pattern_analysis.reporting import export_analysis


class FakePhysics:
    def simulate(self, *, face_deg, path_deg, config, sample_trajectory=False):
        y = 12 * face_deg - 2 * path_deg
        return ShotOutcome(
            200.0,
            y,
            math.hypot(200.0, y),
            face_deg,
            2500.0,
            face_deg - path_deg,
            ((0.0, 0.0), (100.0, y / 2), (200.0, y)),
        )


def test_export_writes_reproducible_data_and_social_graphics(tmp_path) -> None:
    result = run_analysis(AnalysisConfig(n_shots=20, seed=71), physics=FakePhysics())
    paths = export_analysis(result, tmp_path)
    assert set(paths) == {
        "shots_csv",
        "summary_json",
        "receipt_json",
        "overhead_png",
        "dispersion_png",
    }
    assert all(path.exists() for path in paths.values())
    assert paths["overhead_png"].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert paths["dispersion_png"].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert paths["overhead_png"].stat().st_size > 20_000
    assert paths["dispersion_png"].stat().st_size > 20_000
    with paths["shots_csv"].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 60
    assert rows[0]["pattern"] == "Straight"
    summary = json.loads(paths["summary_json"].read_text())
    assert summary["patterns"]["Draw"]["n"] == 20
    ratio = summary["paired_variance_comparisons"]["Draw"]["raw"]
    assert ratio["estimate"] == pytest.approx(1.0)
    assert ratio["lower_95"] == pytest.approx(1.0)
    assert ratio["upper_95"] == pytest.approx(1.0)
    receipt = json.loads(paths["receipt_json"].read_text())
    assert receipt["seed"] == 71
    assert receipt["physics"]["impact_model"] == "rigid_body"
    assert "simplified" in receipt["limitations"][0].lower()
