"""Historical-tour approach scoring scenario contracts."""

from __future__ import annotations

import math
import csv
import json

import pytest

from src.tools.shot_pattern_analysis.scoring import (
    build_broadie_approx_baseline,
    green_expected_strokes,
    putt_probabilities,
    score_approach_endpoints,
    score_saved_bundle,
)

pytestmark = pytest.mark.unit


def test_published_putting_anchors_and_verified_baseline() -> None:
    one_eight, _ = putt_probabilities(8 / 3)
    _, three_forty = putt_probabilities(40 / 3)
    assert one_eight == pytest.approx(0.5, abs=0.06)
    assert three_forty == pytest.approx(0.10, abs=0.015)
    assert green_expected_strokes(33 / 3) == pytest.approx(2.0, abs=0.07)
    one_tap, three_tap = putt_probabilities(1 / 3)
    assert one_tap + three_tap <= 1
    baseline = build_broadie_approx_baseline()
    assert baseline.table_sha256 and len(baseline.table_sha256) == 64
    assert any(p.lie == "fairway" and p.distance_yards == 200 for p in baseline.states)
    assert any(p.lie == "green" and p.distance_yards == 10 for p in baseline.states)
    assert any(
        p.lie == "tee" and p.distance_yards == 400 and p.expected_strokes == 3.99
        for p in baseline.states
    )
    assert any(
        p.lie == "rough" and p.distance_yards == 600 and p.expected_strokes == 5.25
        for p in baseline.states
    )


def test_unholed_tap_in_has_at_least_one_expected_stroke() -> None:
    """A zero-distance endpoint on the green is unholed, not a made shot."""
    assert green_expected_strokes(0.0) == 1.0
    assert all(
        green_expected_strokes(distance_yards) >= 1.0
        for distance_yards in (0.00001, 0.01, 0.053975, 0.1, 1.0)
    )
    baseline = build_broadie_approx_baseline()
    green = [state for state in baseline.states if state.lie == "green"]
    assert green[0].distance_yards == 0.0
    assert green[0].expected_strokes == 1.0
    assert baseline.version == "table9-full-plus-anchor-reconciled-putting/2"


def test_source_backed_scoring_respects_green_and_pair_indices() -> None:
    endpoints = {
        "Straight": [(0, 200.0, 0.0), (1, 200.0, 10.0), (2, 200.0, 20.0)],
        "Draw": [(0, 200.0, 0.0), (1, 200.0, 2.0), (2, 200.0, 10.0)],
        "Fade": [(0, 200.0, 0.0), (1, 200.0, 20.0), (2, 200.0, 30.0)],
    }
    result = score_approach_endpoints(endpoints, target_x_m=200.0, green_radius_m=15)
    assert result["source_backed_status"] == "available"
    assert result["api_scored_shots"] == 9
    assert (
        result["patterns"]["Draw"]["mean_strokes_gained"]
        > result["patterns"]["Fade"]["mean_strokes_gained"]
    )
    assert result["patterns"]["Straight"]["n"] == 3
    assert math.isfinite(result["paired_benefit_vs_straight"]["Draw"]["estimate"])


def test_invalid_scoring_geometry_rejected() -> None:
    with pytest.raises(ValueError):
        green_expected_strokes(float("nan"))
    with pytest.raises(ValueError):
        score_approach_endpoints(
            {"Straight": [(0, 0.0, 0.0)]}, target_x_m=0, green_radius_m=15
        )
    with pytest.raises(ValueError, match="published fairway benchmark"):
        score_approach_endpoints(
            {
                name: [(0, 700.0, 0.0), (1, 700.0, 1.0)]
                for name in ("Straight", "Draw", "Fade")
            },
            target_x_m=700,
            green_radius_m=15,
        )


def test_saved_bundle_exports_green_size_sensitivity(tmp_path) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps({"target_x_m": 200, "config": {"target_radius_m": 15, "seed": 2}})
    )
    with (tmp_path / "shots.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["pattern", "shot_index", "aimed_x_m", "aimed_y_m"]
        )
        writer.writeheader()
        for pattern in ("Straight", "Draw", "Fade"):
            for i, y in enumerate((3, 12, 18)):
                writer.writerow(
                    {
                        "pattern": pattern,
                        "shot_index": i,
                        "aimed_x_m": 200,
                        "aimed_y_m": y,
                    }
                )
    report_path = score_saved_bundle(tmp_path)
    report = json.loads(report_path.read_text())
    assert set(report["green_radius_sensitivity_m"]) == {"10", "15", "20"}
    assert (
        report["green_radius_sensitivity_m"]["20"]["patterns"]["Straight"][
            "mean_strokes_gained"
        ]
        > report["green_radius_sensitivity_m"]["10"]["patterns"]["Straight"][
            "mean_strokes_gained"
        ]
    )


def test_valid_flight_bundle_with_zero_circle_keeps_export(tmp_path) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps({"target_x_m": 200, "config": {"target_radius_m": 0, "seed": 2}})
    )
    with (tmp_path / "shots.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["pattern", "shot_index", "aimed_x_m", "aimed_y_m"]
        )
        writer.writeheader()
        for pattern in ("Straight", "Draw", "Fade"):
            for i in range(2):
                writer.writerow(
                    {
                        "pattern": pattern,
                        "shot_index": i,
                        "aimed_x_m": 200,
                        "aimed_y_m": i + 1,
                    }
                )
    report = json.loads(score_saved_bundle(tmp_path).read_text())
    assert report["status"] == "unavailable"
    assert "positive" in report["reason"]
