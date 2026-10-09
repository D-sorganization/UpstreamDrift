"""Corrected-club hypothetical course scoring contracts."""

from __future__ import annotations

import pytest
import csv
import json

from src.tools.shot_pattern_analysis.scenario_scoring import (
    score_corrected_bundle,
    score_course_endpoints,
)

pytestmark = pytest.mark.unit


def _endpoints() -> dict[str, list[tuple[int, float, float]]]:
    return {
        "Straight": [(0, 200.0, 0.0), (1, 200.0, 20.0)],
        "Draw": [(0, 201.0, 0.0), (1, 201.0, 10.0)],
        "Fade": [(0, 199.0, 0.0), (1, 199.0, 30.0)],
    }


def test_driver_tee_uses_fairway_corridor_and_source_api_knots() -> None:
    result = score_course_endpoints(
        _endpoints(), club_id="driver", target_x_m=200.0,
        hole_distance_m=400.0, fairway_half_width_m=15.0,
    )
    assert result["status"] == "available"
    assert result["scenario_id"] == "tee_fairway_rough"
    assert result["start_lie"] == "tee"
    assert result["scored_shots"] == 6
    assert result["api_evaluated_states"] < result["scored_shots"] * 30
    assert result["patterns"]["Draw"]["fairway_fraction"] == 1.0
    assert result["patterns"]["Fade"]["fairway_fraction"] == 0.5


def test_iron_approach_uses_green_circle_and_outside_rough() -> None:
    result = score_course_endpoints(
        _endpoints(), club_id="seven_iron", target_x_m=200.0,
        green_radius_m=15.0,
    )
    assert result["scenario_id"] == "approach_green_rough"
    assert result["start_lie"] == "fairway"
    assert result["patterns"]["Draw"]["green_fraction"] == 1.0
    assert result["patterns"]["Fade"]["green_fraction"] == 0.5


def test_driver_rejects_course_distance_outside_published_tee_support() -> None:
    with pytest.raises(ValueError, match="support"):
        score_course_endpoints(
            _endpoints(), club_id="driver", target_x_m=200.0,
            hole_distance_m=700.0,
        )


def test_saved_driver_bundle_exports_course_sensitivity_and_receipt(tmp_path) -> None:
    (tmp_path / "summary.json").write_text(json.dumps({
        "target_x_m": 200.0,
        "config": {"club_id": "driver", "seed": 3},
    }))
    (tmp_path / "receipt.json").write_text("{}")
    with (tmp_path / "shots.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pattern", "shot_index", "aimed_x_m", "aimed_y_m"])
        writer.writeheader()
        for name, shots in _endpoints().items():
            for index, x, y in shots:
                writer.writerow({"pattern": name, "shot_index": index, "aimed_x_m": x, "aimed_y_m": y})
    report = json.loads(score_corrected_bundle(tmp_path).read_text())
    assert set(report["hole_distance_sensitivity_m"]) == {"350", "400", "450"}
    assert set(report["fairway_width_sensitivity_m"]) == {"20", "30", "40"}
    assert json.loads((tmp_path / "summary.json").read_text())["course_scoring"]["status"] == "available"
    assert json.loads((tmp_path / "receipt.json").read_text())["scoring_postprocess"]["api_evaluated_states"] > 0
