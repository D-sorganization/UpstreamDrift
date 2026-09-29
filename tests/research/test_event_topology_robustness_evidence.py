"""Immutable evidence checks for the registered #9125 Phase A campaign."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_event_topology_robustness import (
    ARRAY_PATH,
    REGISTERED_DELAYS_S,
    REPORT_PATH,
    STATUS_CODES,
    registered_scenarios,
    validate_report,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_registered_report_passes_fail_closed_validation(
    report: dict[str, object],
) -> None:
    result = validate_report(report)

    assert result == {
        "scenario_count": 4,
        "delay_count": 11,
        "retained_outcome_count": 6380,
    }


def test_raw_arrays_reproduce_every_reported_topology_count(
    report: dict[str, object],
) -> None:
    arrays = np.load(ARRAY_PATH, allow_pickle=False)
    scenarios = report["scenarios"]
    assert isinstance(scenarios, list)
    for scenario, registered in zip(scenarios, registered_scenarios(), strict=True):
        assert isinstance(scenario, dict)
        codes = arrays[f"{registered.name}_status_code"]
        crossings = arrays[f"{registered.name}_crossing_count"]
        assert codes.shape == (len(REGISTERED_DELAYS_S), registered.replicate_count)
        assert crossings.shape == codes.shape
        summaries = scenario["delay_summaries"]
        assert isinstance(summaries, list)
        for delay_index, summary in enumerate(summaries):
            assert isinstance(summary, dict)
            expected_counts = {
                status: int(np.count_nonzero(codes[delay_index] == code))
                for status, code in STATUS_CODES.items()
                if np.any(codes[delay_index] == code)
            }
            assert summary["topology_counts"] == expected_counts
            assert int(crossings[delay_index].sum()) == registered.replicate_count


def test_full_precision_event_arrays_retain_finite_transverse_events() -> None:
    arrays = np.load(ARRAY_PATH, allow_pickle=False)
    for scenario in registered_scenarios():
        event_time = arrays[f"{scenario.name}_event_time_s"][:, :, 0]
        transversality = arrays[f"{scenario.name}_transversality_per_s"][:, :, 0]
        direction = arrays[f"{scenario.name}_direction_code"][:, :, 0]
        assert np.all(np.isfinite(event_time))
        assert np.all(np.isfinite(transversality))
        assert np.all(transversality > 0.0)
        assert np.all(direction == 1)


def test_inference_boundary_suppresses_human_and_strategy_promotion(
    report: dict[str, object],
) -> None:
    availability = report["availability"]
    assert availability == {
        "human_motor_noise": "unavailable",
        "fatigue_interpretation": "unavailable",
        "controller_ranking": "suppressed",
        "coaching_recommendation": "unavailable",
    }
    boundary = str(report["inference_boundary"]).lower()
    assert "synthetic model-scenario" in boundary
    assert "do not establish" in boundary
