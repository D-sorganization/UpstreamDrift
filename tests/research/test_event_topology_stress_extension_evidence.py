"""Immutable raw-evidence checks for the preregistered #9125 Phase B."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_event_topology_robustness import (
    REGISTERED_DELAYS_S,
    STATUS_CODES,
)
from scripts.research.proximal_distal_energy.run_event_topology_stress_extension import (
    ARRAY_PATH,
    REPORT_PATH,
    registered_stress_scenarios,
    validate_report,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_phase_b_report_passes_fixed_stop_rule_validation(
    report: dict[str, object],
) -> None:
    assert validate_report(report) == {
        "scenario_count": 5,
        "delay_count": 11,
        "retained_outcome_count": 10560,
    }
    qualification = report["qualification"]
    assert isinstance(qualification, dict)
    assert qualification["all_summaries_adequate"] is True
    assert qualification["first_registered_scale_with_topology_loss"] == 0.02


def test_phase_b_arrays_reproduce_status_and_crossing_counts(
    report: dict[str, object],
) -> None:
    arrays = np.load(ARRAY_PATH, allow_pickle=False)
    scenarios = report["scenarios"]
    assert isinstance(scenarios, list)
    for record, scenario in zip(scenarios, registered_stress_scenarios(), strict=True):
        assert isinstance(record, dict)
        status = arrays[f"{scenario.name}_status_code"]
        crossing_count = arrays[f"{scenario.name}_crossing_count"]
        event_time = arrays[f"{scenario.name}_event_time_s"]
        assert status.shape == (len(REGISTERED_DELAYS_S), 192)
        assert crossing_count.shape == status.shape
        assert event_time.shape[:2] == status.shape
        summaries = record["delay_summaries"]
        assert isinstance(summaries, list)
        for delay_index, summary in enumerate(summaries):
            assert isinstance(summary, dict)
            raw_counts = {
                label: int(np.count_nonzero(status[delay_index] == code))
                for label, code in STATUS_CODES.items()
                if np.any(status[delay_index] == code)
            }
            assert summary["topology_counts"] == raw_counts
            for replicate_index, count in enumerate(crossing_count[delay_index]):
                retained = event_time[delay_index, replicate_index]
                assert int(np.count_nonzero(np.isfinite(retained))) == int(count)


def test_registered_boundary_cells_retain_null_and_multiple_outcomes(
    report: dict[str, object],
) -> None:
    scenarios = report["scenarios"]
    assert isinstance(scenarios, list)
    first = scenarios[0]
    largest = scenarios[-1]
    assert isinstance(first, dict)
    assert isinstance(largest, dict)
    first_last = first["delay_summaries"][-1]
    largest_last = largest["delay_summaries"][-1]
    assert first_last["topology_counts"] == {  # type: ignore[index]
        "absent": 1,
        "unique_transverse": 191,
    }
    assert largest_last["topology_counts"] == {  # type: ignore[index]
        "absent": 118,
        "multiple": 7,
        "unique_transverse": 67,
    }
    assert largest_last["preservation_interval"] == pytest.approx(  # type: ignore[index]
        [0.00573197, 0.0728071]
    )


def test_phase_b_remains_synthetic_and_nonprescriptive(
    report: dict[str, object],
) -> None:
    availability = report["availability"]
    assert isinstance(availability, dict)
    assert availability["human_motor_noise"] == "unavailable"
    assert availability["fatigue_interpretation"] == "unavailable"
    assert availability["target_accuracy"] == "unavailable"
    assert availability["controller_ranking"] == "suppressed"
    assert availability["coaching_recommendation"] == "unavailable"
