from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.research.proximal_distal_energy.run_event_topology_channel_matrix import (
    ARRAY_PATH,
    REPORT_PATH,
    STATUS_CODES,
    validate_report,
)


def _report() -> dict[str, object]:
    return json.loads(Path(REPORT_PATH).read_text(encoding="utf-8"))


def test_phase_c_report_is_source_current_and_complete() -> None:
    assert validate_report(_report()) == {
        "channel_count": 4,
        "step_control_count": 12,
        "horizon_control_count": 12,
        "retained_noise_outcome_count": 8448,
    }


def test_phase_c_zero_authority_has_no_command_or_noise_authority() -> None:
    with np.load(ARRAY_PATH, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["channel_zero_mask"], [0.0, 0.0])
        assert np.count_nonzero(arrays["channel_zero_command_delta_nm"]) == 0


def test_phase_c_raw_topology_counts_reproduce_every_published_cell() -> None:
    report = _report()
    with np.load(ARRAY_PATH, allow_pickle=False) as arrays:
        for channel in report["channel_maps"]:
            name = channel["channel"]
            statuses = arrays[f"channel_{name}_status_code"]
            for delay_index, summary in enumerate(channel["delay_summaries"]):
                observed = {
                    status: int(np.count_nonzero(statuses[delay_index] == code))
                    for status, code in STATUS_CODES.items()
                    if np.count_nonzero(statuses[delay_index] == code)
                }
                assert observed == summary["topology_counts"]


def test_phase_c_types_horizon_truncation_and_numerical_stability() -> None:
    report = _report()
    qualification = report["qualification"]

    assert qualification["all_step_topology_identities_stable"] is True
    assert qualification["expanded_horizon_topology_stable"] is True
    assert qualification["original_horizon_truncation_channels"] == ["wrist_only"]
    assert qualification["preservation_is_success_probability"] is False


def test_phase_c_keeps_parent_feasibility_and_work_power_separate() -> None:
    report = _report()

    assert report["parent_bounded_outcomes"]["trial_count"] == 28
    assert report["parent_bounded_outcomes"]["feasibility_status_counts"] == {
        "feasible": 22,
        "infeasible": 6,
    }
    assert report["availability"]["work_power"] == "unavailable"
    assert report["availability"]["coaching_recommendation"] == "unavailable"
