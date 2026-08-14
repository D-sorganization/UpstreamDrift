from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
JSON_PATH = DATA / "timing_viability_study.json"
NPZ_PATH = DATA / "timing_viability_study.npz"


def test_timing_viability_evidence_preserves_common_phase_and_all_cohorts() -> None:
    record = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    arrays = np.load(NPZ_PATH)

    assert record["schema_version"] == "timing-viability-adverse-load/v1"
    assert record["parent_epic"] == 8557
    assert record["issue"] == 8623
    assert record["registered_before_preferred_result"] is True
    assert record["policies"] == ["clock", "state_triggered"]
    assert len(record["phase_offsets_s"]) == 5
    assert len(record["load_cases"]) == 6
    assert record["case_count"] == 60
    assert arrays["outcomes"].shape == (2, 6, 5, 7)
    assert arrays["reference_outcomes"].shape == (2, 6, 5, 7)
    assert arrays["normalized_error_trajectories"].shape[:3] == (2, 6, 5)


def test_state_event_surfaces_share_the_clock_policy_nominal_phase_map() -> None:
    record = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    mapping = record["common_phase_mapping"]
    offsets = np.asarray(record["phase_offsets_s"])
    target_times = np.asarray(mapping["target_event_times_s"])
    thresholds = np.asarray(mapping["state_angle_thresholds_rad"])

    assert np.allclose(target_times, record["nominal_event_time_s"] + offsets)
    assert np.all(np.diff(thresholds) > 0.0)
    assert mapping["source"] == "common_nominal_clock_trajectory"


def test_study_uses_common_guards_and_retains_neutral_claim_boundary() -> None:
    record = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    assert set(record["viability_sensitivity"]) == {"strict", "primary", "lenient"}
    assert set(record["recovery_qualified_viability_sensitivity"]) == {
        "strict",
        "primary",
        "lenient",
    }
    for sensitivity in record["viability_sensitivity"].values():
        assert set(sensitivity["policies"]) == {"clock", "state_triggered"}
        for summary in sensitivity["policies"].values():
            assert 0.0 <= summary["robust_viable_fraction"] <= 1.0
            assert summary["robust_contiguous_width_s"] >= 0.0
    assert record["claim_status"]["human_timing_demand"] == "untested"
    assert record["claim_status"]["human_self_correction"] == "untested"
    assert record["claim_status"]["coaching_prescription"] == "unsupported"
    assert record["claim_status"]["model_policy_ordering"] == "clock_larger"
    assert record["claim_status"]["recovery_policy_ordering"] == "no_separation"
    assert (
        record["claim_status"]["registered_sustained_recovery"]
        == "not_observed_in_any_case"
    )
    assert any("delivery proxy" in item for item in record["limitations"])
    assert any("not a golfer population" in item for item in record["limitations"])


def test_representative_half_step_screen_bounds_numerical_interpretation() -> None:
    record = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    for result in record["numerical_sensitivity"].values():
        assert result["refined_step_s"] == result["coarse_step_s"] / 2.0
        assert result["delivery_speed_absolute_difference_m_s"] < 0.2
        assert result["peak_hand_force_relative_difference"] < 0.05
        assert result["refined_normalized_energy_residual"] < 0.05
