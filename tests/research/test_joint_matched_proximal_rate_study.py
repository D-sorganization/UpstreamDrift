"""Contracts for the jointly work- and load-matched proximal-rate screen."""

from __future__ import annotations

import pytest

from scripts.research.proximal_distal_energy.run_joint_matched_proximal_rate_study import (
    run_study,
    write_outputs,
)

pytestmark = pytest.mark.scientific


def test_joint_match_keeps_all_attempts_and_uses_independent_pairs() -> None:
    record = run_study()

    assert record["schema_version"] == "joint-matched-proximal-rate/v1"
    assert record["attempted_program_count"] == 216
    assert record["valid_impact_count"] == 148
    primary = record["primary_match"]
    assert primary["candidate_pair_count"] == 109
    assert primary["independent_pair_count"] == 46
    indices = [
        index
        for pair in primary["pairs"]
        for index in (
            pair["lower_rate_program_index"],
            pair["higher_rate_program_index"],
        )
    ]
    assert len(indices) == len(set(indices))


def test_joint_match_enforces_registered_work_load_and_rate_bounds() -> None:
    primary = run_study()["primary_match"]

    assert primary["minimum_release_rate_separation_rad_s"] == pytest.approx(1.5)
    assert primary["maximum_relative_net_work_difference"] == pytest.approx(0.05)
    assert primary["maximum_relative_positive_work_difference"] == pytest.approx(0.05)
    assert primary["maximum_relative_peak_force_difference"] == pytest.approx(0.10)
    for pair in primary["pairs"]:
        assert pair["release_rate_separation_rad_s"] >= 1.5
        assert pair["relative_net_work_difference"] <= 0.05
        assert pair["relative_positive_work_difference"] <= 0.05
        assert pair["relative_peak_force_difference"] <= 0.10


def test_joint_match_retains_mixed_and_adverse_results() -> None:
    record = run_study()
    primary = record["primary_match"]

    assert primary["higher_rate_faster_pair_count"] == 20
    assert primary["higher_rate_slower_pair_count"] == 26
    assert primary["impact_speed_difference_range_m_s"] == pytest.approx(
        [-3.8481018198, 1.4523463206]
    )
    assert primary["mean_impact_speed_difference_m_s"] == pytest.approx(-0.1272357736)
    assert primary["causal_estimand"] is False
    assert record["conclusion"] == "mixed_nonmonotonic_model_response"


def test_joint_match_reports_tolerance_sensitivity() -> None:
    sensitivity = run_study()["tolerance_sensitivity"]

    assert len(sensitivity) == 9
    keys = {(row["work_tolerance"], row["load_tolerance"]): row for row in sensitivity}
    assert keys[(0.025, 0.05)]["independent_pair_count"] == 16
    assert keys[(0.075, 0.15)]["independent_pair_count"] == 53
    assert all(row["higher_rate_faster_pair_count"] > 0 for row in sensitivity)
    assert all(row["higher_rate_slower_pair_count"] > 0 for row in sensitivity)


def test_joint_match_outputs_are_byte_deterministic(tmp_path) -> None:
    first = write_outputs(tmp_path / "first")
    second = write_outputs(tmp_path / "second")

    assert first.read_bytes() == second.read_bytes()
