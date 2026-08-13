from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_rotating_base_torso_velocity_study import (
    build_study,
)

pytestmark = pytest.mark.unit


def test_study_retains_registered_programs_and_negative_controls() -> None:
    record, arrays = build_study(compact=True)

    assert record["model_tier"] == "planar_rotating_base_two_hand_compliant_club"
    assert record["attempted_case_count"] >= 18
    assert record["valid_case_count"] > 0
    assert record["valid_case_count"] < record["attempted_case_count"]
    assert {row["torso_profile"] for row in record["cases"]} >= {
        "accelerate",
        "constant_rate",
        "decelerate",
    }
    assert {row["matching_rule"] for row in record["cases"]} == {
        "relative_club_rate",
        "absolute_club_rate",
    }
    assert record["negative_controls"]["coincident_grip_max_couple_nm"] < 1e-10
    assert record["negative_controls"]["reversed_grip_couple_sign_reversed"]
    assert arrays["case_impact_speed_m_s"].shape == (record["attempted_case_count"],)


def test_study_reports_closed_ledgers_and_bounded_claims() -> None:
    record, _arrays = build_study(compact=True)
    valid = [row for row in record["cases"] if row["valid"]]

    assert max(abs(row["work_energy_closure_j"]) for row in valid) < 0.08
    assert max(row["maximum_constraint_residual_m"] for row in valid) < 1e-7
    assert record["claims"]["universal_high_torso_velocity_strategy"] in {
        "rejected",
        "not_supported",
    }
    assert record["claims"]["human_coaching_strategy"] == "unsupported"
    assert np.isfinite(record["associations"]["torso_rate_vs_impact_speed_r"])
