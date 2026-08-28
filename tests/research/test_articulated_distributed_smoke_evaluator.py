"""Registered distributed forward-attribution evaluator contracts."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from scripts.research.proximal_distal_energy.articulated_forward_attribution_runner import (
    StudyCase,
)
from scripts.research.proximal_distal_energy.articulated_forward_smoke_evaluator import (
    evaluate_distributed_smoke_case,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    ROOT
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "data"
    / "articulated_forward_attribution_plan.json"
)


def _manifest() -> dict[str, object]:
    manifest = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    design = manifest["design"]
    design["distributed_contact_law"] = {
        "name": "distributed_tension_with_regularized_coulomb_limit",
        "station_count_per_hand": 3,
        "station_width_m": 0.03,
        "tangential_damping_n_s_m": 18.0,
        "friction_coefficient": 0.3,
        "slack_distance_m": 0.0015,
        "static_stick_modeled": False,
    }
    return manifest


def test_distributed_smoke_retains_events_closure_and_claim_boundaries() -> None:
    result = evaluate_distributed_smoke_case(
        StudyCase(
            source_case_index=4,
            source_sample_index=6,
            source_time_s=0.12,
            engine="mujoco",
            variant="nominal",
            time_step_s=0.001,
            case_key="distributed-test",
        ),
        _manifest(),
    )

    assert result["estimand"] == "same_trajectory_descriptive_attribution"
    assert result["contact_model"]["static_stick_modeled"] is False
    assert result["contact_model"]["station_count_per_hand"] == 3
    assert result["events"]["path_model"] == "linear_state_interpolant"
    assert result["events"]["discrete_impulse_modeled"] is False
    assert result["closure"]["pointwise_force_residual"] <= 1.0e-10
    assert np.isfinite(result["closure"]["momentum_relative_residual"])
    assert np.isfinite(result["closure"]["work_relative_residual"])
    assert result["claim_boundary"] == {
        "human_or_coaching_inference": False,
        "static_stick_inference": False,
        "causal_counterfactual": False,
        "smoke_state_representative_of_humans": False,
    }


def test_distributed_variant_can_kill_friction_without_changing_the_base_law() -> None:
    manifest = _manifest()
    nominal = manifest["design"]["variant_parameters"][0]
    nominal["friction_coefficient_factor"] = 0.0

    result = evaluate_distributed_smoke_case(
        StudyCase(
            source_case_index=4,
            source_sample_index=6,
            source_time_s=0.12,
            engine="mujoco",
            variant="nominal",
            time_step_s=0.001,
            case_key="distributed-frictionless-test",
        ),
        manifest,
    )

    assert result["contact_model"]["friction_coefficient"] == 0.0
    assert not any(
        record["kind"].startswith("friction_limit")
        for record in result["events"]["records"]
    )
