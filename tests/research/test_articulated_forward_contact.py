from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_forward_contact import (
    ArticulatedForwardContactConfig,
    ForwardIntegrationCase,
    integrate_articulated_contact,
    mechanical_energy,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def test_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="duration_s"):
        ArticulatedForwardContactConfig(duration_s=0.0)
    with pytest.raises(ValueError, match="time_steps_s"):
        ArticulatedForwardContactConfig(time_steps_s=(0.0005, 0.001))
    with pytest.raises(ValueError, match="retention_threshold_m"):
        ArticulatedForwardContactConfig(retention_threshold_m=np.inf)
    with pytest.raises(ValueError, match="case_indices"):
        ArticulatedForwardContactConfig(case_indices=(0, 0))


def test_zero_preload_trace_converges_under_refinement() -> None:
    model, metadata = build_subject_scaled_model(default_synthetic_profiles()[0])
    with np.load(DATA_DIR / "subject_scaled_closed_contact.npz") as source:
        q = np.asarray(source["solution_q"][0, 6], dtype=float)
        grip_span_m = float(source["case_grip_span_m"][0])
    qd = np.zeros(model.nq)
    config = ArticulatedForwardContactConfig(
        duration_s=0.002,
        time_steps_s=(0.001, 0.0005),
    )

    def integration_case(time_step_s: float) -> ForwardIntegrationCase:
        return ForwardIntegrationCase(
            q=q,
            qd=qd,
            grip_span_m=grip_span_m,
            hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
            time_step_s=time_step_s,
            contact_stiffness=1800.0,
            contact_damping=0.0,
            initial_club_displacement_m=0.0,
            initial_club_velocity_m_s=0.0,
            engine="mujoco",
        )

    coarse = integrate_articulated_contact(
        model, integration_case(config.time_steps_s[0]), config
    )
    fine = integrate_articulated_contact(
        model, integration_case(config.time_steps_s[1]), config
    )

    assert np.all(np.isfinite(coarse["mechanical_energy_j"]))
    assert abs(fine["work_energy_residual_j"][-1]) < abs(
        coarse["work_energy_residual_j"][-1]
    )
    assert mechanical_energy(model, q, qd) == pytest.approx(
        coarse["mechanical_energy_j"][0]
    )


def test_committed_forward_evidence_is_complete_and_bounded() -> None:
    record = json.loads(
        (DATA_DIR / "articulated_forward_contact.json").read_text(encoding="utf-8")
    )
    assert record["schema_version"] == "articulated-forward-contact/v1"
    assert record["design"]["engine_names"] == ["mujoco", "pinocchio"]
    assert record["design"]["bounded_horizon_s"] > 0.0
    assert record["design"]["unilateral_collision_contact"] is False
    assert record["results"]["all_registered_gates_passed"] is True
    assert record["results"]["refinement_direction_passed"] is True
    assert record["claim_boundary"]["human_transfer_or_strategy"] == "untested"


def test_committed_forward_arrays_are_finite_and_typed() -> None:
    with np.load(DATA_DIR / "articulated_forward_contact.npz") as arrays:
        assert arrays["engine_names"].tolist() == ["mujoco", "pinocchio"]
        assert arrays["variant_names"].tolist()[0] == "nominal"
        assert arrays["state_case_index"].size > 0
        assert arrays["retained"].dtype == np.bool_
        for key in arrays.files:
            if arrays[key].dtype.kind in "fc":
                assert np.all(np.isfinite(arrays[key])), key
