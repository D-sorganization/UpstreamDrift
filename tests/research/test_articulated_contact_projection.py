from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from scripts.research.proximal_distal_energy.articulated_contact_projection import (
    ArticulatedContactProjectionConfig,
    evaluate_contact_projection,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

pytestmark = pytest.mark.scientific

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def test_projection_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="contact_stiffness"):
        ArticulatedContactProjectionConfig(contact_stiffness=0.0)
    with pytest.raises(ValueError, match="club_translation_perturbation_m"):
        ArticulatedContactProjectionConfig(club_translation_perturbation_m=np.inf)
    with pytest.raises(ValueError, match="acceleration_relative_tolerance"):
        ArticulatedContactProjectionConfig(acceleration_relative_tolerance=-1.0)


def test_zero_preload_and_perturbed_contact_projection_close_power() -> None:
    model, metadata = build_subject_scaled_model(default_synthetic_profiles()[0])
    with np.load(
        ROOT
        / "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz"
    ) as source:
        q = np.asarray(source["solution_q"][0, 0], dtype=float)
        grip_span_m = float(source["case_grip_span_m"][0])
    qd = np.zeros(model.nq)
    zero = evaluate_contact_projection(
        model,
        q,
        qd,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
        perturb_contact=False,
    )
    perturbed = evaluate_contact_projection(
        model,
        q,
        qd,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
        perturb_contact=True,
    )

    assert zero.maximum_contact_force_n <= 1.0e-6
    assert perturbed.maximum_contact_force_n > 0.0
    assert perturbed.action_reaction_residual_n <= 1.0e-12
    assert abs(perturbed.virtual_power_residual_w) <= 1.0e-12
    assert perturbed.contact_dissipation_power_w <= 0.0
    assert abs(perturbed.coincident_force_couple_nm) <= 1.0e-12
    assert perturbed.reversed_couple_sign_residual_nm <= 1.0e-12


def test_committed_projection_evidence_is_complete_and_bounded() -> None:
    import json

    record = json.loads(
        (DATA_DIR / "articulated_contact_projection.json").read_text(encoding="utf-8")
    )
    assert record["schema_version"] == "articulated-contact-projection/v1"
    assert record["design"]["state_count"] == 234
    assert record["design"]["forward_steps"] == 0
    assert record["results"]["failed_state_count"] == 0
    assert record["results"]["all_registered_gates_passed"] is True
    assert record["claim_boundary"]["forward_trajectory"] == "not_executed"
    assert record["claim_boundary"]["human_transfer_or_strategy"] == "untested"


def test_committed_projection_arrays_are_finite() -> None:
    with np.load(DATA_DIR / "articulated_contact_projection.npz") as arrays:
        assert arrays["initial_acceleration"].shape == (18, 13, 2, 20)
        assert arrays["acceleration_relative_error"].shape == (18, 13)
        assert arrays["contact_dissipation_power_w"].shape == (18, 13)
        assert arrays["engine_names"].tolist() == ["mujoco", "pinocchio"]
        for key in arrays.files:
            if arrays[key].dtype.kind in "fc":
                assert np.all(np.isfinite(arrays[key])), key
