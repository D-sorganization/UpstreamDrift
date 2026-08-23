"""Contracts for corner-consistent articulated atlas authority and models."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_atlas_authority import (
    ArticulatedAtlasAuthority,
    load_default_atlas_authority,
    scientific_model_sha256,
)
from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    finite_difference_kinematics,
)
from scripts.research.proximal_distal_energy.articulated_scaled_authority import (
    load_scaled_authority,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
pytestmark = pytest.mark.scientific


@pytest.fixture(scope="module")
def authority() -> ArticulatedAtlasAuthority:
    scaled = load_scaled_authority(
        DATA / "articulated_scaled_authority_nominal.json",
        DATA / "articulated_scaled_authority_nominal.npz",
    )
    return ArticulatedAtlasAuthority.from_scaled(scaled)


def test_authority_builds_the_registered_case_model(authority) -> None:
    model, metadata = authority.build_case_model(0)
    expected = default_synthetic_profiles()[int(authority.profile_index[0])]

    assert metadata["profile"]["height_m"] == expected.height_m
    assert metadata["profile"]["mass_kg"] == expected.mass_kg
    assert authority.validate_case_model(0, model, metadata) == scientific_model_sha256(
        model
    )
    assert authority.solution_q.shape == (18, 13, 20)
    assert authority.time_s.shape == (13,)


def test_authority_rejects_nominal_model_for_a_scaled_corner(authority) -> None:
    scaled = replace(
        authority,
        height_scale=1.10,
        body_mass_scale=1.15,
    )
    nominal_profile = default_synthetic_profiles()[int(authority.profile_index[0])]
    nominal_model, nominal_metadata = build_subject_scaled_model(nominal_profile)

    with pytest.raises(RuntimeError, match="authority/model scaling"):
        scaled.validate_case_model(0, nominal_model, nominal_metadata)


def test_authority_retains_and_rejects_infeasible_selected_state(authority) -> None:
    feasible = authority.feasible.copy()
    feasible[0, 4] = False
    failure_class = authority.failure_class.copy()
    failure_class[0, 4] = "joint_limit_failure"
    failed = replace(authority, feasible=feasible, failure_class=failure_class)

    assert failed.selected_failures() == (
        {"case_index": 0, "phase_index": 4, "failure_class": "joint_limit_failure"},
    )
    with pytest.raises(RuntimeError, match="selected authority states are infeasible"):
        failed.require_selected_feasible()

    model, metadata = failed.build_case_model(0)
    assert failed.validate_case_model(0, model, metadata) == scientific_model_sha256(
        model
    )
    assert failed.feasible_states((0,), (3, 4, 5)) == ((0, 3), (0, 5))
    with pytest.raises(
        RuntimeError,
        match="case=0, phase=4, failure=joint_limit_failure",
    ):
        failed.require_state_feasible(0, 4)


def test_authority_rejects_invalid_phase_access(authority) -> None:
    with pytest.raises(ValueError, match="phase_index"):
        authority.require_state_feasible(0, 13)
    with pytest.raises(ValueError, match="phase_index"):
        authority.feasible_states((0,), (-1,))


def test_authority_provenance_binds_scales_failures_and_models(authority) -> None:
    feasible = authority.feasible.copy()
    feasible[0, 12] = False
    failure_class = authority.failure_class.copy()
    failure_class[0, 12] = "ik_nonconvergence"
    corner = replace(
        authority,
        feasible=feasible,
        failure_class=failure_class,
        height_scale=0.90,
        body_mass_scale=0.85,
        joint_limit_scale=1.15,
        authority_sha256="a" * 64,
    )

    record = corner.provenance_record()

    assert record["authority_sha256"] == "a" * 64
    assert record["scales"] == {
        "height": 0.90,
        "body_mass": 0.85,
        "joint_limit": 1.15,
    }
    assert record["retained_failures"] == [
        {"case_index": 0, "phase_index": 12, "failure_class": "ik_nonconvergence"}
    ]
    assert set(record["model_sha256"]) == {"0", "8", "9", "17"}
    assert all(len(value) == 64 for value in record["model_sha256"].values())


def test_authority_rejects_unregistered_case_access(authority) -> None:
    with pytest.raises(ValueError, match="selected case"):
        authority.build_case_model(1)


def test_scientific_model_digest_ignores_last_bit_but_not_material_drift(
    authority,
) -> None:
    model, _ = authority.build_case_model(9)
    body = model.bodies[3]
    last_bit = replace(body, mass_kg=np.nextafter(body.mass_kg, np.inf))
    material = replace(body, mass_kg=body.mass_kg + 1e-10)
    last_bit_model = replace(
        model,
        bodies=(*model.bodies[:3], last_bit, *model.bodies[4:]),
    )
    material_model = replace(
        model,
        bodies=(*model.bodies[:3], material, *model.bodies[4:]),
    )

    assert scientific_model_sha256(last_bit_model) == scientific_model_sha256(model)
    assert scientific_model_sha256(material_model) != scientific_model_sha256(model)


def test_default_atlas_authority_uses_governed_structural_nominal() -> None:
    authority = load_default_atlas_authority()
    campaign = json.loads(
        (DATA / "articulated_structural_authority_campaign.json").read_text(
            encoding="utf-8"
        )
    )
    nominal = campaign["corners"][0]

    assert nominal["corner_id"] == "nominal"
    assert authority.authority_sha256 == nominal["authority_sha256"]
    assert authority.selected_failures() == ()
    assert authority.provenance_record()["scales"] == {
        "height": 1.0,
        "body_mass": 1.0,
        "joint_limit": 1.0,
    }


def test_resolved_state_uses_corner_model_and_phase_derivative() -> None:
    scaled = load_scaled_authority(
        DATA / "articulated_structural_authority_height_scale_high.json",
        DATA / "articulated_structural_authority_height_scale_high.npz",
    )
    authority = ArticulatedAtlasAuthority.from_scaled(scaled)
    state = authority.resolve_state(8, 6)
    expected_velocity, _ = finite_difference_kinematics(
        authority.solution_q[8], authority.time_s
    )

    assert state.case_index == 8
    assert state.phase_index == 6
    assert state.authority_sha256 == authority.authority_sha256
    assert state.model_metadata["profile"]["height_m"] == pytest.approx(
        default_synthetic_profiles()[int(authority.profile_index[8])].height_m * 1.10
    )
    np.testing.assert_array_equal(state.q, authority.solution_q[8, 6])
    np.testing.assert_allclose(state.qd, expected_velocity[6], rtol=0.0, atol=0.0)
    assert state.grip_span_m == pytest.approx(authority.grip_span_m[8])


def test_resolved_state_rejects_retained_infeasible_phase() -> None:
    scaled = load_scaled_authority(
        DATA / "articulated_structural_authority_height_scale_low.json",
        DATA / "articulated_structural_authority_height_scale_low.npz",
    )
    authority = ArticulatedAtlasAuthority.from_scaled(scaled)
    failure = authority.selected_failures()[0]

    with pytest.raises(RuntimeError, match="selected authority state is infeasible"):
        authority.resolve_state(
            int(failure["case_index"]),
            int(failure["phase_index"]),
        )
