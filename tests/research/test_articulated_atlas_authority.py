"""Contracts for corner-consistent articulated atlas authority and models."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.articulated_atlas_authority import (
    ArticulatedAtlasAuthority,
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
    assert authority.validate_case_model(0, model, metadata) == model.canonical_hash
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
    assert failed.validate_case_model(0, model, metadata) == model.canonical_hash
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
