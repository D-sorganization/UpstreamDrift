"""Contracts for the subject-scaled spatial contact-geometry atlas."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    SyntheticSubjectProfile,
    build_subject_scaled_model,
    contact_geometry_snapshot,
    run_subject_scaled_geometry_atlas,
)

pytestmark = pytest.mark.scientific
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"


def test_synthetic_subject_profile_fails_closed() -> None:
    with pytest.raises(ValueError, match="height_m"):
        SyntheticSubjectProfile("bad-height", 0.0, 70.0, "M")
    with pytest.raises(ValueError, match="mass_kg"):
        SyntheticSubjectProfile("bad-mass", 1.75, np.nan, "F")
    with pytest.raises(ValueError, match="sex"):
        SyntheticSubjectProfile("bad-sex", 1.75, 70.0, "unknown")
    with pytest.raises(ValueError, match="profile_id"):
        SyntheticSubjectProfile("", 1.75, 70.0, "M")


def test_subject_scaling_uses_canonical_anthropometrics() -> None:
    profile = SyntheticSubjectProfile("female-mid", 1.70, 65.0, "F")
    model, metadata = build_subject_scaled_model(profile)

    assert model.nq == 20
    assert len(model.canonical_hash) == 64
    assert metadata["estimator"] == "de_leva_1996"
    assert metadata["profile"] == {
        "profile_id": "female-mid",
        "height_m": 1.70,
        "mass_kg": 65.0,
        "sex": "F",
    }
    assert metadata["segment_lengths_m"]["upper_arm"] > 0.0
    assert metadata["segment_lengths_m"]["forearm"] > 0.0
    assert metadata["segment_lengths_m"]["hand"] > 0.0
    assert metadata["hand_contact_local_x_m"] > 0.0


def test_contact_geometry_snapshot_exposes_closure_and_rank() -> None:
    profile = SyntheticSubjectProfile("male-mid", 1.75, 75.0, "M")
    model, metadata = build_subject_scaled_model(profile)
    snapshot = contact_geometry_snapshot(
        model,
        time_s=0.20,
        grip_span_m=0.18,
        hand_contact_local_x_m=metadata["hand_contact_local_x_m"],
    )

    assert snapshot.hand_to_grip_distance_m.shape == (2,)
    assert np.all(snapshot.hand_to_grip_distance_m > 0.0)
    assert snapshot.constraint_jacobian.shape == (6, 20)
    assert snapshot.constraint_jacobian_rank == 6
    assert snapshot.constraint_jacobian_minimum_singular_value > 0.0
    assert snapshot.point_force_wrench_map_rank == 5
    assert snapshot.augmented_point_force_wrench_map_rank == 6
    assert snapshot.point_force_wrench_map_nullity == 1


def test_subject_scaled_atlas_retains_adverse_contact_result() -> None:
    record, arrays = run_subject_scaled_geometry_atlas()

    assert record["schema_version"] == "subject-scaled-spatial-geometry/v1"
    assert record["design"]["profile_count"] == 6
    assert record["design"]["grip_span_count"] == 3
    assert record["design"]["case_count"] == 18
    assert record["design"]["time_sample_count"] == 61
    assert record["geometry_tests"]["point_force_map_rank_values"] == [5]
    assert record["geometry_tests"]["augmented_map_rank_values"] == [6]
    assert record["geometry_tests"]["couple_per_span_invariance_residual"] < 1e-12
    assert record["closure_tests"]["minimum_hand_to_grip_distance_m"] > 0.01
    assert record["claim_status"]["subject_scaled_anatomical_contact"] == (
        "not_established_prescribed_state_fails_contact_closure"
    )
    assert record["claim_status"]["human_strategy"] == "untested"
    assert arrays["hand_to_grip_distance_m"].shape == (18, 61, 2)
    assert arrays["constraint_jacobian_singular_values"].shape == (18, 61, 6)
    assert np.all(np.isfinite(arrays["hand_to_grip_distance_m"]))


def test_committed_subject_scaled_atlas_matches_contract() -> None:
    record = json.loads(
        (DATA_DIR / "subject_scaled_spatial_geometry.json").read_text(encoding="utf-8")
    )
    with np.load(DATA_DIR / "subject_scaled_spatial_geometry.npz") as arrays:
        assert record["design"]["case_count"] == arrays["case_profile_index"].size
        assert arrays["hand_to_grip_distance_m"].shape[0] == 18
        assert np.all(np.isfinite(arrays["constraint_jacobian_condition_number"]))
    assert record["limitations"][0].startswith("Synthetic profiles")
    assert record["claim_status"]["human_or_coaching_inference"] == "unsupported"


def test_subject_scaled_evidence_sources_and_figure_are_current() -> None:
    record = json.loads(
        (DATA_DIR / "subject_scaled_spatial_geometry.json").read_text(encoding="utf-8")
    )
    for relative, expected in record["source_sha256"].items():
        assert (
            hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest() == expected
        )
    for suffix in ("pdf", "svg"):
        figure = (
            REPO_ROOT
            / "docs/research/proximal_distal_energy_transfer/figures"
            / f"fig_subject_scaled_spatial_geometry.{suffix}"
        )
        assert figure.stat().st_size > 5_000
