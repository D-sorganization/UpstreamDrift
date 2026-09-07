"""Contracts for the momentum-transfer geometry atlas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.momentum_geometry_atlas import (
    bilateral_force_couple,
    distributed_contact_couple,
    force_velocity_projection,
    relative_link_gates,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _evidence() -> tuple[dict, dict[str, np.ndarray]]:
    record = json.loads((DATA / "momentum_geometry_atlas.json").read_text())
    with np.load(DATA / record["array_artifact"], allow_pickle=False) as artifact:
        arrays = {name: artifact[name].copy() for name in artifact.files}
    return record, arrays


def test_force_velocity_projection_has_exact_null_and_sign_reversal() -> None:
    assert force_velocity_projection(40.0, 3.0, 0.0) == pytest.approx(120.0)
    assert force_velocity_projection(40.0, 3.0, np.pi / 2) == pytest.approx(
        0.0, abs=1e-12
    )
    assert force_velocity_projection(40.0, 3.0, np.pi) == pytest.approx(-120.0)


def test_relative_link_gates_are_orthogonal_and_have_distinct_zeros() -> None:
    angles = np.linspace(-np.pi, np.pi, 401)
    tangential, centripetal = relative_link_gates(angles)

    assert np.max(np.abs(tangential**2 + centripetal**2 - 1.0)) < 1e-12
    assert tangential[200] == pytest.approx(1.0)
    assert centripetal[200] == pytest.approx(0.0, abs=1e-12)
    assert tangential[300] == pytest.approx(0.0, abs=1e-12)
    assert centripetal[300] == pytest.approx(-1.0)


def test_bilateral_couple_obeys_zero_common_mode_and_reversal_controls() -> None:
    axis = np.array([1.0, 0.0, 0.0])
    transverse = np.array([0.0, 1.0, 0.0])
    baseline = bilateral_force_couple(0.24, axis, 50.0 * transverse)
    coincident = bilateral_force_couple(0.0, axis, 50.0 * transverse)
    reversed_arm = bilateral_force_couple(-0.24, axis, 50.0 * transverse)
    axial = bilateral_force_couple(0.24, axis, 50.0 * axis)

    assert baseline[2] == pytest.approx(12.0)
    assert np.linalg.norm(coincident) == pytest.approx(0.0)
    assert reversed_arm == pytest.approx(-baseline)
    assert np.linalg.norm(axial) == pytest.approx(0.0)


def test_generated_atlas_is_complete_frame_invariant_and_model_bounded() -> None:
    record, arrays = _evidence()

    assert record["registered_before_preferred_result"] is True
    assert record["negative_controls"]["maximum_null_residual"] < 1e-12
    assert record["negative_controls"]["maximum_reversal_residual"] < 1e-12
    assert record["frame_audit"]["maximum_power_residual_w"] < 1e-12
    assert record["claim_status"]["universal_human_geometry"] == "untested"
    assert (
        record["claim_status"]["force_magnitude_alone_determines_transfer"]
        == "rejected"
    )
    assert (
        record["cross_tier_controls"]["moving_base_planar"]["coincident_couple_nm"]
        < 1e-12
    )
    assert (
        record["cross_tier_controls"]["spatial_two_engine"]["coincident_couple_nm"]
        < 1e-12
    )
    assert (
        record["cross_tier_controls"]["spatial_two_engine"]["reversal_residual_nm"]
        < 1e-12
    )
    assert set(record["tier_coverage"]) >= {
        "analytical",
        "fixed_hub_planar",
        "moving_base_two_hand",
        "spatial_forward_contact",
        "subject_scaled_articulated_contact",
        "distributed_club",
        "held_out_measurement",
        "subject_scaled",
        "governed_human",
    }
    assert np.all(np.isfinite(arrays["couple_normalized_nm"]))
    source = ROOT / "scripts/research/proximal_distal_energy/momentum_geometry_atlas.py"
    runner = (
        ROOT / "scripts/research/proximal_distal_energy/run_momentum_geometry_atlas.py"
    )
    expected = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source, runner)
    }
    assert record["source_sha256"] == expected


def test_distributed_stations_reduce_to_the_ideal_bilateral_couple() -> None:
    axis = np.array([1.0, 0.0, 0.0])
    transverse = np.array([0.0, 50.0, 0.0])
    stations = np.array([-0.015, 0.0, 0.015])
    offsets = np.concatenate((0.12 + stations, -0.12 + stations))
    forces = np.vstack(
        (np.tile(transverse / 3.0, (3, 1)), np.tile(-transverse / 3.0, (3, 1)))
    )

    distributed = distributed_contact_couple(offsets, axis, forces)

    assert distributed == pytest.approx(
        bilateral_force_couple(0.24, axis, transverse), abs=1e-13
    )


def test_distributed_couple_nulls_and_reverses_with_declared_geometry() -> None:
    axis = np.array([1.0, 0.0, 0.0])
    transverse = np.array([0.0, 25.0, 0.0])
    offsets = np.array([0.12, -0.12])
    opposed = np.vstack((transverse, -transverse))
    common_mode = np.vstack((transverse, transverse))
    axial = np.vstack((np.array([30.0, 0.0, 0.0]), np.array([-30.0, 0.0, 0.0])))

    baseline = distributed_contact_couple(offsets, axis, opposed)

    assert np.linalg.norm(
        distributed_contact_couple(offsets, axis, common_mode)
    ) == pytest.approx(0.0, abs=1e-15)
    assert np.linalg.norm(
        distributed_contact_couple(np.zeros(2), axis, opposed)
    ) == pytest.approx(0.0, abs=1e-15)
    assert np.linalg.norm(
        distributed_contact_couple(offsets, axis, axial)
    ) == pytest.approx(0.0, abs=1e-15)
    assert distributed_contact_couple(-offsets, axis, opposed) == pytest.approx(
        -baseline
    )


def test_distributed_contact_couple_rejects_malformed_inputs() -> None:
    axis = np.array([1.0, 0.0, 0.0])
    forces = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])

    with pytest.raises(ValueError, match="offsets must be"):
        distributed_contact_couple(np.array([0.1]), axis, forces)
    with pytest.raises(ValueError, match="separation axis must have shape"):
        distributed_contact_couple(np.array([0.1, -0.1]), np.array([1.0, 0.0]), forces)
    with pytest.raises(ValueError, match="must be finite"):
        distributed_contact_couple(np.array([0.1, np.nan]), axis, forces)
    with pytest.raises(ValueError, match="nonzero length"):
        distributed_contact_couple(np.array([0.1, -0.1]), np.zeros(3), forces)


def test_atlas_covers_the_subject_scaled_distributed_and_measurement_tiers() -> None:
    record, arrays = _evidence()
    controls = record["cross_tier_controls"]

    subject_scaled = controls["subject_scaled_articulated_contact"]
    assert subject_scaled["closed_configuration_count"] == 234
    assert subject_scaled["closed_maximum_contact_error_m"] < 1e-6
    assert subject_scaled["prescribed_maximum_contact_error_m"] > 0.1
    assert subject_scaled["couple_per_span_invariance_residual"] < 1e-12

    distributed = controls["distributed_club"]
    assert distributed["coincident_station_couple_residual_nm"] == 0.0
    assert distributed["reversed_station_couple_sign_residual_nm"] == 0.0
    assert distributed["grip_trajectory_count"] == 576
    assert distributed["shaft_trajectory_count"] == 384

    measurement = controls["held_out_measurement"]
    assert measurement["net_wrench_only_qualification"] == "fails_by_structure"
    assert measurement["net_wrench_only_normalized_net_wrench_rmse"] < 1e-12
    assert measurement["net_wrench_only_allocation_rmse_n"] > 1.0

    assert record["negative_controls"]["maximum_station_count_couple_deviation_nm"] < (
        1e-12
    )
    assert arrays["distributed_couple_normalized_nm"].shape == (
        arrays["station_count_per_hand"].size,
        arrays["signed_grip_separation_m"].size,
    )
    assert (
        np.max(
            np.abs(
                arrays["distributed_couple_normalized_nm"]
                - arrays["point_pair_couple_normalized_nm"]
            )
        )
        < 1e-12
    )


def test_atlas_does_not_promote_linked_tiers_into_human_or_calibrated_claims() -> None:
    record, _ = _evidence()

    assert (
        record["claim_status"]["station_spread_changes_couple_at_fixed_first_moment"]
        == "rejected"
    )
    assert (
        record["claim_status"]["net_wrench_identifies_bilateral_contact_geometry"]
        == "rejected"
    )
    assert (
        record["claim_status"]["calibrated_subject_scaled_contact_geometry"]
        == "untested"
    )
    assert record["tier_coverage"]["governed_human"].startswith("open_")
    assert record["tier_coverage"]["subject_scaled"].startswith("open_")
    assert any(
        "synthetic sensor qualification" in item for item in record["limitations"]
    )
    assert any("not calibrated anatomy" in item for item in record["limitations"])


pytestmark = pytest.mark.scientific
