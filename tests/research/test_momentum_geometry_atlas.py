"""Contracts for the momentum-transfer geometry atlas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.momentum_geometry_atlas import (
    bilateral_force_couple,
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
        "subject_scaled",
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


pytestmark = pytest.mark.scientific
