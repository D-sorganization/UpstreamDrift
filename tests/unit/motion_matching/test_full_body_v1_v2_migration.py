"""v1 -> v2 trunk coordinate migration by rotation composition (HRG-D1, #13170).

``TorsoInput = lumbar + thoracic axial`` is rejected as a general mapping; the
migration matches the composed pelvis-to-thorax rotation and reports the
residual and the non-uniqueness of the six-coordinate solution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching import full_body_migration as mig
from src.shared.python.motion_matching import full_body_v2 as v2

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
PELVIS = "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/LowerTorso"
THORAX = "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/COMRod"


@pytest.fixture(scope="module")
def specs() -> tuple[dict, dict]:
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    doc = v2.derive_full_body_v2_spec(
        v1, v2.segment_boundary_locations(v1), provenance="HRG-D1 unit fixture"
    )
    return v1, doc


@pytest.fixture(scope="module")
def migrator(specs) -> mig.TrunkMigrator:
    return mig.TrunkMigrator(*specs)


def _relative_rotation(spec: dict, q: dict) -> np.ndarray:
    poses = spec_body_poses(spec, q)
    return (np.linalg.inv(poses[PELVIS]) @ poses[THORAX])[:3, :3]


def _v1_q(torso: float, sx: float, sy: float) -> dict[str, float]:
    return {
        "TorsoInput": np.radians(torso),
        "SpineInputX": np.radians(sx),
        "SpineInputY": np.radians(sy),
        "LEInput": np.radians(-20.0),
        "hip_flexion_r": np.radians(15.0),
    }


def test_zero_maps_to_zero(migrator) -> None:
    frame = migrator.migrate_frame(_v1_q(0.0, 0.0, 0.0))
    for name in v2.V2_TRUNK.coordinates:
        assert frame.q_v2[name] == pytest.approx(0.0, abs=1e-9)
    assert frame.rotation_residual_rad < 1e-9


def test_composed_rotation_is_matched_and_residual_reported(specs, migrator) -> None:
    v1, doc = specs
    q1 = _v1_q(-58.0, 8.0, 25.0)  # top-of-backswing scale twist
    frame = migrator.migrate_frame(q1)
    assert frame.converged
    assert frame.rotation_residual_rad < 1e-8
    target = _relative_rotation(v1, q1)
    got = _relative_rotation(doc, frame.q_v2)
    angle = np.linalg.norm(Rotation.from_matrix(target.T @ got).as_rotvec())
    assert angle == pytest.approx(frame.rotation_residual_rad, abs=1e-10)
    # Non-trunk coordinates carry over by name, untouched.
    assert frame.q_v2["LEInput"] == q1["LEInput"]
    assert frame.q_v2["hip_flexion_r"] == q1["hip_flexion_r"]
    assert not {"TorsoInput", "SpineInputX", "SpineInputY"} & set(frame.q_v2)
    # Two pivots instead of one: the shoulder hub moves, and that is reported.
    assert np.isfinite(frame.hub_position_residual_m)
    assert frame.hub_position_residual_m > 1e-4


def test_non_uniqueness_is_reported_with_witness_solutions(migrator) -> None:
    frame = migrator.migrate_frame(_v1_q(-40.0, 5.0, 20.0))
    assert frame.constraint_rank == 3
    assert frame.null_space_dimension == 3
    # Exact alternatives (all lumbar, all thoracic) satisfy the same rotation
    # yet differ from the chosen split: the migration is not unique.
    assert frame.alternative_residuals_rad[0] < 1e-8
    assert frame.alternative_residuals_rad[1] < 1e-8
    assert frame.alternative_spread_rad > np.radians(10.0)
    assert frame.distribution_source == "prior:bruno2015_regional@1"


def test_prior_split_follows_bruno2015_regional_for_pure_axial_twist(migrator) -> None:
    frame = migrator.migrate_frame(_v1_q(-30.0, 0.0, 0.0))
    lumbar = frame.q_v2["lumbar_axial_rotation"]
    thoracic = frame.q_v2["thoracic_axial_rotation"]
    assert lumbar + thoracic == pytest.approx(np.radians(-30.0), abs=1e-8)
    assert lumbar / (lumbar + thoracic) == pytest.approx(0.17, abs=1e-6)
    assert frame.prior_deviation_rad < 1e-8


def test_rejected_axial_sum_mapping_is_not_exact_in_general(migrator) -> None:
    """Splitting each v1 angle per axis (TorsoInput = lumbar + thoracic axial)."""
    pure = migrator.migrate_frame(_v1_q(-30.0, 0.0, 0.0))
    assert pure.rejected_axial_sum_residual_rad < 1e-9  # single axis: it works
    mixed = migrator.migrate_frame(_v1_q(-58.0, 8.0, 25.0))
    assert mixed.rejected_axial_sum_residual_rad > np.radians(1.0)
    assert mixed.rotation_residual_rad < 1e-8


def test_soft_limit_excess_is_reported_not_enforced(migrator) -> None:
    frame = migrator.migrate_frame(_v1_q(-90.0, 0.0, 0.0))
    assert frame.rotation_residual_rad < 1e-8
    # 83 % of 90 deg is 74.7 deg of thoracic axial rotation against +-50.
    assert frame.soft_limit_excess_deg["thoracic_axial_rotation"] == pytest.approx(
        0.83 * 90.0 - 50.0, abs=1e-5
    )
    assert "lumbar_flexion" not in frame.soft_limit_excess_deg


def test_trajectory_report_summarises_every_frame(migrator, specs) -> None:
    v1, _ = specs
    order = v1["coordinate_order"]
    n = 12
    q = np.zeros((n, len(order)))
    t = np.linspace(0.0, 1.0, n)
    q[:, order.index("TorsoInput")] = np.radians(-58.0) * np.sin(np.pi * t)
    q[:, order.index("SpineInputX")] = np.radians(8.0) * t
    q[:, order.index("SpineInputY")] = np.radians(25.0) * (1 - t)
    report = migrator.migrate_trajectory(q, order)
    assert report.q_v2.shape == (n, len(report.coordinate_order))
    assert report.rotation_residual_rad.shape == (n,)
    doc = report.as_document()
    assert doc["frames"] == n
    assert doc["max_rotation_residual_rad"] < 1e-8
    assert doc["constraint_rank"] == [3]
    assert doc["null_space_dimension"] == [3]
    assert doc["all_converged"] is True
    assert doc["source_spec_sha256"] == report.source_spec_sha256
    assert doc["max_rejected_axial_sum_residual_rad"] > np.radians(1.0)
    json.dumps(doc, allow_nan=False)


def test_preconditions(specs) -> None:
    v1, doc = specs
    with pytest.raises(ValueError, match="full-body-v1"):
        mig.TrunkMigrator(doc, doc)
    with pytest.raises(ValueError, match="full-body-v2"):
        mig.TrunkMigrator(v1, v1)
    other = dict(doc, source_spec_sha256="f" * 64)
    with pytest.raises(ValueError, match="source_spec_sha256"):
        mig.TrunkMigrator(v1, other)
    migrator = mig.TrunkMigrator(v1, doc)
    with pytest.raises(ValueError, match="finite"):
        migrator.migrate_frame({"TorsoInput": float("nan")})
    with pytest.raises(ValueError, match="unknown"):
        migrator.migrate_frame({"lumbar_flexion": 0.0})
    with pytest.raises(ValueError, match="shape"):
        migrator.migrate_trajectory(np.zeros((3, 2)), v1["coordinate_order"])
