"""Unit tests for bare-club vs welded-hand spatial inertia partitioning (NSX-01, #12550)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.biomechanics.club_solids_partition import (
    BARE_CLUB_SOLID_NAMES,
    WELDED_HAND_SOLID_NAMES,
    ClubSolidsPartition,
    SolidSpatialInertia,
    bare_club_kinematics_from_body_frame,
    club_solids_partition,
)

pytestmark = pytest.mark.unit

MODELS_DIR = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
DRIVER_SPEC_PATH = MODELS_DIR / "full_body_spec_anthro_driver.json"
IRON7_SPEC_PATH = MODELS_DIR / "full_body_spec_anthro_iron7.json"


@pytest.fixture(scope="module")
def driver_spec() -> dict:
    return json.loads(DRIVER_SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def iron7_spec() -> dict:
    return json.loads(IRON7_SPEC_PATH.read_text(encoding="utf-8"))


def _hand_aggregate(solids: list[dict]) -> tuple[float, np.ndarray, np.ndarray]:
    """Independent reference aggregation in the club body frame."""
    placed = []
    for solid in solids:
        t = np.asarray(solid["placement"], dtype=np.float64)
        rot = t[:3, :3]
        pos = t[:3, 3]
        mass = float(solid["mass_kg"])
        com = rot @ np.asarray(solid["com_m"], dtype=np.float64) + pos
        inertia = rot @ np.asarray(solid["inertia_com_kg_m2"], dtype=np.float64) @ rot.T
        placed.append((mass, com, inertia))
    total_mass = sum(m for m, _, _ in placed)
    com_total = sum(m * c for m, c, _ in placed) / total_mass
    inertia_total = np.zeros((3, 3), dtype=np.float64)
    for m, c, inertia in placed:
        d = c - com_total
        inertia_total += inertia + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return total_mass, com_total, inertia_total


def test_driver_partition_matches_hand_calculation(driver_spec: dict) -> None:
    partition = club_solids_partition(driver_spec)
    assert isinstance(partition, ClubSolidsPartition)

    bare, hands = partition
    assert bare is partition.bare_club_inertia
    assert hands == partition.welded_hand_inertias

    assert bare.mass_kg == pytest.approx(0.313, abs=1e-12)
    hand_mass = sum(h.mass_kg for h in hands)
    assert hand_mass == pytest.approx(1.1082715706945232, abs=1e-12)
    assert partition.hand_mass_kg == pytest.approx(1.1082715706945232, abs=1e-12)
    assert partition.composite_inertia.mass_kg == pytest.approx(
        0.313 + 1.1082715706945232, abs=1e-12
    )

    club_body = next(
        b for b in driver_spec["bodies"] if b["name"].endswith("Clubface Vector")
    )
    bare_raw = [
        s
        for s in club_body["solids"]
        if s["name"].rsplit("/", 1)[-1] in BARE_CLUB_SOLID_NAMES
    ]
    hand_raw = [
        s
        for s in club_body["solids"]
        if s["name"].rsplit("/", 1)[-1] in WELDED_HAND_SOLID_NAMES
    ]
    assert len(bare_raw) == 3
    assert len(hand_raw) == 3

    expected_mass, expected_com, expected_inertia = _hand_aggregate(bare_raw)
    assert bare.mass_kg == pytest.approx(expected_mass, abs=1e-12)
    np.testing.assert_allclose(bare.com_body_m, expected_com, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        bare.inertia_com_body_kg_m2, expected_inertia, rtol=0, atol=1e-12
    )

    comp_mass, comp_com, comp_inertia = _hand_aggregate(club_body["solids"])
    assert partition.composite_inertia.mass_kg == pytest.approx(comp_mass, abs=1e-12)
    np.testing.assert_allclose(
        partition.composite_inertia.com_body_m, comp_com, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        partition.composite_inertia.inertia_com_body_kg_m2,
        comp_inertia,
        rtol=0,
        atol=1e-12,
    )


def test_iron7_partition_matches_hand_calculation(iron7_spec: dict) -> None:
    partition = club_solids_partition(iron7_spec)
    bare = partition.bare_club_inertia
    assert bare.mass_kg == pytest.approx(0.428, abs=1e-12)
    assert partition.hand_mass_kg == pytest.approx(1.1082715706945232, abs=1e-12)
    assert partition.composite_inertia.mass_kg == pytest.approx(
        0.428 + 1.1082715706945232, abs=1e-12
    )

    club_body = next(
        b for b in iron7_spec["bodies"] if b["name"].endswith("Clubface Vector")
    )
    bare_raw = [
        s
        for s in club_body["solids"]
        if s["name"].rsplit("/", 1)[-1] in BARE_CLUB_SOLID_NAMES
    ]
    _, expected_com, expected_inertia = _hand_aggregate(bare_raw)
    np.testing.assert_allclose(bare.com_body_m, expected_com, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        bare.inertia_com_body_kg_m2, expected_inertia, rtol=0, atol=1e-12
    )


def test_dbc_postcondition_rejects_mismatched_total_club_mass(
    driver_spec: dict,
) -> None:
    corrupted = copy.deepcopy(driver_spec)
    corrupted["club"]["total_mass_kg"] = 1.4212715706945233
    with pytest.raises(ValueError, match="total_mass_kg"):
        club_solids_partition(corrupted)


def test_dbc_rejects_unknown_solid_provenance(driver_spec: dict) -> None:
    corrupted = copy.deepcopy(driver_spec)
    club_body = next(
        b for b in corrupted["bodies"] if b["name"].endswith("Clubface Vector")
    )
    club_body["solids"].append(
        {
            "name": "GolfSwing3D_Kinetic/Club/MysteryCounterweight",
            "mass_kg": 0.05,
            "com_m": [0.0, 0.0, 0.0],
            "inertia_com_kg_m2": np.eye(3).tolist(),
            "placement": np.eye(4).tolist(),
        }
    )
    with pytest.raises(ValueError, match="MysteryCounterweight"):
        club_solids_partition(corrupted)


def test_dbc_rejects_invalid_inputs_and_missing_sections(
    driver_spec: dict,
) -> None:
    with pytest.raises(TypeError, match="Mapping"):
        club_solids_partition(["not_a_mapping"])  # type: ignore[arg-type]

    no_club = copy.deepcopy(driver_spec)
    del no_club["club"]
    with pytest.raises(ValueError, match="club"):
        club_solids_partition(no_club)

    with pytest.raises(ValueError, match="mass_kg"):
        SolidSpatialInertia("bad", 0.0, (0.0, 0.0, 0.0), np.eye(3))


def test_bare_club_kinematics_from_body_frame_transforms_to_world(
    driver_spec: dict,
) -> None:
    partition = club_solids_partition(driver_spec)
    theta = 0.6
    rot_wb = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    p_b = np.array([0.25, -0.4, 1.1], dtype=np.float64)
    v_b = np.array([1.2, -0.5, 0.3], dtype=np.float64)
    omega = np.array([2.0, -3.0, 4.0], dtype=np.float64)
    a_b = np.array([5.0, -6.0, 7.0], dtype=np.float64)
    alpha = np.array([-8.0, 9.0, -10.0], dtype=np.float64)
    gravity = np.array([0.0, 0.0, -9.80665], dtype=np.float64)

    kin = bare_club_kinematics_from_body_frame(
        partition,
        body_position_world_m=p_b,
        body_rotation_world=rot_wb,
        body_linear_velocity_world_m_s=v_b,
        body_angular_velocity_world_rad_s=omega,
        body_linear_acceleration_world_m_s2=a_b,
        body_angular_acceleration_world_rad_s2=alpha,
        gravity_m_s2=gravity,
    )

    c_bare_b = np.asarray(partition.bare_club_inertia.com_body_m, dtype=np.float64)
    r_w = rot_wb @ c_bare_b
    expected_com_w = p_b + r_w
    expected_acc_w = a_b + np.cross(alpha, r_w) + np.cross(omega, np.cross(omega, r_w))
    expected_inertia_w = (
        rot_wb @ partition.bare_club_inertia.inertia_com_body_kg_m2 @ rot_wb.T
    )

    assert kin["mass_kg"] == pytest.approx(0.313, abs=1e-12)
    np.testing.assert_allclose(kin["com_m"], expected_com_w, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        kin["com_acceleration_m_s2"], expected_acc_w, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        kin["inertia_world_kg_m2"], expected_inertia_w, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(kin["angular_velocity_rad_s"], omega, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        kin["angular_acceleration_rad_s2"], alpha, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(kin["gravity_m_s2"], gravity, rtol=0, atol=1e-12)

    with pytest.raises(ValueError, match="orthonormal"):
        bare_club_kinematics_from_body_frame(
            partition,
            body_position_world_m=p_b,
            body_rotation_world=2.0 * np.eye(3),
            body_linear_velocity_world_m_s=v_b,
            body_angular_velocity_world_rad_s=omega,
            body_linear_acceleration_world_m_s2=a_b,
            body_angular_acceleration_world_rad_s2=alpha,
            gravity_m_s2=gravity,
        )
