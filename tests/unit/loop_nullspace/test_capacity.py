"""Unit tests for NSX-04 canonical joint torque-capacity table and loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.loop_nullspace.capacity import (
    CapacitySlotEntry,
    CapacityVector,
    capacity_for,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVER_SPEC_PATH = (
    REPO_ROOT
    / "docs"
    / "development"
    / "full_body_models"
    / "full_body_spec_anthro_driver.json"
)
IRON_SPEC_PATH = (
    REPO_ROOT
    / "docs"
    / "development"
    / "full_body_models"
    / "full_body_spec_anthro_iron7.json"
)


def test_capacity_covers_all_44_coordinates_of_anthro_specs() -> None:
    """Every coordinate in the 44-coordinate driver and iron7 specs is covered."""
    driver_spec = json.loads(DRIVER_SPEC_PATH.read_text(encoding="utf-8"))
    iron_spec = json.loads(IRON_SPEC_PATH.read_text(encoding="utf-8"))

    coords_driver = driver_spec["coordinate_order"]
    coords_iron = iron_spec["coordinate_order"]
    assert len(coords_driver) == 44
    assert coords_driver == coords_iron

    cap = capacity_for(coords_driver, subject=driver_spec["subject"])
    assert isinstance(cap, CapacityVector)
    assert cap.coordinate_order == tuple(coords_driver)
    assert cap.tau_max_nm.shape == (44,)
    assert cap.tau_min_nm.shape == (44,)
    assert np.all(cap.tau_max_nm > 0.0)
    assert np.all(cap.tau_min_nm < 0.0)
    assert not cap.tau_max_nm.flags.writeable
    assert not cap.tau_min_nm.flags.writeable
    assert len(cap.entries) == 44
    assert all(isinstance(e, CapacitySlotEntry) for e in cap.entries)

    # Arm and torso slots must be VERIFIED
    verified_coords = {
        e.coordinate for e in cap.entries if e.verification_status == "VERIFIED"
    }
    expected_arm_torso = {
        "SpineInputX",
        "SpineInputY",
        "TorsoInput",
        "LEInput",
        "LFInput",
        "LSInputX",
        "LSInputY",
        "LSInputZ",
        "LWInputX",
        "LWInputY",
        "REInput",
        "RFInput",
        "RSInputX",
        "RSInputY",
        "RSInputZ",
        "RWInputX",
        "RWInputY",
    }
    assert expected_arm_torso.issubset(verified_coords)


def test_capacity_dbc_raises_listing_all_missing_slots() -> None:
    """Missing actuated coordinates trigger ValueError listing every missing slot."""
    with pytest.raises(ValueError) as exc_info:
        capacity_for(
            ["LEInput", "MissingJointAlpha", "REInput", "MissingJointBeta"],
            unactuated_coordinates=("UnactuatedRootX",),
        )
    msg = str(exc_info.value)
    assert "MissingJointAlpha" in msg
    assert "MissingJointBeta" in msg

    # Unactuated coordinates are exempt from the actuated coverage requirement
    cap = capacity_for(
        ["UnactuatedRootX", "LEInput", "REInput"],
        unactuated_coordinates=("UnactuatedRootX",),
    )
    assert cap.coordinate_order == ("LEInput", "REInput")


def test_anthropometric_scaling_rule_reproduces_cited_ratios() -> None:
    """Anthropometric scaling scales capacities by mass or mass*height relative to 75 kg / 1.78 m."""
    ref_cap = capacity_for(
        ["LEInput", "TorsoInput", "hip_flexion_r"],
        subject={"mass_kg": 75.0, "height_m": 1.78},
    )
    scaled_cap = capacity_for(
        ["LEInput", "TorsoInput", "hip_flexion_r"],
        subject={"mass_kg": 90.0, "height_m": 1.78},
    )
    mass_ratio = 90.0 / 75.0
    assert np.allclose(
        scaled_cap.tau_max_nm, ref_cap.tau_max_nm * mass_ratio, rtol=1e-12
    )
    assert np.allclose(
        scaled_cap.tau_min_nm, ref_cap.tau_min_nm * mass_ratio, rtol=1e-12
    )

    # Also support stature_m key from full_body_spec_anthro_driver.json
    stature_cap = capacity_for(
        ["LEInput"],
        subject={"mass_kg": 78.0, "stature_m": 1.71},
    )
    assert stature_cap.tau_max_nm[0] > 0.0


def test_assumed_and_unverified_slots_surface_explicit_warnings() -> None:
    """Uncited/auxiliary ASSUMED slots carry UNVERIFIED status and surface warnings."""
    driver_spec = json.loads(DRIVER_SPEC_PATH.read_text(encoding="utf-8"))
    cap = capacity_for(driver_spec["coordinate_order"])

    assumed_entries = [
        e
        for e in cap.entries
        if e.status == "ASSUMED" or e.verification_status == "UNVERIFIED"
    ]
    assert len(assumed_entries) > 0
    assert len(cap.warnings) >= len(assumed_entries)
    for entry in assumed_entries:
        assert entry.verification_status == "UNVERIFIED"
        assert entry.warning is not None and len(entry.warning) > 0
        assert any(entry.coordinate in w for w in cap.warnings)
