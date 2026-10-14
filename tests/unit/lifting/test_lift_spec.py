"""Conformance of UpstreamDrift's lift definitions to the parity standard (#11743).

The vendored ``biomech_parity_standard.json`` (Repository_Management v2.x) is
the single source of truth for each lift's ordered named phases, root joint
and start/catch/lockout via-poses.  These tests pin that UpstreamDrift reads
them from the standard through :mod:`lift_spec` instead of hard-coding them.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from src.shared.python.lifting import lift_spec
from src.shared.python.lifting.pack_audit import analysis
from src.shared.python.lifting.pack_audit.names import CANONICAL_COORDINATES, LIFTS

_CANONICAL = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "shared"
    / "python"
    / "lifting"
    / "_canonical"
)
_STANDARD = _CANONICAL / "biomech_parity_standard.json"

# Ordered phase names of standard v2.1.0 (Repository_Management#2025/#2077).
EXPECTED_PHASES: dict[str, tuple[str, ...]] = {
    "squat": ("standing", "descent", "bottom", "ascent", "lockout"),
    "deadlift": ("setup", "first_pull", "knee_pass", "lockout", "descent"),
    "bench_press": ("unrack", "descent", "chest_touch", "press", "lockout"),
    "snatch": (
        "setup",
        "first_pull",
        "second_pull",
        "turnover",
        "catch",
        "recovery",
    ),
    "clean_and_jerk": (
        "setup",
        "first_pull",
        "second_pull",
        "rack_catch",
        "rack_recovery",
        "jerk_dip_drive",
        "overhead_catch",
        "overhead_recovery",
    ),
}


def _raw() -> dict:
    return json.loads(_STANDARD.read_text(encoding="utf-8"))


def test_vendored_standard_matches_its_manifest_digest() -> None:
    manifest = json.loads((_CANONICAL / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "Repository_Management/shared_scripts/model_parity"
    want = manifest["files"]["biomech_parity_standard.json"]
    assert hashlib.sha256(_STANDARD.read_bytes()).hexdigest() == want


def test_vendored_standard_is_v2_with_named_phases() -> None:
    assert int(lift_spec.standard_version().split(".")[0]) >= 2


@pytest.mark.parametrize("lift", LIFTS)
def test_phase_names_match_the_standard_in_order(lift: str) -> None:
    raw = [p["name"] for p in _raw()["exercises"][lift]["phases"]]
    assert lift_spec.lift_phase_names(lift) == tuple(raw)
    assert lift_spec.lift_phase_names(lift) == EXPECTED_PHASES[lift]


@pytest.mark.parametrize("lift", LIFTS)
def test_phases_carry_definitions_and_increasing_fractions(lift: str) -> None:
    phases = lift_spec.lift_phases(lift)
    assert len(phases) == _raw()["exercises"][lift]["phase_count"]
    assert all(p.definition for p in phases)
    fractions = [p.t_frac for p in phases]
    assert fractions[0] == 0.0
    assert all(b > a for a, b in zip(fractions, fractions[1:], strict=False))
    assert fractions[-1] < 1.0


def test_root_joint_policy() -> None:
    assert lift_spec.lift_root_joint("bench_press") == "weld"
    for lift in LIFTS:
        if lift != "bench_press":
            assert lift_spec.lift_root_joint(lift) == "free"


@pytest.mark.parametrize("lift", LIFTS)
def test_via_poses_bind_to_a_named_phase(lift: str) -> None:
    names = lift_spec.lift_phase_names(lift)
    poses = lift_spec.lift_via_pose_names(lift)
    assert "start" in poses and "lockout" in poses
    assert lift_spec.lift_via_pose(lift, "start").phase == names[0]
    for pose in poses:
        assert lift_spec.lift_via_pose(lift, pose).phase in names


@pytest.mark.parametrize("lift", LIFTS)
def test_via_pose_coordinates_are_canonical_and_within_limits(lift: str) -> None:
    limits = {}
    for coord in _raw()["coordinates"]:
        lo, hi = (math.radians(v) for v in coord["limits_deg"])
        for side in _raw()["sides"]:
            limits[coord["name"].format(side=side)] = (lo, hi)
    for pose in lift_spec.lift_via_pose_names(lift):
        via = lift_spec.lift_via_pose(lift, pose)
        assert via.coordinates, f"{lift}/{pose} has no coordinates"
        assert set(via.coordinates) <= set(CANONICAL_COORDINATES)
        for name, angle in via.coordinates.items():
            lo, hi = limits[name]
            assert lo - 1e-9 <= angle <= hi + 1e-9, f"{lift}/{pose}:{name}"
        assert len(via.root_position_m) == 3
        assert len(via.root_orientation_rpy_rad) == 3
        assert len(via.bar_position_m) == 3


def test_unknown_lift_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown lift"):
        lift_spec.lift_phase_names("gait")
    with pytest.raises(ValueError, match="unknown lift"):
        lift_spec.lift_phases("curl")


def test_unknown_via_pose_is_rejected() -> None:
    with pytest.raises(ValueError, match="no via-pose"):
        lift_spec.lift_via_pose("deadlift", "catch")


def test_phase_rows_compare_pack_names_with_the_standard() -> None:
    names = list(EXPECTED_PHASES["deadlift"])
    phase = {
        "bilateral": True,
        "unmapped_keys": [],
        "interpreted_keys": [],
        "canonical_targets": {},
    }
    receipt = {
        "standard_reference": {"phase_count": {"deadlift": 5}},
        "results": {
            "deadlift": {
                "mujoco": {
                    "coordinates": [],
                    "phases": [{**phase, "name": n} for n in names],
                },
                "drake": {
                    "coordinates": [],
                    "phases": [{**phase, "name": n} for n in ("start", "pull")],
                },
            }
        },
    }
    rows = {r["engine"]: r for r in analysis.phase_rows(receipt)}
    assert rows["mujoco"]["standard_names"] == names
    assert rows["mujoco"]["names_match"] is True
    assert rows["drake"]["names_match"] is False
