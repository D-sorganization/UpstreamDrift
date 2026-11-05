"""SHOW-C1 (#14557): sourced angular-velocity limits for the impact envelope.

Peak segment / joint angular velocities for skilled male golfers (driver) were
read in the full text of each cited paper.  A limit built from an observed
group mean carries a stated ASSUMED margin, so it is never "VERIFIED" and a
binding on it is still surfaced as assumption-limited (D32).  Joints with no
source stay UNVERIFIED priors.
"""

from __future__ import annotations

import pytest

from src.shared.python.impact_optimization.physiological_envelope import (
    DEFAULT_LIMITS_PATH,
    load_limits,
)
import yaml

pytestmark = pytest.mark.unit

SOURCED = "SOURCED_WITH_ASSUMED_MARGIN"
MARGIN = 2.0

#: coordinate -> (limit deg/s, observed deg/s, citation id)
EXPECTED = {
    "HipInputZ": (1030.0, 515.0, "10.1371/journal.pone.0331051"),
    "TorsoInput": (1534.0, 767.0, "10.1080/02640410701373543"),
    "thoracic_axial_rotation": (1534.0, 767.0, "10.1080/02640410701373543"),
    "LSInputX": (2276.0, 1138.0, "10.1371/journal.pone.0331051"),
    "LSInputY": (2276.0, 1138.0, "10.1371/journal.pone.0331051"),
    "LSInputZ": (2276.0, 1138.0, "10.1371/journal.pone.0331051"),
    "RWInputX": (3062.0, 1531.0, "ISBS-2012-sweeney-wrist-kinematics-golf-drive"),
    "RWInputY": (3062.0, 1531.0, "ISBS-2012-sweeney-wrist-kinematics-golf-drive"),
    "LWInputX": (1338.0, 669.0, "ISBS-2012-sweeney-wrist-kinematics-golf-drive"),
    "LWInputY": (1338.0, 669.0, "ISBS-2012-sweeney-wrist-kinematics-golf-drive"),
}


def _entries() -> list[dict]:
    raw = yaml.safe_load(DEFAULT_LIMITS_PATH.read_text(encoding="utf-8"))
    return raw["velocity_limits_deg_s"]


def test_sourced_slots_use_the_observed_peak_times_the_stated_margin() -> None:
    limits = load_limits(DEFAULT_LIMITS_PATH)
    for name, (value, _observed, _cite) in EXPECTED.items():
        assert limits[name].value_deg_s == pytest.approx(value), name
        assert limits[name].verification_status == SOURCED, name


def test_every_sourced_entry_records_citation_observation_and_margin() -> None:
    sourced = [e for e in _entries() if e["verification_status"] == SOURCED]
    assert sourced
    for e in sourced:
        c = e["citation"]
        assert c["id"] and c["location"] and c["population"], e["coordinates"]
        assert c["margin"] == MARGIN
        assert float(e["value"]) == pytest.approx(c["observed_deg_s"] * c["margin"])


def test_citation_ids_match_the_expected_sources() -> None:
    by_coord = {n: e for e in _entries() for n in e["coordinates"]}
    for name, (_value, observed, cite) in EXPECTED.items():
        assert by_coord[name]["citation"]["id"] == cite, name
        assert by_coord[name]["citation"]["observed_deg_s"] == observed, name


def test_sourced_slots_stay_surfaced_as_assumed_not_verified() -> None:
    limits = load_limits(DEFAULT_LIMITS_PATH)
    assert all(v.verification_status != "VERIFIED" for v in limits.values())


@pytest.mark.parametrize(
    "name",
    ["lumbar_axial_rotation", "NeckInputZ", "knee_angle_r", "RSInputX", "SpineInputX"],
)
def test_joints_without_a_source_stay_unverified_priors(name: str) -> None:
    assert load_limits(DEFAULT_LIMITS_PATH)[name].verification_status == "UNVERIFIED"
