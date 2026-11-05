"""SHOW-C1 (#14557): sourced angular-velocity limits for the impact envelope.

Peak segment / joint angular velocities for skilled golfers (driver; Myers 2008 does not state sex) were
read in the full text of each cited paper.  A limit built from an observed
group mean is scaled to the 125 mph target (D38) and carries an ASSUMED margin, so it is never "VERIFIED" and a
binding on it is still surfaced as assumption-limited (D32).  Joints with no
source stay UNVERIFIED priors.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError

from src.shared.python.motion_matching.same_input import InputBundle
from src.shared.python.impact_optimization.physiological_envelope import (
    DEFAULT_LIMITS_PATH,
    load_limits,
)
import yaml

pytestmark = pytest.mark.unit

SOURCED = "SOURCED_WITH_ASSUMED_MARGIN"
MARGIN = 2.0
TARGET_MPH = 125.0

#: coordinate -> (limit deg/s, observed deg/s, citation id)
EXPECTED = {
    "HipInputZ": (1030.0, 515.0, "10.1371/journal.pone.0331051"),
    "TorsoInput": (1534.0, 767.0, "10.1080/02640410701373543"),
    "thoracic_axial_rotation": (1534.0, 767.0, "10.1080/02640410701373543"),
    "LSInputX": (2276.0, 1138.0, "10.1371/journal.pone.0331051"),
    "LSInputY": (2276.0, 1138.0, "10.1371/journal.pone.0331051"),
    "LSInputZ": (2276.0, 1138.0, "10.1371/journal.pone.0331051"),
    "RWInputX": (3186.0, 1531.0, "ISBS-2012-sweeney-wrist-kinematics-golf-drive"),
    "RWInputY": (3186.0, 1531.0, "ISBS-2012-sweeney-wrist-kinematics-golf-drive"),
    "LWInputX": (1613.0, 669.0, "ISBS-2012-sweeney-wrist-kinematics-golf-drive"),
    "LWInputY": (1613.0, 669.0, "ISBS-2012-sweeney-wrist-kinematics-golf-drive"),
}


def _entries() -> list[dict]:
    raw = yaml.safe_load(DEFAULT_LIMITS_PATH.read_text(encoding="utf-8"))
    return raw["velocity_limits_deg_s"]


def test_sourced_slots_use_the_d38_limit() -> None:
    limits = load_limits(DEFAULT_LIMITS_PATH)
    for name, (value, _observed, _cite) in EXPECTED.items():
        assert limits[name].value_deg_s == pytest.approx(value), name
        assert limits[name].verification_status == SOURCED, name


def test_every_sourced_entry_follows_the_d38_formula() -> None:
    """limit = max(2 x mean, (mean + 3 SD) x 125 / study club speed)."""
    sourced = [e for e in _entries() if e["verification_status"] == SOURCED]
    assert sourced
    for e in sourced:
        c = e["citation"]
        assert c["id"] and c["location"] and c["population"], e["coordinates"]
        assert c["margin"] == MARGIN
        assert c["target_clubhead_mph"] == TARGET_MPH
        two_x = c["observed_deg_s"] * c["margin"]
        scaled = (
            (c["observed_deg_s"] + 3 * c["observed_sd_deg_s"])
            * TARGET_MPH
            / c["study_clubhead_mph"]
        )
        assert c["two_x_term"] == pytest.approx(two_x, abs=0.1)
        assert c["scaled_term"] == pytest.approx(scaled, abs=0.1)
        assert float(e["value"]) == pytest.approx(max(two_x, scaled), abs=0.5)
        assert c["binding_term"] == ("two_x" if two_x >= scaled else "scaled")


def test_both_branches_of_the_max_are_exercised_by_the_registry() -> None:
    winners = {
        e["citation"]["binding_term"]
        for e in _entries()
        if e["verification_status"] == SOURCED
    }
    assert winners == {"two_x", "scaled"}


def test_lead_wrist_limit_is_pinned_at_the_exact_study_speed() -> None:
    # (669 + 3 * 210) * 125 / 100.66 = 1613.1 (study speed 45 m/s = 100.66 mph)
    assert load_limits(DEFAULT_LIMITS_PATH)["LWInputX"].value_deg_s == 1613.0


BUNDLE = Path(__file__).resolve().parents[3] / (
    "docs/development/full_body_models/evidence/trunk_visuals/driver_v2_bundle.npz"
)
#: sha256 of the recorded bundle the receipt below was taken from.
BUNDLE_SHA256 = "c3e1862b2be2ac116a981a2fed8bf1c22bff0c47b186124de895045e2ea53056"
#: Envelope window origin: recorded steps 1225..1325 (1 ms; impact at ~1325).
WINDOW = slice(1225, 1326)

#: Peak |velocity| over the recorded driver_v2 window (steps 1225..1325, deg/s),
#: from ``reference_v`` (rad/s in the bundle, converted to deg/s).
#: The impact-step (1327) transient is deliberately outside the window: LWInputX
#: 2316, LWInputY 1725, RWInputX 3249, RWInputY 1797 are contact artefacts.
RECORDED_WINDOW_PEAK_DEG_S = {
    "HipInputZ": 423.4,
    "thoracic_axial_rotation": 466.2,
    "LSInputX": 316.8,
    "LSInputY": 428.3,
    "LSInputZ": 1472.6,
    "LWInputX": 1150.7,
    "LWInputY": 1323.8,
    "RWInputX": 1365.1,
    "RWInputY": 1499.2,
}


def test_limits_clear_the_recorded_baseline_in_window_peaks() -> None:
    limits = load_limits(DEFAULT_LIMITS_PATH)
    for name, peak in RECORDED_WINDOW_PEAK_DEG_S.items():
        assert limits[name].value_deg_s >= peak, name


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


def test_receipt_peaks_are_reproduced_from_the_recorded_bundle() -> None:
    """The committed receipt numbers must match the bundle they claim to come from."""
    assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == BUNDLE_SHA256
    bundle = InputBundle.load(BUNDLE)
    order = list(bundle.coordinate_order)
    window = np.rad2deg(np.abs(np.asarray(bundle.reference_v)[WINDOW]))
    for name, peak in RECORDED_WINDOW_PEAK_DEG_S.items():
        assert window[:, order.index(name)].max() == pytest.approx(peak, abs=0.1), name


def test_load_limits_rejects_an_unknown_verification_status(tmp_path: Path) -> None:
    bad = tmp_path / "limits.yaml"
    bad.write_text(
        "\n".join(
            [
                "version: 1",
                "velocity_limits_deg_s:",
                "  - coordinates: [HipInputZ]",
                "    value: 1000",
                "    citation: {id: null, note: x}",
                "    verification_status: PROBABLY_FINE",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractViolationError):
        load_limits(bad)
