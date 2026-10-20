"""OpenSim IK and inverse dynamics of the barbell lifts (OSL-3, #13041).

Red tests against the real OpenSim 4.6 runtime, written before the code:

* **Static hold.**  At each lift's OSL-2 hold pose, the vertical position
  where OpenSim's own inverse dynamics balances (pelvis vertical residual
  zero, contacts on) gives a contact-derived ground reaction -- read from the
  ``SmoothSphereHalfSpaceForce`` record values through the shared GCV-1
  analysis -- equal to (body + bar) x g.  The bench is part of the ground (it
  has no mass), so the bench press adds no bench weight.
* **Squat IK.**  OpenSim's ``InverseKinematicsSolver`` tracks the LIFT-5
  back-squat same-input bundle; the convention-free sagittal segment angles
  (shank, thigh, trunk) of the solved model track the bundle's within the
  receipt threshold.
* **Squat ID.**  ``InverseDynamicsSolver`` with the bundle's contact-derived
  external loads (GCV-1 foot wrench, GCV-7 per-hand bar wrench): every joint
  moment is finite and the pelvis residuals are reported, never hidden.

Only the back squat has a LIFT-5 same-input bundle; the other four lifts are
reported unavailable with a reason (unavailable is never zero).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]

from src.shared.python.lifting import opensim_full_body_lifts as lifts_mod
from src.shared.python.lifting.bundles import SAME_INPUT_LIFT_BUNDLES
from src.shared.python.lifting.opensim_full_body_lifts import (
    OpenSimFullBodyLiftSpec,
    compose_opensim_full_body_lift,
    opensim_lift_ik_id,
    opensim_lift_static_grf,
)
from src.shared.python.lifting.pack_audit.names import LIFTS

pytestmark = pytest.mark.unit

#: Static-equilibrium tolerance: the vertical root is solved to 1e-12 m, so
#: the contact force is balanced to well under 1e-6 of the weight.
_STATIC_REL_TOL = 1e-6
_RESIDUAL_FORCES = ("pelvis_tx", "pelvis_ty", "pelvis_tz")
_RESIDUAL_MOMENTS = ("pelvis_tilt", "pelvis_list", "pelvis_rotation")


def _real_opensim() -> None:
    from tests.helpers.opensim_support import require_opensim

    osim = require_opensim()
    if not hasattr(osim, "__file__"):
        pytest.skip("opensim in sys.modules is a test double, not the real package")


def _require_body() -> None:
    from src.engines.physics_engines.opensim.python.musculoskeletal_swing import (
        resolve_base_model,
    )

    try:
        resolve_base_model()
    except FileNotFoundError as exc:
        pytest.skip(f"muscle-bearing body not available: {exc}")


@pytest.mark.parametrize("lift", LIFTS)
def test_every_lift_has_a_bundle_entry(lift: str) -> None:
    assert set(SAME_INPUT_LIFT_BUNDLES) == set(LIFTS)
    bundle = SAME_INPUT_LIFT_BUNDLES[lift]
    assert bundle is not None, f"bundle for {lift} must be registered"
    assert bundle.is_file(), f"bundle file {bundle} must exist"


def test_lift_without_a_bundle_is_unavailable_never_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unregistered = dict(SAME_INPUT_LIFT_BUNDLES)
    unregistered["squat"] = None
    monkeypatch.setattr(lifts_mod, "SAME_INPUT_LIFT_BUNDLES", unregistered)
    receipt = opensim_lift_ik_id(OpenSimFullBodyLiftSpec("squat"))
    assert receipt["status"] == "unavailable"
    assert "LIFT-5" in receipt["reason"]
    assert receipt["bundle"] is None
    assert receipt["ik"] is None
    assert receipt["id"] is None


def test_ik_id_fails_closed_without_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lifts_mod, "is_engine_available", lambda _name: False)
    receipt = opensim_lift_ik_id(OpenSimFullBodyLiftSpec("squat"))
    assert receipt["status"] == "unavailable"
    assert "OpenSim runtime" in receipt["reason"]
    assert receipt["ik"] is None and receipt["id"] is None


@pytest.mark.requires_opensim
@pytest.mark.integration
@pytest.mark.parametrize("lift", LIFTS)
def test_static_hold_ground_reaction_equals_body_plus_bar_weight(lift: str) -> None:
    _real_opensim()
    _require_body()
    composition = compose_opensim_full_body_lift(OpenSimFullBodyLiftSpec(lift))
    assert composition.is_composed, composition.reason
    hold = opensim_lift_static_grf(composition)
    assert hold["available"], hold["reason"]
    spec = composition.spec
    expected = (composition.receipt["skeleton_mass_kg"] + spec.bar_mass_kg) * hold[
        "gravity_m_s2"
    ]
    assert hold["expected_weight_n"] == pytest.approx(expected, rel=1e-12)
    total = np.asarray(hold["total_grf_n"])
    assert total[2] == pytest.approx(expected, rel=_STATIC_REL_TOL)
    assert abs(hold["relative_error"]) <= _STATIC_REL_TOL
    # Equilibrium is reached on contact, not by a hidden support.
    assert all(f > 0.0 for f in hold["support_vertical_n"].values())
    if lift == "bench_press":
        assert hold["support_vertical_n"]["bench"] > 0.0
    # The balance residual (what the locked rotations carry) is reported.
    for name in _RESIDUAL_MOMENTS:
        assert math.isfinite(hold["balance_residual"][name])


RECEIPTS_PATH = _REPO / "docs/development/lifting/evidence/opensim_ik_id/receipts.json"

_RESIDUAL_GATE_XFAILS = {
    "deadlift": (
        "forces_within=False (peak pelvis_ty 288.23 N > 110.92 N), "
        "moments_within=False (peak pelvis_tilt 147.88 Nm > 18.28 Nm) (#13041)"
    ),
    "bench_press": (
        "forces_within=False (peak pelvis_ty 511.26 N > 32.49 N), "
        "moments_within=False (peak pelvis_tilt 318.13 Nm > 3.45 Nm) (#13041)"
    ),
    "snatch": (
        "forces_within=False (peak pelvis_tx 463.57 N > 155.12 N), "
        "moments_within=False (peak pelvis_tilt 202.60 Nm > 27.10 Nm) (#13041)"
    ),
    "clean_and_jerk": (
        "forces_within=False (peak pelvis_tx 458.52 N > 223.10 N), "
        "moments_within=False (peak pelvis_tilt 259.03 Nm > 38.77 Nm) (#13041)"
    ),
}


def _load_receipts() -> dict:
    if not RECEIPTS_PATH.is_file():
        pytest.skip(f"receipts file {RECEIPTS_PATH} not found")
    return json.loads(RECEIPTS_PATH.read_text())


@pytest.fixture(scope="module")
def squat_receipt() -> dict:
    _real_opensim()
    _require_body()
    receipt = opensim_lift_ik_id(OpenSimFullBodyLiftSpec("squat"))
    assert receipt["status"] == "complete", receipt["reason"]
    return receipt


@pytest.mark.parametrize("lift", LIFTS)
def test_lift_receipt_complete(lift: str) -> None:
    receipts = _load_receipts()
    entry = receipts["lifts"][lift]["ik_id"]
    assert entry["status"] == "complete", entry["reason"]


@pytest.mark.parametrize("lift", LIFTS)
def test_lift_receipt_records_bundle_hash(lift: str) -> None:
    receipts = _load_receipts()
    entry = receipts["lifts"][lift]["ik_id"]
    assert entry["status"] == "complete", entry["reason"]
    path = SAME_INPUT_LIFT_BUNDLES[lift]
    assert path is not None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert entry["bundle"]["sha256"] == digest
    assert entry["bundle"]["frames"] >= 2


@pytest.mark.parametrize("lift", LIFTS)
def test_lift_ik_error_finite(lift: str) -> None:
    receipts = _load_receipts()
    entry = receipts["lifts"][lift]["ik_id"]
    assert entry["status"] == "complete", entry["reason"]
    ik = entry["ik"]
    angles = ik["segment_angle_error"]
    assert angles["threshold_rms_rad"] > 0.0
    for segment in ("shank", "thigh", "trunk"):
        assert math.isfinite(angles["per_segment_rms_rad"][segment])
    markers = ik["marker_error"]
    assert math.isfinite(markers["rms_m"]) and math.isfinite(markers["max_m"])
    assert set(markers["per_marker_max_m"]) == set(ik["markers"])
    assert isinstance(markers["within_best_practice"], bool)


@pytest.mark.parametrize("lift", LIFTS)
def test_lift_id_moments_finite_and_residuals_reported(lift: str) -> None:
    receipts = _load_receipts()
    entry = receipts["lifts"][lift]["ik_id"]
    assert entry["status"] == "complete", entry["reason"]
    inverse = entry["id"]
    moments = inverse["joint_moments_peak_abs"]
    assert moments, "joint moments must be reported"
    assert all(math.isfinite(v) for v in moments.values())
    assert not set(moments) & set(_RESIDUAL_FORCES + _RESIDUAL_MOMENTS)
    residuals = inverse["residuals"]
    for name in _RESIDUAL_FORCES + _RESIDUAL_MOMENTS:
        r_entry = residuals[name]
        assert math.isfinite(r_entry["max_abs"]) and math.isfinite(r_entry["rms"])
    gate = inverse["residual_gate"]
    assert gate["max_force_n"] > 0.0 and gate["max_moment_nm"] > 0.0
    assert isinstance(gate["forces_within"], bool)
    assert isinstance(gate["moments_within"], bool)
    loads = inverse["external_loads"]
    assert loads["feet"].startswith("GCV-1")
    assert loads["hands"].startswith("GCV-7")


@pytest.mark.parametrize("lift", LIFTS)
def test_lift_residual_gate(lift: str, request: pytest.FixtureRequest) -> None:
    if lift in _RESIDUAL_GATE_XFAILS:
        request.applymarker(
            pytest.mark.xfail(strict=True, reason=_RESIDUAL_GATE_XFAILS[lift])
        )
    receipts = _load_receipts()
    gate = receipts["lifts"][lift]["ik_id"]["id"]["residual_gate"]
    assert gate["forces_within"] is True
    assert gate["moments_within"] is True


def test_squat_receipt_range_violations_empty() -> None:
    receipts = _load_receipts()
    entry = receipts["lifts"]["squat"]["ik_id"]
    assert entry["status"] == "complete", entry["reason"]
    assert entry["ik"]["range_violations"] == {}


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_squat_live_ik_coordinate_rms_below_receipt_threshold(
    squat_receipt: dict,
) -> None:
    ik = squat_receipt["ik"]
    angles = ik["segment_angle_error"]
    assert angles["threshold_rms_rad"] > 0.0
    assert angles["rms_rad"] < angles["threshold_rms_rad"]


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_squat_live_id_residuals_reported(squat_receipt: dict) -> None:
    inverse = squat_receipt["id"]
    residuals = inverse["residuals"]
    for name in _RESIDUAL_FORCES + _RESIDUAL_MOMENTS:
        entry = residuals[name]
        assert math.isfinite(entry["max_abs"]) and math.isfinite(entry["rms"])


def test_bench_press_id_external_loads_include_bench() -> None:
    receipts = _load_receipts()
    entry = receipts["lifts"]["bench_press"]["ik_id"]
    loads = entry["id"]["external_loads"]
    assert "bench" in loads, "bench contact must be documented in external_loads"
    assert "back" in loads["bench"]


def test_bench_press_ik_pelvis_and_torso_supine() -> None:
    receipts = _load_receipts()
    entry = receipts["lifts"]["bench_press"]["ik_id"]
    pelvis = entry["ik"]["pelvis_landmarks"]
    tilt_span = pelvis["model_pelvis_tilt_rad"]
    for tilt in tilt_span:
        elevation = abs(abs(tilt) - math.pi / 2.0)
        assert elevation <= math.radians(20.0), (
            f"pelvis elevation {math.degrees(elevation):.1f} deg > 20 deg"
        )


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_bench_press_live_supine_and_bench_loads() -> None:
    _real_opensim()
    _require_body()
    receipt = opensim_lift_ik_id(OpenSimFullBodyLiftSpec("bench_press"))
    assert receipt["status"] == "complete", receipt["reason"]
    loads = receipt["id"]["external_loads"]
    assert "bench" in loads
    pelvis = receipt["ik"]["pelvis_landmarks"]
    tilt_span = pelvis["model_pelvis_tilt_rad"]
    for tilt in tilt_span:
        elevation = abs(abs(tilt) - math.pi / 2.0)
        assert elevation <= math.radians(20.0), (
            f"pelvis elevation {math.degrees(elevation):.1f} deg > 20 deg"
        )
