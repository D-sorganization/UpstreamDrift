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
import math

import numpy as np
import pytest

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


def test_every_lift_has_a_bundle_entry() -> None:
    assert set(SAME_INPUT_LIFT_BUNDLES) == set(LIFTS)
    squat = SAME_INPUT_LIFT_BUNDLES["squat"]
    assert squat is not None and squat.is_file()


@pytest.mark.parametrize(
    "lift", [lift for lift in LIFTS if SAME_INPUT_LIFT_BUNDLES[lift] is None]
)
def test_lift_without_a_bundle_is_unavailable_never_zero(lift: str) -> None:
    receipt = opensim_lift_ik_id(OpenSimFullBodyLiftSpec(lift))
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


@pytest.fixture(scope="module")
def squat_receipt() -> dict:
    _real_opensim()
    _require_body()
    receipt = opensim_lift_ik_id(OpenSimFullBodyLiftSpec("squat"))
    assert receipt["status"] == "complete", receipt["reason"]
    return receipt


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_squat_receipt_records_bundle_hash(squat_receipt: dict) -> None:
    path = SAME_INPUT_LIFT_BUNDLES["squat"]
    assert path is not None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert squat_receipt["bundle"]["sha256"] == digest
    assert squat_receipt["bundle"]["frames"] >= 2


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_squat_ik_coordinate_rms_below_receipt_threshold(squat_receipt: dict) -> None:
    ik = squat_receipt["ik"]
    angles = ik["segment_angle_error"]
    assert angles["threshold_rms_rad"] > 0.0
    assert angles["rms_rad"] < angles["threshold_rms_rad"]
    for segment in ("shank", "thigh", "trunk"):
        assert math.isfinite(angles["per_segment_rms_rad"][segment])
    markers = ik["marker_error"]
    assert math.isfinite(markers["rms_m"]) and math.isfinite(markers["max_m"])
    assert set(markers["per_marker_max_m"]) == set(ik["markers"])
    assert isinstance(markers["within_best_practice"], bool)


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_squat_id_moments_finite_and_residuals_reported(squat_receipt: dict) -> None:
    inverse = squat_receipt["id"]
    moments = inverse["joint_moments_peak_abs"]
    assert moments, "joint moments must be reported"
    assert all(math.isfinite(v) for v in moments.values())
    assert not set(moments) & set(_RESIDUAL_FORCES + _RESIDUAL_MOMENTS)
    residuals = inverse["residuals"]
    for name in _RESIDUAL_FORCES + _RESIDUAL_MOMENTS:
        entry = residuals[name]
        assert math.isfinite(entry["max_abs"]) and math.isfinite(entry["rms"])
    gate = inverse["residual_gate"]
    assert gate["max_force_n"] > 0.0 and gate["max_moment_nm"] > 0.0
    assert isinstance(gate["forces_within"], bool)
    assert isinstance(gate["moments_within"], bool)
    loads = inverse["external_loads"]
    assert loads["feet"].startswith("GCV-1")
    assert loads["hands"].startswith("GCV-7")
