"""Real OpenSim musculoskeletal lift models (OSL-2, #13040; LIFT-6, #11746).

Every test in the ``requires_opensim`` block composes the five barbell lifts
in the real OpenSim 4.6 runtime: the muscle-bearing Rajagopal-Lai-Uhlrich
body scaled to the spec, a barbell of the spec's mass held by both hands
(``WeldJoint`` left, ``WeldConstraint`` right, the LIFT-4 hand interface),
smooth sphere/half-space foot contact and, for the bench press, bench and
back contact.  The counts are read from the built model, never from a list.

The static hold reuses the shared LIFT-4 bar-hold wrench
(``pack_audit.adapters.opensim_bar_hold``) and its #11744 acceptance: the
hand forces sum to the bar weight within 1 %.  The native check reads the
hand-bar reactions OpenSim itself computes with every coordinate locked.

The body is resolved at run time from the ``opensim-models`` submodule (or
``$UPSTREAMDRIFT_MSK_BASE_MODEL``); without it the tests skip with a reason.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from src.shared.python.lifting import opensim_full_body_lifts as lifts_mod
from src.shared.python.lifting.lift_spec import lift_grip, standard_height_m
from src.shared.python.lifting.opensim_full_body_lifts import (
    OpenSimFullBodyLiftSpec,
    compose_opensim_full_body_lift,
    opensim_lift_static_hold,
)
from src.shared.python.lifting.pack_audit.names import LIFTS

pytestmark = pytest.mark.unit

#: #11744 acceptance (the shared bar-hold tolerance): the hand forces sum to
#: the bar weight within 1 %.  Same value as ``test_bar_hold_wrench.py``.
_WEIGHT_REL_TOL = 0.01
#: The composed grip width is solved to the spec width; 1 mm is the bracket
#: tolerance the composer's root finder is asked to meet.
_GRIP_WIDTH_TOL_M = 1e-3
#: The trail WeldConstraint is built at the assembled hand pose, so its
#: residual is floating-point noise; far inside OpenSim_Models' 5 mm bound.
_RIGHT_RESIDUAL_TOL_M = 1e-6
#: Rajagopal-Lai-Uhlrich 2023 carries 80 lower-limb Millard muscles.
_EXPECTED_MUSCLES = 80
#: 22 skeleton bodies plus the barbell.
_EXPECTED_BODIES = 23
#: 22 skeleton joints plus the left hand-bar WeldJoint.
_EXPECTED_JOINTS = 23
#: Two knee CoordinateCouplerConstraints plus the right hand-bar WeldConstraint.
_EXPECTED_CONSTRAINTS = 3
#: Three spheres per foot (heel, forefoot, toe; the MS-20 calibrated layout).
_FOOT_CONTACTS = 6
#: Bench press: four back spheres (two scapular, two sacral) on the bench top.
_BENCH_CONTACTS = 4


def test_composition_fails_closed_when_runtime_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lifts_mod, "is_engine_available", lambda _name: False)
    composition = compose_opensim_full_body_lift(OpenSimFullBodyLiftSpec("squat"))
    assert composition.status == "unqualified"
    assert composition.is_composed is False
    assert composition.barbell_attached is False
    assert composition.reason is not None and "OpenSim runtime" in composition.reason
    for count in (
        composition.body_count,
        composition.joint_count,
        composition.constraint_count,
        composition.muscle_count,
        composition.force_count,
    ):
        assert count is None, "unavailable is never zero"
    hold = opensim_lift_static_hold(composition)
    assert hold["available"] is False
    assert hold["reason"]


def test_non_barbell_lift_is_unqualified_with_reason() -> None:
    composition = compose_opensim_full_body_lift(
        OpenSimFullBodyLiftSpec("split_jerk", bar_mass_kg=100.0)
    )
    assert composition.status == "unqualified"
    assert composition.reason


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


_CACHE: dict[str, object] = {}


def _composed(lift: str):  # type: ignore[no-untyped-def]
    _real_opensim()
    _require_body()
    if lift not in _CACHE:
        _CACHE[lift] = compose_opensim_full_body_lift(OpenSimFullBodyLiftSpec(lift))
    composition = _CACHE[lift]
    assert composition.is_composed, composition.reason  # type: ignore[attr-defined]
    return composition


@pytest.mark.requires_opensim
@pytest.mark.integration
@pytest.mark.parametrize("lift", LIFTS)
def test_lift_model_initialises_with_measured_counts(lift: str) -> None:
    composition = _composed(lift)
    model = composition.native_model
    model.initSystem()  # must not raise
    assert composition.status == "composed"
    assert composition.body_count == _EXPECTED_BODIES
    assert composition.joint_count == _EXPECTED_JOINTS
    assert composition.constraint_count == _EXPECTED_CONSTRAINTS
    assert composition.muscle_count == _EXPECTED_MUSCLES
    assert composition.muscle_count > 0
    # Counts are the model's own, not a template.
    assert composition.body_count == model.getBodySet().getSize()
    assert composition.joint_count == model.getJointSet().getSize()
    assert composition.constraint_count == model.getConstraintSet().getSize()
    assert composition.muscle_count == model.getMuscles().getSize()
    assert composition.force_count == model.getForceSet().getSize()
    assert len(composition.body_joints) == composition.joint_count
    expected_contacts = _FOOT_CONTACTS + (
        _BENCH_CONTACTS if lift == "bench_press" else 0
    )
    assert len(composition.contact_elements) == expected_contacts
    forces = model.getForceSet()
    for name in composition.contact_elements:
        force = forces.get(name)
        assert force.getConcreteClassName() == "SmoothSphereHalfSpaceForce"


@pytest.mark.requires_opensim
@pytest.mark.integration
@pytest.mark.parametrize("lift", LIFTS)
def test_body_is_scaled_and_receipt_records_source(lift: str) -> None:
    composition = _composed(lift)
    spec = composition.spec
    receipt = composition.receipt
    assert receipt["body_source"]["name"] == "RajagopalLaiUhlrich2023"
    assert len(receipt["body_source"]["sha256"]) == 64
    assert receipt["skeleton_mass_kg"] == pytest.approx(spec.body_mass_kg, rel=1e-9)
    assert receipt["bar_mass_kg"] == pytest.approx(spec.bar_mass_kg, rel=1e-12)
    target = lift_grip(lift).width_m * spec.height_m / standard_height_m()
    assert receipt["grip_width_target_m"] == pytest.approx(target, rel=1e-12)
    assert abs(receipt["grip_width_m"] - target) <= _GRIP_WIDTH_TOL_M


@pytest.mark.requires_opensim
@pytest.mark.integration
@pytest.mark.parametrize("lift", LIFTS)
def test_bar_is_attached_to_both_hands(lift: str) -> None:
    composition = _composed(lift)
    assert composition.barbell_attached is True
    attachment = composition.receipt["hand_attachment"]
    assert attachment == {
        "left": "WeldJoint hand_l_to_barbell",
        "right": "WeldConstraint hand_r_to_barbell",
    }


@pytest.mark.requires_opensim
@pytest.mark.integration
@pytest.mark.parametrize("lift", LIFTS)
def test_static_hold_carries_the_bar_weight_through_the_hands(lift: str) -> None:
    composition = _composed(lift)
    hold = opensim_lift_static_hold(composition)
    assert hold["available"], hold["reason"]

    shared = hold["bar_hold"]
    assert shared["available"], shared["reason"]
    assert shared["bar_mass_kg"] == pytest.approx(composition.spec.bar_mass_kg)
    assert shared["relative_error"] <= _WEIGHT_REL_TOL, shared
    assert shared["hand_force_n"]["L"][2] > 0.0
    assert shared["hand_force_n"]["R"][2] > 0.0
    assert shared["hand_grip_residual_m"]["R"] <= _RIGHT_RESIDUAL_TOL_M

    native = hold["native"]
    assert native["bar_weight_n"] == pytest.approx(shared["bar_weight_n"])
    assert native["relative_error"] <= _WEIGHT_REL_TOL, native
    assert native["sum_vertical_n"] > 0.0
    assert math.isfinite(native["hand_force_n"]["L"][2])
    assert math.isfinite(native["hand_force_n"]["R"][2])


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_bench_press_lies_on_the_bench() -> None:
    composition = _composed("bench_press")
    receipt = composition.receipt
    assert receipt["hold_pose"]["supine"] is True
    assert receipt["bench_height_m"] is not None and receipt["bench_height_m"] > 0.0
    assert _composed("deadlift").receipt["bench_height_m"] is None


@pytest.mark.requires_opensim
def test_unsupported_options_fail_closed() -> None:
    _real_opensim()
    base = OpenSimFullBodyLiftSpec("deadlift")
    for spec in (
        replace(base, body_model="MyoFullBody"),
        replace(base, contact_model="hunt_crossley"),
        replace(base, bench_contact=True),
        replace(base, bar_mass_kg=10.0),
    ):
        composition = compose_opensim_full_body_lift(spec)
        assert composition.status == "unqualified", spec
        assert composition.reason, spec
        assert composition.barbell_attached is False
        assert composition.muscle_count is None
