"""Static-hold per-hand bar wrench for the OpenSim lift pack (LIFT-4, #11744).

Reuses the shared GCV-7 grip analysis
(:mod:`src.shared.python.biomechanics.grip_wrench`) through the engine-free
:mod:`.bar_hold_static` -- no second wrench-transport routine.

Method: a rigid two-hand hold of a rigid bar is statically indeterminate,
and every hand-held OpenSim pack exercise (deadlift, bench_press, snatch,
clean_and_jerk) ties the bar to the hands with a rigid zero-DOF
``WeldJoint`` from ``hand_l`` (parent) to a barbell-prefixed body (child)
and a position-level ``WeldConstraint`` between ``hand_r`` (``frame1``) and
a barbell-prefixed body (``frame2``) -- see
``opensim_models.exercises.base.attach_barbell_to_hands`` in the OpenSim_Models
pack. OpenSim cannot supply the hand/hand split directly, so this module
forms the bar's mass and world centre of mass from every ``barbell``-prefixed
body's ``getMass``/``getMassCenter`` (transported to ground with
``findStationLocationInGround``) and the net static-equilibrium wrench the
hands must exert, split by the shared ``allocate_min_norm`` minimum-norm
allocator (through :func:`.bar_hold_static.static_hold_split`).

Grip-frame note (load-bearing, verified against the real pack --
``opensim_models.shared.utils.xml_helpers.add_weld_joint`` /
``opensim_models.shared.utils.constraint_helpers.add_weld_constraint``, and
cross-checked against the pack's own
``tests/parity/test_barbell_grip_residual.py``): ``attach_barbell_to_hands``
always builds the left ``WeldJoint`` with ``location_in_parent=(0, 0, 0)``
on ``hand_l``, and the right ``WeldConstraint`` with
``location_in_body_1=(0, 0, 0)`` on ``hand_r``. So, unlike the Pinocchio
pack -- where only the left hand's own frame is the grip point and the
right grip needs a distinct bar-side anchor frame -- **both** OpenSim hands'
own origins are their grip points directly; there is no bar-side anchor to
disambiguate for either side. The bar-side frames (the WeldJoint's child
frame, and the WeldConstraint's ``frame2``) exist only to measure the
assembly residual: ``hand_grip_residual_m["L"]`` is the distance between
the WeldJoint's parent and child frames, which coincide to floating-point
precision because a ``WeldJoint`` is a zero-DOF joint (kinematically exact,
no assembly needed -- this is a genuine measurement, not a hard-coded
``0.0``, but it should read as numerical noise);
``hand_grip_residual_m["R"]`` is the distance between the WeldConstraint's
``frame1``/``frame2``, which OpenSim's ``initSystem()`` assembles to within
~5 mm (OpenSim_Models#394/#426).

This is a kinematic static-equilibrium calculation, not a simulation: there
is no bar acceleration to report (``bar_linear_accel_mps2`` is always
``None``, never a fabricated ``0.0``) and the force balance
(``relative_error``) is trivially ~0 by construction; the meaningful checks
are the bar mass against ``Anthropometry.bar_total_mass_kg`` and the
symmetry of the split.

Frame: OpenSim is Y-up natively; every returned vector (and the grip points
and gravity used internally) is converted to the canonical z-up frame with
:func:`src.shared.python.lifting.pack_audit.frames.to_canonical`, matching
MuJoCo/Drake/Pinocchio.

Precondition: the pack must weld the bar to both hands with this exact
topology (a ``WeldJoint`` from ``hand_l`` to a barbell-prefixed body, and a
``WeldConstraint`` between ``hand_r`` and a barbell-prefixed body). Squat
welds the bar to the torso instead (no such hand attachment exists) and is
reported ``unavailable``, never as a wrong or zero number.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import opensim as osim

from ..frames import OPENSIM_Y_UP, to_canonical
from ..model import unavailable_bar_hold
from .bar_hold_static import static_hold_result, static_hold_split

_BAR_PREFIX = "barbell"
_LEFT_HAND_BODY = "hand_l"
_RIGHT_HAND_BODY = "hand_r"
#: One WeldJoint (left) + one WeldConstraint (right); see module docstring.
_N_WELDS = 2
_METHOD = (
    "kinematic static equilibrium (no simulation): bar mass/COM from the "
    "getMass/getMassCenter of every barbell-prefixed body transported to "
    "ground by findStationLocationInGround at the pack start pose; "
    "required hand wrench R = -m*g with moment (r_com - r_mid) x R about "
    "the grip midpoint, split by the shared allocate_min_norm (GCV-7)"
)


def _vec3(v: Any) -> np.ndarray:
    return np.array([float(v.get(i)) for i in range(3)], dtype=float)


def _canon_point(frame: Any, state: Any) -> np.ndarray:
    """A ``Frame``'s (or ``Body``'s) ground position, in the canonical frame."""
    point = _vec3(frame.getPositionInGround(state)).tolist()
    return to_canonical(OPENSIM_Y_UP, point)


def _find_left_weld_joint(model: Any) -> Any | None:
    """The ``WeldJoint`` welding ``hand_l`` to a barbell-prefixed body, if any."""
    joints = model.getJointSet()
    for i in range(joints.getSize()):
        joint = joints.get(i)
        if joint.getConcreteClassName() != "WeldJoint":
            continue
        parent = joint.getParentFrame().findBaseFrame().getName()
        child = joint.getChildFrame().findBaseFrame().getName()
        if parent == _LEFT_HAND_BODY and child.startswith(_BAR_PREFIX):
            return joint
    return None


def _find_right_weld_constraint(model: Any) -> Any | None:
    """The ``WeldConstraint`` welding ``hand_r`` to a barbell-prefixed body."""
    constraints = model.getConstraintSet()
    for i in range(constraints.getSize()):
        constraint = constraints.get(i)
        if constraint.getConcreteClassName() != "WeldConstraint":
            continue
        weld = osim.WeldConstraint.safeDownCast(constraint)
        frame1_body = weld.getConnectee("frame1").findBaseFrame().getName()
        frame2_body = weld.getConnectee("frame2").findBaseFrame().getName()
        if frame1_body == _RIGHT_HAND_BODY and frame2_body.startswith(_BAR_PREFIX):
            return weld
    return None


def _bar_mass_and_world_com(model: Any, state: Any) -> tuple[float, np.ndarray]:
    """Bar mass and canonical-frame world centre of mass.

    Sums ``getMass``/``getMassCenter`` over every body whose name starts
    with ``barbell`` (the shaft and both sleeves), transported to ground by
    ``findStationLocationInGround``.

    Raises:
        ValueError: if the barbell bodies carry no positive mass.
    """
    bodies = model.getBodySet()
    total_mass = 0.0
    weighted_com = np.zeros(3)
    for i in range(bodies.getSize()):
        body = bodies.get(i)
        if not body.getName().startswith(_BAR_PREFIX):
            continue
        mass = float(body.getMass())
        if mass <= 0.0:
            continue
        world_com = _vec3(body.findStationLocationInGround(state, body.getMassCenter()))
        total_mass += mass
        weighted_com += mass * world_com
    if total_mass <= 0.0:
        raise ValueError("barbell bodies carry no positive mass")
    com = (weighted_com / total_mass).tolist()
    return total_mass, to_canonical(OPENSIM_Y_UP, com)


def bar_hold_wrench(model: Any, state: Any) -> dict[str, Any]:
    """Per-hand static-hold wrench on the bar for one OpenSim lift pack model.

    Args:
        model: the pack's ``opensim.Model``, already ``initSystem``-ed.
        state: a state realized at least through ``realizePosition`` (the
            adapter passes the pack start state, as ``evaluate(None)`` does).

    Returns:
        A JSON-serialisable dict; see ``EngineAdapter.bar_hold_wrench`` for
        the field contract. Additionally carries ``hand_grip_residual_m``
        (``{"L": ..., "R": ...}``), the distance between each hand's own
        attachment frame and the bar-side frame it is welded to, as a
        diagnostic only (see module docstring).
    """
    left_joint = _find_left_weld_joint(model)
    right_weld = _find_right_weld_constraint(model)
    if left_joint is None or right_weld is None:
        return unavailable_bar_hold(
            "no hand_l WeldJoint / hand_r WeldConstraint to a barbell-prefixed "
            "body: the bar is not hand-held in this pack (e.g. squat welds "
            "the bar to the torso)"
        )

    bar_mass_kg, bar_com_world = _bar_mass_and_world_com(model, state)
    gravity = to_canonical(OPENSIM_Y_UP, _vec3(model.getGravity()).tolist())

    grip_l = _canon_point(left_joint.getParentFrame(), state)
    bar_anchor_l = _canon_point(left_joint.getChildFrame(), state)
    grip_r = _canon_point(right_weld.getConnectee("frame1"), state)
    bar_anchor_r = _canon_point(right_weld.getConnectee("frame2"), state)

    split = static_hold_split(bar_mass_kg, gravity, bar_com_world, grip_l, grip_r)
    return static_hold_result(
        bar_mass_kg,
        split,
        _METHOD,
        _N_WELDS,
        {
            "L": float(np.linalg.norm(grip_l - bar_anchor_l)),
            "R": float(np.linalg.norm(grip_r - bar_anchor_r)),
        },
    )
