"""Static-hold per-hand bar wrench for the Drake lift pack (LIFT-4, #11744).

Reuses the shared GCV-7 grip analysis
(:mod:`src.shared.python.biomechanics.grip_wrench`) through the engine-free
:mod:`.bar_hold_static` -- no second wrench-transport routine.

Method: a rigid two-hand hold of a rigid bar is statically indeterminate.
The Drake pack welds the bar to the left hand with a fixed SDF joint
(``hand_l -> barbell_shaft``; the sleeves are fixed children of the shaft)
and closes the loop to the right hand with a ``biomech:weld`` element
(``hand_r`` / ``barbell_shaft``) that the pack loader turns into a Drake weld
constraint only on a discrete plant. The pack therefore exposes no
hand/hand reaction split, so this module evaluates the bar's mass and world
centre of mass from every ``barbell``-prefixed body's
``default_spatial_inertia`` transported by ``EvalBodyPoseInWorld`` at the
pack start pose, forms the net wrench the hands must exert for static
equilibrium (``R = -m*g``, moment of ``R`` about the grip midpoint balancing
the weight at the bar's centre of mass) and splits it with the shared
``allocate_min_norm`` (through :func:`.bar_hold_static.static_hold_split`).

Grip-frame note (load-bearing, verified against the real pack): both
welds are anchored on the hand frames themselves -- the left fixed joint
places the shaft at ``hand_l`` offset by the grip half-width, and the right
loop-closure is built from the initial-pose transform from ``hand_r`` to the
shaft -- so each hand's own body frame is its grip point.
``hand_grip_residual_m["L"]`` is therefore ``0.0`` by construction (a fixed
joint is kinematically exact; not a measurement), and ``["R"]`` is the
distance between the shaft origin and where the right-hand weld says it
should be, i.e. the loop-closure residual at the start pose (the same
quantity as the pack's ``weld_residuals``).

This is a kinematic static-equilibrium calculation, not a simulation: there
is no bar acceleration to report (``bar_linear_accel_mps2`` is always
``None``, never a fabricated ``0.0``) and the force balance
(``relative_error``) is trivially ~0 by construction; the meaningful checks
are the bar mass against ``Anthropometry.bar_total_mass_kg`` and the
symmetry of the split.

Frame: Drake is z-up natively, matching the canonical frame, so no
conversion is applied. No ``pydrake`` import is needed: the plant is only
queried through methods.

Precondition: the pack must weld the bar to both hands with this topology.
Squat welds the bar to the torso instead and is reported ``unavailable``,
never as a wrong or zero number.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..model import unavailable_bar_hold
from .bar_hold_static import static_hold_result, static_hold_split

_BAR_PREFIX = "barbell"
_LEFT_HAND_BODY = "hand_l"
_RIGHT_HAND_BODY = "hand_r"
#: One fixed joint (left) + one loop-closure weld (right); see module docstring.
_N_WELDS = 2
_METHOD = (
    "kinematic static equilibrium (no simulation): bar mass/COM from the "
    "default_spatial_inertia of every barbell-prefixed body transported to "
    "world by EvalBodyPoseInWorld at the pack start pose; required hand "
    "wrench R = -m*g with moment (r_com - r_mid) x R about the grip midpoint, "
    "split by the shared allocate_min_norm (GCV-7)"
)


def _has_left_bar_joint(plant: Any, model_instance: Any) -> bool:
    """Whether a joint welds ``hand_l`` (parent) to a barbell-prefixed body."""
    for index in plant.GetJointIndices(model_instance):
        joint = plant.get_joint(index)
        parent = joint.parent_body().name()
        child = joint.child_body().name()
        if parent == _LEFT_HAND_BODY and child.startswith(_BAR_PREFIX):
            return True
    return False


def _find_right_weld(weld_specs: Iterable[Any]) -> Any | None:
    """The ``biomech:weld`` loop-closure from ``hand_r`` to the bar, if any."""
    for spec in weld_specs:
        if spec.parent == _RIGHT_HAND_BODY and spec.child.startswith(_BAR_PREFIX):
            return spec
    return None


def _body_translation(plant: Any, context: Any, name: str) -> np.ndarray:
    pose = plant.EvalBodyPoseInWorld(context, plant.GetBodyByName(name))
    return np.array(pose.translation(), dtype=float)


def _bar_mass_and_world_com(
    plant: Any, context: Any, model_instance: Any
) -> tuple[float, np.ndarray]:
    """Bar mass and world centre of mass from the barbell-prefixed bodies.

    Raises:
        ValueError: if the barbell bodies carry no positive mass.
    """
    total_mass = 0.0
    weighted_com = np.zeros(3)
    for index in plant.GetBodyIndices(model_instance):
        body = plant.get_body(index)
        if not body.name().startswith(_BAR_PREFIX):
            continue
        inertia = body.default_spatial_inertia()
        mass = float(inertia.get_mass())
        if mass <= 0.0:
            continue
        pose = plant.EvalBodyPoseInWorld(context, body)
        com_world = pose.multiply(np.asarray(inertia.get_com(), dtype=float))
        total_mass += mass
        weighted_com += mass * np.asarray(com_world, dtype=float)
    if total_mass <= 0.0:
        raise ValueError("barbell bodies carry no positive mass")
    return total_mass, weighted_com / total_mass


def _right_weld_residual_m(plant: Any, context: Any, loaded: Any, weld: Any) -> float:
    """Distance between the shaft origin and its right-hand weld target."""
    parent = plant.EvalBodyPoseInWorld(context, plant.GetBodyByName(weld.parent))
    target = np.array(
        parent.multiply(loaded.weld_poses[weld.name]).translation(), dtype=float
    )
    return float(np.linalg.norm(target - _body_translation(plant, context, weld.child)))


def bar_hold_wrench(loaded: Any) -> dict[str, Any]:
    """Per-hand static-hold wrench on the bar for one Drake lift pack model.

    Args:
        loaded: the pack's ``LoadedExercise`` (``plant``, ``model_instance``,
            ``weld_specs`` and ``weld_poses``), as held by ``DrakeAdapter``.

    Returns:
        A JSON-serialisable dict; see ``EngineAdapter.bar_hold_wrench`` for
        the field contract. Additionally carries ``hand_grip_residual_m``
        (``{"L": ..., "R": ...}``), a diagnostic only (see module docstring).

    Raises:
        TypeError: if ``loaded`` lacks ``plant``/``model_instance``/
            ``weld_specs``.
        ValueError: if the barbell bodies carry no mass, the grip points
            coincide, or the net vertical hand force is ~0.
    """
    for attr in ("plant", "model_instance", "weld_specs"):
        if not hasattr(loaded, attr):
            raise TypeError(f"loaded must provide {attr!r} (a LoadedExercise)")
    plant = loaded.plant
    right_weld = _find_right_weld(loaded.weld_specs)
    if right_weld is None or not _has_left_bar_joint(plant, loaded.model_instance):
        return unavailable_bar_hold(
            "no hand_l fixed joint / hand_r weld to a barbell-prefixed body: "
            "the bar is not hand-held in this pack (e.g. squat welds the bar "
            "to the torso)"
        )

    context = plant.CreateDefaultContext()
    bar_mass_kg, bar_com_world = _bar_mass_and_world_com(
        plant, context, loaded.model_instance
    )
    gravity = np.asarray(plant.gravity_field().gravity_vector(), dtype=float)
    grip_l = _body_translation(plant, context, _LEFT_HAND_BODY)
    grip_r = _body_translation(plant, context, _RIGHT_HAND_BODY)

    split = static_hold_split(bar_mass_kg, gravity, bar_com_world, grip_l, grip_r)
    return static_hold_result(
        bar_mass_kg,
        split,
        _METHOD,
        _N_WELDS,
        {
            # grip_l IS the hand_l frame (see module docstring): exactly 0
            # by construction, not a measurement.
            "L": 0.0,
            "R": _right_weld_residual_m(plant, context, loaded, right_weld),
        },
    )
