"""Static-hold per-hand bar wrench for the MuJoCo lift pack (LIFT-4, #11744).

Reuses the shared GCV-7 grip analysis
(:mod:`src.shared.python.biomechanics.grip_wrench`) through the GCV-8 weld
``efc_force`` extraction (:mod:`src.engines.physics_engines.mujoco.python.grip_efc`)
instead of a second wrench-transport routine.

Method: on a private copy of the model, every DOF that does not belong to a
barbell body gets a very large ``dof_armature`` (the lifter is held rigid so
the hands cannot accelerate), and every barbell geom is excluded from contact
(``geom_contype``/``geom_conaffinity`` set to 0, so the floor/rack never
carries the load).  A single ``mj_forward`` at the pack's start pose and zero
velocity then yields the weld reaction: the physical force each hand must
exert on the bar to hold it still.  The bar's own linear acceleration is
checked (not assumed) so the static-hold reading is verified, never asserted.

Precondition the measurement depends on: at least one matched barbell body
must have ``body_dofnum > 0`` (a genuine joint).  A body with no joint never
enters MuJoCo's equations of motion, so its weight cannot show up in a weld
reaction no matter how it is extracted; this is reported ``unavailable`` with
a precise reason (never a wrong or zero number) rather than assumed away. As
of this writing the MuJoCo_Models lift pack's barbell bodies are built via
``create_barbell_bodies()`` with no joint at all (confirmed empirically),
so every hand-held lift currently reports unavailable for this reason --
see ``tests/unit/lifting/pack_audit/test_bar_hold_wrench.py`` for the full
investigation and the pack-side follow-up this implies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import mujoco
import numpy as np

from src.engines.physics_engines.mujoco.python.grip_efc import grip_analysis_from_efc

from ..model import unavailable_bar_hold

_RIGID_ARMATURE = 1.0e9
_BAR_PREFIX = "barbell"
_METHOD = (
    "rigid-lifter static hold: dof_armature=1e9 on every non-barbell DOF, "
    "barbell geom contacts disabled, weld reaction read from efc_force "
    "(GCV-7/GCV-8)"
)


def _bar_body_ids(model: mujoco.MjModel) -> list[int]:
    return [
        i for i in range(1, model.nbody) if model.body(i).name.startswith(_BAR_PREFIX)
    ]


def _freeze_non_bar_dofs(model: mujoco.MjModel, bar_body_ids: Sequence[int]) -> None:
    """Set a very large ``dof_armature`` on every DOF outside the barbell."""
    bar_dofs: set[int] = set()
    for bid in bar_body_ids:
        start = int(model.body_dofadr[bid])
        bar_dofs.update(range(start, start + int(model.body_dofnum[bid])))
    for dof in range(model.nv):
        if dof not in bar_dofs:
            model.dof_armature[dof] = _RIGID_ARMATURE


def _disable_bar_contacts(model: mujoco.MjModel, bar_body_ids: Sequence[int]) -> None:
    bar_set = set(bar_body_ids)
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) in bar_set:
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 0


def _start_qpos(model: mujoco.MjModel) -> np.ndarray:
    return np.array(model.key_qpos[0] if model.nkey else model.qpos0, dtype=float)


def _bar_linear_accel_mps2(
    model: mujoco.MjModel, data: mujoco.MjData, bar_body_ids: Sequence[int]
) -> float:
    """Worst-case world-frame linear acceleration magnitude over the bar bodies.

    Uses the same ``mj_rnePostConstraint`` + ``mj_objectAcceleration`` pattern
    as ``rigid_body_dynamics/induced_acceleration.py``: ``mj_objectAcceleration``
    returns the proper (accelerometer-style) acceleration, so gravity is added
    back to recover the coordinate (world-frame) acceleration.
    """
    if not bar_body_ids:
        return 0.0
    mujoco.mj_rnePostConstraint(model, data)
    gravity = np.asarray(model.opt.gravity, dtype=float)
    spatial = np.zeros(6)
    worst = 0.0
    for bid in bar_body_ids:
        mujoco.mj_objectAcceleration(
            model, data, mujoco.mjtObj.mjOBJ_BODY, bid, spatial, 0
        )
        accel = np.asarray(spatial[3:6], dtype=float) + gravity
        worst = max(worst, float(np.linalg.norm(accel)))
    return worst


def _hand_weld_names(
    welds: Sequence[Mapping[str, Any]], bar_body_names: set[str]
) -> tuple[dict[str, str], str]:
    """Map hand side -> weld name for welds between a hand and a barbell body.

    Returns ``({}, reason)`` when a hand weld is missing for either side.
    """
    found: dict[str, str] = {}
    for w in welds:
        body1, body2 = str(w["body1"]), str(w["body2"])
        if body1.startswith("hand_") and body2 in bar_body_names:
            found[body1[-1].upper()] = str(w["name"])
        elif body2.startswith("hand_") and body1 in bar_body_names:
            found[body2[-1].upper()] = str(w["name"])
    missing = sorted({"L", "R"} - set(found))
    if not missing:
        return found, ""
    all_bodies = {str(w["body1"]) for w in welds} | {str(w["body2"]) for w in welds}
    other = sorted(all_bodies - bar_body_names - {"world"})
    reason = (
        f"no hand-to-bar weld for side(s) {missing}; "
        f"bar is attached via {other or 'no welds'} instead of both hands"
    )
    return {}, reason


def bar_hold_wrench(xml: str, welds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-hand static-hold wrench on the bar for one MuJoCo lift pack model.

    Args:
        xml: the pack's generated MJCF (the same text the adapter loaded).
        welds: weld descriptors as returned by ``MujocoAdapter._welds()``
            (each a mapping with at least ``name``, ``body1``, ``body2``).

    Returns:
        A JSON-serialisable dict; see ``EngineAdapter.bar_hold_wrench`` for
        the field contract.

    Raises:
        TypeError: if ``xml`` is not a non-empty string.
        ValueError: if ``welds`` is empty, or the resolved bar has no
            positive mass, or the net vertical hand force is ~0 (cannot form
            a left/right split).
    """
    if not isinstance(xml, str) or not xml.strip():
        raise TypeError("xml must be a non-empty string")
    if not welds:
        raise ValueError("welds must be a non-empty sequence of weld descriptors")

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    bar_body_ids = _bar_body_ids(model)
    bar_body_names = {model.body(i).name for i in bar_body_ids}

    weld_names, reason = _hand_weld_names(welds, bar_body_names)
    if not weld_names:
        return unavailable_bar_hold(reason)

    if not any(int(model.body_dofnum[bid]) > 0 for bid in bar_body_ids):
        return unavailable_bar_hold(
            f"barbell bodies {sorted(bar_body_names)} all have body_dofnum=0 "
            "(no joint): the bar is a kinematic fixture in this pack, not a "
            "dynamically free rigid body, so its mass never enters the "
            "equations of motion and the weld reaction cannot reflect its "
            "weight; the pack needs a free joint on the bar for this "
            "measurement to be physically meaningful"
        )

    _freeze_non_bar_dofs(model, bar_body_ids)
    _disable_bar_contacts(model, bar_body_ids)
    data.qpos[:] = _start_qpos(model)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    analysis = grip_analysis_from_efc(model, data, welds=weld_names)
    if analysis.left is None or analysis.right is None:
        return unavailable_bar_hold(
            analysis.unavailable_reason or "weld reaction unavailable"
        )

    bar_mass_kg = float(sum(model.body_mass[i] for i in bar_body_ids))
    gravity_mag = float(np.linalg.norm(model.opt.gravity))
    if bar_mass_kg <= 0.0 or gravity_mag <= 0.0:
        raise ValueError("bar mass and gravity magnitude must both be positive")
    bar_weight_n = bar_mass_kg * gravity_mag

    left_f = np.asarray(analysis.left.force_on_club_n, dtype=float)
    right_f = np.asarray(analysis.right.force_on_club_n, dtype=float)
    sum_vertical_n = float(left_f[2] + right_f[2])
    if abs(sum_vertical_n) < 1e-9:
        raise ValueError("sum of vertical hand forces is ~0; cannot form a split")

    couple = analysis.couple_at_midpoint_nm
    return {
        "available": True,
        "reason": "",
        "split_method": analysis.split_method,
        "bar_mass_kg": bar_mass_kg,
        "bar_weight_n": bar_weight_n,
        "hand_force_n": {"L": left_f.tolist(), "R": right_f.tolist()},
        "sum_vertical_n": sum_vertical_n,
        "relative_error": abs(sum_vertical_n - bar_weight_n) / bar_weight_n,
        "split_left_fraction": float(left_f[2] / sum_vertical_n),
        "couple_at_midpoint_nm": list(couple) if couple is not None else None,
        "bar_linear_accel_mps2": _bar_linear_accel_mps2(model, data, bar_body_ids),
        "method": _METHOD,
        "n_welds": len(weld_names),
    }
