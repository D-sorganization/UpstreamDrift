"""MuJoCo before/after report for a pelvis revision (HRG-C3, #13172).

Three quantities, all computed from the compiled model so they test the
exported physics rather than the spec arithmetic:

* **Hip lever arms.** For each hip joint centre (HJC), the horizontal
  (perpendicular to gravity) vector to the pelvis centre of mass and to the
  centre of mass of everything above the hips (pelvis, trunk, head, arms and
  club). Their product with the weight is the static hip moment of that
  segment group.
* **Hip reaction wrench.** The force and moment about the HJC midpoint that the
  legs must apply to the pelvis for the recorded motion of the superior body.
  Legs are given negligible mass, so the result needs neither the leg motion
  nor ground contact. It is read from ``qfrc_inverse`` through the root
  Jacobian at the midpoint, a point rigid in the pelvis.
* **L5/S1 torques.** ``qfrc_inverse`` of the twist and spine coordinates,
  which depend only on the bodies above L5/S1.

Software correctness only; scientific qualification is separate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
    export_full_body_mjcf,
)

Array = NDArray[np.float64]
LEG_SUFFIXES = ("femur", "tibia", "talus", "calcn", "toes")
HIPS = ("hip_flexion_r", "hip_flexion_l")
L5S1_COORDINATES = ("TorsoInput", "SpineInputX", "SpineInputY")
ROOT = ("TranslationInputX", "TranslationInputY", "TranslationInputZ")
LEG_MASS_KG = 1.0e-9


def compile_spec(spec: Mapping[str, Any], *, massless_legs: bool = False):
    """Compile ``spec`` to a MuJoCo model; optionally give the legs no mass."""
    import mujoco

    xml, _ = export_full_body_mjcf(json.dumps(spec).encode("utf-8"))
    model = mujoco.MjModel.from_xml_string(xml)
    if massless_legs:
        for body in range(model.nbody):
            name = model.body(body).name
            if name.split("_")[0] in LEG_SUFFIXES:
                model.body_mass[body] = LEG_MASS_KG
                model.body_inertia[body] = LEG_MASS_KG * 1.0e-3
    return model


def set_state(
    model: Any, data: Any, coordinates: Sequence[str], q: Array, v: Array, a: Array
) -> None:
    """Load a state given in ``coordinates`` order into ``data`` by joint name."""
    q, v, a = (np.asarray(x, dtype=float) for x in (q, v, a))
    if not (q.shape == v.shape == a.shape == (len(coordinates),)):
        raise ValueError("q, v and a must each have one entry per coordinate")
    for k, name in enumerate(coordinates):
        joint = model.joint(name)
        data.qpos[joint.qposadr[0]] = q[k]
        data.qvel[joint.dofadr[0]] = v[k]
        data.qacc[joint.dofadr[0]] = a[k]


def lever_arms(model: Any, data: Any) -> dict[str, Any]:
    """HJC-to-COM horizontal lever arms at the state in ``data`` (kinematics run)."""
    import mujoco

    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    gravity = np.asarray(model.opt.gravity, dtype=float)
    up = -gravity / np.linalg.norm(gravity)
    superior = [
        b
        for b in range(1, model.nbody)
        if model.body(b).name.split("_")[0] not in LEG_SUFFIXES
        and model.body_mass[b] > 0
    ]
    mass = model.body_mass[superior]
    com_superior = mass @ data.xipos[superior] / mass.sum()
    pelvis = next(
        b for b in superior if model.body(b).parentid[0] == 0
    )  # root-attached body
    com_pelvis = data.xipos[pelvis]
    out: dict[str, Any] = {"superior_mass_kg": float(mass.sum())}
    for side, joint in zip(("r", "l"), HIPS, strict=True):
        hjc = data.xanchor[model.joint(joint).id]

        def horizontal(point: Array, hjc: Array = hjc) -> Array:
            d = point - hjc
            return d - (d @ up) * up

        out[side] = {
            "hjc_world_m": hjc.tolist(),
            "pelvis_com_arm_m": horizontal(com_pelvis).tolist(),
            "superior_com_arm_m": horizontal(com_superior).tolist(),
            "pelvis_com_height_m": float((com_pelvis - hjc) @ up),
            "superior_com_height_m": float((com_superior - hjc) @ up),
        }
    mid = 0.5 * (
        data.xanchor[model.joint(HIPS[0]).id] + data.xanchor[model.joint(HIPS[1]).id]
    )
    out["pelvis_com_to_hjc_midpoint_m"] = float(np.linalg.norm(com_pelvis - mid))
    out["superior_com_arm_midpoint_m"] = (
        (com_superior - mid) - ((com_superior - mid) @ up) * up
    ).tolist()
    out["superior_com_world_m"] = com_superior.tolist()
    return out


def inverse_state(model: Any, data: Any) -> Array:
    """``qfrc_inverse`` at the state in ``data`` (a copy)."""
    import mujoco

    mujoco.mj_inverse(model, data)
    return np.array(data.qfrc_inverse)


def hip_reaction_wrench(model: Any, data: Any, qfrc: Array) -> dict[str, Any]:
    """Force and moment about the HJC midpoint implied by the root generalised
    forces ``qfrc`` (world axes; moment taken about the midpoint).

    Precondition: the root is the six unactuated pelvis coordinates (first
    three world-axis slides, then the Euler hinges), ``data`` holds the state.
    """
    import mujoco

    mid = 0.5 * (
        data.xanchor[model.joint(HIPS[0]).id] + data.xanchor[model.joint(HIPS[1]).id]
    )
    dofs = [model.joint(n).dofadr[0] for n in ROOT] + [
        model.joint(n).dofadr[0] for n in ("HipInputX", "HipInputY", "HipInputZ")
    ]
    pelvis = next(b for b in range(1, model.nbody) if model.body(b).parentid[0] == 0)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, mid, pelvis)
    jac = np.vstack([jacp, jacr])[:, dofs]
    wrench = np.linalg.solve(jac.T, np.asarray(qfrc)[dofs])
    return {
        "point_world_m": mid.tolist(),
        "force_n": wrench[:3].tolist(),
        "moment_nm": wrench[3:].tolist(),
    }


def compare(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    coordinates: Sequence[str],
    q: Array,
    v: Array,
    a: Array,
    frames: Sequence[int],
) -> dict[str, Any]:
    """Lever arms, hip reaction wrenches and L5/S1 torques for both specs.

    ``q``, ``v`` and ``a`` are (frames, coordinates) arrays shared by both
    specs (same generalised trajectory, not refitted). Returns per-frame
    results plus peak magnitudes.
    """
    import mujoco

    full = {k: compile_spec(s) for k, s in (("before", before), ("after", after))}
    light = {
        k: compile_spec(s, massless_legs=True)
        for k, s in (("before", before), ("after", after))
    }
    out: dict[str, Any] = {"frames": list(map(int, frames)), "coordinates": "same"}
    for key in ("before", "after"):
        rows = []
        for f in frames:
            row: dict[str, Any] = {"frame": int(f)}
            data = mujoco.MjData(full[key])
            set_state(full[key], data, coordinates, q[f], v[f], a[f])
            row["lever_arms"] = lever_arms(full[key], data)
            qfrc = inverse_state(full[key], data)
            row["l5s1_torque_nm"] = {
                n: float(qfrc[full[key].joint(n).dofadr[0]]) for n in L5S1_COORDINATES
            }
            lite = mujoco.MjData(light[key])
            set_state(light[key], lite, coordinates, q[f], v[f], a[f])
            mujoco.mj_kinematics(light[key], lite)
            mujoco.mj_comPos(light[key], lite)
            row["hip_reaction"] = hip_reaction_wrench(
                light[key], lite, inverse_state(light[key], lite)
            )
            rows.append(row)
        out[key] = rows
    return out
