"""Independent angular-momentum and COM oracles (MuJoCo + NumPy only).

Does not import ``mujoco_segment_angular_momentum`` or ``body_angular_momentum_about_point``.
Used by #15435 acceptance tests so regressions cannot track the provider implementation.
"""

from __future__ import annotations

import mujoco
import numpy as np

from src.shared.python.body_part_viz.mujoco_segment_group_catalog import (
    is_excluded_body_name,
)


def oracle_mass_weighted_com_and_velocity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qvel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mass-weighted COM position and velocity, excluding ball-like bodies."""
    total_mass = 0.0
    weighted_pos = np.zeros(3, dtype=float)
    weighted_vel = np.zeros(3, dtype=float)
    qvel_arr = np.asarray(qvel, dtype=float)
    for body_id in range(1, model.nbody):
        name = model.body(body_id).name or ""
        mass = float(model.body_mass[body_id])
        if mass <= 0.0 or is_excluded_body_name(name):
            continue
        pos = np.asarray(data.xipos[body_id], dtype=float)
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBodyCom(model, data, jacp, None, body_id)
        vel = jacp @ qvel_arr
        weighted_pos += mass * pos
        weighted_vel += mass * vel
        total_mass += mass
    if total_mass <= 0.0:
        raise ValueError("oracle COM: no qualifying bodies")
    return weighted_pos / total_mass, weighted_vel / total_mass


def _body_angular_momentum_about_reference(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    qvel: np.ndarray,
    ref_com: np.ndarray,
    ref_vel: np.ndarray,
) -> np.ndarray:
    """Explicit H_i = m (r_i - r_C) × (v_i - v_C) + I_world ω_i."""
    mass = float(model.body_mass[body_id])
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)
    com_pos = np.asarray(data.xipos[body_id], dtype=float)
    com_vel = jacp @ qvel
    ang_vel = jacr @ qvel
    inertia_diag = np.asarray(model.body_inertia[body_id], dtype=float)
    rot = np.asarray(data.ximat[body_id], dtype=float).reshape(3, 3)
    i_world = rot @ np.diag(inertia_diag) @ rot.T
    orbital = mass * np.cross(com_pos - ref_com, com_vel - ref_vel)
    spin = i_world @ ang_vel
    return orbital + spin


def oracle_system_angular_momentum(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qvel: np.ndarray,
) -> np.ndarray:
    """Sum of per-body H about the oracle whole-body COM (ball excluded)."""
    ref_com, ref_vel = oracle_mass_weighted_com_and_velocity(model, data, qvel)
    qvel_arr = np.asarray(qvel, dtype=float)
    total = np.zeros(3, dtype=float)
    for body_id in range(1, model.nbody):
        name = model.body(body_id).name or ""
        mass = float(model.body_mass[body_id])
        if mass <= 0.0 or not np.isfinite(model.body_inertia[body_id]).all():
            continue
        if is_excluded_body_name(name):
            continue
        total += _body_angular_momentum_about_reference(
            model, data, body_id, qvel_arr, ref_com, ref_vel
        )
    return total


def oracle_club_angular_momentum(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qvel: np.ndarray,
    club_body_names: frozenset[str],
) -> np.ndarray:
    """Club-group H from the same oracle formula, summed over member bodies."""
    ref_com, ref_vel = oracle_mass_weighted_com_and_velocity(model, data, qvel)
    qvel_arr = np.asarray(qvel, dtype=float)
    total = np.zeros(3, dtype=float)
    for body_id in range(1, model.nbody):
        name = model.body(body_id).name
        if not name or name not in club_body_names:
            continue
        mass = float(model.body_mass[body_id])
        if mass <= 0.0:
            continue
        total += _body_angular_momentum_about_reference(
            model, data, body_id, qvel_arr, ref_com, ref_vel
        )
    return total


def oracle_ball_mass_weight(
    model: mujoco.MjModel,
) -> tuple[int, float]:
    """Return (body_id, mass) for the first excluded ball body, if any."""
    for body_id in range(1, model.nbody):
        name = model.body(body_id).name or ""
        if is_excluded_body_name(name) and float(model.body_mass[body_id]) > 0.0:
            return body_id, float(model.body_mass[body_id])
    raise LookupError("no massive ball body in model")
