"""Kinetics from the fitted kinematics, for any registered model (#9714).

Given a continuous joint-angle trajectory q(t) and segment masses, the
generalised equations of motion are

    tau = M(q) qdd + C(q, qd) qd + G(q)

with the inertia matrix built from the segment centres of mass,
``M = sum_i m_i J_i^T J_i`` (point masses at the segment midpoints; rod
inertia about the segment axis is not modelled and the doc says so), the
Coriolis term from the derivative of ``M`` along the motion,
``C qd = dM/dt qd - 1/2 d(qd^T M qd)/dq``, and gravity
``G = -sum_i m_i J_i^T g``. Everything is evaluated numerically from the
model's forward kinematics, so it works for every :class:`ModelSpec` with no
engine dependency. Inverse dynamics gives the torques the fitted motion
needs; a forward replay integrates those torques back and reports how far it
drifts from the fitted angles, which is the acceptance metric of #9714: a
replay error that stays small says the kinematics and kinetics agree.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import require

from .kinematics import ArticulatedModel

Array = npt.NDArray[np.float64]
GRAVITY = np.array([0.0, -9.81, 0.0])
#: Segment mass as a fraction of body mass, by the segment's length name
#: (de Leva 1996 adjusted Zatsiorsky values, rounded; head+neck lumped).
MASS_FRACTIONS: dict[str, float] = {
    "lower_torso": 0.11,
    "upper_torso": 0.32,
    "head": 0.07,
    "hub_to_shoulder": 0.0,  # the strut is a linkage, not a limb
    "upper_arm": 0.027,
    "forearm": 0.022,
    "hip_half": 0.0,
    "thigh": 0.14,
    "shank": 0.043,
    "arm": 0.049,  # pendulum arm = upper arm + forearm
}


@dataclass(frozen=True)
class Segments:
    """Each massive segment: child joint index, parent joint index, mass (kg)."""

    child: tuple[int, ...]
    parent: tuple[int, ...]
    mass_kg: tuple[float, ...]


def segment_masses(model: ArticulatedModel, body_mass_kg: float) -> Segments:
    """Masses per segment from :data:`MASS_FRACTIONS`; unknown names get 0."""
    require(body_mass_kg > 0, "body mass must be positive", body_mass_kg)
    child, parent, mass = [], [], []
    for i, j in enumerate(model.joints):
        if j.parent is None or j.length is None:
            continue
        child.append(i)
        parent.append(model.index[j.parent])
        mass.append(body_mass_kg * MASS_FRACTIONS.get(j.length, 0.0))
    return Segments(tuple(child), tuple(parent), tuple(mass))


def _com_positions(model: ArticulatedModel, q: Array, seg: Segments) -> Array:
    pos = model.forward(q)
    return 0.5 * (pos[:, list(seg.child)] + pos[:, list(seg.parent)])


def _com_jacobian(
    model: ArticulatedModel, q: Array, seg: Segments, eps: float
) -> Array:
    """``(T, S, 3, n)`` d COM / d q by central differences, all frames at once."""
    t, n = q.shape
    out = np.zeros((t, len(seg.child), 3, n))
    for d in range(n):
        plus, minus = q.copy(), q.copy()
        plus[:, d] += eps
        minus[:, d] -= eps
        out[:, :, :, d] = (
            _com_positions(model, plus, seg) - _com_positions(model, minus, seg)
        ) / (2 * eps)
    return out


def inertia(
    model: ArticulatedModel, q: Array, seg: Segments, eps: float = 1e-6
) -> Array:
    """``M(q)`` ``(T, n, n)``: point masses at segment centres."""
    jac = _com_jacobian(model, q, seg, eps)
    m = np.asarray(seg.mass_kg)
    return np.einsum("s,tsij,tsik->tjk", m, jac, jac)


def gravity_torque(
    model: ArticulatedModel, q: Array, seg: Segments, eps: float = 1e-6
) -> Array:
    """``G(q)`` ``(T, n)``: generalised force holding the segments up."""
    jac = _com_jacobian(model, q, seg, eps)
    m = np.asarray(seg.mass_kg)
    return -np.einsum("s,tsij,i->tj", m, jac, GRAVITY)


def _derivatives(q: Array, fps: float) -> tuple[Array, Array]:
    qd = np.gradient(q, 1.0 / fps, axis=0)
    qdd = np.gradient(qd, 1.0 / fps, axis=0)
    return qd, qdd


def inverse_dynamics(
    model: ArticulatedModel,
    q: Array,
    fps: float,
    seg: Segments,
    *,
    eps: float = 1e-5,
) -> dict[str, Array]:
    """Torques ``tau`` ``(T, n)`` that produce ``q(t)`` under gravity.

    Preconditions: ``q`` is ``(T >= 3, n_dof)``, positive fps. The Coriolis
    term uses the time derivative of ``M`` and the configuration gradient of
    the kinetic energy, both by finite differences.
    """
    q = np.asarray(q, dtype=float)
    require(q.ndim == 2 and q.shape[0] >= 3 and q.shape[1] == model.n_dof, "q shape")
    require(fps > 0, "fps must be positive", fps)
    qd, qdd = _derivatives(q, fps)
    m_q = inertia(model, q, seg)
    dm_dt = np.gradient(m_q, 1.0 / fps, axis=0)
    # d/dq (1/2 qd^T M qd) by central differences over each configuration DOF.
    grad = np.zeros_like(q)
    for d in range(model.n_dof):
        plus, minus = q.copy(), q.copy()
        plus[:, d] += eps
        minus[:, d] -= eps
        ke_plus = 0.5 * np.einsum("ti,tij,tj->t", qd, inertia(model, plus, seg), qd)
        ke_minus = 0.5 * np.einsum("ti,tij,tj->t", qd, inertia(model, minus, seg), qd)
        grad[:, d] = (ke_plus - ke_minus) / (2 * eps)
    coriolis = np.einsum("tij,tj->ti", dm_dt, qd) - grad
    g = gravity_torque(model, q, seg)
    tau = np.einsum("tij,tj->ti", m_q, qdd) + coriolis + g
    return {
        "tau": tau,
        "inertia": m_q,
        "coriolis": coriolis,
        "gravity": g,
        "qd": qd,
        "qdd": qdd,
    }


def forward_replay(
    model: ArticulatedModel,
    q: Array,
    fps: float,
    seg: Segments,
    tau: Array,
    *,
    damping: float = 1e-9,
) -> dict[str, Any]:
    """Integrate ``tau`` from the fitted initial state; report drift from ``q``.

    Semi-implicit Euler at the capture rate, with ``M`` and the non-inertial
    terms evaluated at the fitted configuration of each step (a linearised
    replay: it tests that the torques reproduce the accelerations, which is
    the consistency #9714 asks for, without accumulating the chaotic drift a
    free multi-link pendulum shows over seconds). Returns per-DOF RMS error,
    the worst DOF and the replayed trajectory.
    """
    q = np.asarray(q, dtype=float)
    require(tau.shape == q.shape, "tau must match q", (tau.shape, q.shape))
    dt = 1.0 / fps
    parts = inverse_dynamics(model, q, fps, seg)
    m_q, coriolis, g = parts["inertia"], parts["coriolis"], parts["gravity"]
    replay = np.zeros_like(q)
    replay[0] = q[0]
    velocity = parts["qd"][0].copy()
    for t in range(1, q.shape[0]):
        rhs = tau[t - 1] - coriolis[t - 1] - g[t - 1] - damping * velocity
        accel = np.linalg.solve(m_q[t - 1] + 1e-9 * np.eye(q.shape[1]), rhs)
        velocity = velocity + accel * dt
        replay[t] = replay[t - 1] + velocity * dt
    err = replay - q
    rms = np.sqrt(np.mean(err**2, axis=0))
    worst = int(np.argmax(rms))
    return {
        "replay": replay,
        "rms_per_dof": rms,
        "worst_dof": model.dof_names[worst],
        "worst_rms": float(rms[worst]),
        "max_abs_error": float(np.max(np.abs(err))),
    }


def kinetics_report(
    model: ArticulatedModel,
    q: Array,
    fps: float,
    *,
    body_mass_kg: float,
    names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Torques, peak values and the replay check as a JSON-ready record."""
    seg = segment_masses(model, body_mass_kg)
    parts = inverse_dynamics(model, q, fps, seg)
    replay = forward_replay(model, q, fps, seg, parts["tau"])
    tau = parts["tau"]
    peak = {
        name: float(np.max(np.abs(tau[:, i]))) for i, name in enumerate(model.dof_names)
    }
    rms = {
        name: float(replay["rms_per_dof"][i]) for i, name in enumerate(model.dof_names)
    }
    return {
        "schema_version": "model-kinetics/1.0.0",
        "model": model.spec.name,
        "body_mass_kg": body_mass_kg,
        "fps": fps,
        "assumptions": (
            "point masses at segment midpoints (de Leva fractions), no rod inertia, "
            "linearised semi-implicit Euler replay at the capture rate"
        ),
        "dof_names": list(model.dof_names),
        "simscape_names": dict(names or {}),
        "tau": tau.tolist(),
        "peak_torque": peak,
        "replay_rms_per_dof": rms,
        "replay_worst_dof": replay["worst_dof"],
        "replay_max_abs_error": replay["max_abs_error"],
    }
