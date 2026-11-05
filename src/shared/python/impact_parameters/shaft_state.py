"""Flexible-shaft state of a MuJoCo rollout for the optimiser loop (SHOW-C4, #14560).

The R6 chain shaft (``motion_matching/club_shaft_chain.py``) is already part of
the MuJoCo full-body export.  This module reads its state back out of a
rollout so the fit and the optimiser can report kick/droop at impact and
penalise shaft exploits (wrist-snap, shaft resonance, #14556 panel review).

Tip deflection is the R11 definition (``flex_shaft/r11/chain_deflection.py``)
evaluated with MuJoCo kinematics: the displacement of the tip site caused by
the bent shaft, ``p(q) - p(q with the chain coordinates zeroed)``, in the world
frame.  It is split in the target frame (:class:`TargetFrame`):

* ``lead_m``: component along the tip velocity; positive = the head is ahead
  of the straight shaft (forward kick), negative = lag.
* ``droop_m``: component along gravity (down); positive = head below the
  straight-shaft position.  It is the gravity component of the whole bend, so
  it mixes in the lead/lag bend when the shaft is not vertical; use
  ``toe_down_m`` for the toe-down of the shaft literature.

* ``toe_down_m`` (#14600): component along the normal of the plane spanned by
  the shaft axis and the tip velocity, oriented toward gravity; positive = the
  head is out of the swing plane on the ground side.  This is the toe-down
  deflection of the shaft literature; ``droop_m`` is the gravity component of
  the whole deflection and so also carries the lead/lag bend (the shaft is far
  from vertical at impact).

Also exposed per sample: the deflection rate (a resonance proxy) and the
elastic strain energy ``1/2 sum k q^2`` of the chain springs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.contracts import ensure, require

from .target_frame import TargetFrame

_MIN_SPEED_MPS = 1e-6


@dataclass(frozen=True)
class ShaftImpactState:
    """Per-sample shaft state (SI units, world frame for vectors)."""

    times_s: np.ndarray
    tip_deflection_m: np.ndarray
    lead_m: np.ndarray
    droop_m: np.ndarray
    tip_deflection_rate_mps: np.ndarray
    strain_energy_j: np.ndarray
    face_delta_deg: np.ndarray | None = None
    loft_delta_deg: np.ndarray | None = None
    toe_down_m: np.ndarray | None = None

    @property
    def peak_deflection_m(self) -> float:
        """Largest tip deflection magnitude over the rollout."""
        return float(np.max(np.linalg.norm(self.tip_deflection_m, axis=1)))

    def at_impact(self, index: int) -> dict[str, Any]:
        """Report the shaft state at one sample (the impact index)."""
        require(0 <= index < len(self.times_s), "impact index out of range", index)
        shaft_angles = {}
        if self.face_delta_deg is not None and self.loft_delta_deg is not None:
            shaft_angles = {
                "face_delta_deg": float(self.face_delta_deg[index]),
                "loft_delta_deg": float(self.loft_delta_deg[index]),
            }
        toe_down = (
            {}
            if self.toe_down_m is None
            else {"toe_down_m": float(self.toe_down_m[index])}
        )
        return {
            **shaft_angles,
            **toe_down,
            "time_s": float(self.times_s[index]),
            "lead_m": float(self.lead_m[index]),
            "droop_m": float(self.droop_m[index]),
            "tip_deflection_m": float(np.linalg.norm(self.tip_deflection_m[index])),
            "tip_deflection_rate_mps": float(self.tip_deflection_rate_mps[index]),
            "strain_energy_j": float(self.strain_energy_j[index]),
        }


def decompose_deflection(
    deflection: np.ndarray, velocity: np.ndarray, frame: TargetFrame
) -> tuple[float, float]:
    """``(lead_m, droop_m)`` of a world-frame tip deflection.

    ``lead`` is along ``velocity`` (0.0 when the tip is at rest: undefined),
    ``droop`` is along ``-up`` of ``frame``.
    """
    speed = float(np.linalg.norm(velocity))
    lead = 0.0 if speed < _MIN_SPEED_MPS else float(deflection @ velocity) / speed
    return lead, -float(deflection @ frame.z_t)


def toe_down_component(
    deflection: np.ndarray,
    velocity: np.ndarray,
    shaft_axis: np.ndarray,
    frame: TargetFrame,
) -> float:
    """Toe-down part of a world-frame tip deflection (metres, + = ground side).

    The normal of the (shaft axis, tip velocity) plane, oriented along
    ``-up`` of ``frame``, dotted with ``deflection``. Returns 0.0 when the
    plane is undefined (tip at rest, velocity parallel to the shaft) or the
    normal is horizontal (no ground side).
    """
    normal = np.cross(velocity, shaft_axis)
    length = float(np.linalg.norm(normal))
    if length < _MIN_SPEED_MPS:
        return 0.0
    toward_ground = -float(normal @ frame.z_t)
    if abs(toward_ground) < _MIN_SPEED_MPS * length:
        return 0.0
    return float(deflection @ normal) / length * math.copysign(1.0, toward_ground)


def face_loft_deltas(
    bent_normal: np.ndarray, straight_normal: np.ndarray, frame: TargetFrame
) -> tuple[float, float]:
    """``(face_delta_deg, loft_delta_deg)`` of the face normal caused by the bend.

    Face + = more open (GCV-15 heading of the normal), loft + = normal tilted up.
    """

    def angles(n: np.ndarray) -> tuple[float, float]:
        unit = n / np.linalg.norm(n)
        y_t = np.cross(frame.z_t, frame.x_t)
        heading = np.degrees(np.arctan2(-(unit @ y_t), unit @ frame.x_t))
        return float(heading), float(np.degrees(np.arcsin(unit @ frame.z_t)))

    (face_b, loft_b), (face_s, loft_s) = angles(bent_normal), angles(straight_normal)
    return face_b - face_s, loft_b - loft_s


def shaft_state_series(
    model: Any,
    times_s: object,
    qpos: object,
    qvel: object,
    chain_dofs: object,
    tip_site: str,
    frame: TargetFrame,
    face_normal_site: object | None = None,
) -> ShaftImpactState:
    """Shaft state along a MuJoCo rollout of a chain-shaft model.

    ``chain_dofs`` are the chain coordinates' dof indices (1-DOF joints, so
    they are also the ``qpos`` columns); ``tip_site`` marks the head.
    ``face_normal_site`` (the face normal in the tip-site frame) adds the dynamic
    face and loft change caused by the bent shaft.

    Raises:
        ValueError: shapes of ``times_s``/``qpos``/``qvel`` disagree, the dofs
            are out of range, or the site is unknown.
    """
    import mujoco

    t = np.asarray(times_s, dtype=float)
    q, v = np.asarray(qpos, dtype=float), np.asarray(qvel, dtype=float)
    dofs = np.asarray(chain_dofs, dtype=int)
    require(q.shape == v.shape, "qpos and qvel must have the same shape", q.shape)
    require(
        q.ndim == 2 and q.shape == (t.shape[0], model.nq),
        "qpos must be (len(times_s), nq)",
        q.shape,
    )
    require(
        dofs.size > 0 and bool(np.all((dofs >= 0) & (dofs < model.nv))),
        "chain_dofs out of range",
    )
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip_site)
    require(site >= 0, f"MuJoCo model has no site {tip_site!r}")
    normal = None if face_normal_site is None else np.asarray(face_normal_site, float)
    require(
        normal is None or normal.shape == (3,), "face_normal_site must be a 3-vector"
    )
    stiffness = np.asarray(model.jnt_stiffness)[np.asarray(model.dof_jntid)[dofs]]
    data = mujoco.MjData(model)
    jacp = np.zeros((3, model.nv))
    n = t.shape[0]
    deflection, lead, droop = np.zeros((n, 3)), np.zeros(n), np.zeros(n)
    toe_down = np.zeros(n)
    angles = np.zeros((n, 2))
    for i in range(n):
        data.qpos[:], data.qvel[:] = q[i], v[i]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        tip, bent_rot = (
            data.site_xpos[site].copy(),
            data.site_xmat[site].reshape(3, 3).copy(),
        )
        mujoco.mj_jacSite(model, data, jacp, None, site)
        data.qpos[dofs] = 0.0
        mujoco.mj_kinematics(model, data)
        deflection[i] = tip - data.site_xpos[site]
        if normal is not None:
            straight_rot = data.site_xmat[site].reshape(3, 3)
            angles[i] = face_loft_deltas(
                bent_rot @ normal, straight_rot @ normal, frame
            )
        tip_velocity = jacp @ v[i]
        lead[i], droop[i] = decompose_deflection(deflection[i], tip_velocity, frame)
        shaft_axis = data.site_xmat[site].reshape(3, 3)[:, 1]  # straight shaft
        toe_down[i] = toe_down_component(deflection[i], tip_velocity, shaft_axis, frame)
    rate = (
        np.linalg.norm(np.gradient(deflection, t, axis=0), axis=1)
        if n > 1
        else np.zeros(n)
    )
    energy = 0.5 * np.sum(stiffness * q[:, dofs] ** 2, axis=1)
    ensure(bool(np.all(np.isfinite(deflection))), "deflection must be finite")
    face, loft = (None, None) if normal is None else (angles[:, 0], angles[:, 1])
    return ShaftImpactState(
        t, deflection, lead, droop, rate, energy, face, loft, toe_down
    )
