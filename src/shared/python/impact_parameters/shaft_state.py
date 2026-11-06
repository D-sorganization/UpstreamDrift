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
  straight-shaft position.  It is the gravity component of the whole bend:
  the deflection splits into ``lead`` (along the tip velocity), an in-swing-
  plane part perpendicular to the velocity and the swing-plane normal
  (``toe_down_m``), and each enters through its ``e_i . up`` cosine
  (:func:`droop_decomposition`).  In the C4 replay the in-plane perpendicular
  part, not the lead, dominates the negative droop.
* ``toe_down_m`` (#14600): component along the normal of the plane spanned by
  the shaft axis and the tip velocity, oriented toward gravity; positive = the
  head is out of the swing plane on the ground side.  This is the toe-down
  deflection of the shaft literature; ``droop_m`` is a different quantity (the
  gravity component of the whole deflection).

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
    tip_velocity_mps: np.ndarray | None = None
    shaft_axis: np.ndarray | None = None
    frame: TargetFrame | None = None

    @property
    def peak_deflection_m(self) -> float:
        """Largest tip deflection magnitude over the rollout."""
        return float(np.max(np.linalg.norm(self.tip_deflection_m, axis=1)))

    def _decomposition_report(self, index: int) -> dict[str, Any]:
        """``{"droop_decomposition": {...}}`` when the sample has one."""
        if (
            self.tip_velocity_mps is None
            or self.shaft_axis is None
            or self.frame is None
        ):
            return {}
        report = droop_decomposition(
            self.tip_deflection_m[index],
            self.tip_velocity_mps[index],
            self.shaft_axis[index],
            self.frame,
        )
        return {} if report is None else {"droop_decomposition": report}

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
            **self._decomposition_report(index),
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


def droop_decomposition(
    deflection: np.ndarray,
    velocity: np.ndarray,
    shaft_axis: np.ndarray,
    frame: TargetFrame,
) -> dict[str, float] | None:
    """How a tip deflection enters ``droop_m`` (#14600).

    Orthonormal basis: ``e1`` along the tip velocity, ``e2`` the shaft axis
    with its ``e1`` part removed, ``e3 = e1 x e2`` (the swing-plane normal).
    ``droop = -d . up = sum_i -d_i (e_i . up)``; each ``term_*`` is one such
    sum member (m) and ``residual_m`` is ``droop_m`` minus their sum.  Also
    returns ``v_dot_z``, ``v_dot_s`` and ``s_dot_z`` (cosines against ``up``
    and the shaft axis) and the coordinates ``lead_m`` (= d1), ``in_plane_perp_m``
    (d2), ``normal_m`` (d3).  ``None`` when the basis is undefined (tip at rest
    or velocity parallel to the shaft).
    """
    speed = float(np.linalg.norm(velocity))
    shaft = np.asarray(shaft_axis, dtype=float)
    shaft = shaft / np.linalg.norm(shaft)
    if speed < _MIN_SPEED_MPS:
        return None
    e1 = np.asarray(velocity, dtype=float) / speed
    perp = shaft - float(shaft @ e1) * e1
    if float(np.linalg.norm(perp)) < 1e-9:
        return None
    e2 = perp / np.linalg.norm(perp)
    e3 = np.cross(e1, e2)
    up = np.asarray(frame.z_t, dtype=float)
    coords = [float(deflection @ e) for e in (e1, e2, e3)]
    terms = [-c * float(e @ up) for c, e in zip(coords, (e1, e2, e3), strict=True)]
    droop = -float(deflection @ up)
    return {
        "v_dot_z": float(e1 @ up),
        "v_dot_s": float(e1 @ shaft),
        "s_dot_z": float(shaft @ up),
        "lead_m": coords[0],
        "in_plane_perp_m": coords[1],
        "normal_m": coords[2],
        "term_lead_m": terms[0],
        "term_in_plane_perp_m": terms[1],
        "term_normal_m": terms[2],
        "droop_m": droop,
        "residual_m": droop - sum(terms),
    }


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


def _chain_axis(model: Any, data: Any, dofs: np.ndarray) -> np.ndarray:
    """Unit shaft direction (grip to tip) of the straightened chain.

    Taken from the first and last chain joint anchors, not from the tip site's
    frame: the full-body ``Clubhead`` site is a face frame whose y axis is
    perpendicular to the shaft (#14600), so it cannot name the shaft direction.
    A single-joint chain falls back to its body's local y axis.
    """
    joints = np.asarray(model.dof_jntid)[dofs]
    span = data.xanchor[joints[-1]] - data.xanchor[joints[0]]
    length = float(np.linalg.norm(span))
    if length > 1e-9:
        return span / length
    return data.xmat[int(model.jnt_bodyid[joints[0]])].reshape(3, 3)[:, 1].copy()


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
    toe_down, tip_vel, axes = np.zeros(n), np.zeros((n, 3)), np.zeros((n, 3))
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
        shaft_axis = _chain_axis(model, data, dofs)  # straight shaft
        toe_down[i] = toe_down_component(deflection[i], tip_velocity, shaft_axis, frame)
        tip_vel[i], axes[i] = tip_velocity, shaft_axis
    rate = (
        np.linalg.norm(np.gradient(deflection, t, axis=0), axis=1)
        if n > 1
        else np.zeros(n)
    )
    energy = 0.5 * np.sum(stiffness * q[:, dofs] ** 2, axis=1)
    ensure(bool(np.all(np.isfinite(deflection))), "deflection must be finite")
    face, loft = (None, None) if normal is None else (angles[:, 0], angles[:, 1])
    return ShaftImpactState(
        t,
        deflection,
        lead,
        droop,
        rate,
        energy,
        face,
        loft,
        toe_down,
        tip_vel,
        axes,
        frame,
    )
