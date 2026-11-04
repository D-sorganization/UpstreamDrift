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
  straight-shaft position.

Also exposed per sample: the deflection rate (a resonance proxy) and the
elastic strain energy ``1/2 sum k q^2`` of the chain springs.
"""

from __future__ import annotations

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

    @property
    def peak_deflection_m(self) -> float:
        """Largest tip deflection magnitude over the rollout."""
        return float(np.max(np.linalg.norm(self.tip_deflection_m, axis=1)))

    def at_impact(self, index: int) -> dict[str, Any]:
        """Report the shaft state at one sample (the impact index)."""
        require(0 <= index < len(self.times_s), "impact index out of range", index)
        return {
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


def shaft_state_series(
    model: Any,
    times_s: object,
    qpos: object,
    qvel: object,
    chain_dofs: object,
    tip_site: str,
    frame: TargetFrame,
) -> ShaftImpactState:
    """Shaft state along a MuJoCo rollout of a chain-shaft model.

    ``chain_dofs`` are the chain coordinates' dof indices (1-DOF joints, so
    they are also the ``qpos`` columns); ``tip_site`` marks the head.

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
    stiffness = np.asarray(model.jnt_stiffness)[np.asarray(model.dof_jntid)[dofs]]
    data = mujoco.MjData(model)
    jacp = np.zeros((3, model.nv))
    n = t.shape[0]
    deflection, lead, droop = np.zeros((n, 3)), np.zeros(n), np.zeros(n)
    for i in range(n):
        data.qpos[:], data.qvel[:] = q[i], v[i]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        tip = data.site_xpos[site].copy()
        mujoco.mj_jacSite(model, data, jacp, None, site)
        data.qpos[dofs] = 0.0
        mujoco.mj_kinematics(model, data)
        deflection[i] = tip - data.site_xpos[site]
        lead[i], droop[i] = decompose_deflection(deflection[i], jacp @ v[i], frame)
    rate = (
        np.linalg.norm(np.gradient(deflection, t, axis=0), axis=1)
        if n > 1
        else np.zeros(n)
    )
    energy = 0.5 * np.sum(stiffness * q[:, dofs] ** 2, axis=1)
    ensure(bool(np.all(np.isfinite(deflection))), "deflection must be finite")
    return ShaftImpactState(t, deflection, lead, droop, rate, energy)
