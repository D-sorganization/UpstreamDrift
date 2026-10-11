"""Joint-equality couplings of a MuJoCo musculoskeletal model (issue #11644).

MyoFullBody keeps OpenSim's dependent coordinates (knee translations, scapula
and clavicle chain, intervertebral joints, ``shoulder1_r2``) as MuJoCo
``mjEQ_JOINT`` equality constraints.  A constraint solver would only satisfy
them softly, so for kinematics and moment arms this module substitutes them
exactly: ``q1 = q1_0 + c0 + c1 x + c2 x^2 + ...`` with ``x = q2 - q2_0``
(MuJoCo's joint-equality law, ``q_0 = model.qpos0``).  The chain rule through
the substitution gives moment arms with respect to the *independent*
coordinates only, which is what OpenSim reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.contracts import require

_POLY_TERMS = 5


@dataclass(frozen=True)
class JointCoupling:
    """Exact substitution of the joint equalities of one model.

    Attributes:
        qpos0: reference coordinates (``model.qpos0``).
        dependent: qpos addresses of the dependent joints, in solve order.
        driver: qpos address driving each dependent joint, ``-1`` if fixed.
        poly: ``(n_dependent, 5)`` polynomial coefficients.
    """

    qpos0: np.ndarray
    dependent: tuple[int, ...]
    driver: tuple[int, ...]
    poly: np.ndarray

    def __post_init__(self) -> None:
        require(self.poly.shape == (len(self.dependent), _POLY_TERMS), "bad poly shape")
        require(len(self.driver) == len(self.dependent), "driver/dependent mismatch")

    @classmethod
    def from_model(cls, model: Any) -> JointCoupling:
        """Collect the active joint equalities of ``model``.

        Raises:
            ValueError: if a dependency is circular or a joint is not 1-DOF.
        """
        import mujoco

        rows: dict[int, tuple[int, np.ndarray]] = {}
        for e in range(model.neq):
            if model.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            dep, drv = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
            for joint in (dep, drv):
                if joint >= 0 and model.jnt_type[joint] not in (
                    mujoco.mjtJoint.mjJNT_HINGE,
                    mujoco.mjtJoint.mjJNT_SLIDE,
                ):
                    raise ValueError("joint equalities must couple 1-DOF joints")
            adr = int(model.jnt_qposadr[dep])
            src = int(model.jnt_qposadr[drv]) if drv >= 0 else -1
            rows[adr] = (src, np.asarray(model.eq_data[e, :_POLY_TERMS], dtype=float))
        order: list[int] = []
        pending = set(rows)
        while pending:
            ready = [a for a in sorted(pending) if rows[a][0] not in pending]
            if not ready:
                raise ValueError("circular joint equality chain")
            order.extend(ready)
            pending.difference_update(ready)
        return cls(
            qpos0=np.asarray(model.qpos0, dtype=float).copy(),
            dependent=tuple(order),
            driver=tuple(rows[a][0] for a in order),
            poly=np.array([rows[a][1] for a in order]).reshape(-1, _POLY_TERMS),
        )

    @property
    def n_q(self) -> int:
        return int(self.qpos0.shape[0])

    def free_mask(self) -> np.ndarray:
        """Boolean ``(nq,)``: True for coordinates that are not dependent."""
        mask = np.ones(self.n_q, dtype=bool)
        mask[list(self.dependent)] = False
        return mask

    def _levels(self) -> list[np.ndarray]:
        """Indices (into ``dependent``) grouped so each group's drivers are ready."""
        cached = self.__dict__.get("_level_cache")
        if cached is not None:
            return cached  # type: ignore[no-any-return]
        pending = set(range(len(self.dependent)))
        done: set[int] = set()
        adr_of = {a: i for i, a in enumerate(self.dependent)}
        levels: list[np.ndarray] = []
        while pending:
            ready = [
                i
                for i in sorted(pending)
                if self.driver[i] < 0
                or adr_of.get(self.driver[i], -1) in done
                or self.driver[i] not in adr_of
            ]
            levels.append(np.array(ready, dtype=int))
            done.update(ready)
            pending.difference_update(ready)
        object.__setattr__(self, "_level_cache", levels)
        return levels

    def expand(self, q: np.ndarray) -> np.ndarray:
        """Copy of ``q`` with every dependent coordinate set from its driver."""
        require(q.shape == (self.n_q,), "q has the wrong size")
        out = q.astype(float).copy()
        adr = np.array(self.dependent, dtype=int)
        src = np.array(self.driver, dtype=int)
        for level in self._levels():
            drive = src[level]
            x = np.where(drive < 0, 0.0, out[drive] - self.qpos0[drive])
            coef = self.poly[level]
            value = coef[:, 4]
            for k in (3, 2, 1, 0):
                value = value * x + coef[:, k]
            out[adr[level]] = self.qpos0[adr[level]] + value
        return out

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """``d q_full / d q_free``, shape ``(nq, nq)`` (zero columns if dependent).

        ``q`` is the (expanded) coordinate vector at which the slopes are taken.
        """
        require(q.shape == (self.n_q,), "q has the wrong size")
        jac = np.eye(self.n_q)
        for adr in self.dependent:
            jac[adr, adr] = 0.0
        for adr, src, coef in zip(self.dependent, self.driver, self.poly, strict=True):
            if src < 0:
                continue
            x = q[src] - self.qpos0[src]
            slope = float(
                np.polynomial.polynomial.polyval(
                    x, np.polynomial.polynomial.polyder(coef)
                )
            )
            jac[adr, :] = slope * jac[src, :]
        return jac
