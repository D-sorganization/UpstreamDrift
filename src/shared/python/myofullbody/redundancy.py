"""Per-frame static optimisation on MyoFullBody (issue #11645).

Hybrid formulation (the spec owns the dynamics, MyoFullBody owns the muscles):

* The spec inverse dynamics already fixed the 44 joint efforts ``tau`` that
  reproduce the swing in the same-input bundle.
* MyoFullBody contributes only the muscle *moment arms* and *force capacity*:
  at the mapped pose, MuJoCo's ``actuator_moment`` and the force-length-velocity
  curves give, per muscle, the force at zero and at full activation.
* The map from MyoFullBody coordinates to spec coordinates is the rate map
  ``Phi`` of :mod:`mapping`, so ``tau_spec = Phi^T qfrc_myo`` conserves power.
  Muscle ``m`` therefore contributes ``R[m, c] * F_m`` to spec coordinate ``c``
  with ``R = (-actuator_moment) @ Phi`` (MuJoCo force is negative in tension).

Each frame solves ``min sum(a^2) + w * sum(reserve^2)`` with ``0 <= a <= 1`` and
``tau = R^T (F_passive + a * F_active) + reserve`` (the bounded least-squares of
:func:`musculoskeletal_static_opt.solve_frame`).  Reserve actuators exist for
every spec coordinate; their size is the measure of what the muscles cannot do.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from src.shared.python.contracts import require

if TYPE_CHECKING:
    from src.engines.physics_engines.opensim.python import (
        musculoskeletal_static_opt as so,
    )

Array: TypeAlias = np.ndarray

ROOT_PREFIXES = ("TranslationInput", "HipInput")
GROUP_NAMES = ("trunk", "arms", "neck", "legs")
RESERVE_RMS_FRACTION_LIMIT = 0.10
ID_RELATIVE_RMS_LIMIT = 0.01
TORQUE_CLOSURE_LIMIT_NM = 1e-6
UNCOVERED_GROUPS = ("neck",)


def coordinate_groups(coordinate_order: tuple[str, ...]) -> dict[str, list[int]]:
    """Spec column indices of the non-root coordinates by body group.

    The groups are ``trunk`` (spine and torso), ``arms`` (scapula, shoulder,
    elbow, forearm, wrist), ``neck`` (no MyoFullBody muscles) and ``legs``.

    Raises:
        ValueError: if a non-root coordinate belongs to no group.
    """
    out: dict[str, list[int]] = {name: [] for name in GROUP_NAMES}
    for i, name in enumerate(coordinate_order):
        if name.startswith(ROOT_PREFIXES):
            continue
        if name.startswith(("Spine", "Torso")):
            out["trunk"].append(i)
        elif name.startswith("Neck"):
            out["neck"].append(i)
        elif name.startswith(("L", "R")) and name[1:].startswith(
            ("E", "F", "Scap", "S", "W")
        ):
            out["arms"].append(i)
        elif "_" in name:
            out["legs"].append(i)
        else:
            raise ValueError(f"coordinate {name!r} belongs to no body group")
    return out


@dataclass(frozen=True)
class FrameBasis:
    """Muscle capacity and moment arms at one pose.

    Attributes:
        active: ``(nm,)`` force per unit activation (N), >= 0.
        passive: ``(nm,)`` force at zero activation (N), >= 0.
        moment: ``(nm, nc)`` spec-space moment arms (m or rad/rad).
        phi: ``(nv_myo, 44)`` rate map used (kept for the independent check).
    """

    active: Array
    passive: Array
    moment: Array
    phi: Array


class MuscleBasis:
    """MyoFullBody muscle capacity and spec-space moment arms from MuJoCo."""

    def __init__(
        self,
        mapper: Any,
        columns: list[int],
        velocity_map: Callable[[Array, Any], Array] | None = None,
    ) -> None:
        import mujoco

        require(len(columns) > 0, "columns must be non-empty")
        self.mapper = mapper
        self._velocity_map = velocity_map or mapper.velocity_map
        self.model = mapper.model
        self.data = mapper.data
        self.columns = list(columns)
        self.names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        self._moment = np.zeros((self.model.nu, self.model.nv))

    def _forces(self, act: float) -> Array:
        import mujoco

        self.data.act[:] = act
        self.data.ctrl[:] = act
        mujoco.mj_fwdActuation(self.model, self.data)
        return np.asarray(self.data.actuator_force).copy()

    def set_state(self, q_spec: Array, v_spec: Array, pose: Any) -> Array:
        """Place MyoFullBody at the mapped pose with rates ``Phi v``; returns ``Phi``."""
        import mujoco

        phi = self._velocity_map(q_spec, pose)
        self.data.qpos[:] = pose.qpos
        self.data.qvel[:] = phi @ v_spec
        mujoco.mj_fwdPosition(self.model, self.data)
        mujoco.mj_fwdVelocity(self.model, self.data)
        return phi

    def evaluate(self, q_spec: Array, v_spec: Array, pose: Any) -> FrameBasis:
        """Capacity and moment arms at the pose.

        Postconditions: ``active >= 0`` and ``passive >= 0`` (forces in tension
        are positive here) and the actuation is linear in activation, which the
        runner verifies against MuJoCo at the solved activations.
        """
        import mujoco

        phi = self.set_state(q_spec, v_spec, pose)
        f0 = self._forces(0.0)
        f1 = self._forces(1.0)
        mujoco.mju_sparse2dense(
            self._moment,
            self.data.actuator_moment,
            self.data.moment_rownnz,
            self.data.moment_rowadr,
            self.data.moment_colind,
        )
        moment = -self._moment @ phi[:, self.columns]
        return FrameBasis(np.maximum(f0 - f1, 0.0), np.maximum(-f0, 0.0), moment, phi)

    def generalised_force(self, activation: Array, phi: Array) -> Array:
        """Independent route: spec generalised force of muscles at ``activation``.

        Uses MuJoCo's own ``qfrc_actuator`` (not the linear model) mapped by
        ``Phi^T``; the state of the last :meth:`evaluate` call is reused.
        """
        import mujoco

        self.data.act[:] = activation
        self.data.ctrl[:] = activation
        mujoco.mj_fwdActuation(self.model, self.data)
        return phi[:, self.columns].T @ np.asarray(self.data.qfrc_actuator)


def solve_frame(
    active: Array,
    passive: Array,
    moment: Array,
    tau: Array,
    *,
    reserve_weight: float = 100.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> so.FrameSolution:
    """Same problem as ``musculoskeletal_static_opt.solve_frame``, solved fast.

    ``min sum(a^2) + w * sum(reserve^2)`` with ``0 <= a <= 1`` and
    ``reserve = tau - moment^T (passive + a * active)``.  The bounded
    least-squares (BVLS) of that module takes about a minute for 416 muscles;
    here a projected Newton method (Bertsekas) uses the Woodbury identity on the
    free set, so each Newton step solves a system of the size of the joint count.

    Returns:
        A solution with ``success`` set when the projected-gradient residual is
        below ``tol``.  The two solvers agree to solver tolerance (tested).

    Raises:
        ValueError: on shape mismatch, non-finite input or a non-positive weight.
    """
    nm = active.shape[0]
    require(passive.shape == (nm,), "passive must be (nm,)")
    require(moment.ndim == 2 and moment.shape[0] == nm, "moment must be (nm, nc)")
    require(tau.shape == (moment.shape[1],), "tau must be (nc,)")
    require(reserve_weight > 0.0, "reserve_weight must be positive")
    for array in (active, passive, moment, tau):
        require(bool(np.isfinite(array).all()), "inputs must be finite")
    w = float(reserve_weight)
    gain = moment.T * active[None, :]
    offset = tau - moment.T @ passive

    def value(a: Array) -> float:
        r = offset - gain @ a
        return 0.5 * float(a @ a) + 0.5 * w * float(r @ r)

    a = np.zeros(nm)
    converged = False
    for _ in range(max_iter):
        grad = a - w * (gain.T @ (offset - gain @ a))
        gap = np.abs(a - np.clip(a - grad, 0.0, 1.0))
        if gap.max() < tol:
            converged = True
            break
        eps = min(1e-3, float(np.linalg.norm(gap)))
        fixed = ((a <= eps) & (grad > 0.0)) | ((a >= 1.0 - eps) & (grad < 0.0))
        free = ~fixed
        g_free = gain[:, free]
        small = np.eye(gain.shape[0]) / w + g_free @ g_free.T
        x = grad[free]
        step = np.zeros(nm)
        step[free] = -(x - g_free.T @ np.linalg.solve(small, g_free @ x))
        f0, alpha = value(a), 1.0
        for _ in range(40):
            trial = np.clip(a + alpha * step, 0.0, 1.0)
            if value(trial) <= f0 + 1e-4 * float(grad @ (trial - a)):
                break
            alpha *= 0.5
        improvement = f0 - value(trial)
        a = trial
        if improvement <= 1e-12 * (1.0 + f0):
            converged = True  # no representable decrease left (ill-conditioned tail)
            break
    reserve = offset - gain @ a
    cost = float(a @ a + w * (reserve @ reserve))
    from src.engines.physics_engines.opensim.python import (
        musculoskeletal_static_opt as so_runtime,
    )

    return so_runtime.FrameSolution(a, reserve, cost, converged)


def group_reserve_metrics(
    reserve: Array, tau: Array, groups: dict[str, list[int]], columns: list[int]
) -> dict[str, dict[str, float]]:
    """Reserve RMS and peak (N m) per group, with the group's effort RMS and ratio.

    ``reserve`` and ``tau`` are ``(frames, nc)`` over ``columns`` (spec indices).
    """
    require(reserve.shape == tau.shape, "reserve and tau must have the same shape")
    where = {c: i for i, c in enumerate(columns)}
    out: dict[str, dict[str, float]] = {}
    for name, cols in groups.items():
        idx = [where[c] for c in cols if c in where]
        if not idx:
            continue
        r, t = reserve[:, idx], tau[:, idx]
        rms_r = float(np.sqrt(np.mean(r**2)))
        rms_t = float(np.sqrt(np.mean(t**2)))
        out[name] = {
            "reserve_rms_nm": rms_r,
            "reserve_peak_nm": float(np.abs(r).max()),
            "effort_rms_nm": rms_t,
            "effort_peak_nm": float(np.abs(t).max()),
            "reserve_over_effort_rms": rms_r / rms_t if rms_t > 0.0 else 0.0,
            "coordinates": len(idx),
        }
    return out


def qualification(
    metrics: dict[str, dict[str, float]],
    solver_ok: bool,
    id_ok: bool,
    limit: float = RESERVE_RMS_FRACTION_LIMIT,
    uncovered: tuple[str, ...] = UNCOVERED_GROUPS,
) -> dict[str, Any]:
    """Fail-closed status: ``NOT_QUALIFIED`` unless every covered group is within limit.

    Groups in ``uncovered`` (default :data:`UNCOVERED_GROUPS`) have no actuator at
    all, so their entire effort is reserve by construction; they are reported but
    declared as an explicit scope limitation rather than silently passed.  With a
    torque-actuated neck (:mod:`neck`) the caller passes ``uncovered=()`` and the
    neck is held to the same limit as every other group.

    Raises:
        ValueError: if ``limit`` is not positive.
    """
    require(limit > 0.0, "limit must be positive")
    covered = {k: v for k, v in metrics.items() if k not in uncovered}
    failing = sorted(
        k for k, v in covered.items() if v["reserve_over_effort_rms"] > limit
    )
    reasons = [f"reserve RMS above {limit:.0%} of effort RMS: {k}" for k in failing]
    if not covered:
        reasons.append("no muscle-covered coordinate group")
    if not solver_ok:
        reasons.append("bounded least-squares did not converge on every frame")
    if not id_ok:
        reasons.append("inverse-dynamics consistency check failed")
    return {
        "status": "NOT_QUALIFIED" if reasons else "QUALIFIED_SOFTWARE_ONLY",
        "reasons": reasons,
        "reserve_rms_over_effort_rms_limit": limit,
        "uncovered_groups_scope_limitation": [g for g in uncovered if g in metrics],
    }
