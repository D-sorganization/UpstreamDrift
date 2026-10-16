"""MuJoCo-muscle (MyoSuite) muscle frame for the co-contraction experiment (NSX-94).

MyoSuite muscles are MuJoCo ``muscle`` actuators on spatial tendons.  Their
tension is

    T(a, L, V) = -F0 ( a FL(L) FV(V) + FP(L) )           (MuJoCo sign: force < 0),

*linear in the activation* at fixed length and velocity, and the tendon is
**rigid** (the actuator length is the tendon length; there is no series elastic
element).  So the frame is exact for the secant law, the fibre stiffness is
``k_m = dT/dL`` at the posture and the series stiffness is the ``k_T -> inf``
limit ``k_MTU = k_m``.  Tendon compliance is *not* modelled, which is why the
capability registry declares this engine ``PARTIAL``
(:func:`default_muscle_registry`).

Frame at a posture.  The caller's ``MjData`` is never modified; a scratch copy
is evaluated at zero velocity (so ``FV = FV(0)`` and the torque is the posture's
static muscle torque).  ``R = -actuator_moment`` (MuJoCo's moment is ``dL/dq``).
``k_m`` is obtained per muscle as the least-squares ratio of ``-dT_k/dq`` to
``R_k`` over the chosen coordinates -- exact when ``T_k`` depends on ``q`` only
through ``L_k``, which holds for a rigid tendon -- by central differences of
``q``.

Non-claims: MuJoCo's generic muscle curves are not a golfer's muscles; a
descending-limb length makes ``k_m`` negative, and that fails closed downstream
(:class:`StiffnessNotQualified`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.evidence import (
    Array,
    InputProvenance,
    ModelProvenance,
)
from src.shared.python.loop_nullspace.muscle_frame import MuscleFrame, TendonKind

__all__ = ["mujoco_muscle_frame"]

DEFAULT_GRID = (0.0, 0.5, 1.0)
Q_STEP = 1e-6
"""Central-difference step (rad or m) for ``dT/dq`` and ``dR/dq``: truncation
``h^2 T'''/6`` and round-off ``eps T / h ~ 1e-10 T`` are both far below the
``1e-4`` relative agreement the tests demand."""

_MUSCLE_GAIN = 2  # mjtGain.mjGAIN_MUSCLE (checked against the enum in the tests)
_NO_DYNAMICS = 0  # mjDYN_NONE


def _model_name(model: Any) -> str:
    first = bytes(model.names).split(b"\x00", 1)[0].decode()
    return first or "mujoco-model"


def _resolve(model: Any, kind: str, names: Sequence[str] | None) -> list[int]:
    import mujoco

    obj = {
        "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
        "joint": mujoco.mjtObj.mjOBJ_JOINT,
    }[kind]
    if names is None:
        return []
    out = []
    for name in names:
        idx = mujoco.mj_name2id(model, obj, name)
        if idx < 0:
            raise ValueError(f"{kind}s not in the model: {name!r}")
        out.append(int(idx))
    return out


def _muscle_ids(model: Any, names: Sequence[str] | None) -> list[int]:
    ids = _resolve(model, "actuator", names)
    if names is not None:
        return ids
    return [
        i for i in range(model.nu) if int(model.actuator_gaintype[i]) == _MUSCLE_GAIN
    ]


def _dof_ids(model: Any, names: Sequence[str] | None) -> tuple[list[int], list[str]]:
    import mujoco

    if names is None:
        dofs = list(range(model.nv))
        label = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.dof_jntid[d]))
            for d in dofs
        ]
        return dofs, [str(x) for x in label]
    joints = _resolve(model, "joint", names)
    return [int(model.jnt_dofadr[j]) for j in joints], list(names)


class _Probe:
    """Evaluate muscle tension and moment arms at a posture and activation."""

    def __init__(self, model: Any, data: Any, muscles: list[int], dofs: list[int]):
        import mujoco

        self._mj = mujoco
        self.model = model
        self.muscles = muscles
        self.dofs = dofs
        self.work = mujoco.MjData(model)
        self.qpos0 = np.array(data.qpos, dtype=np.float64)

    def at(self, activation: float, dq: Array | None = None) -> tuple[Array, Array]:
        """``(tension (m,), R (m, n))`` at ``qpos0`` retracted by ``dq``."""
        model, work = self.model, self.work
        work.qpos[:] = self.qpos0
        work.qvel[:] = 0.0
        if dq is not None:
            full = np.zeros(model.nv)
            full[self.dofs] = dq
            self._mj.mj_integratePos(model, work.qpos, full, 1.0)
        for i in self.muscles:
            if int(model.actuator_dyntype[i]) == _NO_DYNAMICS:
                work.ctrl[i] = activation
            else:
                work.act[int(model.actuator_actadr[i])] = activation
        self._mj.mj_forward(model, work)
        moment = np.zeros((model.nu, model.nv))
        self._mj.mju_sparse2dense(
            moment,
            work.actuator_moment,
            work.moment_rownnz,
            work.moment_rowadr,
            work.moment_colind,
        )
        arms = -moment[np.ix_(self.muscles, self.dofs)]
        tension = -np.asarray(work.actuator_force)[self.muscles]
        return tension, arms


def _fiber_stiffness(probe: _Probe, level: float, arms: Array) -> Array:
    """``k_m = dT/dL`` per muscle from ``dT/dq`` (module doc)."""
    n = len(probe.dofs)
    d_t = np.zeros((len(probe.muscles), n))
    for j in range(n):
        step = np.zeros(n)
        step[j] = Q_STEP
        plus, _ = probe.at(level, step)
        minus, _ = probe.at(level, -step)
        d_t[:, j] = (plus - minus) / (2.0 * Q_STEP)
    denom = np.sum(arms**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_fib = np.where(denom > 0.0, -np.sum(d_t * arms, axis=1) / denom, 0.0)
    return np.asarray(k_fib)


def _arm_jacobian(probe: _Probe, level: float) -> Array:
    n = len(probe.dofs)
    out = np.zeros((len(probe.muscles), n, n))
    for j in range(n):
        step = np.zeros(n)
        step[j] = Q_STEP
        out[:, :, j] = (probe.at(level, step)[1] - probe.at(level, -step)[1]) / (
            2.0 * Q_STEP
        )
    return out


def mujoco_muscle_frame(
    model: Any,
    data: Any,
    *,
    muscles: Sequence[str] | None = None,
    coordinates: Sequence[str] | None = None,
    grid: Sequence[float] = DEFAULT_GRID,
    with_geometry: bool = False,
    engine: str = "myosuite",
    source: str = "mujoco-posture",
) -> MuscleFrame:
    """The rigid-tendon :class:`MuscleFrame` of a MuJoCo muscle model at ``data``'s posture.

    Preconditions (``ValueError``): the model has ``muscle`` actuators, named
    muscles and joints exist, ``grid`` increases strictly from 0 to 1, and every
    selected muscle has positive ``F0``.  ``data`` is not modified.
    Postconditions: ``force_grid`` is the tension at zero velocity (``>= 0``),
    the tendon is ``RIGID``, ``tendon_stiffness_grid`` is ``None`` and
    ``fiber_stiffness_grid`` is ``dT/dL`` (possibly negative on the descending
    limb; the stiffness layer then fails closed).
    """
    ids = _muscle_ids(model, muscles)
    if not ids:
        raise ValueError("the model has no muscle actuators")
    dofs, dof_names = _dof_ids(model, coordinates)
    probe = _Probe(model, data, ids, dofs)
    levels = np.asarray(grid, dtype=np.float64)
    forces, k_fib = [], []
    arms = probe.at(0.0)[1]
    for level in levels:
        tension, level_arms = probe.at(float(level))
        forces.append(np.maximum(tension, 0.0))
        k_fib.append(_fiber_stiffness(probe, float(level), level_arms))
    f_max = np.array([float(model.actuator_gainprm[i, 2]) for i in ids])
    names = tuple(str(model.actuator(i).name) for i in ids)
    return MuscleFrame(
        names=names,
        coordinates=tuple(dof_names),
        moment_arms=arms,
        activation_grid=levels,
        force_grid=np.array(forces),
        max_isometric_force=f_max,
        tendon=TendonKind.RIGID,
        engine=engine,
        model_provenance=ModelProvenance(_model_name(model), "1"),
        input_provenance=InputProvenance("mujoco-posture", source),
        fiber_stiffness_grid=np.array(k_fib),
        tendon_stiffness_grid=None,
        moment_arm_jacobian=_arm_jacobian(probe, 0.0) if with_geometry else None,
    )
