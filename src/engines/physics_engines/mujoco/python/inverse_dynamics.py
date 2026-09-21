"""MuJoCo-native inverse dynamics and computed-torque tracking via mj_inverse.

Computes constraint-consistent inverse torques accounting for floating-base
coordinates, equality constraints (dual-grip weld), and ground contact reaction forces.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.contracts import postcondition, precondition

if TYPE_CHECKING:
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )

logger = logging.getLogger(__name__)

Controller = Callable[[float, np.ndarray, np.ndarray], np.ndarray]


class MujocoInverseDynamics:
    """MuJoCo-native inverse dynamics stage using ``mj_inverse``."""

    def __init__(self, model: NativeMujocoFullBodyModel) -> None:
        """Initialize inverse dynamics with a native MuJoCo full-body model adapter."""
        mj: Any = import_module("mujoco")
        self._mj = mj
        self.adapter = model
        self.model = model.model
        self.data = mj.MjData(self.model)

        self.coordinate_order: list[str] = list(model.coordinate_order)
        self.nv: int = len(self.coordinate_order)
        self.spec_to_mj: np.ndarray = np.array(
            [model._indices[name] for name in self.coordinate_order], dtype=int
        )
        self.mj_to_spec: np.ndarray = np.zeros(self.nv, dtype=int)
        for spec_idx, mj_dof in enumerate(self.spec_to_mj):
            self.mj_to_spec[mj_dof] = spec_idx
        self.root_mj_dofs: np.ndarray = self.spec_to_mj[:6]
        self.act_mj_dofs: np.ndarray = self.spec_to_mj[6:]

    @precondition(
        lambda self, q, v, a, **kwargs: (
            np.asarray(q).ndim == 1
            and np.asarray(v).ndim == 1
            and np.asarray(a).ndim == 1
            and len(q) == self.nv
            and len(v) == self.nv
            and len(a) == self.nv
            and np.isfinite(q).all()
            and np.isfinite(v).all()
            and np.isfinite(a).all()
        ),
        "q, v, and a must be finite 1D arrays matching coordinate count",
    )
    @postcondition(
        lambda result: bool(
            np.asarray(result).ndim == 1
            and len(result) == 41
            and np.isfinite(result).all()
            and np.all(np.asarray(result)[:6] == 0.0)
        ),
        "Computed torques must be finite with zero root effort",
    )
    def compute_inverse_torques(
        self,
        q: np.ndarray,
        v: np.ndarray,
        a: np.ndarray,
        *,
        compensate_contact: bool = True,
    ) -> np.ndarray:
        """Evaluate constraint-consistent inverse torques at state (q, v) with acceleration a.

        Args:
            q: Joint positions in pipeline coordinate order.
            v: Joint velocities in pipeline coordinate order.
            a: Target joint accelerations in pipeline coordinate order.
            compensate_contact: When True, subtracts ground reaction forces.

        Returns:
            Torques in pipeline coordinate order with zero root effort.
        """
        mj = self._mj
        self.data.qpos[self.spec_to_mj] = q
        self.data.qvel[self.spec_to_mj] = v
        self.data.qacc[self.spec_to_mj] = a
        mj.mj_inverse(self.model, self.data)

        if compensate_contact:
            coord_map = {
                name: float(q[i]) for i, name in enumerate(self.coordinate_order)
            }
            rate_map = {
                name: float(v[i]) for i, name in enumerate(self.coordinate_order)
            }
            _, tau_contact, _ = self.adapter.generalized_forces(coord_map, rate_map)
            tau_mj = self.data.qfrc_inverse - tau_contact
        else:
            tau_mj = self.data.qfrc_inverse.copy()

        tau_mj[self.root_mj_dofs] = 0.0
        return tau_mj[self.spec_to_mj].copy()

    @precondition(
        lambda self, q, v, wanted_acceleration, **kwargs: (
            np.asarray(q).ndim == 1
            and np.asarray(v).ndim == 1
            and np.asarray(wanted_acceleration).ndim == 1
            and len(q) == self.nv
            and len(v) == self.nv
            and len(wanted_acceleration) in (self.nv - 6, self.nv)
            and np.isfinite(q).all()
            and np.isfinite(v).all()
            and np.isfinite(wanted_acceleration).all()
        ),
        "State and wanted accelerations must be finite with valid dimensions",
    )
    def inverse_dynamics(
        self,
        q: np.ndarray,
        v: np.ndarray,
        wanted_acceleration: np.ndarray,
        *,
        compensate_contact: bool = True,
    ) -> np.ndarray:
        """Compute actuated joint torques producing target accelerations."""
        target = np.asarray(wanted_acceleration, dtype=float)
        a_full = np.zeros(self.nv, dtype=float)
        if target.shape == (self.nv - 6,):
            a_full[6:] = target
        else:
            a_full[:] = target
        return self.compute_inverse_torques(
            q, v, a_full, compensate_contact=compensate_contact
        )

    def create_tracking_controller(
        self,
        sim: Any,
        time_ref: Sequence[float] | np.ndarray,
        q_ref: np.ndarray,
        *,
        omega_rad_s: float | np.ndarray = 25.0,
        zeta: float = 1.0,
        balance: tuple[float, float] | None = None,
        root_regulation: tuple[float, float] | None = None,
        acceleration_feedforward: float = 1.0,
    ) -> Controller:
        """Construct a computed-torque tracking controller matching FullBodySimulator contract."""
        from src.shared.python.motion_matching import full_body_forward_dynamics as fs

        times = np.asarray(time_ref, dtype=float)
        reference = np.asarray(q_ref, dtype=float)
        if (
            times.ndim != 1
            or np.any(np.diff(times) <= 0)
            or reference.shape != (times.size, self.nv)
            or not np.isfinite(reference).all()
        ):
            raise ValueError("Reference times must increase with one finite q row each")

        if not 0.0 <= acceleration_feedforward <= 1.0:
            raise ValueError("acceleration_feedforward must lie in [0, 1]")

        fs._check_gains(omega_rad_s, zeta, balance)
        fs._check_gains(1.0, 1.0, root_regulation)

        if times.size > 1:
            velocity = np.gradient(reference, times, axis=0)
            acceleration = acceleration_feedforward * np.gradient(
                velocity, times, axis=0
            )
        else:
            velocity = np.zeros_like(reference)
            acceleration = np.zeros_like(reference)

        act = sim.actuated
        omega = np.broadcast_to(np.asarray(omega_rad_s, dtype=float), (self.nv,))[act]

        def sample(table: np.ndarray, t: float) -> np.ndarray:
            return np.array(
                [np.interp(t, times, table[:, k]) for k in range(self.nv)], dtype=float
            )

        def controller(t: float, q: np.ndarray, v: np.ndarray) -> np.ndarray:
            q_t, v_t, a_t = (
                sample(reference, t),
                sample(velocity, t),
                sample(acceleration, t),
            )
            wanted_act = (
                a_t[act]
                + 2.0 * zeta * omega * (v_t[act] - v[act])
                + (omega**2) * (q_t[act] - q[act])
            )
            if balance is not None:
                com_ref = sim.centre_of_mass(q_t)[0]
                wanted_act = wanted_act + fs._balance_acceleration(
                    sim, q, v, com_ref, balance
                )
            if root_regulation is not None:
                wanted_act = (
                    wanted_act
                    + fs._root_regulation_acceleration(
                        sim, q, v, q_t, v_t, root_regulation
                    )[act]
                )

            a_full = np.zeros(self.nv, dtype=float)
            a_full[:6] = a_t[:6]
            a_full[act] = wanted_act
            return self.compute_inverse_torques(q, v, a_full, compensate_contact=True)

        return controller
