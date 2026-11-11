"""Joint-level Hill MTU actuator (agonist/antagonist pair).

Millard et al. (2013) series tendon equilibrium; Thelen (2003) curves via
:class:`HillMuscleModel`; activation dynamics per :class:`ActivationDynamics`.
Units: m, N, N·m, N·m/rad, N·m·s/rad; activation and excitation in [0, 1].
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from src.shared.python.biomechanics.activation_dynamics import ActivationDynamics
from src.shared.python.biomechanics.hill_muscle import HillMuscleModel, MuscleState
from src.shared.python.biomechanics.muscle_equilibrium import EquilibriumSolver
from src.shared.python.core.contracts import ensure, require
from src.shared.python.loop_nullspace.muscle_stiffness import (
    StiffnessNotQualified,
    series_stiffness,
)


class TendonMode(str, enum.Enum):
    RIGID = "rigid"
    COMPLIANT = "compliant"


@dataclass(frozen=True)
class MTUJointGeometry:
    moment_arm_agonist: float
    moment_arm_antagonist: float
    l_mt_nominal_agonist: float
    l_mt_nominal_antagonist: float

    def __post_init__(self) -> None:
        require(
            self.moment_arm_agonist != 0.0, "agonist r nonzero", self.moment_arm_agonist
        )
        require(
            self.moment_arm_antagonist != 0.0,
            "antagonist r nonzero",
            self.moment_arm_antagonist,
        )
        if self.moment_arm_agonist * self.moment_arm_antagonist > 0.0:
            raise ValueError("moment arms must have opposite signs")

    def muscle_tendon_length_agonist(self, q: float) -> float:
        return self.l_mt_nominal_agonist - self.moment_arm_agonist * q

    def muscle_tendon_length_antagonist(self, q: float) -> float:
        return self.l_mt_nominal_antagonist - self.moment_arm_antagonist * q


@dataclass
class MTUJointState:
    a_agonist: float
    a_antagonist: float
    l_m_agonist: float
    l_m_antagonist: float


@dataclass(frozen=True)
class MTULinearisation:
    K_q: float
    D_q: float
    d_tau_d_a: tuple[float, float]
    qualified: bool


class MTUJointActuator:
    """Hill MTU pair at one revolute joint (constant moment arms)."""

    def __init__(
        self,
        agonist: HillMuscleModel,
        antagonist: HillMuscleModel,
        geometry: MTUJointGeometry,
        *,
        tendon_mode: TendonMode = TendonMode.RIGID,
        activation: ActivationDynamics | None = None,
    ) -> None:
        self.agonist = agonist
        self.antagonist = antagonist
        self.geometry = geometry
        self.tendon_mode = TendonMode(tendon_mode)
        self._activation = activation or ActivationDynamics()
        self._solver_ag = EquilibriumSolver(agonist)
        self._solver_ant = EquilibriumSolver(antagonist)

    def _line_force(
        self, muscle: HillMuscleModel, a: float, l_m: float, v_ce: float, l_mt: float
    ) -> float:
        if self.tendon_mode is TendonMode.RIGID:
            return muscle.compute_force(MuscleState(a, l_m, v_ce, l_MT=l_mt))
        cos_a = float(np.cos(muscle.params.pennation_angle))
        f_t = muscle.tendon_force((l_mt - l_m * cos_a) / muscle.params.l_slack)
        return float(max(0.0, f_t * muscle.params.F_max))

    def _muscle_torque(
        self,
        muscle: HillMuscleModel,
        r: float,
        q: float,
        qdot: float,
        a: float,
        l_m: float,
    ) -> float:
        ag = muscle is self.agonist
        l_mt = (
            self.geometry.muscle_tendon_length_agonist(q)
            if ag
            else self.geometry.muscle_tendon_length_antagonist(q)
        )
        v_ce = 0.0
        if self.tendon_mode is TendonMode.COMPLIANT:
            solver = self._solver_ag if ag else self._solver_ant
            v_ce = solver.solve_fiber_velocity(l_mt, -r * qdot, a, l_m)
        return float(r * self._line_force(muscle, a, l_m, v_ce, l_mt))

    def torque(self, q: float, qdot: float, state: MTUJointState) -> float:
        require(0.0 <= state.a_agonist <= 1.0, "a_ag in [0,1]", state.a_agonist)
        require(0.0 <= state.a_antagonist <= 1.0, "a_ant in [0,1]", state.a_antagonist)
        g = self.geometry
        tau = self._muscle_torque(
            self.agonist,
            g.moment_arm_agonist,
            q,
            qdot,
            state.a_agonist,
            state.l_m_agonist,
        )
        tau += self._muscle_torque(
            self.antagonist,
            g.moment_arm_antagonist,
            q,
            qdot,
            state.a_antagonist,
            state.l_m_antagonist,
        )
        ensure(np.isfinite(tau), "torque finite", tau)
        return tau

    def derivatives(
        self, u: np.ndarray, q: float, qdot: float, state: MTUJointState
    ) -> np.ndarray:
        dadt = (
            self._activation.compute_derivative(float(u[0]), state.a_agonist),
            self._activation.compute_derivative(float(u[1]), state.a_antagonist),
        )
        ldot = (0.0, 0.0)
        if self.tendon_mode is TendonMode.COMPLIANT:
            g = self.geometry
            ldot = (
                self._solver_ag.solve_fiber_velocity(
                    g.muscle_tendon_length_agonist(q),
                    -g.moment_arm_agonist * qdot,
                    state.a_agonist,
                    state.l_m_agonist,
                ),
                self._solver_ant.solve_fiber_velocity(
                    g.muscle_tendon_length_antagonist(q),
                    -g.moment_arm_antagonist * qdot,
                    state.a_antagonist,
                    state.l_m_antagonist,
                ),
            )
        return np.array((*dadt, *ldot), dtype=np.float64)

    def _fiber_force(
        self, muscle: HillMuscleModel, a: float, l_m: float, v_ce: float
    ) -> float:
        return muscle.compute_force(MuscleState(a, l_m, v_ce, l_MT=0.0))

    def _fiber_stiffness(
        self, muscle: HillMuscleModel, a: float, l_m: float, v_ce: float
    ) -> float:
        eps = 1e-7
        return (
            self._fiber_force(muscle, a, l_m + eps, v_ce)
            - self._fiber_force(muscle, a, l_m - eps, v_ce)
        ) / (2.0 * eps)

    def _tendon_stiffness(
        self, muscle: HillMuscleModel, l_mt: float, l_m: float
    ) -> float:
        cos_a = float(np.cos(muscle.params.pennation_angle))
        l_t, slack = l_mt - l_m * cos_a, muscle.params.l_slack
        eps = 1e-7
        scale = muscle.params.F_max / (2.0 * eps)
        return (
            float(
                muscle.tendon_force((l_t + eps) / slack)
                - muscle.tendon_force((l_t - eps) / slack)
            )
            * scale
        )

    def series_fiber_tendon_stiffness(
        self, q: float, qdot: float, state: MTUJointState
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        out: list[tuple[float, float, float]] = []
        spec = (
            (
                self.agonist,
                self.geometry.moment_arm_agonist,
                state.a_agonist,
                state.l_m_agonist,
                self.geometry.muscle_tendon_length_agonist,
            ),
            (
                self.antagonist,
                self.geometry.moment_arm_antagonist,
                state.a_antagonist,
                state.l_m_antagonist,
                self.geometry.muscle_tendon_length_antagonist,
            ),
        )
        for muscle, r, a, l_m, mt_fn in spec:
            l_mt = mt_fn(q)
            solver = self._solver_ag if muscle is self.agonist else self._solver_ant
            v_ce = (
                0.0
                if self.tendon_mode is TendonMode.RIGID
                else solver.solve_fiber_velocity(l_mt, -r * qdot, a, l_m)
            )
            k_m = self._fiber_stiffness(muscle, a, l_m, v_ce)
            k_t = (
                np.inf
                if self.tendon_mode is TendonMode.RIGID
                else self._tendon_stiffness(muscle, l_mt, l_m)
            )
            out.append(
                (
                    k_m,
                    float(k_t) if np.isfinite(k_t) else float("inf"),
                    float(series_stiffness([k_m], [k_t])[0]),
                )
            )
        return out[0], out[1]

    def _stiffness_qualified(self, state: MTUJointState, q: float, qdot: float) -> bool:
        _ = (q, qdot)
        spec = (
            (self.agonist, state.a_agonist, state.l_m_agonist),
            (self.antagonist, state.a_antagonist, state.l_m_antagonist),
        )
        for muscle, a, l_m in spec:
            l_norm = l_m / muscle.params.l_opt
            if a > 0 and muscle.force_length_active(
                l_norm + 1e-4
            ) < muscle.force_length_active(l_norm - 1e-4):
                return False
            if (
                self.tendon_mode is TendonMode.COMPLIANT
                and self._fiber_stiffness(muscle, a, l_m, 0.0) < 0.0
            ):
                return False
        return True

    def linearise(
        self, q: float, qdot: float, state: MTUJointState
    ) -> MTULinearisation:
        if not self._stiffness_qualified(state, q, qdot):
            raise StiffnessNotQualified("descending limb or negative fibre stiffness")
        h_q, h_v = 1e-7, 1e-7
        t0 = self.torque(q, qdot, state)
        k_q = (
            self.torque(q + h_q, qdot, state) - self.torque(q - h_q, qdot, state)
        ) / (2 * h_q)
        d_q = (
            self.torque(q, qdot + h_v, state) - self.torque(q, qdot - h_v, state)
        ) / (2 * h_v)
        st_ag = MTUJointState(
            state.a_agonist + h_v,
            state.a_antagonist,
            state.l_m_agonist,
            state.l_m_antagonist,
        )
        st_ant = MTUJointState(
            state.a_agonist,
            state.a_antagonist + h_v,
            state.l_m_agonist,
            state.l_m_antagonist,
        )
        return MTULinearisation(
            float(k_q),
            float(d_q),
            (
                (self.torque(q, qdot, st_ag) - t0) / h_v,
                (self.torque(q, qdot, st_ant) - t0) / h_v,
            ),
            True,
        )
