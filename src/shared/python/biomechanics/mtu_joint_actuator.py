"""Joint-level Hill MTU actuator (agonist/antagonist pair).

Millard et al. (2013) series tendon; Thelen (2003) via :class:`HillMuscleModel`.
Units: m, N, N·m; activation in [0, 1].
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
        require(self.moment_arm_agonist != 0.0, "agonist r", self.moment_arm_agonist)
        require(
            self.moment_arm_antagonist != 0.0,
            "antagonist r",
            self.moment_arm_antagonist,
        )
        if self.moment_arm_agonist * self.moment_arm_antagonist > 0.0:
            raise ValueError("moment arms must have opposite signs")

    def l_mt(self, agonist: bool, q: float) -> float:
        if agonist:
            return self.l_mt_nominal_agonist - self.moment_arm_agonist * q
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
    def __init__(
        self,
        agonist: HillMuscleModel,
        antagonist: HillMuscleModel,
        geometry: MTUJointGeometry,
        *,
        tendon_mode: TendonMode = TendonMode.RIGID,
        activation: ActivationDynamics | None = None,
    ) -> None:
        self.agonist, self.antagonist = agonist, antagonist
        self.geometry, self.tendon_mode = geometry, TendonMode(tendon_mode)
        self._activation = activation or ActivationDynamics()
        self._solver_ag, self._solver_ant = (
            EquilibriumSolver(agonist),
            EquilibriumSolver(antagonist),
        )

    def _solver(self, muscle: HillMuscleModel) -> EquilibriumSolver:
        return self._solver_ag if muscle is self.agonist else self._solver_ant

    def _line_force(
        self, muscle: HillMuscleModel, a: float, l_m: float, v_ce: float, l_mt: float
    ) -> float:
        if self.tendon_mode is TendonMode.RIGID:
            return muscle.compute_force(MuscleState(a, l_m, v_ce, l_MT=l_mt))
        pa = muscle.params
        f_t = muscle.tendon_force(
            (l_mt - l_m * np.cos(pa.pennation_angle)) / pa.l_slack
        )
        return float(max(0.0, f_t * pa.F_max))

    def _muscle_torque(
        self, agonist: bool, q: float, qdot: float, a: float, l_m: float
    ) -> float:
        g, muscle = self.geometry, self.agonist if agonist else self.antagonist
        r = g.moment_arm_agonist if agonist else g.moment_arm_antagonist
        l_mt = g.l_mt(agonist, q)
        v_ce = 0.0
        if self.tendon_mode is TendonMode.COMPLIANT:
            v_ce = self._solver(muscle).solve_fiber_velocity(l_mt, -r * qdot, a, l_m)
        return float(r * self._line_force(muscle, a, l_m, v_ce, l_mt))

    def torque(self, q: float, qdot: float, state: MTUJointState) -> float:
        require(0.0 <= state.a_agonist <= 1.0, "a_ag", state.a_agonist)
        require(0.0 <= state.a_antagonist <= 1.0, "a_ant", state.a_antagonist)
        tau = self._muscle_torque(True, q, qdot, state.a_agonist, state.l_m_agonist)
        tau += self._muscle_torque(
            False, q, qdot, state.a_antagonist, state.l_m_antagonist
        )
        ensure(np.isfinite(tau), "torque", tau)
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
                    g.l_mt(True, q),
                    -g.moment_arm_agonist * qdot,
                    state.a_agonist,
                    state.l_m_agonist,
                ),
                self._solver_ant.solve_fiber_velocity(
                    g.l_mt(False, q),
                    -g.moment_arm_antagonist * qdot,
                    state.a_antagonist,
                    state.l_m_antagonist,
                ),
            )
        return np.array((*dadt, *ldot), dtype=np.float64)

    def _fiber_stiffness(
        self, muscle: HillMuscleModel, a: float, l_m: float, v_ce: float
    ) -> float:
        eps = 1e-7

        def force(lm: float) -> float:
            return muscle.compute_force(MuscleState(a, lm, v_ce, l_MT=0.0))

        return (force(l_m + eps) - force(l_m - eps)) / (2.0 * eps)

    def _tendon_stiffness(
        self, muscle: HillMuscleModel, l_mt: float, l_m: float
    ) -> float:
        pa, eps = muscle.params, 1e-7
        l_t = l_mt - l_m * np.cos(pa.pennation_angle)
        df = muscle.tendon_force((l_t + eps) / pa.l_slack) - muscle.tendon_force(
            (l_t - eps) / pa.l_slack
        )
        return float(df * pa.F_max / (2.0 * eps))

    def _stiffness_triple(
        self, agonist: bool, q: float, qdot: float, a: float, l_m: float
    ) -> tuple[float, float, float]:
        g, muscle = self.geometry, self.agonist if agonist else self.antagonist
        r = g.moment_arm_agonist if agonist else g.moment_arm_antagonist
        l_mt = g.l_mt(agonist, q)
        v_ce = (
            0.0
            if self.tendon_mode is TendonMode.RIGID
            else self._solver(muscle).solve_fiber_velocity(l_mt, -r * qdot, a, l_m)
        )
        k_m = self._fiber_stiffness(muscle, a, l_m, v_ce)
        k_t = (
            np.inf
            if self.tendon_mode is TendonMode.RIGID
            else self._tendon_stiffness(muscle, l_mt, l_m)
        )
        k_mtu = float(series_stiffness([k_m], [k_t])[0])
        return k_m, float(k_t) if np.isfinite(k_t) else float("inf"), k_mtu

    def series_fiber_tendon_stiffness(
        self, q: float, qdot: float, state: MTUJointState
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return (
            self._stiffness_triple(True, q, qdot, state.a_agonist, state.l_m_agonist),
            self._stiffness_triple(
                False, q, qdot, state.a_antagonist, state.l_m_antagonist
            ),
        )

    def _stiffness_qualified(self, state: MTUJointState) -> bool:
        for muscle, a, l_m in (
            (self.agonist, state.a_agonist, state.l_m_agonist),
            (self.antagonist, state.a_antagonist, state.l_m_antagonist),
        ):
            ln = l_m / muscle.params.l_opt
            if a > 0 and muscle.force_length_active(
                ln + 1e-4
            ) < muscle.force_length_active(ln - 1e-4):
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
        if not self._stiffness_qualified(state):
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
        st_an = MTUJointState(
            state.a_agonist,
            state.a_antagonist + h_v,
            state.l_m_agonist,
            state.l_m_antagonist,
        )
        d_ag, d_an = (
            (self.torque(q, qdot, st_ag) - t0) / h_v,
            (self.torque(q, qdot, st_an) - t0) / h_v,
        )
        return MTULinearisation(
            float(k_q), float(d_q), (float(d_ag), float(d_an)), True
        )
