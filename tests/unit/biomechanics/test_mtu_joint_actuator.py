"""Tests for MTU joint actuator (Refs #13887)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from src.shared.python.biomechanics.activation_dynamics import ActivationDynamics
from src.shared.python.biomechanics.hill_muscle import HillMuscleModel, MuscleParameters
from src.shared.python.biomechanics.multi_muscle import AntagonistPair, MuscleGroup
from src.shared.python.biomechanics.mtu_joint_actuator import (
    MTUJointActuator,
    MTUJointGeometry,
    MTUJointState,
    TendonMode,
)
from src.shared.python.biomechanics.muscle_equilibrium import EquilibriumSolver
from src.shared.python.loop_nullspace.muscle_stiffness import series_stiffness


def _equilibrium_state(
    actuator: MTUJointActuator, q: float, a_ag: float, a_ant: float
) -> MTUJointState:
    sag, sant = (
        EquilibriumSolver(actuator.agonist),
        EquilibriumSolver(actuator.antagonist),
    )
    l_ag = actuator.geometry.l_mt(True, q)
    l_ant = actuator.geometry.l_mt(False, q)
    return MTUJointState(
        a_ag,
        a_ant,
        sag.solve_fiber_length(l_ag, a_ag),
        sant.solve_fiber_length(l_ant, a_ant),
    )


def _pair_fixture() -> tuple[MTUJointActuator, AntagonistPair]:
    ag_p = MuscleParameters(F_max=1000.0, l_opt=0.10, l_slack=0.25, v_max=10.0)
    ant_p = MuscleParameters(F_max=1200.0, l_opt=0.10, l_slack=0.25, v_max=10.0)
    ag_m = HillMuscleModel(ag_p)
    ant_m = HillMuscleModel(ant_p)
    r_ag, r_ant = 0.04, -0.035
    geom = MTUJointGeometry(
        moment_arm_agonist=r_ag,
        moment_arm_antagonist=r_ant,
        l_mt_nominal_agonist=ag_p.l_opt + ag_p.l_slack,
        l_mt_nominal_antagonist=ant_p.l_opt + ant_p.l_slack,
    )
    actuator = MTUJointActuator(
        ag_m,
        ant_m,
        geom,
        tendon_mode=TendonMode.RIGID,
        activation=ActivationDynamics(),
    )
    flexors = MuscleGroup("flex", enable_rust=False)
    flexors.add_muscle("ag", ag_m, r_ag)
    extensors = MuscleGroup("ext", enable_rust=False)
    extensors.add_muscle("ant", ant_m, r_ant)
    pair = AntagonistPair(flexors, extensors, enable_rust=False)
    return actuator, pair


class TestRigidModeMatchesAntagonistPair:
    def test_rigid_mode_matches_antagonist_pair(self) -> None:
        actuator, pair = _pair_fixture()
        q, qdot = 0.15, 0.0
        state = MTUJointState(
            a_agonist=0.6, a_antagonist=0.25, l_m_agonist=0.10, l_m_antagonist=0.10
        )
        tau_act = actuator.torque(q, qdot, state)
        tau_pair = pair.compute_net_torque(
            {"ag": state.a_agonist},
            {"ant": state.a_antagonist},
            {
                "ag": (state.l_m_agonist, 0.0),
                "ant": (state.l_m_antagonist, 0.0),
            },
        )
        assert tau_act == pytest.approx(tau_pair, rel=1e-12, abs=1e-12)


class TestCompliantEquilibrium:
    def test_compliant_equilibrium_residual(self) -> None:
        ag_p = MuscleParameters(F_max=800.0, l_opt=0.11, l_slack=0.22, v_max=8.0)
        ant_p = MuscleParameters(F_max=900.0, l_opt=0.12, l_slack=0.24, v_max=9.0)
        geom = MTUJointGeometry(0.04, -0.035, 0.33, 0.36)
        act = MTUJointActuator(
            HillMuscleModel(ag_p),
            HillMuscleModel(ant_p),
            geom,
            tendon_mode=TendonMode.COMPLIANT,
        )
        rng = np.random.default_rng(13887)
        checked = 0
        attempts = 0
        while checked < 20 and attempts < 200:
            attempts += 1
            q = float(rng.uniform(-0.05, 0.45))
            a_ag = float(rng.uniform(0.1, 0.9))
            a_ant = float(rng.uniform(0.1, 0.9))
            l_mt_ag = act.geometry.l_mt(True, q)
            l_mt_ant = act.geometry.l_mt(False, q)
            ok = True
            for muscle, l_mt, a in (
                (act.agonist, l_mt_ag, a_ag),
                (act.antagonist, l_mt_ant, a_ant),
            ):
                solver = EquilibriumSolver(muscle)
                l_ce = solver.solve_fiber_length(l_mt, a)
                residual = solver._equilibrium_residual(l_ce, l_mt, a)
                if abs(residual) >= 1e-9:
                    ok = False
                    break
            if ok:
                checked += 1
        assert checked == 20


class TestLinearisation:
    def test_linearisation_matches_fd(self) -> None:
        actuator, _ = _pair_fixture()
        actuator.tendon_mode = TendonMode.COMPLIANT
        q, qdot = 0.2, 0.5
        state = MTUJointState(0.5, 0.35, 0.10, 0.10)
        lin = actuator.linearise(q, qdot, state)
        h_q, h_v = 1e-7, 1e-7
        k_fd = (
            actuator.torque(q + h_q, qdot, state)
            - actuator.torque(q - h_q, qdot, state)
        ) / (2 * h_q)
        d_fd = (
            actuator.torque(q, qdot + h_v, state)
            - actuator.torque(q, qdot - h_v, state)
        ) / (2 * h_v)
        assert lin.K_q == pytest.approx(k_fd, rel=1e-5)
        assert lin.D_q == pytest.approx(d_fd, rel=1e-5)


class TestSeriesStiffness:
    def test_series_stiffness_below_both(self) -> None:
        actuator, _ = _pair_fixture()
        state = MTUJointState(0.7, 0.3, 0.10, 0.10)
        k_ag, k_ant = actuator.series_fiber_tendon_stiffness(0.0, 0.0, state)
        for k_m, k_t, k_mtu in (k_ag, k_ant):
            assert k_mtu <= min(k_m, k_t) + 1e-9
            np.testing.assert_allclose(k_mtu, series_stiffness(k_m, k_t), rtol=1e-9)


class TestActivationBounds:
    def test_activation_bounds(self) -> None:
        actuator, _ = _pair_fixture()
        state = MTUJointState(0.0, 0.0, 0.10, 0.10)
        dt = 0.001
        for u_ag, u_ant in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
            state = MTUJointState(0.0, 0.0, 0.10, 0.10)
            u = np.array([u_ag, u_ant])
            for _ in range(500):
                dot = actuator.derivatives(u, 0.0, 0.0, state)
                state.a_agonist = np.clip(state.a_agonist + dot[0] * dt, 0.0, 1.0)
                state.a_antagonist = np.clip(state.a_antagonist + dot[1] * dt, 0.0, 1.0)
            assert 0.0 <= state.a_agonist <= 1.0
            assert 0.0 <= state.a_antagonist <= 1.0


class TestCocontractionStiffness:
    def test_cocontraction_raises_kq_at_fixed_net_torque(self) -> None:
        actuator, _ = _pair_fixture()
        actuator.tendon_mode = TendonMode.COMPLIANT
        q, qdot = 0.0, 0.0
        low = _equilibrium_state(actuator, q, 0.45, 0.15)
        tau_target = actuator.torque(q, qdot, low)

        def torque_for_ant(a_ant: float) -> float:
            st = _equilibrium_state(actuator, q, 0.65, a_ant)
            return actuator.torque(q, qdot, st)

        a_lo, a_hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (a_lo + a_hi)
            if torque_for_ant(mid) > tau_target:
                a_lo = mid
            else:
                a_hi = mid
        high = _equilibrium_state(actuator, q, 0.65, 0.5 * (a_lo + a_hi))
        assert actuator.torque(q, qdot, high) == pytest.approx(tau_target, rel=1e-5)
        k_low = actuator.linearise(q, qdot, low).K_q
        k_high = actuator.linearise(q, qdot, high).K_q
        assert abs(k_high) > abs(k_low)
