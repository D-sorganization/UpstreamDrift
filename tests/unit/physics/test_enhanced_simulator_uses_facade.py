"""Unit tests for EnhancedBallFlightSimulator routing through the Rust facade (issue #13903).

Verifies that EnhancedBallFlightSimulator routes deterministic drag and Magnus
calculations through the existing Rust facade (`compute_total_force`), achieving
high-throughput force evaluation while preserving gusts and environment
randomization in Python.

TDD progression:
1. test_deterministic_forces_route_through_facade: asserts compute_total_force is
   called for each RK4 stage during simulation. Red on main.
2. test_trajectory_equivalence_deterministic: carry, apex, and flight time match
   between facade path and reference path within 1e-6 relative tolerance (#13866).
3. test_pure_python_fallback_equivalence: wheel absent / Python fallback passes
   the same equivalence test within 1e-6 relative tolerance.
4. test_force_provider_injection: custom force_provider parameter is respected.
5. test_trajectory_with_gusts_and_randomization_seeded: stochastic terms remain
   in Python and produce deterministic outputs given identical seeds.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from src.shared.python.physics.aerodynamics import (
    AerodynamicsConfig,
    RandomizationConfig,
    WindConfig,
)
import src.shared.python.physics.aerodynamics._rust_facade as rust_facade
from src.shared.python.physics.ball_enhanced_simulator import (
    EnhancedBallFlightSimulator,
)
from src.shared.python.physics.ball_flight_physics import (
    BallProperties,
    EnvironmentalConditions,
    LaunchConditions,
)

pytestmark = pytest.mark.unit

# Parity tolerance aligned with #13866 (1e-6 relative)
PARITY_TOLERANCE_REL = 1e-6


@pytest.fixture
def standard_launch() -> LaunchConditions:
    """Representative driver launch condition."""
    return LaunchConditions(
        velocity=70.0,
        launch_angle=math.radians(12.0),
        azimuth_angle=0.0,
        spin_rate=2500.0,
        spin_axis=np.array([0.0, -1.0, 0.0]),
    )


class TestEnhancedSimulatorRustFacadeRouting:
    """TDD tests for routing deterministic aero through the Rust facade."""

    def test_deterministic_forces_route_through_facade(
        self,
        standard_launch: LaunchConditions,
    ) -> None:
        """compute_total_force must be called for each RK4 stage when simulating.

        Red on main where EnhancedBallFlightSimulator only invokes
        AerodynamicsEngine.compute_forces directly in Python.
        """
        wind_cfg = WindConfig(gusts_enabled=False)
        sim = EnhancedBallFlightSimulator(wind_config=wind_cfg)

        with patch.object(
            rust_facade,
            "compute_total_force",
            wraps=rust_facade.compute_total_force,
        ) as spy_force:
            dt = 0.01
            max_time = 0.05
            trajectory = sim.simulate_trajectory(
                standard_launch, max_time=max_time, dt=dt
            )

            num_steps = len(trajectory) - 1
            assert num_steps >= 4, (
                f"Expected at least 4 integration steps, got {num_steps}"
            )

            # 4 RK4 stages (k1, k2, k3, k4) per step must route through compute_total_force
            expected_min_calls = 4 * num_steps
            assert spy_force.call_count >= expected_min_calls, (
                f"compute_total_force called {spy_force.call_count} times, expected >= {expected_min_calls}"
            )

    def test_trajectory_equivalence_deterministic(
        self,
        standard_launch: LaunchConditions,
    ) -> None:
        """Carry, apex, and flight time match reference within 1e-6 relative tolerance."""
        env = EnvironmentalConditions()
        ball = BallProperties(cd0=0.25, spin_decay_rate=0.05)
        wind = WindConfig(gusts_enabled=False)

        sim = EnhancedBallFlightSimulator(ball=ball, environment=env, wind_config=wind)
        traj = sim.simulate_trajectory(standard_launch, max_time=10.0, dt=0.01)

        carry = sim.calculate_carry_distance(traj)
        apex = sim.calculate_max_height(traj)
        flight_time = traj[-1].time

        # Reference: simulate with an injected Python force provider that explicitly
        # calls AerodynamicsEngine.compute_forces to obtain the baseline trajectory.
        ref_sim = EnhancedBallFlightSimulator(
            ball=ball,
            environment=env,
            wind_config=wind,
            force_provider=lambda v, s, t, p: ref_sim._aero_engine.compute_forces(
                v, s, t=t, position=p
            )["total"],
        )
        ref_traj = ref_sim.simulate_trajectory(standard_launch, max_time=10.0, dt=0.01)

        ref_carry = ref_sim.calculate_carry_distance(ref_traj)
        ref_apex = ref_sim.calculate_max_height(ref_traj)
        ref_flight_time = ref_traj[-1].time

        assert carry == pytest.approx(ref_carry, rel=PARITY_TOLERANCE_REL, abs=1e-5)
        assert apex == pytest.approx(ref_apex, rel=PARITY_TOLERANCE_REL, abs=1e-5)
        assert flight_time == pytest.approx(
            ref_flight_time, rel=PARITY_TOLERANCE_REL, abs=1e-5
        )

    def test_pure_python_fallback_equivalence(
        self,
        standard_launch: LaunchConditions,
    ) -> None:
        """When Rust wheel is absent, pure-Python fallback passes the same equivalence."""
        env = EnvironmentalConditions()
        ball = BallProperties(cd0=0.25, spin_decay_rate=0.05)
        wind = WindConfig(gusts_enabled=False)

        with patch.object(rust_facade, "_RUST_AVAILABLE", False):
            sim = EnhancedBallFlightSimulator(
                ball=ball, environment=env, wind_config=wind
            )
            traj = sim.simulate_trajectory(standard_launch, max_time=10.0, dt=0.01)

            carry = sim.calculate_carry_distance(traj)
            apex = sim.calculate_max_height(traj)
            flight_time = traj[-1].time

            ref_sim = EnhancedBallFlightSimulator(
                ball=ball,
                environment=env,
                wind_config=wind,
                force_provider=lambda v, s, t, p: ref_sim._aero_engine.compute_forces(
                    v, s, t=t, position=p
                )["total"],
            )
            ref_traj = ref_sim.simulate_trajectory(
                standard_launch, max_time=10.0, dt=0.01
            )

            ref_carry = ref_sim.calculate_carry_distance(ref_traj)
            ref_apex = ref_sim.calculate_max_height(ref_traj)
            ref_flight_time = ref_traj[-1].time

            assert carry == pytest.approx(ref_carry, rel=PARITY_TOLERANCE_REL, abs=1e-5)
            assert apex == pytest.approx(ref_apex, rel=PARITY_TOLERANCE_REL, abs=1e-5)
            assert flight_time == pytest.approx(
                ref_flight_time, rel=PARITY_TOLERANCE_REL, abs=1e-5
            )

    def test_force_provider_injection(
        self,
        standard_launch: LaunchConditions,
    ) -> None:
        """Injected force_provider is invoked during simulation and controls forces."""
        calls: list[tuple[float, np.ndarray]] = []

        def custom_provider(
            v: np.ndarray,
            s: np.ndarray,
            t: float,
            p: np.ndarray,
        ) -> np.ndarray:
            calls.append((t, v.copy()))
            # Constant drag-like force opposing velocity
            speed = float(np.linalg.norm(v))
            if speed > 1e-6:
                return -0.1 * v
            return np.zeros(3)

        sim = EnhancedBallFlightSimulator(force_provider=custom_provider)
        traj = sim.simulate_trajectory(standard_launch, max_time=0.05, dt=0.01)

        assert len(calls) > 0, "Custom force_provider was never called"
        assert sim.force_provider is custom_provider

    def test_trajectory_with_gusts_and_randomization_seeded(
        self,
        standard_launch: LaunchConditions,
    ) -> None:
        """Gusts and randomization enabled under fixed seed produce deterministic results."""
        wind = WindConfig(
            base_velocity=np.array([5.0, 0.0, 0.0]),
            gusts_enabled=True,
            gust_intensity=0.5,
            gust_frequency=0.2,
        )
        rand = RandomizationConfig(
            enabled=True,
            air_density_variance=0.02,
            temperature_variance=2.0,
            wind_variance=0.1,
        )

        sim1 = EnhancedBallFlightSimulator(
            wind_config=wind, randomization_config=rand, seed=12345
        )
        sim2 = EnhancedBallFlightSimulator(
            wind_config=wind, randomization_config=rand, seed=12345
        )

        traj1 = sim1.simulate_trajectory(standard_launch, max_time=2.0, dt=0.01)
        traj2 = sim2.simulate_trajectory(standard_launch, max_time=2.0, dt=0.01)

        carry1 = sim1.calculate_carry_distance(traj1)
        carry2 = sim2.calculate_carry_distance(traj2)
        assert carry1 == pytest.approx(carry2, abs=1e-12)

        apex1 = sim1.calculate_max_height(traj1)
        apex2 = sim2.calculate_max_height(traj2)
        assert apex1 == pytest.approx(apex2, abs=1e-12)

    def test_aero_disabled_produces_zero_force(
        self,
        standard_launch: LaunchConditions,
    ) -> None:
        """When aerodynamics is disabled, force provider emits zero force."""
        aero = AerodynamicsConfig(enabled=False)
        sim = EnhancedBallFlightSimulator(aero_config=aero)

        zero_force = sim._default_force_provider(
            np.array([50.0, 0.0, 10.0]),
            np.array([0.0, 200.0, 0.0]),
            0.0,
            np.zeros(3),
        )
        np.testing.assert_allclose(zero_force, np.zeros(3), atol=1e-15)
