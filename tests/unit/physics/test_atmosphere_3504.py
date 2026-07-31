"""Tests for atmospheric and drag-crisis helpers added under issue #3504.

Covers:
- ``cd_dimpled_sphere`` Reynolds-number behaviour and continuity.
- ``air_density`` ISA-atmosphere model and input validation.
- Trajectory integrator picks up altitude-dependent air density and
  produces longer carries at higher altitude (less drag).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from src.shared.python.physics.atmosphere import (
    DRY_AIR_R_SPECIFIC_J_KG_K,
    ISA_RHO0_KG_M3,
    MAX_VALID_ALTITUDE_M,
    MAX_VALID_REYNOLDS,
    MAX_VALID_TEMPERATURE_C,
    MIN_VALID_ALTITUDE_M,
    MIN_VALID_REYNOLDS,
    MIN_VALID_TEMPERATURE_C,
    air_density,
    cd_dimpled_sphere,
)

# ---------------------------------------------------------------------------
# Drag crisis
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDragCrisisCdDimpledSphere:
    """Verify the Reynolds-dependent drag coefficient model."""

    def test_drag_crisis_dip(self) -> None:
        """Cd at Re=1e4 must be higher than at Re=1e5 (drag crisis)."""
        cd_low = cd_dimpled_sphere(1e4)
        cd_post_crisis = cd_dimpled_sphere(1e5)
        assert cd_low > cd_post_crisis, (
            f"Expected drag-crisis dip but Cd(1e4)={cd_low:.3f} <= "
            f"Cd(1e5)={cd_post_crisis:.3f}"
        )

    def test_drag_crisis_continuity_no_jumps(self) -> None:
        """Cd should be continuous: no jumps > 0.05 between adjacent samples.

        We sample log-uniformly so the test catches both the crisis and the
        recovery transitions.
        """
        log_re = np.linspace(math.log10(1e3), math.log10(1e7), 401)
        cds = [cd_dimpled_sphere(10.0**lr) for lr in log_re]
        diffs = np.diff(cds)
        max_jump = float(np.max(np.abs(diffs)))
        assert max_jump < 0.05, (
            f"Cd discontinuity detected: max adjacent-sample jump "
            f"{max_jump:.4f} >= 0.05"
        )

    def test_drag_crisis_envelope(self) -> None:
        """Cd must stay in [0.15, 0.55] across the supported Re range."""
        for re in [1e3, 1e4, 4e4, 5e4, 7e4, 1e5, 2e5, 5e5, 1e6, 5e6, 1e7]:
            cd = cd_dimpled_sphere(re)
            assert (
                0.15 <= cd <= 0.55
            ), f"Cd({re:g})={cd:.3f} outside the supported [0.15, 0.55]"

    def test_drag_crisis_minimum_near_post_crisis(self) -> None:
        """Minimum Cd should sit near Re ~ 1e5 -- 3e5 with Cd ~ 0.20 -- 0.26."""
        re_grid = np.logspace(math.log10(5e4), math.log10(1e6), 201)
        cds = [cd_dimpled_sphere(r) for r in re_grid]
        min_idx = int(np.argmin(cds))
        re_at_min = float(re_grid[min_idx])
        cd_at_min = float(cds[min_idx])
        assert 8e4 <= re_at_min <= 5e5, (
            f"Drag-crisis minimum at Re={re_at_min:.2e} is outside the "
            f"expected band [8e4, 5e5]"
        )
        assert (
            0.20 <= cd_at_min <= 0.28
        ), f"Cd at minimum = {cd_at_min:.3f} is outside the expected [0.20, 0.28] band"

    def test_drag_crisis_invalid_inputs(self) -> None:
        """Reynolds outside supported range or non-positive must raise."""
        with pytest.raises(ValueError):
            cd_dimpled_sphere(0.0)
        with pytest.raises(ValueError):
            cd_dimpled_sphere(-1.0)
        with pytest.raises(ValueError):
            cd_dimpled_sphere(MIN_VALID_REYNOLDS / 10.0)
        with pytest.raises(ValueError):
            cd_dimpled_sphere(MAX_VALID_REYNOLDS * 10.0)
        with pytest.raises(ValueError):
            cd_dimpled_sphere(float("nan"))
        with pytest.raises(ValueError):
            cd_dimpled_sphere(float("inf"))

    def test_drag_crisis_type_check(self) -> None:
        """Non-numeric Reynolds must raise TypeError."""
        with pytest.raises(TypeError):
            cd_dimpled_sphere("100000")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ISA air density
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAirDensityISA:
    """Verify the ISA-troposphere air density helper."""

    def test_air_density_sea_level_15c(self) -> None:
        """rho(0 m, 15 C) must match 1.225 kg/m^3 within 0.005."""
        rho = air_density(0.0, temperature_c=15.0)
        assert (
            abs(rho - ISA_RHO0_KG_M3) <= 0.005
        ), f"rho(0,15C)={rho:.4f}, expected {ISA_RHO0_KG_M3} +/- 0.005"

    def test_air_density_drops_with_altitude(self) -> None:
        """Density at 1500 m must be roughly 12--18% lower than sea level."""
        rho_sea = air_density(0.0)
        rho_high = air_density(1500.0)
        ratio = rho_high / rho_sea
        assert (
            0.82 <= ratio <= 0.88
        ), f"Density ratio at 1500m = {ratio:.3f}, expected ~0.85 (12--18% drop)"

    def test_air_density_monotonic_decreasing(self) -> None:
        """Density must decrease monotonically up the troposphere."""
        alts = [0.0, 500.0, 1000.0, 2000.0, 4000.0, 6000.0, 8000.0]
        rhos = [air_density(a) for a in alts]
        for prev, curr in zip(rhos, rhos[1:], strict=False):
            assert curr < prev, f"Density not strictly decreasing: {rhos}"

    def test_air_density_temperature_effect(self) -> None:
        """Hot day should have lower density than cold day at same altitude."""
        rho_hot = air_density(0.0, temperature_c=35.0)
        rho_cold = air_density(0.0, temperature_c=-5.0)
        assert (
            rho_hot < rho_cold
        ), f"Expected hot day rho ({rho_hot:.4f}) < cold day rho ({rho_cold:.4f})"

    def test_air_density_pressure_override(self) -> None:
        """Pressure override must reproduce p / (R*T) at sea level."""
        custom_p = 95_000.0  # ~storm low pressure
        rho = air_density(0.0, temperature_c=15.0, pressure_pa=custom_p)
        expected = custom_p / (DRY_AIR_R_SPECIFIC_J_KG_K * (15.0 + 273.15))
        assert math.isclose(rho, expected, rel_tol=1e-12)

    def test_air_density_invalid_altitude_high(self) -> None:
        """Altitudes above MAX_VALID_ALTITUDE_M must raise ValueError."""
        with pytest.raises(ValueError):
            air_density(MAX_VALID_ALTITUDE_M + 1.0)

    def test_air_density_invalid_altitude_low(self) -> None:
        """Altitudes below MIN_VALID_ALTITUDE_M must raise ValueError."""
        with pytest.raises(ValueError):
            air_density(MIN_VALID_ALTITUDE_M - 1.0)

    def test_air_density_invalid_temperature(self) -> None:
        """Temperatures outside [-50, 60] C must raise ValueError."""
        with pytest.raises(ValueError):
            air_density(0.0, temperature_c=MAX_VALID_TEMPERATURE_C + 5.0)
        with pytest.raises(ValueError):
            air_density(0.0, temperature_c=MIN_VALID_TEMPERATURE_C - 5.0)

    def test_air_density_invalid_pressure(self) -> None:
        """Non-positive pressure override must raise ValueError."""
        with pytest.raises(ValueError):
            air_density(0.0, pressure_pa=0.0)
        with pytest.raises(ValueError):
            air_density(0.0, pressure_pa=-1.0)

    def test_air_density_nan_inf(self) -> None:
        """Non-finite altitude or temperature must raise ValueError."""
        with pytest.raises(ValueError):
            air_density(float("nan"))
        with pytest.raises(ValueError):
            air_density(0.0, temperature_c=float("inf"))


# ---------------------------------------------------------------------------
# Trajectory integration: altitude reduces drag and lengthens carry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrajectoryAltitudeIntegration:
    """End-to-end check that altitude-aware density actually reaches the
    integrator and produces longer carries at high altitude."""

    @staticmethod
    def _carry_distance(traj: list) -> float:
        if not traj:
            return 0.0
        last = traj[-1]
        return float(math.hypot(last.position[0], last.position[1]))

    def test_carry_longer_at_altitude(self) -> None:
        """Same launch -> more carry at 1500 m than at sea level."""
        from src.shared.python.physics.aerodynamics import AerodynamicsConfig
        from src.shared.python.physics.ball_enhanced_simulator import (
            EnhancedBallFlightSimulator,
        )
        from src.shared.python.physics.ball_launch_conditions import (
            EnvironmentalConditions,
            LaunchConditions,
        )

        launch = LaunchConditions(
            velocity=70.0,
            launch_angle=math.radians(12.0),
            spin_rate=2700.0,
        )
        aero = AerodynamicsConfig(
            drag_enabled=True,
            lift_enabled=True,
            magnus_enabled=False,
        )

        env_sea = EnvironmentalConditions.from_altitude(
            altitude_m=0.0, temperature_c=15.0
        )
        env_high = EnvironmentalConditions.from_altitude(
            altitude_m=1500.0, temperature_c=15.0
        )

        sim_sea = EnhancedBallFlightSimulator(
            environment=env_sea,
            aero_config=aero,
            seed=0,
            track_altitude_density=True,
        )
        sim_high = EnhancedBallFlightSimulator(
            environment=env_high,
            aero_config=aero,
            seed=0,
            track_altitude_density=True,
        )

        traj_sea = sim_sea.simulate_trajectory(launch, max_time=8.0, dt=0.01)
        traj_high = sim_high.simulate_trajectory(launch, max_time=8.0, dt=0.01)

        carry_sea = self._carry_distance(traj_sea)
        carry_high = self._carry_distance(traj_high)

        assert carry_high > carry_sea, (
            f"Expected carry_high ({carry_high:.1f} m) > "
            f"carry_sea ({carry_sea:.1f} m) at 1500 m altitude"
        )
        # And the gain should be material -- at least 1% but no more than 50%.
        gain = (carry_high - carry_sea) / carry_sea
        assert (
            0.01 < gain < 0.5
        ), f"Altitude carry gain {gain * 100:.2f}% is outside the expected 1--50% band"

    def test_track_altitude_density_disabled_uses_constant_rho(self) -> None:
        """With ``track_altitude_density=False`` the engine keeps its initial rho."""
        from src.shared.python.physics.aerodynamics import AerodynamicsConfig
        from src.shared.python.physics.ball_enhanced_simulator import (
            EnhancedBallFlightSimulator,
        )
        from src.shared.python.physics.ball_launch_conditions import (
            EnvironmentalConditions,
            LaunchConditions,
        )

        launch = LaunchConditions(
            velocity=60.0,
            launch_angle=math.radians(15.0),
        )
        env = EnvironmentalConditions(air_density=1.0)
        sim = EnhancedBallFlightSimulator(
            environment=env,
            aero_config=AerodynamicsConfig(),
            seed=0,
            track_altitude_density=False,
        )
        # Run a few steps; the engine's _current_air_density must not change.
        sim.simulate_trajectory(launch, max_time=0.05, dt=0.01)
        assert math.isclose(sim._aero_engine._current_air_density, 1.0, rel_tol=1e-12)

    def test_track_altitude_density_default_off_preserves_caller_density(
        self,
    ) -> None:
        """Default behavior must not overwrite an explicitly supplied density.

        Regression for the API-compat concern raised on PR #3522: a caller
        that passes a humidity/weather-calibrated ``air_density`` should see
        that value used, not the ISA-derived value.
        """
        from src.shared.python.physics.aerodynamics import AerodynamicsConfig
        from src.shared.python.physics.ball_enhanced_simulator import (
            EnhancedBallFlightSimulator,
        )
        from src.shared.python.physics.ball_launch_conditions import (
            EnvironmentalConditions,
            LaunchConditions,
        )

        launch = LaunchConditions(velocity=50.0, launch_angle=math.radians(10.0))
        custom_rho = 1.111  # not equal to ISA at any of the standard altitudes
        env = EnvironmentalConditions(air_density=custom_rho, altitude=1500.0)
        sim = EnhancedBallFlightSimulator(
            environment=env,
            aero_config=AerodynamicsConfig(),
            seed=0,
            # No track_altitude_density argument -> default must be False.
        )
        sim.simulate_trajectory(launch, max_time=0.05, dt=0.01)
        assert math.isclose(
            sim._aero_engine._current_air_density, custom_rho, rel_tol=1e-12
        ), "Default-init simulator must not overwrite caller-supplied air_density"

    def test_pressure_override_propagates_to_per_step_updates(self) -> None:
        """A non-default ``sea_level_pressure_pa`` must reach each step."""
        from src.shared.python.physics.aerodynamics import AerodynamicsConfig
        from src.shared.python.physics.atmosphere import air_density
        from src.shared.python.physics.ball_enhanced_simulator import (
            EnhancedBallFlightSimulator,
        )
        from src.shared.python.physics.ball_launch_conditions import (
            EnvironmentalConditions,
            LaunchConditions,
        )

        # Construct a low-pressure environment at altitude 0 so the pressure
        # term dominates the density derivation.
        low_pressure_pa = 95000.0
        env = EnvironmentalConditions.from_altitude(
            altitude_m=0.0, temperature_c=15.0, pressure_pa=low_pressure_pa
        )
        # Sanity: factory stored the pressure on the dataclass.
        assert env.sea_level_pressure_pa == low_pressure_pa
        # Sanity: construction-time density used the pressure override.
        expected_rho = air_density(0.0, 15.0, low_pressure_pa)
        assert math.isclose(env.air_density, expected_rho, rel_tol=1e-12)

        sim = EnhancedBallFlightSimulator(
            environment=env,
            aero_config=AerodynamicsConfig(),
            seed=0,
            track_altitude_density=True,
        )
        launch = LaunchConditions(velocity=50.0, launch_angle=math.radians(10.0))
        sim.simulate_trajectory(launch, max_time=0.05, dt=0.01)
        # After tracking begins, the per-step update must use the same
        # pressure override -- so density at z=0 stays at expected_rho.
        # (At t~0.05s with launch_angle=10deg, z is small but >0; we accept
        # a small ISA-driven decrease, but the value must be much closer to
        # the override-derived density than to the ISA-default density.)
        isa_default_rho = air_density(0.0, 15.0)  # 101325 Pa
        actual = sim._aero_engine._current_air_density
        # Override-derived density should be the dominant signal.
        assert abs(actual - expected_rho) < abs(actual - isa_default_rho), (
            f"per-step density {actual:.4f} closer to ISA default "
            f"{isa_default_rho:.4f} than to override-derived {expected_rho:.4f} "
            "-- pressure override is being dropped"
        )


# ---------------------------------------------------------------------------
# DragModel integration with cd_dimpled_sphere
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDragModelUsesDragCrisis:
    """The DragModel.get_effective_coefficient should now reflect the dip."""

    def test_drag_model_low_re_higher_than_high_re(self) -> None:
        from src.shared.python.physics.aerodynamics._models import DragModel

        model = DragModel(reynolds_correction=True)
        # Choose two velocities that bracket the drag crisis at sea-level rho.
        slow = np.array([3.0, 0.0, 0.0])  # very low Re
        fast = np.array([60.0, 0.0, 0.0])  # well past the crisis
        cd_slow = model.get_effective_coefficient(slow)
        cd_fast = model.get_effective_coefficient(fast)
        assert cd_slow > cd_fast, (
            f"DragModel Cd(slow)={cd_slow:.3f} should exceed "
            f"Cd(fast)={cd_fast:.3f} via the drag-crisis model"
        )
