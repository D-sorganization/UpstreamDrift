"""Unit tests for aerodynamics module.

TDD tests covering:
- AerodynamicsConfig: Toggle on/off, tunable parameters
- WindModel: Random wind, gusts, direction changes
- EnvironmentRandomizer: Stochastic environment simulation
- AerodynamicsEngine: Unified force calculations with toggle support

Following Pragmatic Programmer principles:
- Reversible: All effects can be toggled
- Reusable: Modular, composable components
- DRY: No duplication in test setup
- Orthogonal: Independent test cases
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from src.shared.python.aerodynamics import (
    AerodynamicsConfig,
    AerodynamicsEngine,
    DragModel,
    EnvironmentRandomizer,
    LiftModel,
    MagnusModel,
    RandomizationConfig,
    TurbulenceModel,
    WindGust,
    WindModel,
    WindConfig,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Test Fixtures (DRY - shared setup)
# =============================================================================


@pytest.fixture
def default_aero_config() -> AerodynamicsConfig:
    """Default aerodynamics configuration with all effects enabled."""
    return AerodynamicsConfig()


@pytest.fixture
def disabled_aero_config() -> AerodynamicsConfig:
    """Aerodynamics configuration with all effects disabled."""
    return AerodynamicsConfig(
        drag_enabled=False,
        lift_enabled=False,
        magnus_enabled=False,
    )


@pytest.fixture
def default_wind_config() -> WindConfig:
    """Default wind configuration."""
    return WindConfig()


@pytest.fixture
def gusty_wind_config() -> WindConfig:
    """Wind configuration with gusts enabled."""
    return WindConfig(
        base_velocity=np.array([5.0, 0.0, 0.0]),
        gusts_enabled=True,
        gust_intensity=0.3,
        gust_frequency=0.1,
    )


@pytest.fixture
def default_wind_model() -> WindModel:
    """Default wind model instance."""
    return WindModel()


@pytest.fixture
def default_randomizer() -> EnvironmentRandomizer:
    """Default environment randomizer."""
    return EnvironmentRandomizer()


@pytest.fixture
def default_engine() -> AerodynamicsEngine:
    """Default aerodynamics engine."""
    return AerodynamicsEngine()


@pytest.fixture
def typical_velocity() -> NDArray[np.floating]:
    """Typical golf ball velocity (driver shot)."""
    return np.array([65.0, 0.0, 15.0])  # m/s


@pytest.fixture
def typical_spin() -> NDArray[np.floating]:
    """Typical backspin (2500 rpm converted to rad/s)."""
    omega = 2500 * 2 * np.pi / 60  # rad/s
    return np.array([0.0, -omega, 0.0])  # Backspin axis


# =============================================================================
# AerodynamicsConfig Tests
# =============================================================================


class TestAerodynamicsConfig:
    """Tests for AerodynamicsConfig dataclass."""

    def test_default_all_enabled(self, default_aero_config: AerodynamicsConfig) -> None:
        """Test default configuration has all effects enabled."""
        assert default_aero_config.drag_enabled is True
        assert default_aero_config.lift_enabled is True
        assert default_aero_config.magnus_enabled is True
        assert default_aero_config.enabled is True

    def test_master_toggle_disabled(self) -> None:
        """Test master toggle disables all effects."""
        config = AerodynamicsConfig(enabled=False)
        assert config.is_drag_active() is False
        assert config.is_lift_active() is False
        assert config.is_magnus_active() is False

    def test_individual_toggles(self) -> None:
        """Test individual effect toggles work independently (Orthogonal)."""
        # Only drag
        config = AerodynamicsConfig(
            drag_enabled=True,
            lift_enabled=False,
            magnus_enabled=False,
        )
        assert config.is_drag_active() is True
        assert config.is_lift_active() is False
        assert config.is_magnus_active() is False

    def test_tunable_coefficients(self) -> None:
        """Test that aerodynamic coefficients are tunable."""
        config = AerodynamicsConfig(
            drag_coefficient=0.30,
            lift_coefficient=0.20,
            magnus_coefficient=0.35,
        )
        assert config.drag_coefficient == pytest.approx(0.30)
        assert config.lift_coefficient == pytest.approx(0.20)
        assert config.magnus_coefficient == pytest.approx(0.35)

    def test_coefficient_bounds_validation(self) -> None:
        """Test coefficient validation (must be positive)."""
        with pytest.raises(ValueError, match="coefficient"):
            AerodynamicsConfig(drag_coefficient=-0.1)

    def test_spin_decay_rate_tunable(self) -> None:
        """Test spin decay rate is configurable."""
        config = AerodynamicsConfig(spin_decay_rate=0.08)
        assert config.spin_decay_rate == pytest.approx(0.08)

    def test_reynolds_number_correction_toggle(self) -> None:
        """Test Reynolds number correction can be toggled."""
        config_on = AerodynamicsConfig(reynolds_correction_enabled=True)
        config_off = AerodynamicsConfig(reynolds_correction_enabled=False)
        assert config_on.reynolds_correction_enabled is True
        assert config_off.reynolds_correction_enabled is False

    def test_immutability(self, default_aero_config: AerodynamicsConfig) -> None:
        """Test config is immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            default_aero_config.drag_enabled = False  # type: ignore[misc]

    def test_copy_with_changes(self, default_aero_config: AerodynamicsConfig) -> None:
        """Test creating modified copy (Reversible pattern)."""
        modified = default_aero_config.with_changes(drag_enabled=False)
        assert modified.drag_enabled is False
        assert default_aero_config.drag_enabled is True  # Original unchanged


# =============================================================================
# DragModel Tests
# =============================================================================


class TestDragModel:
    """Tests for drag force model."""

    def test_drag_opposes_motion(
        self, typical_velocity: NDArray[np.floating]
    ) -> None:
        """Test drag force opposes velocity direction."""
        model = DragModel()
        drag = model.calculate(typical_velocity, air_density=1.225)

        # Drag should have opposite sign to velocity components
        for i in range(3):
            if abs(typical_velocity[i]) > 0.01:
                assert np.sign(drag[i]) == -np.sign(typical_velocity[i])

    def test_drag_proportional_to_speed_squared(self) -> None:
        """Test drag magnitude scales with v^2."""
        model = DragModel()
        v1 = np.array([10.0, 0.0, 0.0])
        v2 = np.array([20.0, 0.0, 0.0])  # 2x velocity

        drag1 = np.linalg.norm(model.calculate(v1, air_density=1.225))
        drag2 = np.linalg.norm(model.calculate(v2, air_density=1.225))

        # Drag should be ~4x for 2x velocity
        assert drag2 / drag1 == pytest.approx(4.0, rel=0.1)

    def test_zero_velocity_zero_drag(self) -> None:
        """Test zero velocity produces zero drag."""
        model = DragModel()
        drag = model.calculate(np.zeros(3), air_density=1.225)
        np.testing.assert_array_almost_equal(drag, np.zeros(3))

    def test_drag_coefficient_effect(self) -> None:
        """Test higher drag coefficient increases drag."""
        v = np.array([50.0, 0.0, 0.0])

        model_low = DragModel(base_coefficient=0.20)
        model_high = DragModel(base_coefficient=0.40)

        drag_low = np.linalg.norm(model_low.calculate(v, air_density=1.225))
        drag_high = np.linalg.norm(model_high.calculate(v, air_density=1.225))

        assert drag_high > drag_low

    def test_reynolds_number_correction(self) -> None:
        """Test Reynolds number affects drag coefficient."""
        model = DragModel(reynolds_correction=True)

        # Low speed (lower Re)
        v_low = np.array([10.0, 0.0, 0.0])
        # High speed (higher Re)
        v_high = np.array([70.0, 0.0, 0.0])

        cd_low = model.get_effective_coefficient(v_low, air_density=1.225)
        cd_high = model.get_effective_coefficient(v_high, air_density=1.225)

        # At high Re (turbulent), golf ball has lower Cd
        assert cd_high < cd_low


# =============================================================================
# LiftModel Tests
# =============================================================================


class TestLiftModel:
    """Tests for lift force model."""

    def test_lift_perpendicular_to_velocity(
        self, typical_velocity: NDArray[np.floating],
        typical_spin: NDArray[np.floating],
    ) -> None:
        """Test lift is perpendicular to velocity."""
        model = LiftModel()
        lift = model.calculate(typical_velocity, typical_spin, air_density=1.225)

        # Dot product should be near zero
        dot = np.dot(lift, typical_velocity)
        assert abs(dot) < 1e-6 * np.linalg.norm(lift) * np.linalg.norm(typical_velocity)

    def test_backspin_creates_upward_lift(self) -> None:
        """Test backspin generates upward lift (golf ball trajectory)."""
        model = LiftModel()
        velocity = np.array([50.0, 0.0, 0.0])  # Forward
        backspin = np.array([0.0, -260.0, 0.0])  # ~2500 rpm backspin

        lift = model.calculate(velocity, backspin, air_density=1.225)

        # Lift should have positive z component
        assert lift[2] > 0

    def test_no_spin_no_lift(self) -> None:
        """Test zero spin produces zero lift."""
        model = LiftModel()
        lift = model.calculate(
            np.array([50.0, 0.0, 0.0]),
            np.zeros(3),
            air_density=1.225
        )
        np.testing.assert_array_almost_equal(lift, np.zeros(3))

    def test_lift_increases_with_spin(self) -> None:
        """Test higher spin rate increases lift."""
        model = LiftModel()
        velocity = np.array([50.0, 0.0, 0.0])

        low_spin = np.array([0.0, -100.0, 0.0])  # Low spin
        high_spin = np.array([0.0, -300.0, 0.0])  # High spin

        lift_low = np.linalg.norm(model.calculate(velocity, low_spin, air_density=1.225))
        lift_high = np.linalg.norm(model.calculate(velocity, high_spin, air_density=1.225))

        assert lift_high > lift_low


# =============================================================================
# MagnusModel Tests
# =============================================================================


class TestMagnusModel:
    """Tests for Magnus force model."""

    def test_magnus_direction_spin_cross_velocity(
        self,
        typical_velocity: NDArray[np.floating],
        typical_spin: NDArray[np.floating],
    ) -> None:
        """Test Magnus force direction is omega x v."""
        model = MagnusModel()
        magnus = model.calculate(typical_velocity, typical_spin, air_density=1.225)

        # Magnus should be in direction of spin x velocity
        expected_dir = np.cross(typical_spin, typical_velocity)
        if np.linalg.norm(expected_dir) > 1e-10:
            expected_dir = expected_dir / np.linalg.norm(expected_dir)
            magnus_dir = magnus / (np.linalg.norm(magnus) + 1e-10)
            np.testing.assert_array_almost_equal(magnus_dir, expected_dir, decimal=5)

    def test_sidespin_causes_lateral_deflection(self) -> None:
        """Test sidespin (hook/slice) creates lateral Magnus force."""
        model = MagnusModel()
        velocity = np.array([50.0, 0.0, 0.0])  # Forward
        sidespin = np.array([0.0, 0.0, 200.0])  # Spin about z-axis

        magnus = model.calculate(velocity, sidespin, air_density=1.225)

        # Should have significant y-component (lateral)
        assert abs(magnus[1]) > 0

    def test_coefficient_tunability(self) -> None:
        """Test Magnus coefficient can be tuned."""
        velocity = np.array([50.0, 0.0, 0.0])
        spin = np.array([0.0, -200.0, 0.0])

        model_low = MagnusModel(coefficient=0.15)
        model_high = MagnusModel(coefficient=0.35)

        mag_low = np.linalg.norm(model_low.calculate(velocity, spin, air_density=1.225))
        mag_high = np.linalg.norm(model_high.calculate(velocity, spin, air_density=1.225))

        assert mag_high > mag_low


# =============================================================================
# WindConfig Tests
# =============================================================================


class TestWindConfig:
    """Tests for WindConfig dataclass."""

    def test_default_no_wind(self, default_wind_config: WindConfig) -> None:
        """Test default is zero wind."""
        np.testing.assert_array_equal(
            default_wind_config.base_velocity,
            np.zeros(3)
        )

    def test_wind_direction_normalized(self) -> None:
        """Test wind direction is normalized."""
        config = WindConfig(
            base_velocity=np.array([10.0, 5.0, 0.0])
        )
        direction = config.direction
        assert np.linalg.norm(direction) == pytest.approx(1.0)

    def test_wind_speed_property(self) -> None:
        """Test wind speed calculation."""
        config = WindConfig(
            base_velocity=np.array([3.0, 4.0, 0.0])  # 3-4-5 triangle
        )
        assert config.speed == pytest.approx(5.0)


# =============================================================================
# WindModel Tests
# =============================================================================


class TestWindModel:
    """Tests for wind model with gusts and randomness."""

    def test_constant_wind_no_gusts(self) -> None:
        """Test constant wind without gusts."""
        config = WindConfig(
            base_velocity=np.array([5.0, 0.0, 0.0]),
            gusts_enabled=False,
        )
        model = WindModel(config)

        # Should return constant wind
        wind1 = model.get_wind_at(t=0.0, position=np.zeros(3))
        wind2 = model.get_wind_at(t=1.0, position=np.zeros(3))

        np.testing.assert_array_almost_equal(wind1, wind2)
        np.testing.assert_array_almost_equal(wind1, config.base_velocity)

    def test_gusts_add_variation(self) -> None:
        """Test that gusts add temporal variation."""
        config = WindConfig(
            base_velocity=np.array([5.0, 0.0, 0.0]),
            gusts_enabled=True,
            gust_intensity=0.5,
        )
        model = WindModel(config, seed=42)

        # Sample at different times
        winds = [model.get_wind_at(t=t, position=np.zeros(3)) for t in np.linspace(0, 10, 20)]

        # Should have variation
        wind_speeds = [np.linalg.norm(w) for w in winds]
        assert max(wind_speeds) != min(wind_speeds)

    def test_gust_intensity_bounds(self) -> None:
        """Test gust intensity affects magnitude."""
        config_low = WindConfig(
            base_velocity=np.array([5.0, 0.0, 0.0]),
            gusts_enabled=True,
            gust_intensity=0.1,
        )
        config_high = WindConfig(
            base_velocity=np.array([5.0, 0.0, 0.0]),
            gusts_enabled=True,
            gust_intensity=0.9,
        )

        model_low = WindModel(config_low, seed=42)
        model_high = WindModel(config_high, seed=42)

        # Sample many times
        speeds_low = [
            np.linalg.norm(model_low.get_wind_at(t=t, position=np.zeros(3)))
            for t in np.linspace(0, 10, 100)
        ]
        speeds_high = [
            np.linalg.norm(model_high.get_wind_at(t=t, position=np.zeros(3)))
            for t in np.linspace(0, 10, 100)
        ]

        # Higher intensity should have more variance
        assert np.std(speeds_high) > np.std(speeds_low)

    def test_reproducibility_with_seed(self) -> None:
        """Test same seed produces same wind sequence."""
        config = WindConfig(
            base_velocity=np.array([5.0, 0.0, 0.0]),
            gusts_enabled=True,
        )

        model1 = WindModel(config, seed=123)
        model2 = WindModel(config, seed=123)

        for t in [0.0, 0.5, 1.0, 2.0]:
            w1 = model1.get_wind_at(t=t, position=np.zeros(3))
            w2 = model2.get_wind_at(t=t, position=np.zeros(3))
            np.testing.assert_array_almost_equal(w1, w2)

    def test_different_seeds_different_wind(self) -> None:
        """Test different seeds produce different sequences."""
        config = WindConfig(
            base_velocity=np.array([5.0, 0.0, 0.0]),
            gusts_enabled=True,
            gust_intensity=0.5,
        )

        model1 = WindModel(config, seed=111)
        model2 = WindModel(config, seed=222)

        # At least some samples should differ
        differences = []
        for t in np.linspace(0, 5, 10):
            w1 = model1.get_wind_at(t=t, position=np.zeros(3))
            w2 = model2.get_wind_at(t=t, position=np.zeros(3))
            differences.append(np.linalg.norm(w1 - w2))

        assert max(differences) > 0.1  # Some difference exists

    def test_altitude_wind_gradient(self) -> None:
        """Test wind increases with altitude (wind shear)."""
        config = WindConfig(
            base_velocity=np.array([5.0, 0.0, 0.0]),
            altitude_gradient=True,
            gradient_factor=0.1,  # 10% per 10m
        )
        model = WindModel(config)

        wind_low = model.get_wind_at(t=0.0, position=np.array([0.0, 0.0, 0.0]))
        wind_high = model.get_wind_at(t=0.0, position=np.array([0.0, 0.0, 30.0]))

        assert np.linalg.norm(wind_high) > np.linalg.norm(wind_low)


# =============================================================================
# WindGust Tests
# =============================================================================


class TestWindGust:
    """Tests for individual wind gust events."""

    def test_gust_creation(self) -> None:
        """Test gust can be created with parameters."""
        gust = WindGust(
            start_time=1.0,
            duration=2.0,
            peak_velocity=np.array([3.0, 1.0, 0.0]),
        )
        assert gust.start_time == 1.0
        assert gust.duration == 2.0
        assert gust.end_time == 3.0

    def test_gust_envelope_shape(self) -> None:
        """Test gust has smooth envelope (ramps up and down)."""
        gust = WindGust(
            start_time=0.0,
            duration=2.0,
            peak_velocity=np.array([10.0, 0.0, 0.0]),
        )

        # At start, should be zero
        assert np.linalg.norm(gust.get_velocity_at(0.0)) == pytest.approx(0.0, abs=0.01)

        # At middle, should be at peak
        mid_vel = gust.get_velocity_at(1.0)
        assert np.linalg.norm(mid_vel) == pytest.approx(10.0, rel=0.1)

        # At end, should be zero
        assert np.linalg.norm(gust.get_velocity_at(2.0)) == pytest.approx(0.0, abs=0.01)


# =============================================================================
# TurbulenceModel Tests
# =============================================================================


class TestTurbulenceModel:
    """Tests for turbulence model."""

    def test_turbulence_zero_intensity(self) -> None:
        """Test zero turbulence intensity produces no perturbation."""
        model = TurbulenceModel(intensity=0.0)
        perturb = model.get_perturbation(t=0.0, position=np.zeros(3))
        np.testing.assert_array_almost_equal(perturb, np.zeros(3))

    def test_turbulence_varies_with_time(self) -> None:
        """Test turbulence varies over time."""
        model = TurbulenceModel(intensity=1.0, seed=42)

        perturbations = [
            model.get_perturbation(t=t, position=np.zeros(3))
            for t in [0.0, 0.1, 0.2, 0.5, 1.0]
        ]

        # Check that not all are equal
        norms = [np.linalg.norm(p) for p in perturbations]
        assert len(set(norms)) > 1  # Multiple unique values

    def test_turbulence_intensity_scaling(self) -> None:
        """Test intensity scales the perturbation magnitude."""
        low_model = TurbulenceModel(intensity=0.5, seed=42)
        high_model = TurbulenceModel(intensity=2.0, seed=42)

        low_perturb = low_model.get_perturbation(t=0.5, position=np.zeros(3))
        high_perturb = high_model.get_perturbation(t=0.5, position=np.zeros(3))

        # Higher intensity should give larger magnitude
        assert np.linalg.norm(high_perturb) > np.linalg.norm(low_perturb)


# =============================================================================
# RandomizationConfig Tests
# =============================================================================


class TestRandomizationConfig:
    """Tests for environment randomization configuration."""

    def test_default_no_randomization(self) -> None:
        """Test default config has no randomization."""
        config = RandomizationConfig()
        assert config.enabled is False
        assert config.air_density_variance == 0.0
        assert config.temperature_variance == 0.0

    def test_enable_randomization(self) -> None:
        """Test enabling randomization."""
        config = RandomizationConfig(
            enabled=True,
            air_density_variance=0.05,
            temperature_variance=2.0,
            wind_variance=0.1,
        )
        assert config.enabled is True
        assert config.air_density_variance == 0.05


# =============================================================================
# EnvironmentRandomizer Tests
# =============================================================================


class TestEnvironmentRandomizer:
    """Tests for environment randomization."""

    def test_no_randomization_returns_base(self) -> None:
        """Test disabled randomization returns base values."""
        config = RandomizationConfig(enabled=False)
        randomizer = EnvironmentRandomizer(config)

        base_density = 1.225
        result = randomizer.randomize_air_density(base_density)

        assert result == base_density

    def test_randomization_changes_values(self) -> None:
        """Test enabled randomization modifies values."""
        config = RandomizationConfig(
            enabled=True,
            air_density_variance=0.1,
        )
        randomizer = EnvironmentRandomizer(config, seed=42)

        base_density = 1.225
        results = [randomizer.randomize_air_density(base_density) for _ in range(100)]

        # Should have variation
        assert min(results) != max(results)
        # Should be centered around base
        assert np.mean(results) == pytest.approx(base_density, rel=0.1)

    def test_reproducibility_with_seed(self) -> None:
        """Test same seed gives same randomization."""
        config = RandomizationConfig(enabled=True, air_density_variance=0.1)

        r1 = EnvironmentRandomizer(config, seed=999)
        r2 = EnvironmentRandomizer(config, seed=999)

        for _ in range(10):
            assert r1.randomize_air_density(1.225) == r2.randomize_air_density(1.225)

    def test_randomize_wind_config(self) -> None:
        """Test wind configuration randomization."""
        config = RandomizationConfig(
            enabled=True,
            wind_variance=0.2,
            wind_direction_variance=0.1,  # radians
        )
        randomizer = EnvironmentRandomizer(config, seed=42)

        base_wind = WindConfig(base_velocity=np.array([5.0, 0.0, 0.0]))
        randomized = randomizer.randomize_wind_config(base_wind)

        # Should be different but similar
        base_speed = base_wind.speed
        randomized_speed = randomized.speed

        assert randomized_speed != base_speed
        assert abs(randomized_speed - base_speed) < base_speed * 0.5  # Within 50%

    def test_snapshot_creates_consistent_environment(self) -> None:
        """Test snapshot creates consistent random environment."""
        config = RandomizationConfig(
            enabled=True,
            air_density_variance=0.05,
            temperature_variance=2.0,
        )
        randomizer = EnvironmentRandomizer(config, seed=42)

        snapshot = randomizer.create_snapshot(
            base_air_density=1.225,
            base_temperature=15.0,
        )

        # Snapshot should have consistent values
        assert snapshot.air_density == snapshot.air_density  # Same on re-access
        assert hasattr(snapshot, 'air_density')
        assert hasattr(snapshot, 'temperature')


# =============================================================================
# AerodynamicsEngine Tests
# =============================================================================


class TestAerodynamicsEngine:
    """Tests for unified aerodynamics engine."""

    def test_default_engine_creates_forces(
        self,
        default_engine: AerodynamicsEngine,
        typical_velocity: NDArray[np.floating],
        typical_spin: NDArray[np.floating],
    ) -> None:
        """Test engine computes all force components."""
        forces = default_engine.compute_forces(typical_velocity, typical_spin)

        assert 'drag' in forces
        assert 'lift' in forces
        assert 'magnus' in forces
        assert 'total' in forces

        # Total should be sum of components
        expected_total = forces['drag'] + forces['lift'] + forces['magnus']
        np.testing.assert_array_almost_equal(forces['total'], expected_total)

    def test_disabled_config_zero_forces(
        self,
        typical_velocity: NDArray[np.floating],
        typical_spin: NDArray[np.floating],
    ) -> None:
        """Test disabled aerodynamics produces zero forces."""
        config = AerodynamicsConfig(enabled=False)
        engine = AerodynamicsEngine(config)

        forces = engine.compute_forces(typical_velocity, typical_spin)

        np.testing.assert_array_almost_equal(forces['total'], np.zeros(3))

    def test_individual_force_toggles(
        self,
        typical_velocity: NDArray[np.floating],
        typical_spin: NDArray[np.floating],
    ) -> None:
        """Test individual force components can be toggled (Orthogonal)."""
        # Drag only
        config = AerodynamicsConfig(
            drag_enabled=True,
            lift_enabled=False,
            magnus_enabled=False,
        )
        engine = AerodynamicsEngine(config)
        forces = engine.compute_forces(typical_velocity, typical_spin)

        assert np.linalg.norm(forces['drag']) > 0
        np.testing.assert_array_almost_equal(forces['lift'], np.zeros(3))
        np.testing.assert_array_almost_equal(forces['magnus'], np.zeros(3))

    def test_wind_affects_relative_velocity(
        self,
        typical_velocity: NDArray[np.floating],
        typical_spin: NDArray[np.floating],
    ) -> None:
        """Test wind modifies effective velocity for force calculation."""
        engine_no_wind = AerodynamicsEngine()
        wind_model = WindModel(WindConfig(base_velocity=np.array([10.0, 0.0, 0.0])))
        engine_with_wind = AerodynamicsEngine(wind_model=wind_model)

        forces_no_wind = engine_no_wind.compute_forces(typical_velocity, typical_spin)
        forces_with_wind = engine_with_wind.compute_forces(typical_velocity, typical_spin)

        # Forces should differ
        assert not np.allclose(forces_no_wind['drag'], forces_with_wind['drag'])

    def test_compute_acceleration(
        self,
        default_engine: AerodynamicsEngine,
        typical_velocity: NDArray[np.floating],
        typical_spin: NDArray[np.floating],
    ) -> None:
        """Test acceleration computation includes mass."""
        mass = 0.0459  # kg

        forces = default_engine.compute_forces(typical_velocity, typical_spin)
        accel = default_engine.compute_acceleration(typical_velocity, typical_spin, mass)

        expected_accel = forces['total'] / mass
        np.testing.assert_array_almost_equal(accel, expected_accel)

    def test_spin_decay_computation(self, default_engine: AerodynamicsEngine) -> None:
        """Test spin decay over time."""
        initial_spin = np.array([0.0, -260.0, 0.0])  # rad/s
        dt = 0.1  # seconds

        decayed_spin = default_engine.compute_spin_decay(initial_spin, dt)

        # Spin should decrease
        assert np.linalg.norm(decayed_spin) < np.linalg.norm(initial_spin)

    def test_spin_decay_rate_configurable(self) -> None:
        """Test spin decay rate affects decay speed."""
        initial_spin = np.array([0.0, -260.0, 0.0])
        dt = 0.5

        slow_decay = AerodynamicsEngine(
            AerodynamicsConfig(spin_decay_rate=0.02)
        )
        fast_decay = AerodynamicsEngine(
            AerodynamicsConfig(spin_decay_rate=0.2)
        )

        spin_slow = slow_decay.compute_spin_decay(initial_spin, dt)
        spin_fast = fast_decay.compute_spin_decay(initial_spin, dt)

        # Fast decay should result in less spin
        assert np.linalg.norm(spin_fast) < np.linalg.norm(spin_slow)

    def test_with_randomization(
        self,
        typical_velocity: NDArray[np.floating],
        typical_spin: NDArray[np.floating],
    ) -> None:
        """Test engine with environment randomization."""
        config = AerodynamicsConfig()
        rand_config = RandomizationConfig(enabled=True, air_density_variance=0.1)

        engine = AerodynamicsEngine(
            config,
            randomization=EnvironmentRandomizer(rand_config, seed=42),
        )

        # Get multiple force samples
        forces_samples = [
            engine.compute_forces(typical_velocity, typical_spin, resample=True)
            for _ in range(10)
        ]

        # Should have variation
        drag_mags = [np.linalg.norm(f['drag']) for f in forces_samples]
        assert min(drag_mags) != max(drag_mags)


# =============================================================================
# Integration Tests
# =============================================================================


class TestAerodynamicsIntegration:
    """Integration tests for aerodynamics system."""

    def test_full_trajectory_with_aerodynamics(self) -> None:
        """Test aerodynamics can be used in trajectory simulation."""
        config = AerodynamicsConfig(
            drag_enabled=True,
            lift_enabled=True,
            magnus_enabled=True,
        )
        wind_config = WindConfig(
            base_velocity=np.array([2.0, 0.0, 0.0]),
            gusts_enabled=True,
            gust_intensity=0.2,
        )

        engine = AerodynamicsEngine(
            config,
            wind_model=WindModel(wind_config, seed=42),
        )

        # Simulate a simple trajectory step
        position = np.zeros(3)
        velocity = np.array([60.0, 0.0, 20.0])
        spin = np.array([0.0, -250.0, 0.0])
        mass = 0.0459
        dt = 0.01

        # Step
        accel = engine.compute_acceleration(velocity, spin, mass, t=0.0, position=position)
        new_velocity = velocity + accel * dt
        new_position = position + new_velocity * dt
        new_spin = engine.compute_spin_decay(spin, dt)

        # Velocity should have changed
        assert not np.allclose(velocity, new_velocity)
        # Position should have advanced
        assert new_position[0] > position[0]
        # Spin should have decayed
        assert np.linalg.norm(new_spin) < np.linalg.norm(spin)

    def test_aerodynamics_toggle_comparison(self) -> None:
        """Test comparing trajectories with/without aerodynamics."""
        velocity = np.array([60.0, 0.0, 20.0])
        spin = np.array([0.0, -250.0, 0.0])
        mass = 0.0459

        # With aerodynamics
        engine_on = AerodynamicsEngine(AerodynamicsConfig(enabled=True))
        accel_on = engine_on.compute_acceleration(velocity, spin, mass)

        # Without aerodynamics
        engine_off = AerodynamicsEngine(AerodynamicsConfig(enabled=False))
        accel_off = engine_off.compute_acceleration(velocity, spin, mass)

        # Aerodynamics should add deceleration (drag) and lift
        # Without aerodynamics, acceleration should be zero
        np.testing.assert_array_almost_equal(accel_off, np.zeros(3))
        assert np.linalg.norm(accel_on) > 0

    def test_reversibility_toggle_on_off(self) -> None:
        """Test aerodynamics can be toggled on/off (Reversible pattern)."""
        velocity = np.array([60.0, 0.0, 20.0])
        spin = np.array([0.0, -250.0, 0.0])

        config_on = AerodynamicsConfig(enabled=True)
        config_off = config_on.with_changes(enabled=False)
        config_back_on = config_off.with_changes(enabled=True)

        engine1 = AerodynamicsEngine(config_on)
        engine2 = AerodynamicsEngine(config_back_on)

        forces1 = engine1.compute_forces(velocity, spin)
        forces2 = engine2.compute_forces(velocity, spin)

        # Should be identical
        np.testing.assert_array_almost_equal(forces1['total'], forces2['total'])
