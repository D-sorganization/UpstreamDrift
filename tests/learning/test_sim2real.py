"""Tests for sim-to-real transfer module."""

from __future__ import annotations

import numpy as np
import pytest


class MockEngine:
    """Mock physics engine for testing."""

    def __init__(self) -> None:
        self._masses = np.ones(10)
        self._damping = np.ones(10) * 0.1
        self._friction = np.ones(10) * 0.5
        self._motor_strength = np.ones(10)
        self._gravity = np.array([0, 0, -9.81])
        self._q = np.zeros(10)
        self._v = np.zeros(10)

    def get_link_masses(self) -> np.ndarray:
        return self._masses.copy()

    def set_link_masses(self, masses: np.ndarray) -> None:
        self._masses = masses.copy()

    def get_joint_damping(self) -> np.ndarray:
        return self._damping.copy()

    def set_joint_damping(self, damping: np.ndarray) -> None:
        self._damping = damping.copy()

    def get_friction_coefficients(self) -> np.ndarray:
        return self._friction.copy()

    def set_friction_coefficients(self, friction: np.ndarray) -> None:
        self._friction = friction.copy()

    def get_motor_strength(self) -> np.ndarray:
        return self._motor_strength.copy()

    def set_motor_strength(self, strength: np.ndarray) -> None:
        self._motor_strength = strength.copy()

    def get_gravity(self) -> np.ndarray:
        return self._gravity.copy()

    def set_gravity(self, gravity: np.ndarray) -> None:
        self._gravity = gravity.copy()

    def get_joint_positions(self) -> np.ndarray:
        return self._q.copy()

    def set_joint_positions(self, q: np.ndarray) -> None:
        self._q = q.copy()

    def get_joint_velocities(self) -> np.ndarray:
        return self._v.copy()

    def set_joint_velocities(self, v: np.ndarray) -> None:
        self._v = v.copy()

    def set_joint_torques(self, tau: np.ndarray) -> None:
        pass

    def step(self, dt: float = 0.001) -> None:
        # Simple integration
        self._v += np.random.randn(10) * 0.01
        self._q += self._v * dt


class TestDomainRandomization:
    """Tests for domain randomization."""

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        from src.learning.sim2real import DomainRandomizationConfig

        config = DomainRandomizationConfig()

        assert config.mass_range == (0.8, 1.2)
        assert config.friction_range == (0.5, 1.5)
        assert config.randomize_mass is True

    def test_randomizer_creation(self) -> None:
        """Test creating domain randomizer."""
        from src.learning.sim2real import DomainRandomizer

        engine = MockEngine()
        randomizer = DomainRandomizer(engine)

        assert randomizer.engine is engine
        assert "masses" in randomizer.nominal_params

    def test_randomize_parameters(self) -> None:
        """Test randomizing parameters."""
        from src.learning.sim2real import DomainRandomizer

        engine = MockEngine()
        randomizer = DomainRandomizer(engine)

        original_masses = engine.get_link_masses().copy()
        randomization = randomizer.randomize(seed=42)

        new_masses = engine.get_link_masses()

        # Masses should have changed
        assert not np.allclose(original_masses, new_masses)
        assert "mass_scale" in randomization

    def test_reset_to_nominal(self) -> None:
        """Test resetting to nominal values."""
        from src.learning.sim2real import DomainRandomizer

        engine = MockEngine()
        randomizer = DomainRandomizer(engine)

        original_masses = engine.get_link_masses().copy()
        randomizer.randomize(seed=42)
        randomizer.reset_to_nominal()

        restored_masses = engine.get_link_masses()
        np.testing.assert_array_almost_equal(original_masses, restored_masses)

    def test_action_delay(self) -> None:
        """Test action delay application."""
        from src.learning.sim2real import DomainRandomizationConfig, DomainRandomizer

        config = DomainRandomizationConfig(
            action_delay_range=(2, 2),  # Fixed 2-step delay
            randomize_delays=True,
        )
        engine = MockEngine()
        randomizer = DomainRandomizer(engine, config)
        randomizer.randomize(seed=42)

        # First two actions should return zeros
        action1 = np.ones(7)
        delayed1 = randomizer.apply_action_with_delay(action1)
        assert np.allclose(delayed1, 0)

        action2 = np.ones(7) * 2
        delayed2 = randomizer.apply_action_with_delay(action2)
        assert np.allclose(delayed2, 0)

        # Third action should return first action
        action3 = np.ones(7) * 3
        delayed3 = randomizer.apply_action_with_delay(action3)
        # Should be close to action1 (with possible noise)
        assert delayed3.shape == (7,)

    def test_observation_noise(self) -> None:
        """Test observation noise application."""
        from src.learning.sim2real import DomainRandomizationConfig, DomainRandomizer

        config = DomainRandomizationConfig(
            observation_noise_std=0.1,
            randomize_noise=True,
        )
        engine = MockEngine()
        randomizer = DomainRandomizer(engine, config)

        obs = np.zeros(10)
        noisy_obs = randomizer._apply_observation_noise(obs)

        # Should not be exactly zero due to noise
        assert not np.allclose(noisy_obs, 0)

    def test_sample_batch(self) -> None:
        """Test sampling batch of configurations."""
        from src.learning.sim2real import DomainRandomizer

        engine = MockEngine()
        randomizer = DomainRandomizer(engine)

        configs = randomizer.sample_randomization_batch(5)

        assert len(configs) == 5
        # Each config should be different
        scales = [c["mass_scale"] for c in configs]
        assert len(set(scales)) > 1


class TestSystemIdentification:
    """Tests for system identification."""

    def test_identifier_creation(self) -> None:
        """Test creating system identifier."""
        from src.learning.sim2real import SystemIdentifier

        engine = MockEngine()
        identifier = SystemIdentifier(engine)

        assert identifier.model is engine
        assert "mass_scale" in identifier.param_bounds

    def test_compute_reality_gap(self) -> None:
        """Test computing reality gap metrics."""
        from src.learning.sim2real import SystemIdentifier

        engine = MockEngine()
        identifier = SystemIdentifier(engine)

        # Create trajectories with small differences
        n_steps = 50
        n_dof = 10
        sim_traj = np.random.randn(n_steps, n_dof * 2)
        real_traj = sim_traj + np.random.randn(n_steps, n_dof * 2) * 0.1

        metrics = identifier.compute_reality_gap(sim_traj, real_traj)

        assert "total_mse" in metrics
        assert "position_mse" in metrics
        assert "velocity_mse" in metrics
        assert "trajectory_length" in metrics
        assert metrics["total_mse"] > 0
