"""Wave 6 coverage: src.learning.sim2real.domain_randomization."""

from __future__ import annotations

import numpy as np
import pytest

from src.learning.sim2real.domain_randomization import (
    DomainRandomizationConfig,
    DomainRandomizer,
)


class FakeEngine:
    """Minimal physics-engine stand-in implementing the optional accessors."""

    def __init__(self, n: int = 3) -> None:
        self.masses = np.ones(n)
        self.damping = np.ones(n) * 0.1
        self.friction = np.ones(n) * 0.5
        self.motor_strength = np.ones(n) * 2.0
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.actuator_gains = np.ones(n)

    def get_link_masses(self) -> np.ndarray:
        return self.masses

    def set_link_masses(self, v: np.ndarray) -> None:
        self.masses = v

    def get_joint_damping(self) -> np.ndarray:
        return self.damping

    def set_joint_damping(self, v: np.ndarray) -> None:
        self.damping = v

    def get_friction_coefficients(self) -> np.ndarray:
        return self.friction

    def set_friction_coefficients(self, v: np.ndarray) -> None:
        self.friction = v

    def get_motor_strength(self) -> np.ndarray:
        return self.motor_strength

    def set_motor_strength(self, v: np.ndarray) -> None:
        self.motor_strength = v

    def get_gravity(self) -> np.ndarray:
        return self.gravity

    def set_gravity(self, v: np.ndarray) -> None:
        self.gravity = v

    def get_actuator_gains(self) -> np.ndarray:
        return self.actuator_gains


class TestConfig:
    def test_defaults(self) -> None:
        c = DomainRandomizationConfig()
        assert c.mass_range == (0.8, 1.2)
        assert c.randomize_mass is True


class TestRandomizer:
    def test_engine_none_raises(self) -> None:
        with pytest.raises(ValueError):
            DomainRandomizer(None)  # type: ignore[arg-type]

    def test_store_nominal(self) -> None:
        eng = FakeEngine()
        r = DomainRandomizer(eng)
        assert "masses" in r.nominal_params
        assert "actuator_gains" in r.nominal_params

    def test_randomize_with_seed_repeatable(self) -> None:
        eng = FakeEngine()
        r = DomainRandomizer(eng)
        a = r.randomize(seed=42)
        eng2 = FakeEngine()
        r2 = DomainRandomizer(eng2)
        b = r2.randomize(seed=42)
        assert a["mass_scale"] == b["mass_scale"]

    def test_randomize_populates_keys(self) -> None:
        r = DomainRandomizer(FakeEngine())
        rand = r.randomize(seed=1)
        for key in (
            "mass_scale",
            "friction_scale",
            "damping_scale",
            "motor_scale",
            "gravity",
            "action_delay",
            "observation_delay",
        ):
            assert key in rand

    def test_reset_to_nominal(self) -> None:
        eng = FakeEngine()
        r = DomainRandomizer(eng)
        r.randomize(seed=3)
        r.reset_to_nominal()
        np.testing.assert_array_equal(eng.masses, np.ones(3))
        assert r.get_current_randomization() == {}

    def test_apply_action_with_delay_zero(self) -> None:
        r = DomainRandomizer(FakeEngine())
        # delay disabled
        r._action_delay = 0
        action = np.array([1.0, 2.0])
        out = r.apply_action_with_delay(action)
        # noise applied with default std 0.01 -> shape preserved
        assert out.shape == action.shape

    def test_apply_action_with_delay_buffers(self) -> None:
        r = DomainRandomizer(FakeEngine())
        r.config.randomize_delays = True
        r.config.randomize_noise = False
        r._action_delay = 2
        a1 = np.array([1.0])
        a2 = np.array([2.0])
        a3 = np.array([3.0])
        # First two go into the buffer and return zeros
        out1 = r.apply_action_with_delay(a1)
        out2 = r.apply_action_with_delay(a2)
        out3 = r.apply_action_with_delay(a3)
        np.testing.assert_array_equal(out1, [0.0])
        np.testing.assert_array_equal(out2, [0.0])
        np.testing.assert_array_equal(out3, [1.0])

    def test_apply_action_noise_disabled(self) -> None:
        r = DomainRandomizer(FakeEngine())
        r.config.randomize_noise = False
        a = np.array([1.0])
        out = r._apply_action_noise(a)
        np.testing.assert_array_equal(out, a)

    def test_apply_action_none_raises(self) -> None:
        r = DomainRandomizer(FakeEngine())
        with pytest.raises(ValueError):
            r.apply_action_with_delay(None)  # type: ignore[arg-type]

    def test_get_observation_with_delay(self) -> None:
        r = DomainRandomizer(FakeEngine())
        r.config.randomize_delays = True
        r.config.randomize_noise = False
        r._observation_delay = 1
        first = np.array([1.0])
        second = np.array([2.0])
        out1 = r.get_observation_with_delay(first)
        out2 = r.get_observation_with_delay(second)
        # first call returns buffered first obs
        np.testing.assert_array_equal(out1, first)
        np.testing.assert_array_equal(out2, first)

    def test_get_observation_no_delay(self) -> None:
        r = DomainRandomizer(FakeEngine())
        r.config.randomize_delays = False
        r.config.randomize_noise = False
        out = r.get_observation_with_delay(np.array([1.0]))
        np.testing.assert_array_equal(out, [1.0])

    def test_sample_batch(self) -> None:
        r = DomainRandomizer(FakeEngine())
        batch = r.sample_randomization_batch(3)
        assert len(batch) == 3
