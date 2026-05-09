"""Tests for src.shared.python.data_io.dataset_generator (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.data_io.dataset_generator import (
    ControlProfile,
    ParameterRange,
)


class TestParameterRange:
    def test_dataset_generator_construction(self) -> None:
        pr = ParameterRange(name="mass", min_val=0.5, max_val=2.0)
        assert pr.name == "mass"
        assert pr.min_val == pytest.approx(0.5)
        assert pr.max_val == pytest.approx(2.0)

    def test_default_distribution_is_uniform(self) -> None:
        pr = ParameterRange(name="x", min_val=0.0, max_val=1.0)
        assert pr.distribution == "uniform"

    def test_dataset_generator_default_num_points(self) -> None:
        pr = ParameterRange(name="x", min_val=0.0, max_val=1.0)
        assert pr.num_points == 10

    def test_min_gt_max_raises(self) -> None:
        with pytest.raises(ValueError):
            ParameterRange(name="x", min_val=2.0, max_val=1.0)

    def test_equal_min_max_allowed(self) -> None:
        pr = ParameterRange(name="x", min_val=1.0, max_val=1.0)
        assert pr.min_val == pytest.approx(pr.max_val)

    def test_invalid_distribution_raises(self) -> None:
        with pytest.raises(ValueError):
            ParameterRange(
                name="x", min_val=0.0, max_val=1.0, distribution="exponential"
            )

    def test_uniform_sample_in_range(self) -> None:
        pr = ParameterRange(name="x", min_val=0.0, max_val=1.0, distribution="uniform")
        rng = np.random.default_rng(42)
        for _ in range(20):
            val = pr.sample(rng)
            assert 0.0 <= val <= 1.0

    def test_normal_sample_clipped_to_range(self) -> None:
        pr = ParameterRange(name="x", min_val=0.0, max_val=1.0, distribution="normal")
        rng = np.random.default_rng(42)
        for _ in range(20):
            val = pr.sample(rng)
            assert 0.0 <= val <= 1.0

    def test_linspace_sample_in_range(self) -> None:
        pr = ParameterRange(
            name="x", min_val=0.0, max_val=1.0, distribution="linspace", num_points=5
        )
        rng = np.random.default_rng(42)
        val = pr.sample(rng)
        assert 0.0 <= val <= 1.0

    def test_linspace_returns_correct_shape(self) -> None:
        pr = ParameterRange(name="x", min_val=0.0, max_val=1.0, num_points=10)
        result = pr.linspace()
        assert len(result) == 10

    def test_linspace_starts_at_min(self) -> None:
        pr = ParameterRange(name="x", min_val=0.5, max_val=1.5, num_points=5)
        result = pr.linspace()
        assert result[0] == pytest.approx(0.5)

    def test_linspace_ends_at_max(self) -> None:
        pr = ParameterRange(name="x", min_val=0.5, max_val=1.5, num_points=5)
        result = pr.linspace()
        assert result[-1] == pytest.approx(1.5)


class TestControlProfile:
    def test_dataset_generator_construction(self) -> None:
        cp = ControlProfile(name="zero_torque")
        assert cp.name == "zero_torque"

    def test_default_profile_type(self) -> None:
        cp = ControlProfile(name="my_profile")
        assert cp.profile_type == "zero"

    def test_custom_profile_type(self) -> None:
        cp = ControlProfile(name="sin_profile", profile_type="sinusoidal")
        assert cp.profile_type == "sinusoidal"

    def test_dataset_generator_default_parameters_empty(self) -> None:
        cp = ControlProfile(name="my_profile")
        assert cp.parameters == {}

    def test_dataset_generator_custom_parameters(self) -> None:
        cp = ControlProfile(
            name="step_profile",
            profile_type="step",
            parameters={"amplitude": 10.0, "t_step": 0.5},
        )
        assert cp.parameters["amplitude"] == pytest.approx(10.0)
        assert cp.parameters["t_step"] == pytest.approx(0.5)
