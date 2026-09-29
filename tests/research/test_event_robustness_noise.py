"""Common-random-number contracts for issue #9125."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.event_robustness_noise import (
    RobustnessNoiseConfig,
    generate_common_random_perturbations,
)

pytestmark = pytest.mark.unit


def test_antithetic_noise_is_deterministic_centered_and_immutable() -> None:
    config = RobustnessNoiseConfig(
        seed=9125,
        replicate_count=8,
        initial_state_sd=(0.01, 0.02, 0.1, 0.2),
        command_sd_nm=(0.5, 0.25),
        guard_offset_sd=0.005,
    )

    first = generate_common_random_perturbations(config, control_sample_count=5)
    second = generate_common_random_perturbations(config, control_sample_count=5)

    np.testing.assert_array_equal(first.initial_state_delta, second.initial_state_delta)
    np.testing.assert_array_equal(first.command_delta_nm, second.command_delta_nm)
    np.testing.assert_array_equal(first.guard_offset_delta, second.guard_offset_delta)
    np.testing.assert_allclose(first.initial_state_delta.mean(axis=0), 0.0, atol=1e-16)
    np.testing.assert_allclose(first.command_delta_nm.mean(axis=0), 0.0, atol=1e-16)
    assert first.guard_offset_delta.mean() == pytest.approx(0.0, abs=1e-16)
    assert first.initial_state_delta.flags.writeable is False
    assert first.command_delta_nm.flags.writeable is False
    assert first.guard_offset_delta.flags.writeable is False


def test_common_standardized_draws_scale_each_declared_channel() -> None:
    unit = generate_common_random_perturbations(
        RobustnessNoiseConfig(
            seed=7,
            replicate_count=6,
            initial_state_sd=(1.0, 1.0, 1.0, 1.0),
            command_sd_nm=(1.0, 1.0),
            guard_offset_sd=1.0,
        ),
        control_sample_count=3,
    )
    scaled = generate_common_random_perturbations(
        RobustnessNoiseConfig(
            seed=7,
            replicate_count=6,
            initial_state_sd=(2.0, 3.0, 4.0, 5.0),
            command_sd_nm=(6.0, 7.0),
            guard_offset_sd=8.0,
        ),
        control_sample_count=3,
    )

    np.testing.assert_allclose(
        scaled.initial_state_delta,
        unit.initial_state_delta * np.array([2.0, 3.0, 4.0, 5.0]),
    )
    np.testing.assert_allclose(
        scaled.command_delta_nm,
        unit.command_delta_nm * np.array([6.0, 7.0]),
    )
    np.testing.assert_allclose(scaled.guard_offset_delta, unit.guard_offset_delta * 8.0)


def test_zero_noise_retains_explicit_zero_arrays() -> None:
    result = generate_common_random_perturbations(
        RobustnessNoiseConfig(seed=1, replicate_count=4),
        control_sample_count=2,
    )

    np.testing.assert_array_equal(result.initial_state_delta, np.zeros((4, 4)))
    np.testing.assert_array_equal(result.command_delta_nm, np.zeros((4, 2, 2)))
    np.testing.assert_array_equal(result.guard_offset_delta, np.zeros(4))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1, "replicate_count": 4},
        {"seed": 1, "replicate_count": 3},
        {"seed": 1, "replicate_count": 0},
        {"seed": 1, "replicate_count": 4, "command_sd_nm": (-1.0, 0.0)},
        {"seed": 1, "replicate_count": 4, "guard_offset_sd": np.nan},
        {"seed": 1, "replicate_count": 4, "initial_state_sd": (1.0, 2.0)},
    ],
)
def test_invalid_noise_contracts_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RobustnessNoiseConfig(**kwargs)


@pytest.mark.parametrize("control_sample_count", [0, -1, True])
def test_invalid_noise_sample_count_fails_closed(control_sample_count: int) -> None:
    with pytest.raises(ValueError):
        generate_common_random_perturbations(
            RobustnessNoiseConfig(seed=1, replicate_count=4),
            control_sample_count=control_sample_count,
        )
