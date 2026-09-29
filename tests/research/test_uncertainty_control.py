"""Contracts for coupled uncertainty, identifiability, and control."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.uncertainty_control import (
    ActuatorLimits,
    ControlProgram,
    delayed_control_law,
    latin_hypercube,
    nondominated_indices,
    partial_rank_correlations,
    planar_two_hand_wrench_map,
)

pytestmark = pytest.mark.scientific


def test_latin_hypercube_is_deterministic_and_stratified() -> None:
    first = latin_hypercube(12, 5, seed=8426)
    second = latin_hypercube(12, 5, seed=8426)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (12, 5)
    assert np.all((first > 0.0) & (first < 1.0))
    for column in first.T:
        bins = np.floor(column * 12).astype(int)
        np.testing.assert_array_equal(np.sort(bins), np.arange(12))


def test_delayed_control_respects_delay_rate_and_velocity_limits() -> None:
    program = ControlProgram(
        name="bounded_test",
        wrist_onset_s=0.08,
        early_wrist_nm=-2.0,
        late_wrist_nm=5.0,
        shoulder_scale=1.0,
        elbow_scale=1.0,
        impedance_nms_rad=0.2,
    )
    limits = ActuatorLimits(
        delay_s=0.03,
        time_constant_s=0.025,
        maximum_torque_rate_nm_s=80.0,
        concentric_velocity_rad_s=18.0,
        eccentric_torque_ratio=1.25,
    )
    law = delayed_control_law(program, limits, duration_s=0.24, step_s=0.002)
    q = np.zeros(10)
    qdot = np.zeros(10)

    early = law(0.02, q, qdot)
    assert early.right_wrist_nm == pytest.approx(0.0)
    assert early.left_wrist_nm == pytest.approx(0.0)

    samples = np.array(
        [law(float(t), q, qdot).right_wrist_nm for t in np.arange(0.0, 0.242, 0.002)]
    )
    assert np.max(np.abs(np.diff(samples) / 0.002)) <= 80.0 + 1e-10

    fast_qdot = np.zeros(10)
    fast_qdot[8] = 40.0
    bounded = law(0.22, q, fast_qdot)
    assert abs(bounded.right_wrist_nm) < abs(law(0.22, q, qdot).right_wrist_nm)


def test_two_hand_net_wrench_cannot_identify_all_individual_force_components() -> None:
    mapping = planar_two_hand_wrench_map(0.065, -0.065)

    assert mapping.shape == (3, 4)
    assert np.linalg.matrix_rank(mapping) == 3
    assert mapping.shape[1] - np.linalg.matrix_rank(mapping) == 1


def test_partial_rank_correlation_recovers_conditional_direction() -> None:
    design = latin_hypercube(80, 3, seed=7)
    output = 3.0 * design[:, 0] - 2.0 * design[:, 1] + 0.01 * design[:, 2]
    coefficients = partial_rank_correlations(design, output)

    assert coefficients[0] > 0.95
    assert coefficients[1] < -0.90
    assert abs(coefficients[2]) < 0.2


def test_nondominated_indices_preserve_tradeoffs() -> None:
    # All objectives are minimized after explicit sign normalization.
    objectives = np.array(
        [
            [-10.0, 4.0, 2.0],
            [-9.0, 3.0, 1.0],
            [-8.0, 5.0, 3.0],
            [-7.0, 6.0, 4.0],
        ]
    )

    assert nondominated_indices(objectives) == (0, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"delay_s": -0.01},
        {"time_constant_s": 0.0},
        {"maximum_torque_rate_nm_s": 0.0},
        {"concentric_velocity_rad_s": 0.0},
        {"eccentric_torque_ratio": 0.5},
    ],
)
def test_actuator_limits_fail_closed(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ActuatorLimits(**kwargs)
