"""Exact dynamics replay for the #9125 delay/noise study."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.event_robustness_noise import (
    RobustnessNoiseConfig,
    generate_common_random_perturbations,
)
from scripts.research.proximal_distal_energy.event_robustness_study import (
    evaluate_delay_noise_topology,
)
from scripts.research.proximal_distal_energy.event_topology_robustness import (
    DelayContinuationConfig,
    DelayInterpolation,
    EventTopologyStatus,
)
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from scripts.research.proximal_distal_energy.trajectory_control_authority import (
    GuardCrossingConfig,
)
from src.shared.python.simulation_backends import GolfModelParams

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def registered_case() -> tuple[np.ndarray, float, GuardCrossingConfig]:
    dt_s = 2e-3
    controls = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10).controls(
        round(0.40 / dt_s), dt_s
    )
    guard = GuardCrossingConfig(
        guard_gradient=(1.0, 1.0, 0.0, 0.0),
        guard_tolerance=1e-10,
        time_tolerance_s=1e-12,
        transversality_threshold=1e-8,
    )
    return controls, dt_s, guard


def test_zero_noise_replays_each_delay_without_fabricating_variability(
    registered_case: tuple[np.ndarray, float, GuardCrossingConfig],
) -> None:
    controls, dt_s, guard = registered_case
    delay_config = DelayContinuationConfig(
        delays_s=(0.0, 0.05),
        common_horizon_s=0.45,
        interpolation=DelayInterpolation.LINEAR_NODAL,
    )
    perturbations = generate_common_random_perturbations(
        RobustnessNoiseConfig(seed=9125, replicate_count=4),
        control_sample_count=225,
    )

    result = evaluate_delay_noise_topology(
        params=GolfModelParams.default(),
        initial_state=(-2.2, -1.57, 0.0, 0.0),
        controls=controls,
        dt_s=dt_s,
        guard=guard,
        delay_config=delay_config,
        perturbations=perturbations,
    )

    assert result.replicate_count == 4
    assert len(result.nominal.outcomes) == 2
    assert len(result.outcomes) == 8
    for delay_index, nominal in enumerate(result.nominal.outcomes):
        retained = result.outcomes[delay_index * 4 : (delay_index + 1) * 4]
        assert all(item.delay_s == nominal.delay_s for item in retained)
        assert all(
            item.topology.status is EventTopologyStatus.UNIQUE_TRANSVERSE
            for item in retained
        )
        assert all(
            item.topology.events[0].time_s
            == pytest.approx(nominal.topology.events[0].time_s, abs=1e-12)
            for item in retained
        )


def test_antithetic_guard_uncertainty_is_retained_per_replicate(
    registered_case: tuple[np.ndarray, float, GuardCrossingConfig],
) -> None:
    controls, dt_s, guard = registered_case
    perturbations = generate_common_random_perturbations(
        RobustnessNoiseConfig(
            seed=22,
            replicate_count=4,
            guard_offset_sd=0.01,
        ),
        control_sample_count=200,
    )

    result = evaluate_delay_noise_topology(
        params=GolfModelParams.default(),
        initial_state=(-2.2, -1.57, 0.0, 0.0),
        controls=controls,
        dt_s=dt_s,
        guard=guard,
        delay_config=DelayContinuationConfig(
            delays_s=(0.0,),
            common_horizon_s=0.40,
        ),
        perturbations=perturbations,
    )

    assert tuple(item.replicate_index for item in result.outcomes) == (0, 1, 2, 3)
    assert all(
        item.topology.status is EventTopologyStatus.UNIQUE_TRANSVERSE
        for item in result.outcomes
    )
    times = np.array([item.topology.events[0].time_s for item in result.outcomes])
    assert np.ptp(times) > 0.0


def test_mismatched_command_noise_horizon_fails_closed(
    registered_case: tuple[np.ndarray, float, GuardCrossingConfig],
) -> None:
    controls, dt_s, guard = registered_case
    perturbations = generate_common_random_perturbations(
        RobustnessNoiseConfig(seed=2, replicate_count=4),
        control_sample_count=199,
    )

    with pytest.raises(ValueError, match="common horizon"):
        evaluate_delay_noise_topology(
            params=GolfModelParams.default(),
            initial_state=(-2.2, -1.57, 0.0, 0.0),
            controls=controls,
            dt_s=dt_s,
            guard=guard,
            delay_config=DelayContinuationConfig(
                delays_s=(0.0,),
                common_horizon_s=0.40,
            ),
            perturbations=perturbations,
        )
