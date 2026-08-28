"""Adequacy-gated topology summaries for issue #9125."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.event_robustness_noise import (
    RobustnessNoiseConfig,
    generate_common_random_perturbations,
)
from scripts.research.proximal_distal_energy.event_robustness_study import (
    DelayNoiseTopologyResult,
    evaluate_delay_noise_topology,
)
from scripts.research.proximal_distal_energy.event_robustness_summary import (
    TopologyAdequacyConfig,
    summarize_topology_by_delay,
)
from scripts.research.proximal_distal_energy.event_topology_robustness import (
    DelayContinuationConfig,
)
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from scripts.research.proximal_distal_energy.trajectory_control_authority import (
    GuardCrossingConfig,
)
from src.shared.python.simulation_backends import GolfModelParams

pytestmark = pytest.mark.unit


def _zero_noise_result(replicate_count: int) -> DelayNoiseTopologyResult:
    dt_s = 2e-3
    controls = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10).controls(
        round(0.40 / dt_s), dt_s
    )
    perturbations = generate_common_random_perturbations(
        RobustnessNoiseConfig(seed=9125, replicate_count=replicate_count),
        control_sample_count=200,
    )
    return evaluate_delay_noise_topology(
        params=GolfModelParams.default(),
        initial_state=(-2.2, -1.57, 0.0, 0.0),
        controls=controls,
        dt_s=dt_s,
        guard=GuardCrossingConfig(
            guard_gradient=(1.0, 1.0, 0.0, 0.0),
            guard_tolerance=1e-10,
            time_tolerance_s=1e-12,
            transversality_threshold=1e-8,
        ),
        delay_config=DelayContinuationConfig(
            delays_s=(0.0,),
            common_horizon_s=0.40,
        ),
        perturbations=perturbations,
    )


def test_small_antithetic_design_retains_counts_but_suppresses_fraction() -> None:
    summaries = summarize_topology_by_delay(
        _zero_noise_result(4),
        config=TopologyAdequacyConfig(required_independent_pairs=96),
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.independent_pair_count == 2
    assert summary.preserved_pair_count == 2
    assert summary.adequacy_passed is False
    assert summary.preservation_fraction is None
    assert summary.preservation_interval is None
    assert dict(summary.topology_counts) == {"unique_transverse": 4}


def test_registered_pair_count_exposes_bounded_preservation_interval() -> None:
    summaries = summarize_topology_by_delay(
        _zero_noise_result(192),
        config=TopologyAdequacyConfig(
            required_independent_pairs=96,
            maximum_interval_half_width=0.10,
            confidence=0.95,
        ),
    )

    summary = summaries[0]
    assert summary.independent_pair_count == 96
    assert summary.preserved_pair_count == 96
    assert summary.adequacy_passed is True
    assert summary.preservation_fraction == pytest.approx(1.0)
    assert summary.preservation_interval is not None
    lower, upper = summary.preservation_interval
    assert 0.95 < lower < 1.0
    assert upper == pytest.approx(1.0)
    assert (upper - lower) / 2.0 <= 0.10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"required_independent_pairs": 0},
        {"required_independent_pairs": 2, "maximum_interval_half_width": 0.0},
        {"required_independent_pairs": 2, "confidence": 1.0},
        {"required_independent_pairs": 2, "confidence": np.nan},
    ],
)
def test_invalid_adequacy_contracts_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TopologyAdequacyConfig(**kwargs)
