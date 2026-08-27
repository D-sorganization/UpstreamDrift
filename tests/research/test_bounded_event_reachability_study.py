"""Registered continuation and adverse-control study contracts for #9124."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.bounded_event_multiple_shooting import (
    MultipleShootingConfig,
    MultipleShootingStatus,
)
from scripts.research.proximal_distal_energy.bounded_event_reachability import (
    FeasibilityStatus,
)
from scripts.research.proximal_distal_energy.run_bounded_event_reachability import (
    BASE_DT_S,
    REGISTERED_SEGMENT_COUNT,
    build_problem,
    registered_channels,
    registered_targets,
    run_trial,
    study_matrix,
)

pytestmark = pytest.mark.unit


def test_registered_continuation_matrix_is_symmetric_and_channel_matched() -> None:
    targets = registered_targets()
    channels = registered_channels()
    matrix = study_matrix()

    assert len(targets) == 7
    assert len(channels) == 4
    assert len(matrix) == 28
    assert len({target.name for target in targets}) == len(targets)
    assert {channel.name for channel in channels} == {
        "both",
        "shoulder_only",
        "wrist_only",
        "zero",
    }
    assert sum(target.amplitude_rad == 0.0 for target in targets) == 1
    nonzero = {(target.direction, target.amplitude_rad) for target in targets}
    for amplitude in (5e-4, 1e-3, 2e-3):
        assert (-1, amplitude) in nonzero
        assert (1, amplitude) in nonzero
    for target in targets:
        matched = [candidate for candidate, _ in matrix if candidate == target]
        assert len(matched) == len(channels)


def test_registered_channel_bounds_are_explicit_killswitches() -> None:
    channels = {channel.name: channel for channel in registered_channels()}

    assert channels["zero"].bounds.is_zero_authority
    assert channels["shoulder_only"].bounds.lower_nm[1] == 0.0
    assert channels["shoulder_only"].bounds.upper_nm[1] == 0.0
    assert channels["wrist_only"].bounds.lower_nm[0] == 0.0
    assert channels["wrist_only"].bounds.upper_nm[0] == 0.0
    assert not channels["both"].bounds.is_zero_authority


@pytest.mark.parametrize(
    ("channel_name", "expected"),
    (
        ("both", FeasibilityStatus.FEASIBLE),
        ("shoulder_only", FeasibilityStatus.FEASIBLE),
        ("wrist_only", FeasibilityStatus.FEASIBLE),
        ("zero", FeasibilityStatus.INFEASIBLE),
    ),
)
def test_matched_anchor_target_respects_channel_killswitches(
    channel_name: str,
    expected: FeasibilityStatus,
) -> None:
    target = next(
        target for target in registered_targets() if target.name == "plus_0p001"
    )
    channel = next(
        channel for channel in registered_channels() if channel.name == channel_name
    )
    problem = build_problem(
        dt_s=BASE_DT_S,
        target=target,
        channel=channel,
    )
    config = MultipleShootingConfig(
        segment_count=REGISTERED_SEGMENT_COUNT,
        max_iterations=60,
        constraint_tolerance=2e-6,
        objective_tolerance=1e-10,
        seed=0,
    )

    record, result = run_trial(
        problem=problem, target=target, channel=channel, config=config
    )

    assert result.replay is not None
    assert result.replay.feasibility_status is expected
    assert record["target_name"] == target.name
    assert record["channel"] == channel.name
    assert record["dt_s"] == BASE_DT_S
    assert record["segment_count"] == REGISTERED_SEGMENT_COUNT
    if expected is FeasibilityStatus.FEASIBLE:
        assert result.status is MultipleShootingStatus.CONVERGED
        assert record["replay_tangent_residual"] <= problem.tangent_tolerance
    else:
        assert result.status is MultipleShootingStatus.INFEASIBLE
        assert record["solver_success"] is False


def test_adverse_initial_state_is_part_of_problem_identity() -> None:
    target = next(
        target for target in registered_targets() if target.name == "plus_0p001"
    )
    channel = next(
        channel for channel in registered_channels() if channel.name == "both"
    )
    base = build_problem(dt_s=BASE_DT_S, target=target, channel=channel)
    adverse = np.asarray(base.initial_state, dtype=float)
    adverse += np.array([0.002, -0.002, 0.0, 0.0])
    shifted = build_problem(
        dt_s=BASE_DT_S,
        target=target,
        channel=channel,
        initial_state=tuple(adverse),
    )

    assert shifted.initial_state != base.initial_state
    np.testing.assert_array_equal(shifted.target_event_state, base.target_event_state)
    restored = replace(shifted, initial_state=base.initial_state)
    assert restored.initial_state == base.initial_state
    assert restored.target_event_state == base.target_event_state
    np.testing.assert_array_equal(restored.nominal_controls, base.nominal_controls)
