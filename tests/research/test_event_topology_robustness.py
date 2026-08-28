"""Global event-topology contracts for issue #9125."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.event_topology_robustness import (
    CommandDelayConfig,
    CrossingDirection,
    DelayContinuationConfig,
    DelayInterpolation,
    EventTopologyStatus,
    apply_command_delay,
    enumerate_crossing_brackets,
    evaluate_delay_continuation,
    replay_global_event_topology,
)
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from scripts.research.proximal_distal_energy.trajectory_control_authority import (
    GuardCrossingConfig,
)
from src.shared.python.simulation_backends import GolfModelParams

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("values", "directions"),
    [
        ([-2.0, -1.0, -0.5], ()),
        ([-1.0, 1.0, 2.0], (CrossingDirection.POSITIVE,)),
        ([1.0, -1.0, -2.0], (CrossingDirection.NEGATIVE,)),
        (
            [-1.0, 1.0, -1.0, 1.0],
            (
                CrossingDirection.POSITIVE,
                CrossingDirection.NEGATIVE,
                CrossingDirection.POSITIVE,
            ),
        ),
    ],
)
def test_manufactured_brackets_retain_direction_and_multiplicity(
    values: list[float], directions: tuple[CrossingDirection, ...]
) -> None:
    times = np.arange(len(values), dtype=float)

    result = enumerate_crossing_brackets(times, values, zero_tolerance=1e-12)

    assert tuple(item.direction for item in result.brackets) == directions
    assert result.initial_on_guard is False


def test_exact_sample_zero_is_counted_once() -> None:
    result = enumerate_crossing_brackets(
        [0.0, 1.0, 2.0, 3.0],
        [-1.0, 0.0, 1.0, 2.0],
        zero_tolerance=1e-12,
    )

    assert len(result.brackets) == 1
    assert result.brackets[0].sample_index == 0
    assert result.brackets[0].direction is CrossingDirection.POSITIVE


def test_initial_on_guard_fails_closed() -> None:
    result = enumerate_crossing_brackets(
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        zero_tolerance=1e-12,
    )

    assert result.initial_on_guard is True
    assert result.brackets == ()


@pytest.mark.parametrize(
    ("time_s", "guard_values"),
    [
        ([0.0], [1.0]),
        ([0.0, 0.0], [-1.0, 1.0]),
        ([0.0, 1.0], [np.nan, 1.0]),
    ],
)
def test_invalid_crossing_grids_fail_closed(
    time_s: list[float], guard_values: list[float]
) -> None:
    with pytest.raises(ValueError):
        enumerate_crossing_brackets(time_s, guard_values, zero_tolerance=1e-12)


def test_registered_nominal_rollout_has_one_positive_transverse_crossing() -> None:
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

    topology = replay_global_event_topology(
        params=GolfModelParams.default(),
        initial_state=(-2.2, -1.57, 0.0, 0.0),
        controls=controls,
        dt_s=dt_s,
        guard=guard,
    )

    assert topology.status is EventTopologyStatus.UNIQUE_TRANSVERSE
    assert topology.crossing_count == 1
    assert topology.events[0].direction is CrossingDirection.POSITIVE
    assert topology.events[0].time_s == pytest.approx(0.349256, abs=1e-6)
    assert topology.events[0].transversality_per_s > 0.0


def test_negated_guard_retains_the_same_event_as_negative_direction() -> None:
    dt_s = 2e-3
    controls = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10).controls(
        round(0.40 / dt_s), dt_s
    )
    guard = GuardCrossingConfig(
        guard_gradient=(-1.0, -1.0, 0.0, 0.0),
        guard_tolerance=1e-10,
        time_tolerance_s=1e-12,
        transversality_threshold=1e-8,
    )

    topology = replay_global_event_topology(
        params=GolfModelParams.default(),
        initial_state=(-2.2, -1.57, 0.0, 0.0),
        controls=controls,
        dt_s=dt_s,
        guard=guard,
    )

    assert topology.status is EventTopologyStatus.UNIQUE_TRANSVERSE
    assert topology.crossing_count == 1
    assert topology.events[0].direction is CrossingDirection.NEGATIVE
    assert topology.events[0].time_s == pytest.approx(0.349256, abs=1e-6)
    assert topology.events[0].transversality_per_s < 0.0


def test_short_rollout_retains_absent_topology() -> None:
    dt_s = 2e-3
    controls = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10).controls(50, dt_s)
    guard = GuardCrossingConfig(
        guard_gradient=(1.0, 1.0, 0.0, 0.0),
        guard_tolerance=1e-10,
        time_tolerance_s=1e-12,
        transversality_threshold=1e-8,
    )

    topology = replay_global_event_topology(
        params=GolfModelParams.default(),
        initial_state=(-2.2, -1.57, 0.0, 0.0),
        controls=controls,
        dt_s=dt_s,
        guard=guard,
    )

    assert topology.status is EventTopologyStatus.ABSENT
    assert topology.crossing_count == 0
    assert topology.events == ()


def test_zero_delay_is_an_immutable_exact_replay() -> None:
    controls = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    original = controls.copy()

    delayed = apply_command_delay(
        controls,
        dt_s=0.1,
        config=CommandDelayConfig(delay_s=0.0),
    )

    np.testing.assert_array_equal(delayed, controls)
    np.testing.assert_array_equal(controls, original)
    assert delayed.flags.writeable is False


@pytest.mark.parametrize("delay_s", [0.05, 0.10])
def test_zero_order_hold_delay_is_causal(delay_s: float) -> None:
    controls = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    delayed = apply_command_delay(
        controls,
        dt_s=0.1,
        config=CommandDelayConfig(
            delay_s=delay_s,
            interpolation=DelayInterpolation.ZERO_ORDER_HOLD,
        ),
    )

    np.testing.assert_array_equal(
        delayed,
        np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]]),
    )


def test_linear_nodal_delay_uses_declared_prehistory_and_fraction() -> None:
    controls = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])

    delayed = apply_command_delay(
        controls,
        dt_s=0.1,
        config=CommandDelayConfig(
            delay_s=0.05,
            interpolation=DelayInterpolation.LINEAR_NODAL,
            prehistory_control=(-1.0, -10.0),
        ),
    )

    np.testing.assert_allclose(
        delayed,
        np.array([[-1.0, -10.0], [2.0, 20.0], [4.0, 40.0]]),
        rtol=0.0,
        atol=1e-15,
    )


def test_extended_output_uses_declared_posthistory_without_truncating_delay() -> None:
    controls = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    delayed = apply_command_delay(
        controls,
        dt_s=0.1,
        output_sample_count=5,
        config=CommandDelayConfig(
            delay_s=0.1,
            posthistory_control=(-2.0, -20.0),
        ),
    )

    np.testing.assert_array_equal(
        delayed,
        np.array(
            [
                [0.0, 0.0],
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [-2.0, -20.0],
            ]
        ),
    )


def test_linear_nodal_tail_interpolates_to_declared_posthistory() -> None:
    controls = np.array([[1.0, 10.0], [3.0, 30.0]])

    delayed = apply_command_delay(
        controls,
        dt_s=0.1,
        output_sample_count=3,
        config=CommandDelayConfig(
            delay_s=0.05,
            interpolation=DelayInterpolation.LINEAR_NODAL,
            posthistory_control=(-1.0, -10.0),
        ),
    )

    np.testing.assert_allclose(
        delayed,
        np.array([[0.0, 0.0], [2.0, 20.0], [1.0, 10.0]]),
        rtol=0.0,
        atol=1e-15,
    )


@pytest.mark.parametrize("delay_s", [-0.1, np.inf, np.nan])
def test_invalid_delay_fails_closed(delay_s: float) -> None:
    with pytest.raises(ValueError):
        CommandDelayConfig(delay_s=delay_s)


def test_delay_operator_rejects_invalid_sampling_contract() -> None:
    with pytest.raises(ValueError):
        apply_command_delay(
            [[1.0, 2.0]],
            dt_s=0.0,
            config=CommandDelayConfig(delay_s=0.1),
        )
    with pytest.raises(ValueError):
        apply_command_delay(
            [[1.0, 2.0]],
            dt_s=0.1,
            output_sample_count=0,
            config=CommandDelayConfig(delay_s=0.1),
        )


@pytest.mark.parametrize(
    ("delays_s", "horizon_s"),
    [
        ((0.01, 0.02), 0.5),
        ((0.0, 0.02, 0.01), 0.5),
        ((0.0, 0.01), 0.0),
        ((0.0, np.nan), 0.5),
    ],
)
def test_invalid_delay_continuation_contract_fails_closed(
    delays_s: tuple[float, ...], horizon_s: float
) -> None:
    with pytest.raises(ValueError):
        DelayContinuationConfig(delays_s=delays_s, common_horizon_s=horizon_s)


def test_delay_continuation_requires_horizon_to_retain_complete_program() -> None:
    dt_s = 2e-3
    controls = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10).controls(
        round(0.40 / dt_s), dt_s
    )

    with pytest.raises(ValueError, match="complete delayed command program"):
        evaluate_delay_continuation(
            params=GolfModelParams.default(),
            initial_state=(-2.2, -1.57, 0.0, 0.0),
            controls=controls,
            dt_s=dt_s,
            guard=GuardCrossingConfig(
                guard_gradient=(1.0, 1.0, 0.0, 0.0),
            ),
            config=DelayContinuationConfig(
                delays_s=(0.0, 0.10),
                common_horizon_s=0.45,
            ),
        )


def test_registered_delay_continuation_uses_common_global_horizon() -> None:
    dt_s = 2e-3
    controls = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10).controls(
        round(0.40 / dt_s), dt_s
    )
    config = DelayContinuationConfig(
        delays_s=(0.0, 0.05, 0.10),
        common_horizon_s=0.50,
        interpolation=DelayInterpolation.LINEAR_NODAL,
    )

    result = evaluate_delay_continuation(
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
        config=config,
    )

    assert result.output_sample_count == 250
    assert result.zero_delay_control_residual == pytest.approx(0.0)
    assert tuple(item.delay_s for item in result.outcomes) == config.delays_s
    assert all(
        item.topology.status is EventTopologyStatus.UNIQUE_TRANSVERSE
        for item in result.outcomes
    )
    event_times = [item.topology.events[0].time_s for item in result.outcomes]
    assert event_times == sorted(event_times)
    assert event_times[0] == pytest.approx(0.349256, abs=1e-6)
    assert event_times[-1] == pytest.approx(0.435413, abs=1e-6)
