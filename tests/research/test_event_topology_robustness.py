"""Scientific contracts for registered event-topology robustness (#9125)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.event_topology_robustness import (
    BASE_DT_S,
    EventTopologyClass,
    PerturbationScenario,
    applied_control_history,
    enumerate_guard_crossings,
    registered_nominal_inputs,
    run_event_topology_suite,
    simulate_perturbed_downswing,
)

pytestmark = pytest.mark.scientific


def test_registered_nominal_uses_protected_unique_forward_guard() -> None:
    initial, controls = registered_nominal_inputs()
    event = simulate_perturbed_downswing(initial, controls, BASE_DT_S)

    assert event.topology_class is EventTopologyClass.UNIQUE_FORWARD
    assert event.crossing_count == 1
    assert event.first_crossing_time_s == pytest.approx(0.3493, abs=5e-4)
    assert event.crossings[0].is_transverse is True
    assert event.crossings[0].is_forward is True


def test_delay_uses_explicit_zero_prehistory_before_nominal_command() -> None:
    _, controls = registered_nominal_inputs(dt_s=0.002, horizon_s=0.1)
    delayed = applied_control_history(
        controls,
        dt_s=0.002,
        scenario=PerturbationScenario(delay_s=0.006),
    )

    np.testing.assert_array_equal(delayed[:3], np.zeros((3, 2)))
    np.testing.assert_allclose(delayed[3:], controls[:-3], atol=0.0, rtol=0.0)


def test_delay_prehistory_is_explicit_and_channel_masked() -> None:
    _, controls = registered_nominal_inputs(dt_s=0.002, horizon_s=0.1)
    delayed = applied_control_history(
        controls,
        dt_s=0.002,
        scenario=PerturbationScenario(
            delay_s=0.004,
            pre_delay_command_nm=(3.0, -2.0),
            channel_mask=(1.0, 0.0),
        ),
    )

    np.testing.assert_array_equal(delayed[:2], np.array([[3.0, 0.0], [3.0, 0.0]]))
    assert np.all(delayed[:, 1] == 0.0)


def test_crossing_enumerator_types_reversed_multiple_and_grazing() -> None:
    dt_s = 0.01
    forward = np.array([[-0.1, 0.0, 1.0, 0.0], [0.1, 0.0, 1.0, 0.0]], dtype=float)
    reversed_states = forward[::-1].copy()
    reversed_states[:, 2] = -1.0
    multiple = np.array(
        [
            [-0.2, 0.0, 1.0, 0.0],
            [0.1, 0.0, 1.0, 0.0],
            [0.2, 0.0, -1.0, 0.0],
            [-0.1, 0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    grazing = np.array([[-0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]], dtype=float)

    assert (
        enumerate_guard_crossings(reversed_states, dt_s).topology_class
        is EventTopologyClass.REVERSED_DIRECTION
    )
    assert (
        enumerate_guard_crossings(multiple, dt_s).topology_class
        is EventTopologyClass.MULTIPLE_CROSSINGS
    )
    assert (
        enumerate_guard_crossings(grazing, dt_s).topology_class
        is EventTopologyClass.GRAZING
    )


def test_zero_and_wrist_only_channel_topologies_are_both_enforced() -> None:
    summary = run_event_topology_suite()
    topologies = dict(summary.channel_topologies)

    assert topologies == {
        "both": "UNIQUE_FORWARD",
        "shoulder_only": "UNIQUE_FORWARD",
        "wrist_only": "UNIQUE_FORWARD",
        "zero": "ZERO_CROSSINGS",
    }
    assert summary.channel_coverage_passed is True


def test_scenario_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        PerturbationScenario(delay_s=-0.001)
    with pytest.raises(ValueError, match="zero or one"):
        PerturbationScenario(channel_mask=(0.5, 1.0))
