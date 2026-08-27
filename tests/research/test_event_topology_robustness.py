"""Unit tests for event topology and delay/noise robustness (#9125)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.event_topology_robustness import (
    EventTopologyClass,
    enumerate_guard_crossings,
    run_event_topology_suite,
    simulate_perturbed_downswing,
)
from scripts.research.proximal_distal_energy.trajectory_control_authority import (
    generate_nominal_downswing_trajectory,
)

pytestmark = pytest.mark.scientific


def test_nominal_downswing_has_unique_forward_crossing() -> None:
    states, controls = generate_nominal_downswing_trajectory(dt=0.002, steps=140)
    ev = enumerate_guard_crossings(states, dt=0.002)

    assert ev.topology_class == EventTopologyClass.UNIQUE_FORWARD
    assert ev.crossing_count == 1
    assert ev.first_crossing_time_s is not None
    assert ev.crossings[0].is_transverse is True
    assert ev.crossings[0].is_forward is True


def test_multiple_crossings_and_reversed_direction_detection() -> None:
    # Construct synthetic states with an oscillation crossing 3.5 twice
    dt = 0.01
    times = np.linspace(0, 1.0, 100)
    # theta1 = 3.5 + 0.5 * sin(2*pi*t) -> crosses 3.5 at t=0, t=0.5, t=1.0
    theta1 = 3.5 + 0.5 * np.sin(2 * np.pi * times)
    omega1 = 0.5 * 2 * np.pi * np.cos(2 * np.pi * times)
    states = np.column_stack(
        [theta1, np.zeros_like(theta1), omega1, np.zeros_like(theta1)]
    )

    ev = enumerate_guard_crossings(states, dt=dt)
    assert ev.crossing_count >= 2
    assert ev.topology_class == EventTopologyClass.MULTIPLE_CROSSINGS


def test_zero_perturbation_reproduces_nominal() -> None:
    states, controls = generate_nominal_downswing_trajectory(dt=0.002, steps=140)
    z0 = states[0]
    dt = 0.002

    ev_zero = simulate_perturbed_downswing(z0, controls, dt=dt)
    assert ev_zero.topology_class == EventTopologyClass.UNIQUE_FORWARD
    assert ev_zero.crossing_count == 1
