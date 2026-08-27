"""Event topology and delay/noise robustness mapping (#9125).

Evaluates the global event topology (unique, multiple, reversed, grazing, zero crossings)
under actuator delay, state/command perturbations, and event-surface variations for the
canonical analytical double pendulum.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.trajectory_control_authority import (
    continuous_dynamics,
    discrete_rk4_step,
    generate_nominal_downswing_trajectory,
)
from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumParameters,
)

FloatArray = npt.NDArray[np.float64]

INFERENCE_BOUNDARY = (
    "This event-topology and robustness mapping establishes only mathematical scenario "
    "properties for the declared analytical double pendulum under synthetic perturbations. "
    "It cannot establish human motor variability, neuromuscular delay, fatigue effects, "
    "controller rankings, or coaching technique advice."
)


class EventTopologyClass(str, Enum):
    """Classification of guard-surface crossings over a finite horizon."""

    ZERO_CROSSINGS = "ZERO_CROSSINGS"
    UNIQUE_FORWARD = "UNIQUE_FORWARD"
    MULTIPLE_CROSSINGS = "MULTIPLE_CROSSINGS"
    REVERSED_DIRECTION = "REVERSED_DIRECTION"
    GRAZING = "GRAZING"


@dataclass(frozen=True, slots=True)
class CrossingEvent:
    """A detected guard surface crossing."""

    time_s: float
    state_at_crossing: tuple[float, float, float, float]
    normal_velocity: float
    is_transverse: bool
    is_forward: bool


@dataclass(frozen=True, slots=True)
class TopologyEvaluation:
    """Outcome of global event enumeration for a single trajectory."""

    topology_class: EventTopologyClass
    crossing_count: int
    first_crossing_time_s: float | None
    crossings: tuple[CrossingEvent, ...]
    terminal_state: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class EventTopologyRobustnessSummary:
    """Aggregated evidence for issue #9125."""

    zero_perturbation_reproduces_nominal: bool
    nominal_first_crossing_time_s: float
    max_tolerated_delay_s: float
    noise_robustness_retained_unique_fraction: float
    channel_coverage_passed: bool
    step_refinement_stable: bool
    total_trials: int
    inference_boundary: str = INFERENCE_BOUNDARY


DEFAULT_DELIVERY_ANGLE_RAD = 3.5


def default_guard(state: FloatArray, event_offset_rad: float = 0.0) -> float:
    """Standard delivery guard: h(z) = theta1 - (DEFAULT_DELIVERY_ANGLE_RAD + offset)."""
    return float(state[0] - (DEFAULT_DELIVERY_ANGLE_RAD + event_offset_rad))


def enumerate_guard_crossings(
    states: FloatArray,
    dt: float,
    event_offset_rad: float = 0.0,
    transverse_tolerance: float = 1e-3,
    params: DoublePendulumParameters | None = None,
) -> TopologyEvaluation:
    """Detect and enumerate all zero-crossings of the delivery guard."""
    K = len(states)
    h_vals = [default_guard(states[k], event_offset_rad) for k in range(K)]
    crossings: list[CrossingEvent] = []

    for k in range(K - 1):
        h1 = h_vals[k]
        h2 = h_vals[k + 1]

        # Check for sign change
        if (h1 <= 0.0 and h2 > 0.0) or (h1 >= 0.0 and h2 < 0.0) or h1 == 0.0:
            # Linear interpolation for crossing fraction
            denom = abs(h2 - h1)
            frac = abs(h1) / denom if denom > 1e-12 else 0.0
            t_cross = (k + frac) * dt
            z_cross = states[k] + frac * (states[k + 1] - states[k])

            # Normal velocity along guard normal n = [1, 0, 0, 0] is omega1 (z[2])
            normal_vel = float(z_cross[2])
            is_transverse = abs(normal_vel) >= transverse_tolerance
            is_forward = normal_vel > 0.0

            crossings.append(
                CrossingEvent(
                    time_s=float(t_cross),
                    state_at_crossing=(
                        float(z_cross[0]),
                        float(z_cross[1]),
                        float(z_cross[2]),
                        float(z_cross[3]),
                    ),
                    normal_velocity=normal_vel,
                    is_transverse=is_transverse,
                    is_forward=is_forward,
                )
            )

    count = len(crossings)
    if count == 0:
        top_class = EventTopologyClass.ZERO_CROSSINGS
    elif any(not c.is_transverse for c in crossings):
        top_class = EventTopologyClass.GRAZING
    elif count == 1:
        top_class = (
            EventTopologyClass.UNIQUE_FORWARD
            if crossings[0].is_forward
            else EventTopologyClass.REVERSED_DIRECTION
        )
    else:
        top_class = EventTopologyClass.MULTIPLE_CROSSINGS

    first_t = crossings[0].time_s if count > 0 else None
    return TopologyEvaluation(
        topology_class=top_class,
        crossing_count=count,
        first_crossing_time_s=first_t,
        crossings=tuple(crossings),
        terminal_state=(
            float(states[-1][0]),
            float(states[-1][1]),
            float(states[-1][2]),
            float(states[-1][3]),
        ),
    )


def simulate_perturbed_downswing(
    initial_state: FloatArray,
    nominal_controls: FloatArray,
    dt: float,
    delay_s: float = 0.0,
    state_perturbation: FloatArray | None = None,
    control_noise_std: float = 0.0,
    event_offset_rad: float = 0.0,
    channel_mask: FloatArray | None = None,
    seed: int = 42,
    params: DoublePendulumParameters | None = None,
) -> TopologyEvaluation:
    """Roll out analytical double pendulum under actuator delay and perturbations."""
    p = params or DoublePendulumParameters.default()
    rng = np.random.default_rng(seed)
    K = len(nominal_controls)

    mask = (
        channel_mask
        if channel_mask is not None
        else np.array([1.0, 1.0], dtype=np.float64)
    )

    # Initial state with perturbation
    z = initial_state.copy()
    if state_perturbation is not None:
        z += state_perturbation

    delay_steps = int(round(delay_s / dt)) if dt > 0 else 0
    states = [z.copy()]

    for k in range(K):
        # Delayed command
        delayed_k = max(0, k - delay_steps)
        u_base = nominal_controls[delayed_k] * mask

        if control_noise_std > 0.0:
            noise = rng.normal(0.0, control_noise_std, size=2) * mask
            u = u_base + noise
        else:
            u = u_base

        z = discrete_rk4_step(z, u, dt, p)
        states.append(z.copy())

    states_arr = np.array(states, dtype=np.float64)
    return enumerate_guard_crossings(
        states_arr, dt, event_offset_rad=event_offset_rad, params=p
    )


def run_event_topology_suite() -> EventTopologyRobustnessSummary:
    """Run full delay continuation, noise study, and channel controls."""
    states_nom, controls_nom = generate_nominal_downswing_trajectory(
        dt=0.002, steps=140
    )
    z0 = states_nom[0]
    dt = 0.002

    # 1. Zero perturbation check
    eval_zero = simulate_perturbed_downswing(z0, controls_nom, dt)
    reproduces_nominal = (
        eval_zero.topology_class == EventTopologyClass.UNIQUE_FORWARD
        and eval_zero.crossing_count == 1
    )
    nom_crossing_t = eval_zero.first_crossing_time_s or 0.0

    # 2. Step refinement control: dt = 0.001 vs dt = 0.002
    states_fine, controls_fine = generate_nominal_downswing_trajectory(
        dt=0.001, steps=280
    )
    eval_fine = enumerate_guard_crossings(states_fine, dt=0.001)
    step_refinement_stable = (
        eval_fine.topology_class == EventTopologyClass.UNIQUE_FORWARD
        and eval_fine.crossing_count == 1
        and abs((eval_fine.first_crossing_time_s or 0.0) - nom_crossing_t) < 0.01
    )

    # 3. Actuator delay continuation: test delays in [0.002, 0.010, 0.020, 0.050]
    delays = [0.002, 0.006, 0.010, 0.020, 0.040]
    max_tolerated = 0.0
    for d in delays:
        ev_d = simulate_perturbed_downswing(z0, controls_nom, dt, delay_s=d)
        if ev_d.topology_class == EventTopologyClass.UNIQUE_FORWARD:
            max_tolerated = d
        else:
            break

    # 4. Common-random-number noise & event-surface perturbation study
    seeds = [101, 102, 103, 104, 105, 106, 107, 108]
    unique_count = 0
    for s in seeds:
        ev_noise = simulate_perturbed_downswing(
            z0,
            controls_nom,
            dt,
            control_noise_std=2.0,  # 2 N*m torque noise
            event_offset_rad=0.01,  # ~0.5 deg event surface shift
            seed=s,
        )
        if ev_noise.topology_class == EventTopologyClass.UNIQUE_FORWARD:
            unique_count += 1
    noise_fraction = unique_count / len(seeds)

    # 5. Channel coverage: both, shoulder-only, wrist-only, zero-authority
    ev_both = simulate_perturbed_downswing(
        z0, controls_nom, dt, channel_mask=np.array([1.0, 1.0])
    )
    ev_sh = simulate_perturbed_downswing(
        z0, controls_nom, dt, channel_mask=np.array([1.0, 0.0])
    )
    ev_wr = simulate_perturbed_downswing(
        z0, controls_nom, dt, channel_mask=np.array([0.0, 1.0])
    )
    ev_zero = simulate_perturbed_downswing(
        z0, controls_nom, dt, channel_mask=np.array([0.0, 0.0])
    )

    channel_coverage_passed = (
        ev_both.crossing_count >= 1
        and ev_sh.crossing_count >= 1
        and ev_zero.topology_class
        in (EventTopologyClass.ZERO_CROSSINGS, EventTopologyClass.UNIQUE_FORWARD)
    )

    total_trials = 1 + 1 + len(delays) + len(seeds) + 4

    return EventTopologyRobustnessSummary(
        zero_perturbation_reproduces_nominal=reproduces_nominal,
        nominal_first_crossing_time_s=nom_crossing_t,
        max_tolerated_delay_s=max_tolerated,
        noise_robustness_retained_unique_fraction=noise_fraction,
        channel_coverage_passed=channel_coverage_passed,
        step_refinement_stable=step_refinement_stable,
        total_trials=total_trials,
    )
