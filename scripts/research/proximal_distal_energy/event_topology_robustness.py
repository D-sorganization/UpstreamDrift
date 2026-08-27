"""Registered event-topology and perturbation robustness mapping (#9125).

The study reuses the protected #9123 analytical model, torque program, and
delivery guard. Perturbations are synthetic model scenarios; they are not
measurements of human delay, variability, fatigue, or control strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.phase_event_stability import (
    rollout_state_history,
)
from scripts.research.proximal_distal_energy.run_experiments import INITIAL_Q
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from src.shared.python.simulation_backends import GolfModelParams

FloatArray = npt.NDArray[np.float64]

GUARD_GRADIENT = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)
BASE_DT_S = 0.0005
HORIZON_S = 0.55
TRANSVERSALITY_TOLERANCE_PER_S = 1e-4
INFERENCE_BOUNDARY = (
    "This event-topology mapping establishes only finite-horizon properties "
    "of the declared analytical double pendulum under synthetic perturbation "
    "scenarios. It does not establish human motor variability, neuromuscular "
    "delay, fatigue, controller ranking, passive torque, or coaching advice."
)


class EventTopologyClass(str, Enum):
    """Classification of all registered guard crossings in one horizon."""

    ZERO_CROSSINGS = "ZERO_CROSSINGS"
    UNIQUE_FORWARD = "UNIQUE_FORWARD"
    MULTIPLE_CROSSINGS = "MULTIPLE_CROSSINGS"
    REVERSED_DIRECTION = "REVERSED_DIRECTION"
    GRAZING = "GRAZING"


@dataclass(frozen=True, slots=True)
class CrossingEvent:
    """One linearly refined sample-bracket crossing."""

    time_s: float
    state_at_crossing: tuple[float, float, float, float]
    normal_velocity_per_s: float
    is_transverse: bool
    is_forward: bool


@dataclass(frozen=True, slots=True)
class TopologyEvaluation:
    """All finite-horizon crossings and their aggregate topology."""

    topology_class: EventTopologyClass
    crossing_count: int
    first_crossing_time_s: float | None
    crossings: tuple[CrossingEvent, ...]
    terminal_state: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PerturbationScenario:
    """Immutable synthetic perturbation and actuator-history contract."""

    delay_s: float = 0.0
    state_perturbation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    control_noise_std_nm: float = 0.0
    event_offset_rad: float = 0.0
    channel_mask: tuple[float, float] = (1.0, 1.0)
    seed: int = 42
    pre_delay_command_nm: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        values = (
            self.delay_s,
            self.control_noise_std_nm,
            self.event_offset_rad,
            *self.state_perturbation,
            *self.channel_mask,
            *self.pre_delay_command_nm,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("perturbation scenario values must be finite")
        if self.delay_s < 0.0 or self.control_noise_std_nm < 0.0:
            raise ValueError("delay and noise scale must be nonnegative")
        if any(value not in (0.0, 1.0) for value in self.channel_mask):
            raise ValueError("channel_mask entries must be zero or one")


@dataclass(frozen=True, slots=True)
class EventTopologyRobustnessSummary:
    """Aggregated scenario evidence for issue #9125."""

    zero_perturbation_reproduces_nominal: bool
    nominal_first_crossing_time_s: float
    max_tolerated_delay_s: float
    noise_robustness_retained_unique_fraction: float
    channel_topologies: tuple[tuple[str, str], ...]
    channel_coverage_passed: bool
    step_refinement_stable: bool
    total_trials: int
    inference_boundary: str = INFERENCE_BOUNDARY


def registered_nominal_inputs(
    *, dt_s: float = BASE_DT_S, horizon_s: float = HORIZON_S
) -> tuple[FloatArray, FloatArray]:
    """Build the #9123 nominal initial state and registered torque program."""

    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if not math.isfinite(horizon_s) or horizon_s <= 0.0:
        raise ValueError("horizon_s must be finite and positive")
    step_count = int(round(horizon_s / dt_s))
    if not math.isclose(step_count * dt_s, horizon_s, abs_tol=1e-12, rel_tol=0.0):
        raise ValueError("dt_s must divide horizon_s")
    initial = np.array([*INITIAL_Q, 0.0, 0.0], dtype=float)
    controls = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10).controls(
        step_count, dt_s
    )
    return initial, np.asarray(controls, dtype=float)


def applied_control_history(
    nominal_controls: npt.ArrayLike,
    *,
    dt_s: float,
    scenario: PerturbationScenario,
) -> FloatArray:
    """Apply declared delay, prehistory, channel mask, and seeded noise."""

    controls = np.asarray(nominal_controls, dtype=float)
    if controls.ndim != 2 or controls.shape[1] != 2 or controls.shape[0] == 0:
        raise ValueError("nominal_controls must be a nonempty (N, 2) array")
    if not np.all(np.isfinite(controls)):
        raise ValueError("nominal_controls must be finite")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")

    delay_steps = int(round(scenario.delay_s / dt_s))
    applied = np.broadcast_to(
        np.asarray(scenario.pre_delay_command_nm, dtype=float), controls.shape
    ).copy()
    if delay_steps < controls.shape[0]:
        applied[delay_steps:] = controls[: controls.shape[0] - delay_steps]
    mask = np.asarray(scenario.channel_mask, dtype=float)
    applied *= mask[np.newaxis, :]
    if scenario.control_noise_std_nm > 0.0:
        rng = np.random.default_rng(scenario.seed)
        noise = rng.normal(0.0, scenario.control_noise_std_nm, size=applied.shape)
        applied += noise * mask[np.newaxis, :]
    return applied


def _guard_values(states: FloatArray, event_offset_rad: float) -> FloatArray:
    return states @ GUARD_GRADIENT - event_offset_rad


def enumerate_guard_crossings(
    states: npt.ArrayLike,
    dt_s: float,
    *,
    event_offset_rad: float = 0.0,
    transverse_tolerance_per_s: float = TRANSVERSALITY_TOLERANCE_PER_S,
) -> TopologyEvaluation:
    """Enumerate every crossing of the protected #9123 delivery guard."""

    state = np.asarray(states, dtype=float)
    if state.ndim != 2 or state.shape[1] != 4 or state.shape[0] < 2:
        raise ValueError("states must be a finite (N>=2, 4) array")
    if not np.all(np.isfinite(state)):
        raise ValueError("states must be finite")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if not math.isfinite(event_offset_rad):
        raise ValueError("event_offset_rad must be finite")
    if transverse_tolerance_per_s <= 0.0:
        raise ValueError("transverse tolerance must be positive")

    guard = _guard_values(state, event_offset_rad)
    crossings: list[CrossingEvent] = []
    for index, (before, after) in enumerate(zip(guard[:-1], guard[1:], strict=True)):
        if not ((before < 0.0 <= after) or (before > 0.0 >= after)):
            continue
        denominator = after - before
        fraction = 0.0 if denominator == 0.0 else float(-before / denominator)
        event_state = state[index] + fraction * (state[index + 1] - state[index])
        normal_velocity = float(event_state[2] + event_state[3])
        transverse = abs(normal_velocity) >= transverse_tolerance_per_s
        crossings.append(
            CrossingEvent(
                time_s=(index + fraction) * dt_s,
                state_at_crossing=tuple(float(value) for value in event_state),
                normal_velocity_per_s=normal_velocity,
                is_transverse=transverse,
                is_forward=normal_velocity > 0.0,
            )
        )

    topology = _classify_crossings(crossings)
    return TopologyEvaluation(
        topology_class=topology,
        crossing_count=len(crossings),
        first_crossing_time_s=None if not crossings else crossings[0].time_s,
        crossings=tuple(crossings),
        terminal_state=tuple(float(value) for value in state[-1]),
    )


def _classify_crossings(crossings: list[CrossingEvent]) -> EventTopologyClass:
    if not crossings:
        return EventTopologyClass.ZERO_CROSSINGS
    if any(not event.is_transverse for event in crossings):
        return EventTopologyClass.GRAZING
    if len(crossings) > 1:
        return EventTopologyClass.MULTIPLE_CROSSINGS
    if crossings[0].is_forward:
        return EventTopologyClass.UNIQUE_FORWARD
    return EventTopologyClass.REVERSED_DIRECTION


def simulate_perturbed_downswing(
    initial_state: npt.ArrayLike,
    nominal_controls: npt.ArrayLike,
    dt_s: float,
    *,
    scenario: PerturbationScenario = PerturbationScenario(),
    params: GolfModelParams | None = None,
) -> TopologyEvaluation:
    """Replay a perturbation scenario through the protected analytical backend."""

    initial = np.asarray(initial_state, dtype=float)
    if initial.shape != (4,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_state must be a finite four-vector")
    perturbed = initial + np.asarray(scenario.state_perturbation, dtype=float)
    applied = applied_control_history(nominal_controls, dt_s=dt_s, scenario=scenario)
    _, states = rollout_state_history(
        params or GolfModelParams.default(),
        initial_state=perturbed,
        controls=applied,
        dt_s=dt_s,
    )
    return enumerate_guard_crossings(
        states,
        dt_s,
        event_offset_rad=scenario.event_offset_rad,
    )


def _topology_map(
    initial: FloatArray, controls: FloatArray, dt_s: float
) -> dict[str, TopologyEvaluation]:
    masks = {
        "both": (1.0, 1.0),
        "shoulder_only": (1.0, 0.0),
        "wrist_only": (0.0, 1.0),
        "zero": (0.0, 0.0),
    }
    return {
        name: simulate_perturbed_downswing(
            initial,
            controls,
            dt_s,
            scenario=PerturbationScenario(channel_mask=mask),
        )
        for name, mask in masks.items()
    }


def run_event_topology_suite() -> EventTopologyRobustnessSummary:
    """Execute registered nominal, refinement, delay, noise, and channel cases."""

    initial, controls = registered_nominal_inputs()
    nominal = simulate_perturbed_downswing(initial, controls, BASE_DT_S)
    nominal_time = nominal.first_crossing_time_s or 0.0
    reproduces = nominal.topology_class is EventTopologyClass.UNIQUE_FORWARD

    fine_initial, fine_controls = registered_nominal_inputs(dt_s=BASE_DT_S / 2.0)
    fine = simulate_perturbed_downswing(fine_initial, fine_controls, BASE_DT_S / 2.0)
    refinement = (
        fine.topology_class is EventTopologyClass.UNIQUE_FORWARD
        and fine.first_crossing_time_s is not None
        and abs(fine.first_crossing_time_s - nominal_time) < 5e-4
    )

    delays = (0.002, 0.006, 0.010, 0.020, 0.040)
    delay_results = [
        simulate_perturbed_downswing(
            initial,
            controls,
            BASE_DT_S,
            scenario=PerturbationScenario(delay_s=delay),
        )
        for delay in delays
    ]
    tolerated = [
        delay
        for delay, result in zip(delays, delay_results, strict=True)
        if result.topology_class is EventTopologyClass.UNIQUE_FORWARD
    ]

    seeds = (101, 102, 103, 104, 105, 106, 107, 108)
    noise_results = [
        simulate_perturbed_downswing(
            initial,
            controls,
            BASE_DT_S,
            scenario=PerturbationScenario(
                control_noise_std_nm=2.0,
                event_offset_rad=0.01,
                seed=seed,
            ),
        )
        for seed in seeds
    ]
    noise_fraction = sum(
        result.topology_class is EventTopologyClass.UNIQUE_FORWARD
        for result in noise_results
    ) / len(noise_results)

    channels = _topology_map(initial, controls, BASE_DT_S)
    expected_channels = {
        "both": EventTopologyClass.UNIQUE_FORWARD,
        "shoulder_only": EventTopologyClass.UNIQUE_FORWARD,
        "wrist_only": EventTopologyClass.UNIQUE_FORWARD,
        "zero": EventTopologyClass.ZERO_CROSSINGS,
    }
    coverage = all(
        channels[name].topology_class is expected
        for name, expected in expected_channels.items()
    )
    channel_topologies = tuple(
        (name, channels[name].topology_class.value) for name in expected_channels
    )
    return EventTopologyRobustnessSummary(
        zero_perturbation_reproduces_nominal=reproduces,
        nominal_first_crossing_time_s=nominal_time,
        max_tolerated_delay_s=max(tolerated, default=0.0),
        noise_robustness_retained_unique_fraction=noise_fraction,
        channel_topologies=channel_topologies,
        channel_coverage_passed=coverage,
        step_refinement_stable=refinement,
        total_trials=2 + len(delays) + len(seeds) + len(channels),
    )


__all__ = [
    "BASE_DT_S",
    "CrossingEvent",
    "EventTopologyClass",
    "EventTopologyRobustnessSummary",
    "GUARD_GRADIENT",
    "HORIZON_S",
    "INFERENCE_BOUNDARY",
    "PerturbationScenario",
    "TopologyEvaluation",
    "applied_control_history",
    "enumerate_guard_crossings",
    "registered_nominal_inputs",
    "run_event_topology_suite",
    "simulate_perturbed_downswing",
]
