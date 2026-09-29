"""Exact global topology replay for matched delay/noise scenarios.

This module retains every synthetic replicate as a typed topology outcome. It
does not estimate a human success probability or establish physiological
robustness, fatigue tolerance, controller superiority, or coaching guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.event_robustness_noise import (
    CommonRandomPerturbations,
)
from scripts.research.proximal_distal_energy.event_topology_robustness import (
    CommandDelayConfig,
    DelayContinuationConfig,
    DelayContinuationResult,
    GlobalEventTopology,
    apply_command_delay,
    evaluate_delay_continuation,
    replay_global_event_topology,
)
from scripts.research.proximal_distal_energy.trajectory_control_authority import (
    GuardCrossingConfig,
)
from src.shared.python.simulation_backends import GolfModelParams


@dataclass(frozen=True, slots=True)
class DelayNoiseTopologyOutcome:
    """One retained replicate at one declared physical delay."""

    delay_s: float
    replicate_index: int
    topology: GlobalEventTopology

    def __post_init__(self) -> None:
        if not math.isfinite(self.delay_s) or self.delay_s < 0.0:
            raise ValueError("delay_s must be finite and nonnegative")
        if self.replicate_index < 0:
            raise ValueError("replicate_index must be nonnegative")


@dataclass(frozen=True, slots=True)
class DelayNoiseTopologyResult:
    """Nominal reference plus all matched stochastic-scenario replays."""

    nominal: DelayContinuationResult
    outcomes: tuple[DelayNoiseTopologyOutcome, ...]
    replicate_count: int

    def __post_init__(self) -> None:
        expected = len(self.nominal.outcomes) * self.replicate_count
        if self.replicate_count < 1 or len(self.outcomes) != expected:
            raise ValueError("outcome count must match delays times replicates")


def _shifted_guard(
    guard: GuardCrossingConfig, offset_delta: float
) -> GuardCrossingConfig:
    return GuardCrossingConfig(
        guard_gradient=guard.guard_gradient,
        guard_offset=guard.guard_offset + offset_delta,
        guard_tolerance=guard.guard_tolerance,
        time_tolerance_s=guard.time_tolerance_s,
        transversality_threshold=guard.transversality_threshold,
        max_iterations=guard.max_iterations,
    )


def evaluate_delay_noise_topology(
    *,
    params: GolfModelParams,
    initial_state: npt.ArrayLike,
    controls: npt.ArrayLike,
    dt_s: float,
    guard: GuardCrossingConfig,
    delay_config: DelayContinuationConfig,
    perturbations: CommonRandomPerturbations,
) -> DelayNoiseTopologyResult:
    """Replay matched perturbations across every registered delay case."""

    initial = np.asarray(initial_state, dtype=float)
    commands = np.asarray(controls, dtype=float)
    if initial.shape != (4,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_state must contain four finite values")
    nominal = evaluate_delay_continuation(
        params=params,
        initial_state=initial,
        controls=commands,
        dt_s=dt_s,
        guard=guard,
        config=delay_config,
    )
    output_count = nominal.output_sample_count
    if perturbations.command_delta_nm.shape[1] != output_count:
        raise ValueError("command perturbations must match the common horizon")
    replicate_count = perturbations.initial_state_delta.shape[0]
    outcomes: list[DelayNoiseTopologyOutcome] = []
    for delay_s in delay_config.delays_s:
        delayed = apply_command_delay(
            commands,
            dt_s=dt_s,
            output_sample_count=output_count,
            config=CommandDelayConfig(
                delay_s=delay_s,
                interpolation=delay_config.interpolation,
                prehistory_control=delay_config.prehistory_control,
                posthistory_control=delay_config.posthistory_control,
            ),
        )
        for replicate_index in range(replicate_count):
            topology = replay_global_event_topology(
                params=params,
                initial_state=(
                    initial + perturbations.initial_state_delta[replicate_index]
                ),
                controls=(delayed + perturbations.command_delta_nm[replicate_index]),
                dt_s=dt_s,
                guard=_shifted_guard(
                    guard,
                    float(perturbations.guard_offset_delta[replicate_index]),
                ),
            )
            outcomes.append(
                DelayNoiseTopologyOutcome(delay_s, replicate_index, topology)
            )
    return DelayNoiseTopologyResult(nominal, tuple(outcomes), replicate_count)


__all__ = [
    "DelayNoiseTopologyOutcome",
    "DelayNoiseTopologyResult",
    "evaluate_delay_noise_topology",
]
