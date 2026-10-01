"""Simulation timing policy and step schedule planning (R11).

Defines unified endpoint-inclusive sampling, step schedules, remainder step
handling, and truthful horizon calculation across REST and WebSocket simulations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SimulationTimingPlan:
    """Immutable plan for simulation integration steps and sampling clocks.

    Attributes:
        requested_duration: Horizon in seconds requested by caller.
        timestep: Base integration timestep in seconds.
        step_count: Total number of discrete integration steps to execute.
        step_sizes: Sequence of timestep sizes (dt) for each integration step.
        integrated_duration: Truthful total simulated time (sum of step_sizes).
        retained_samples: Total state samples retained (step_count + 1 for endpoint-inclusive).
        has_remainder_step: True if a fractional final step is executed.
        remainder_dt: Size of fractional final step in seconds, if applicable.
        is_divisible: True if requested_duration is an integer multiple of timestep.
        sampling_policy: Identifier for sampling convention ('endpoint_inclusive').
    """

    requested_duration: float
    timestep: float
    step_count: int
    step_sizes: tuple[float, ...]
    integrated_duration: float
    retained_samples: int
    has_remainder_step: bool
    remainder_dt: float | None
    is_divisible: bool
    sampling_policy: str = "endpoint_inclusive"

    def __iter__(self):
        yield self.timestep
        yield self.step_count
        yield self.retained_samples


def engine_supports_variable_step(engine: Any) -> bool:
    """Return whether a physics engine or backend supports variable-sized integration steps.

    Args:
        engine: Physics engine, backend, or adapter instance.

    Returns:
        True if the engine can accept variable dt in step(dt), False otherwise.
    """
    if engine is None:
        return True

    # Check explicit attribute on engine
    if hasattr(engine, "supports_variable_step"):
        return bool(engine.supports_variable_step)

    # Check backend capabilities
    caps = getattr(engine, "capabilities", None)
    if caps is not None and hasattr(caps, "supports_variable_step"):
        return bool(caps.supports_variable_step)

    # Default to True for known dynamic engines
    return True


def compute_simulation_timing(
    duration: float,
    timestep: float,
    allow_remainder_step: bool = True,
    engine: Any = None,
) -> SimulationTimingPlan:
    """Compute an exact, truthful step schedule for a simulation horizon.

    Handles floating-point boundaries, non-divisible durations, sub-step horizons,
    and engine step capabilities without silently relabeling state times.

    Args:
        duration: Requested simulation horizon in seconds (> 0).
        timestep: Base integration step size in seconds (> 0).
        allow_remainder_step: Whether fractional remainder steps are permitted.
        engine: Optional physics engine to probe for variable step support.

    Returns:
        SimulationTimingPlan containing exact step sizes, truthful integrated horizon,
        and sample counts.

    Raises:
        ValueError: If duration or timestep are non-positive or non-finite.
    """
    if not isinstance(duration, (int, float)) or not math.isfinite(duration):
        raise ValueError(f"Duration must be finite, got {duration!r}")
    if duration <= 0:
        raise ValueError(f"Duration must be positive, got {duration!r}")

    if not isinstance(timestep, (int, float)) or not math.isfinite(timestep):
        raise ValueError(f"Timestep must be finite, got {timestep!r}")
    if timestep <= 0:
        raise ValueError(f"Timestep must be positive, got {timestep!r}")

    duration = float(duration)
    timestep = float(timestep)

    # Check if backend permits variable final steps
    can_variable_step = allow_remainder_step and engine_supports_variable_step(engine)

    # 1. Divisible check with floating-point tolerance (e.g. 0.03 / 0.01 = 2.9999999999999996)
    ratio = duration / timestep
    rounded_steps = round(ratio)
    if rounded_steps > 0 and math.isclose(
        rounded_steps * timestep, duration, rel_tol=1e-9, abs_tol=1e-12
    ):
        step_count = rounded_steps
        step_sizes = (timestep,) * step_count
        integrated_duration = round(step_count * timestep, 9)
        return SimulationTimingPlan(
            requested_duration=duration,
            timestep=timestep,
            step_count=step_count,
            step_sizes=step_sizes,
            integrated_duration=integrated_duration,
            retained_samples=step_count + 1,
            has_remainder_step=False,
            remainder_dt=None,
            is_divisible=True,
        )

    # 2. Sub-step duration (duration < timestep)
    if duration < timestep:
        if can_variable_step:
            step_sizes = (duration,)
            return SimulationTimingPlan(
                requested_duration=duration,
                timestep=timestep,
                step_count=1,
                step_sizes=step_sizes,
                integrated_duration=duration,
                retained_samples=2,
                has_remainder_step=True,
                remainder_dt=duration,
                is_divisible=False,
            )
        # Fixed-step only cannot step smaller than timestep; report 0 steps or 1 full step
        return SimulationTimingPlan(
            requested_duration=duration,
            timestep=timestep,
            step_count=0,
            step_sizes=(),
            integrated_duration=0.0,
            retained_samples=1,
            has_remainder_step=False,
            remainder_dt=None,
            is_divisible=False,
        )

    # 3. Non-divisible duration (duration > timestep)
    n_full = int(math.floor(duration / timestep))
    rem = round(duration - (n_full * timestep), 12)

    # Check if remainder is effectively zero within float precision
    if math.isclose(rem, 0.0, abs_tol=1e-12) or math.isclose(
        rem, timestep, abs_tol=1e-12
    ):
        step_count = n_full if math.isclose(rem, 0.0, abs_tol=1e-12) else n_full + 1
        step_sizes = (timestep,) * step_count
        integrated_duration = round(step_count * timestep, 9)
        return SimulationTimingPlan(
            requested_duration=duration,
            timestep=timestep,
            step_count=step_count,
            step_sizes=step_sizes,
            integrated_duration=integrated_duration,
            retained_samples=step_count + 1,
            has_remainder_step=False,
            remainder_dt=None,
            is_divisible=True,
        )

    if can_variable_step and rem > 0:
        step_count = n_full + 1
        step_sizes = (timestep,) * n_full + (rem,)
        integrated_duration = round(duration, 9)
        return SimulationTimingPlan(
            requested_duration=duration,
            timestep=timestep,
            step_count=step_count,
            step_sizes=step_sizes,
            integrated_duration=integrated_duration,
            retained_samples=step_count + 1,
            has_remainder_step=True,
            remainder_dt=rem,
            is_divisible=False,
        )

    # Fixed step only: execute n_full steps, truthfully report executed horizon
    step_count = n_full
    step_sizes = (timestep,) * step_count
    integrated_duration = round(n_full * timestep, 9)
    return SimulationTimingPlan(
        requested_duration=duration,
        timestep=timestep,
        step_count=step_count,
        step_sizes=step_sizes,
        integrated_duration=integrated_duration,
        retained_samples=step_count + 1,
        has_remainder_step=False,
        remainder_dt=None,
        is_divisible=False,
    )
