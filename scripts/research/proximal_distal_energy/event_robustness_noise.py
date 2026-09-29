"""Antithetic common-random-number inputs for issue #9125.

These perturbations are declared synthetic model scenarios.  They are not
estimates of human motor noise, fatigue, skill, injury, or physiology.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]


def _standard_deviations(
    name: str, values: tuple[float, ...], *, size: int
) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if len(converted) != size or not all(
        math.isfinite(value) and value >= 0.0 for value in converted
    ):
        raise ValueError(f"{name} must contain {size} finite nonnegative values")
    return converted


def _readonly(values: npt.ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=float).copy()
    if not np.all(np.isfinite(array)):
        raise ValueError("generated perturbations must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class RobustnessNoiseConfig:
    """Immutable antithetic perturbation design with one declared seed."""

    seed: int
    replicate_count: int
    initial_state_sd: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    command_sd_nm: tuple[float, ...] = (0.0, 0.0)
    guard_offset_sd: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        if (
            isinstance(self.replicate_count, bool)
            or not isinstance(self.replicate_count, int)
            or self.replicate_count < 2
            or self.replicate_count % 2 != 0
        ):
            raise ValueError("replicate_count must be a positive even integer")
        initial_sd = _standard_deviations(
            "initial_state_sd", self.initial_state_sd, size=4
        )
        command_sd = _standard_deviations("command_sd_nm", self.command_sd_nm, size=2)
        if not math.isfinite(self.guard_offset_sd) or self.guard_offset_sd < 0.0:
            raise ValueError("guard_offset_sd must be finite and nonnegative")
        object.__setattr__(self, "initial_state_sd", initial_sd)
        object.__setattr__(self, "command_sd_nm", command_sd)


@dataclass(frozen=True, slots=True)
class CommonRandomPerturbations:
    """Matched physical perturbations for every compared experimental case."""

    initial_state_delta: FloatArray
    command_delta_nm: FloatArray
    guard_offset_delta: FloatArray

    def __post_init__(self) -> None:
        state = _readonly(self.initial_state_delta)
        command = _readonly(self.command_delta_nm)
        guard = _readonly(self.guard_offset_delta)
        replicate_count = state.shape[0] if state.ndim == 2 else -1
        if state.shape != (replicate_count, 4):
            raise ValueError("initial_state_delta must have shape (R, 4)")
        if (
            command.ndim != 3
            or command.shape[:1] != (replicate_count,)
            or command.shape[2] != 2
        ):
            raise ValueError("command_delta_nm must have shape (R, N, 2)")
        if guard.shape != (replicate_count,):
            raise ValueError("guard_offset_delta must have shape (R,)")
        object.__setattr__(self, "initial_state_delta", state)
        object.__setattr__(self, "command_delta_nm", command)
        object.__setattr__(self, "guard_offset_delta", guard)


def generate_common_random_perturbations(
    config: RobustnessNoiseConfig,
    *,
    control_sample_count: int,
) -> CommonRandomPerturbations:
    """Generate deterministic paired draws and scale them to declared units."""

    if (
        isinstance(control_sample_count, bool)
        or not isinstance(control_sample_count, int)
        or control_sample_count < 1
    ):
        raise ValueError("control_sample_count must be a positive integer")
    half_count = config.replicate_count // 2
    generator = np.random.default_rng(config.seed)
    state_standard = generator.standard_normal((half_count, 4))
    command_standard = generator.standard_normal((half_count, control_sample_count, 2))
    guard_standard = generator.standard_normal(half_count)

    state = np.concatenate((state_standard, -state_standard), axis=0)
    command = np.concatenate((command_standard, -command_standard), axis=0)
    guard = np.concatenate((guard_standard, -guard_standard), axis=0)
    state *= np.asarray(config.initial_state_sd, dtype=float)[np.newaxis, :]
    command *= np.asarray(config.command_sd_nm, dtype=float)[np.newaxis, np.newaxis, :]
    guard *= config.guard_offset_sd
    return CommonRandomPerturbations(state, command, guard)


__all__ = [
    "CommonRandomPerturbations",
    "RobustnessNoiseConfig",
    "generate_common_random_perturbations",
]
