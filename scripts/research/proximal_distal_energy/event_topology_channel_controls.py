"""Coordinate-explicit channel-mask controls for event-topology studies.

The masks are countermodels for generalized-coordinate authority. They are not
anatomical isolation experiments and cannot identify a wrist, arm, hand, or
scapular contribution in a human swing.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.event_robustness_noise import (
    CommonRandomPerturbations,
)
from scripts.research.proximal_distal_energy.event_topology_robustness import (
    GlobalEventTopology,
)
from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    clubhead_speed,
)
from src.shared.python.simulation_backends import GolfModelParams

FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ChannelMask:
    """Named binary authority mask in shoulder/wrist torque coordinates."""

    name: str
    values: tuple[float, float]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("channel-mask name must be nonempty")
        converted = tuple(float(value) for value in self.values)
        if len(converted) != 2 or any(value not in (0.0, 1.0) for value in converted):
            raise ValueError("channel-mask values must be binary")
        object.__setattr__(self, "values", converted)


def registered_channel_masks() -> tuple[ChannelMask, ...]:
    """Return the preregistered generalized-coordinate authority masks."""

    return (
        ChannelMask("both", (1.0, 1.0)),
        ChannelMask("shoulder_only", (1.0, 0.0)),
        ChannelMask("wrist_only", (0.0, 1.0)),
        ChannelMask("zero", (0.0, 0.0)),
    )


def apply_channel_mask(controls: npt.ArrayLike, mask: ChannelMask) -> FloatArray:
    """Return an immutable masked copy of an ``(N, 2)`` command history."""

    commands = np.asarray(controls, dtype=float)
    if (
        commands.ndim != 2
        or commands.shape[0] < 1
        or commands.shape[1] != 2
        or not np.all(np.isfinite(commands))
    ):
        raise ValueError("controls must be a nonempty finite (N, 2) array")
    masked = commands * np.asarray(mask.values, dtype=float)[np.newaxis, :]
    masked.setflags(write=False)
    return masked


def mask_common_random_perturbations(
    perturbations: CommonRandomPerturbations,
    mask: ChannelMask,
) -> CommonRandomPerturbations:
    """Mask command noise while preserving matched state and guard draws."""

    command = (
        perturbations.command_delta_nm
        * np.asarray(mask.values)[np.newaxis, np.newaxis, :]
    )
    return CommonRandomPerturbations(
        perturbations.initial_state_delta,
        command,
        perturbations.guard_offset_delta,
    )


def event_metric_records(
    topology: GlobalEventTopology,
    params: GolfModelParams,
) -> list[dict[str, Any]]:
    """Keep kinematics alongside, but never in place of, topology identity."""

    inertials = PlanarInertials.from_params(params)
    records: list[dict[str, Any]] = []
    for event in topology.events:
        state = np.asarray(event.state, dtype=float)
        speed = clubhead_speed(
            inertials,
            state[np.newaxis, :2],
            state[np.newaxis, 2:],
        )
        value = float(speed[0])
        if not math.isfinite(value):
            raise ValueError("event clubhead speed must be finite")
        records.append(
            {
                "direction": event.direction.value,
                "sample_index": event.sample_index,
                "event_time_s": event.time_s,
                "event_state": state.tolist(),
                "guard_residual": event.guard_residual,
                "transversality_per_s": event.transversality_per_s,
                "near_grazing": event.near_grazing,
                "clubhead_speed_m_s": value,
            }
        )
    return records


__all__ = [
    "ChannelMask",
    "apply_channel_mask",
    "event_metric_records",
    "mask_common_random_perturbations",
    "registered_channel_masks",
]
