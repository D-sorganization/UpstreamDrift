"""Fixed-pivot vs prescribed moving-hub identities and external-work accounting (CO-04)."""

from __future__ import annotations

from enum import Enum

import numpy as np

from src.shared.python.tour_baselines.calibration import (
    MovingHubMotion,
    compute_moving_hub_power,
)

__all__ = [
    "HubMode",
    "hub_variant_id",
    "account_external_hub_work",
]


class HubMode(str, Enum):
    """Hub actuation classification for reduced pendulum matches."""

    FIXED_PIVOT = "fixed_pivot"
    PRESCRIBED_MOVING_HUB = "prescribed_moving_hub"


def hub_variant_id(model_id: str, hub_mode: HubMode) -> str:
    """Return a stable distinct identity for model + hub mode."""
    if not model_id:
        raise ValueError("model_id must be non-empty")
    if not isinstance(hub_mode, HubMode):
        raise TypeError("hub_mode must be a HubMode")
    return f"{model_id}.{hub_mode.value}"


def account_external_hub_work(
    *,
    times: np.ndarray,
    hub_positions: np.ndarray,
    hub_reaction_forces: np.ndarray,
    hub_mode: HubMode,
    measured_work_joules: float | None = None,
) -> MovingHubMotion:
    """Account prescribed-hub external work; unmeasured fixed pivots report None."""
    if hub_mode is HubMode.FIXED_PIVOT:
        t_arr = np.asarray(times, dtype=np.float64)
        if t_arr.ndim != 1 or t_arr.size < 2:
            raise ValueError("times must be 1-D with >= 2 samples")
        n = int(t_arr.size)
        zeros = np.zeros((n, 2), dtype=np.float64)
        work_val = (
            float(measured_work_joules) if measured_work_joules is not None else None
        )
        return MovingHubMotion(
            times=t_arr.copy(),
            positions=zeros,
            velocity=zeros.copy(),
            power_watts=np.zeros(n, dtype=np.float64),
            total_work_joules=work_val,
            is_moving_hub=False,
        )
    if hub_mode is not HubMode.PRESCRIBED_MOVING_HUB:
        raise ValueError(f"unsupported hub_mode={hub_mode!r}")
    if measured_work_joules is not None:
        raise ValueError(
            "measured_work_joules cannot be passed for PRESCRIBED_MOVING_HUB"
        )
    return compute_moving_hub_power(times, hub_positions, hub_reaction_forces)
