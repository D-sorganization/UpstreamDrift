"""Model-agnostic timing-viability classification and region summaries.

The functions in this module compare policies only on a declared common phase
coordinate and common outcome guards.  A larger model timing region is not
evidence of lower neural timing demand or a coaching advantage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

REQUIRED_METRICS = (
    "delivery_speed_m_s",
    "face_path_error_deg",
    "peak_hand_force_n",
    "effort_proxy_nms",
    "returned_to_viable_set",
    "normalized_energy_residual",
    "realized_event_time_s",
)


@dataclass(frozen=True, slots=True)
class ViabilityLimits:
    """Predeclared guards relative to one common load-matched baseline."""

    speed_fraction_min: float = 0.95
    face_error_allowance_deg: float = 2.0
    peak_force_ratio_max: float = 1.10
    effort_ratio_max: float = 1.10
    normalized_energy_residual_max: float = 0.05
    require_sustained_recovery: bool = True

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.speed_fraction_min,
                self.face_error_allowance_deg,
                self.peak_force_ratio_max,
                self.effort_ratio_max,
                self.normalized_energy_residual_max,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("viability limits must be finite")
        if not 0.0 < self.speed_fraction_min <= 1.0:
            raise ValueError("speed_fraction_min must be in (0, 1]")
        if self.face_error_allowance_deg < 0.0:
            raise ValueError("face_error_allowance_deg must be nonnegative")
        if self.peak_force_ratio_max < 1.0:
            raise ValueError("peak_force_ratio_max must be at least one")
        if self.effort_ratio_max < 1.0:
            raise ValueError("effort_ratio_max must be at least one")
        if self.normalized_energy_residual_max <= 0.0:
            raise ValueError("normalized_energy_residual_max must be positive")


def _metric_indices(metric_names: tuple[str, ...]) -> dict[str, int]:
    if len(metric_names) != len(set(metric_names)):
        raise ValueError("metric_names must be unique")
    missing = set(REQUIRED_METRICS).difference(metric_names)
    if missing:
        raise ValueError(
            f"metric_names are missing required entries: {sorted(missing)}"
        )
    return {name: metric_names.index(name) for name in REQUIRED_METRICS}


def viability_mask(
    outcomes: npt.ArrayLike,
    baseline: npt.ArrayLike,
    metric_names: tuple[str, ...],
    limits: ViabilityLimits,
) -> BoolArray:
    """Classify every row against common performance, cost, and closure guards.

    The postcondition is one Boolean value per outcome row.  Non-finite rows
    are retained and classified non-viable rather than silently dropped.
    """

    values = np.asarray(outcomes, dtype=np.float64)
    reference = np.asarray(baseline, dtype=np.float64)
    indices = _metric_indices(metric_names)
    if values.ndim != 2 or reference.shape != (values.shape[1],):
        raise ValueError("outcomes must be 2-D and baseline must label one row")
    if values.shape[1] != len(metric_names):
        raise ValueError("metric_names must label every outcome column")
    if not np.all(np.isfinite(reference)):
        raise ValueError("baseline must contain only finite values")
    force_reference = reference[indices["peak_hand_force_n"]]
    effort_reference = reference[indices["effort_proxy_nms"]]
    if force_reference <= 0.0 or effort_reference <= 0.0:
        raise ValueError("baseline force and effort must be positive")

    finite = np.all(np.isfinite(values), axis=1)
    viable = finite.copy()
    viable &= values[:, indices["delivery_speed_m_s"]] >= (
        limits.speed_fraction_min * reference[indices["delivery_speed_m_s"]]
    )
    viable &= values[:, indices["face_path_error_deg"]] <= (
        reference[indices["face_path_error_deg"]] + limits.face_error_allowance_deg
    )
    viable &= values[:, indices["peak_hand_force_n"]] <= (
        limits.peak_force_ratio_max * force_reference
    )
    viable &= values[:, indices["effort_proxy_nms"]] <= (
        limits.effort_ratio_max * effort_reference
    )
    viable &= values[:, indices["normalized_energy_residual"]] <= (
        limits.normalized_energy_residual_max
    )
    if limits.require_sustained_recovery:
        viable &= values[:, indices["returned_to_viable_set"]] >= 0.5
    return viable


def largest_contiguous_width_s(
    phase_offsets_s: npt.ArrayLike, viable: npt.ArrayLike
) -> float:
    """Return the covered span of the largest contiguous viable grid run."""

    offsets = np.asarray(phase_offsets_s, dtype=np.float64)
    mask = np.asarray(viable, dtype=np.bool_)
    if offsets.ndim != 1 or offsets.size < 2 or mask.shape != offsets.shape:
        raise ValueError("phase offsets and viable mask must be equal 1-D arrays")
    if not np.all(np.isfinite(offsets)) or np.any(np.diff(offsets) <= 0.0):
        raise ValueError("phase offsets must be finite and strictly increasing")
    best = 0.0
    start: int | None = None
    for index, item in enumerate(np.append(mask, False)):
        if item and start is None:
            start = index
        elif not item and start is not None:
            best = max(best, float(offsets[index - 1] - offsets[start]))
            start = None
    return best


def summarize_timing_viability(
    phase_offsets_s: npt.ArrayLike,
    outcomes_by_load: npt.ArrayLike,
    baselines_by_load: npt.ArrayLike,
    *,
    load_names: tuple[str, ...],
    metric_names: tuple[str, ...],
    limits: ViabilityLimits,
) -> dict[str, Any]:
    """Summarize load-specific and robust-intersection timing regions."""

    offsets = np.asarray(phase_offsets_s, dtype=np.float64)
    outcomes = np.asarray(outcomes_by_load, dtype=np.float64)
    baselines = np.asarray(baselines_by_load, dtype=np.float64)
    expected = (len(load_names), offsets.size, len(metric_names))
    if outcomes.shape != expected:
        raise ValueError(f"outcomes_by_load must have shape {expected}")
    if baselines.shape != (len(load_names), len(metric_names)):
        raise ValueError("baselines_by_load has an incompatible shape")
    masks = np.empty((len(load_names), offsets.size), dtype=np.bool_)
    per_load: dict[str, Any] = {}
    for index, name in enumerate(load_names):
        if not name.strip():
            raise ValueError("load_names must be nonempty")
        masks[index] = viability_mask(
            outcomes[index], baselines[index], metric_names, limits
        )
        per_load[name] = {
            "viable_mask": masks[index].tolist(),
            "viable_fraction": float(np.mean(masks[index])),
            "contiguous_width_s": largest_contiguous_width_s(offsets, masks[index]),
        }
    robust = np.all(masks, axis=0)
    return {
        "limits": asdict(limits),
        "per_load": per_load,
        "robust_viable_mask": robust.tolist(),
        "robust_viable_fraction": float(np.mean(robust)),
        "robust_contiguous_width_s": largest_contiguous_width_s(offsets, robust),
    }
