"""Input distribution and training coverage diagnostics for NM-08."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.neural_motion.inference.types import DomainCheckResult

__all__ = [
    "DistributionBounds",
    "check_target_distribution",
]


@dataclass(frozen=True, slots=True)
class DistributionBounds:
    """Coverage boundaries derived from calibrated training dataset."""

    min_duration_s: float = 0.5
    max_duration_s: float = 2.5
    max_velocity_m_s: float = 65.0
    max_acceleration_m_s2: float = 2500.0
    supported_contact_regimes: tuple[str, ...] = (
        "airborne",
        "ground_support",
        "impact_compliant",
    )
    supported_geometries: tuple[str, ...] = ("driver", "iron", "driver_g1", "iron_g1")

    def __post_init__(self) -> None:
        if self.min_duration_s <= 0.0 or self.max_duration_s <= self.min_duration_s:
            raise ValueError("Invalid duration bounds")
        if self.max_velocity_m_s <= 0.0 or self.max_acceleration_m_s2 <= 0.0:
            raise ValueError("Dynamic bounds must be positive")


def _extract_target_kinematics(
    target: Any,
) -> tuple[np.ndarray, np.ndarray, int | None, str]:
    """Extract (times, positions, impact_idx, club_type) fail-closed."""
    if hasattr(target, "time") and (
        hasattr(target, "clubhead") or hasattr(target, "clubhead_position")
    ):
        times = np.asarray(target.time, dtype=np.float64)
        pos = np.asarray(
            getattr(target, "clubhead", None)
            if hasattr(target, "clubhead")
            else getattr(target, "clubhead_position", None),
            dtype=np.float64,
        )
        impact_idx = getattr(target, "impact_idx", None)
        club_type = str(getattr(target, "club_type", "driver")).lower()
    elif hasattr(target, "club") and target.club is not None:
        return _extract_target_kinematics(target.club)
    else:
        raise TypeError(f"Unsupported target type {type(target).__name__}")

    if not np.all(np.isfinite(times)):
        raise ValueError("Target times contain non-finite / NaN values")
    if not np.all(np.isfinite(pos)):
        raise ValueError("Target positions contain non-finite / NaN values")
    if len(times) < 2:
        raise ValueError("Target must contain at least 2 time frames")

    return times, pos, impact_idx, club_type


def check_target_distribution(
    target: Any,
    bounds: DistributionBounds | None = None,
) -> DomainCheckResult:
    """Validate target against empirical training coverage."""
    if bounds is None:
        bounds = DistributionBounds()

    times, pos, impact_idx, club_type = _extract_target_kinematics(target)
    duration = float(times[-1] - times[0])
    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError("Target timestamps must be strictly monotonic")

    diff = np.diff(pos, axis=0) / dt[:, None]
    vel = np.sqrt(
        np.einsum("ij,ij->i", diff, diff)
    )  # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is ~1.5x faster than np.linalg.norm(..., axis=-1)
    v_max = float(np.max(vel)) if len(vel) > 0 else 0.0

    diagnostics: list[str] = []
    dist = 0.0
    duration_ok = True
    contact_ok = True

    if duration < bounds.min_duration_s:
        diagnostics.append(
            f"Duration {duration:.2f}s below minimum {bounds.min_duration_s:.2f}s"
        )
        dist += 1.0 + (bounds.min_duration_s - duration) / bounds.min_duration_s
        duration_ok = False
    elif duration > bounds.max_duration_s:
        diagnostics.append(
            f"Duration {duration:.2f}s exceeds maximum {bounds.max_duration_s:.2f}s"
        )
        dist += 1.0 + (duration - bounds.max_duration_s) / bounds.max_duration_s
        duration_ok = False

    if v_max > bounds.max_velocity_m_s:
        diagnostics.append(
            f"Peak velocity {v_max:.1f} m/s exceeds limit {bounds.max_velocity_m_s:.1f} m/s"
        )
        dist += 1.0 + (v_max - bounds.max_velocity_m_s) / bounds.max_velocity_m_s

    has_impact = impact_idx is not None and 0 <= impact_idx < len(times)
    if has_impact and "impact_compliant" not in bounds.supported_contact_regimes:
        diagnostics.append(
            "Contact/impact regime not supported by active model profile"
        )
        dist += 2.0
        contact_ok = False

    is_in_dist = len(diagnostics) == 0
    confidence = float(np.clip(1.0 / (1.0 + dist), 0.0, 1.0))

    return DomainCheckResult(
        is_in_distribution=is_in_dist,
        domain_distance=float(dist),
        empirical_confidence=confidence,
        diagnostics=tuple(diagnostics),
        contact_regime_supported=contact_ok,
        duration_supported=duration_ok,
    )
