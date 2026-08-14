"""Trajectory metrics: dig/skid, depth trace, divot profile (issue #8614).

These metrics characterize the club's path through the sand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.core.contracts import require

__all__ = [
    "DivotProfile",
    "TrajectoryMetrics",
    "compute_trajectory_metrics",
]


@dataclass(frozen=True, slots=True)
class DivotProfile:
    """Geometric profile of the divot carved by the club.

    Attributes:
        entry_x_m: Horizontal position where club enters sand [m].
        entry_time_s: Time of entry [s].
        max_depth_m: Maximum depth below surface [m] (positive downward).
        max_depth_x_m: Horizontal position of maximum depth [m].
        max_depth_time_s: Time of maximum depth [s].
        exit_x_m: Horizontal position where club exits sand [m].
        exit_time_s: Time of exit [s].
        length_m: Horizontal distance from entry to exit [m].
        volume_m3: Estimated divot volume [m^3].
        mass_kg: Estimated displaced sand mass [kg].
    """

    entry_x_m: float
    entry_time_s: float
    max_depth_m: float
    max_depth_x_m: float
    max_depth_time_s: float
    exit_x_m: float
    exit_time_s: float
    length_m: float
    volume_m3: float
    mass_kg: float


@dataclass(frozen=True, slots=True)
class TrajectoryMetrics:
    """Complete trajectory analysis.

    Attributes:
        is_digging: True if club depth is increasing (going deeper).
        is_skidding: True if club is sliding at constant depth.
        dig_skid_index: -1 = pure dig, +1 = pure skid, 0 = balanced.
        divot: Geometric divot profile.
        depth_trace_m: Depth vs time (positive = below surface).
        distance_trace_m: Horizontal distance vs time.
        time_trace_s: Time array for traces.
    """

    is_digging: bool
    is_skidding: bool
    dig_skid_index: float
    divot: DivotProfile
    depth_trace_m: np.ndarray
    distance_trace_m: np.ndarray
    time_trace_s: np.ndarray


def compute_trajectory_metrics(
    t: np.ndarray,
    positions: np.ndarray,
    forces: np.ndarray,
    head_mass_kg: float,
    *,
    sole_width_m: float = 0.015,
    bulk_density_kg_m3: float = 1550.0,
) -> TrajectoryMetrics:
    """Compute trajectory metrics from clubhead position trace.

    Args:
        t: Time array (T,) [s].
        positions: Clubhead positions (T, 3) [m], z positive upward.
        forces: Contact forces (T, 3) [N].
        head_mass_kg: Clubhead mass [kg].
        sole_width_m: Sole width for volume estimate [m].
        bulk_density_kg_m3: Sand bulk density for mass estimate [kg/m^3].

    Returns:
        TrajectoryMetrics with dig/skid analysis and divot profile.
    """
    require(len(t) == len(positions), "t and positions must have same length")
    require(len(t) == len(forces), "t and forces must have same length")
    require(head_mass_kg > 0, "head mass must be positive")

    if len(t) == 0:
        empty_divot = DivotProfile(
            entry_x_m=0.0,
            entry_time_s=0.0,
            max_depth_m=0.0,
            max_depth_x_m=0.0,
            max_depth_time_s=0.0,
            exit_x_m=0.0,
            exit_time_s=0.0,
            length_m=0.0,
            volume_m3=0.0,
            mass_kg=0.0,
        )
        return TrajectoryMetrics(
            is_digging=False,
            is_skidding=False,
            dig_skid_index=0.0,
            divot=empty_divot,
            depth_trace_m=np.array([]),
            distance_trace_m=np.array([]),
            time_trace_s=np.array([]),
        )

    positions = np.asarray(positions)
    t = np.asarray(t)

    x = positions[:, 0]
    z = positions[:, 2]

    depth = np.maximum(0.0, -z)
    distance = x - x[0]

    entry_x, entry_t = _find_entry_point(t, x, z)
    max_depth, max_depth_x, max_depth_t, max_idx = _find_max_depth(t, x, z)
    exit_x, exit_t = _find_exit_point(t, x, z)

    length = exit_x - entry_x if exit_x > entry_x else 0.0

    is_digging, is_skidding, dig_skid_index = _compute_dig_skid(t, z, max_idx)

    volume = _estimate_divot_volume(depth, distance, sole_width_m)
    mass = volume * bulk_density_kg_m3

    divot = DivotProfile(
        entry_x_m=entry_x,
        entry_time_s=entry_t,
        max_depth_m=max_depth,
        max_depth_x_m=max_depth_x,
        max_depth_time_s=max_depth_t,
        exit_x_m=exit_x,
        exit_time_s=exit_t,
        length_m=length,
        volume_m3=volume,
        mass_kg=mass,
    )

    return TrajectoryMetrics(
        is_digging=is_digging,
        is_skidding=is_skidding,
        dig_skid_index=dig_skid_index,
        divot=divot,
        depth_trace_m=depth,
        distance_trace_m=distance,
        time_trace_s=t,
    )


def _find_entry_point(
    t: np.ndarray, x: np.ndarray, z: np.ndarray
) -> tuple[float, float]:
    """Find where the club first enters the sand (z crosses 0 going down)."""
    for i in range(len(z) - 1):
        if z[i] >= 0 and z[i + 1] < 0:
            frac = z[i] / (z[i] - z[i + 1]) if z[i] != z[i + 1] else 0.0
            entry_x = x[i] + frac * (x[i + 1] - x[i])
            entry_t = t[i] + frac * (t[i + 1] - t[i])
            return entry_x, entry_t
    if z[0] < 0:
        return float(x[0]), float(t[0])
    return float(x[-1]), float(t[-1])


def _find_max_depth(
    t: np.ndarray, x: np.ndarray, z: np.ndarray
) -> tuple[float, float, float, int]:
    """Find the maximum depth (lowest z) point."""
    depth = -z
    max_idx = int(np.argmax(depth))
    max_depth = max(0.0, float(depth[max_idx]))
    return max_depth, float(x[max_idx]), float(t[max_idx]), max_idx


def _find_exit_point(
    t: np.ndarray, x: np.ndarray, z: np.ndarray
) -> tuple[float, float]:
    """Find where the club exits the sand (z crosses 0 going up)."""
    for i in range(len(z) - 1, 0, -1):
        if z[i] >= 0 and z[i - 1] < 0:
            frac = -z[i - 1] / (z[i] - z[i - 1]) if z[i] != z[i - 1] else 0.0
            exit_x = x[i - 1] + frac * (x[i] - x[i - 1])
            exit_t = t[i - 1] + frac * (t[i] - t[i - 1])
            return exit_x, exit_t
    if z[-1] < 0:
        return float(x[-1]), float(t[-1])
    return float(x[0]), float(t[0])


def _compute_dig_skid(
    t: np.ndarray, z: np.ndarray, max_idx: int
) -> tuple[bool, bool, float]:
    """Determine if the club is digging or skidding."""
    if len(t) < 2:
        return False, False, 0.0

    submerged = z < 0
    if not np.any(submerged):
        return False, False, 0.0

    dz = np.diff(z)
    dt = np.diff(t)
    dt = np.where(dt > 0, dt, 1e-9)
    vz = dz / dt

    digging_mask = (vz < -0.001) & submerged[:-1]
    skidding_mask = (np.abs(vz) < 0.001) & submerged[:-1]

    dig_time = np.sum(dt[digging_mask])
    skid_time = np.sum(dt[skidding_mask])
    total_contact = np.sum(dt[submerged[:-1]])

    is_digging = dig_time > 0.3 * total_contact if total_contact > 0 else False
    is_skidding = skid_time > 0.3 * total_contact if total_contact > 0 else False

    if total_contact > 0:
        dig_skid_index = (skid_time - dig_time) / total_contact
    else:
        dig_skid_index = 0.0

    return is_digging, is_skidding, float(np.clip(dig_skid_index, -1.0, 1.0))


def _estimate_divot_volume(
    depth: np.ndarray, distance: np.ndarray, sole_width_m: float
) -> float:
    """Estimate divot volume using trapezoidal integration."""
    if len(depth) < 2:
        return 0.0

    dx = np.diff(distance)
    avg_depth = (depth[:-1] + depth[1:]) / 2.0

    volume = float(np.sum(avg_depth * np.abs(dx) * sole_width_m))
    return max(0.0, volume)
