"""Kinematic analysis services for C3D data."""

import numpy as np
import numpy.typing as npt


def compute_marker_statistics(
    time: npt.NDArray[np.float64] | None, pos: npt.NDArray[np.float64]
) -> dict[str, float]:
    """
    Compute basic kinematic quantities for a single marker trajectory:
    - total path length
    - max speed
    - mean speed
    """
    if pos is None:
        raise ValueError("pos must be provided")
    if pos.shape[0] < 2 or time is None or len(time) != pos.shape[0]:
        return {
            "path_length": np.nan,
            "max_speed": np.nan,
            "mean_speed": np.nan,
        }

    dt = np.diff(time)
    dt[dt <= 0] = np.nan  # avoid division by zero

    disp = np.diff(pos, axis=0)  # (N-1, 3)
    # ⚡ Bolt: einsum is ~2x faster than sum(square(..., dtype=float), axis=1)
    # and avoids casting errors or temporary array allocations
    segment_length = np.sqrt(np.einsum("...i,...i->...", disp, disp, dtype=float))

    # Calculate speed, handling Potential NaN from dt logic
    # Speed is segment_length / dt
    # If dt is NaN, speed implies NaN.
    # But existing implementation was: speed = segment_length / dt
    # We stick to the logic but ensure consistent types

    with np.errstate(invalid="ignore", divide="ignore"):
        speed = segment_length / dt

    path_length = np.nansum(segment_length)

    # Check if we have valid speed data before computing stats
    # speed might be all NaNs if dt was all NaNs
    valid_speed = speed[np.isfinite(speed)]

    if valid_speed.size > 0:
        max_speed = np.max(valid_speed)
        mean_speed = np.mean(valid_speed)
    else:
        max_speed = np.nan
        mean_speed = np.nan

    return {
        "path_length": float(path_length),
        "max_speed": float(max_speed),
        "mean_speed": float(mean_speed),
    }
