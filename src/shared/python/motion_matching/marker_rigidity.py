"""Optimistic marker error bound with every attachment frame independently rigid."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.body_part_viz.fitters._kabsch import kabsch_rotation


def rigid_marker_lower_bound(
    offsets_m: NDArray[np.float64],
    observed_m: NDArray[np.float64],
    frames: Sequence[str],
    valid: NDArray[np.bool_],
) -> dict[str, Any]:
    """Relax all joints/closures; fit each fixed constellation without scaling.

    The result is a lower bound for these fixed offsets and equal marker weights,
    not for a model with recalibrated offsets or changing marker assignments.
    """
    offsets, points, mask = (
        np.asarray(offsets_m),
        np.asarray(observed_m),
        np.asarray(valid),
    )
    if (
        offsets.ndim != 2
        or offsets.shape[1] != 3
        or points.shape != offsets.shape
        or len(frames) != len(offsets)
        or mask.shape != (len(offsets),)
        or mask.dtype != np.bool_
        or not mask.any()
        or any(not isinstance(f, str) or not f for f in frames)
        or not np.isfinite(offsets).all()
        or not np.isfinite(points[mask]).all()
    ):
        raise ValueError("Invalid fixed marker constellations or observations")
    groups = []
    sum_squared = 0.0
    frame_array = np.asarray(frames)
    for frame in dict.fromkeys(frames):
        selected = (frame_array == frame) & mask
        count = int(selected.sum())
        if not count:
            continue
        source, target = offsets[selected], points[selected]
        p, q = source - source.mean(axis=0), target - target.mean(axis=0)
        rotation = kabsch_rotation(p, q)
        squared = float(np.sum((p @ rotation.T - q) ** 2))
        sum_squared += squared
        groups.append(
            {
                "frame": frame,
                "count": count,
                "rms_lower_bound_m": float(np.sqrt(squared / count)),
            }
        )
    return {
        "observed_count": int(mask.sum()),
        "rms_lower_bound_m": float(np.sqrt(sum_squared / mask.sum())),
        "groups": groups,
    }
