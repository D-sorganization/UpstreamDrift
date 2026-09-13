"""Select timed interior state guesses without overriding the initial condition."""

from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
from numpy.typing import NDArray


def select_shooting_states(
    document: Mapping[str, Any],
    requested_times: Sequence[float],
    *,
    model_sha256: str,
    dimension: int,
) -> dict[float, NDArray[np.float64]]:
    """Return detached q/v seeds; caller must verify engine closure independently.

    Samples must match within 1e-12 seconds. No interpolation, acceleration
    inference, dynamic-feasibility claim or initial-state replacement occurs.
    """
    if (
        document.get("model_sha256") != model_sha256
        or type(dimension) is not int
        or dimension < 1
    ):
        raise ValueError("Model identity or state dimension is invalid")
    times = np.asarray(document["times_s"], dtype=float)
    q = np.asarray(document["coordinates"], dtype=float)
    v = np.asarray(document["rates"], dtype=float)
    requested = np.asarray(requested_times, dtype=float)
    if (
        times.ndim != 1
        or not times.size
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Seed clock must be finite and strictly ordered")
    if (
        q.shape != (len(times), dimension)
        or v.shape != q.shape
        or not np.isfinite(q).all()
        or not np.isfinite(v).all()
    ):
        raise ValueError("Seed positions and rates must have finite matching shapes")
    if (
        requested.ndim != 1
        or not requested.size
        or not np.isfinite(requested).all()
        or np.any(requested <= 0)
        or np.any(np.diff(requested) <= 0)
    ):
        raise ValueError(
            "Only strictly ordered positive interior node times are allowed"
        )
    result = {}
    for time in requested:
        matches = np.flatnonzero(np.isclose(times, time, atol=1e-12, rtol=0))
        if len(matches) != 1:
            raise ValueError("Requested node must identify exactly one seed sample")
        index = matches[0]
        result[float(time)] = np.concatenate((q[index], v[index]))
    return result
