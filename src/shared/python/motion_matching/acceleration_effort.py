"""Allocate efforts through a constrained forward-acceleration response."""

from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class AccelerationEffortResult(NamedTuple):
    """Allocation and residual; an unreachable component is never hidden."""

    effort: Array
    achieved_acceleration: Array
    acceleration_error: Array
    response_rank: int


def allocate_acceleration_effort(
    response: Array,
    free_acceleration: Array,
    desired_acceleration: Array,
    *,
    acceleration_scales: Array,
    effort_scales: Array,
    rcond: float = 1e-8,
) -> AccelerationEffortResult:
    """Minimize scaled acceleration error, then scaled effort norm.

    The engine supplies a0 and B from its constrained forward dynamics at the
    CURRENT state, so a=a0+B*u retains constraint reactions. This does not
    assume a prescribed trajectory is dynamically feasible or enforce effort
    bounds. The caller must forward-integrate and audit the resulting motion.
    """
    matrix = np.asarray(response, dtype=float)
    free, desired, scales, controls = [
        np.asarray(x, dtype=float)
        for x in (
            free_acceleration,
            desired_acceleration,
            acceleration_scales,
            effort_scales,
        )
    ]
    if (
        matrix.ndim != 2
        or not all(matrix.shape)
        or free.shape != (matrix.shape[0],)
        or desired.shape != free.shape
        or scales.shape != free.shape
        or controls.shape != (matrix.shape[1],)
    ):
        raise ValueError("Acceleration response and vector shapes must agree")
    if (
        not all(np.isfinite(x).all() for x in (matrix, free, desired, scales, controls))
        or np.any(scales <= 0)
        or np.any(controls <= 0)
        or not np.isfinite(rcond)
        or not 0 < rcond < 1
    ):
        raise ValueError(
            "Finite data, positive scales and a relative rank cutoff are required"
        )
    scaled = matrix * controls[None, :] / scales[:, None]
    normalized, _, rank, _ = np.linalg.lstsq(
        scaled, (desired - free) / scales, rcond=rcond
    )
    effort = normalized * controls
    achieved = free + matrix @ effort
    error = achieved - desired
    for value in (effort, achieved, error):
        if not np.isfinite(value).all():
            raise ValueError("Effort allocation overflowed")
        value.setflags(write=False)
    return AccelerationEffortResult(effort, achieved, error, int(rank))
