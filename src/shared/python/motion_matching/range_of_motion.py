"""Human ranges of motion for the full-body coordinates and violation flags.

Matching must never put the golfer where a human cannot go. ``HUMAN_RANGES_DEG``
lists conservative adult ranges (Kapandji, Rajagopal 2016 for the legs) in
the anthropometric document's sign conventions: elbow flexion negative,
right scapula elevation negative, wrist cock positive toward the elbow pit
(radial), wrist Y flexion/extension about the palm normal.
Coordinates that are Euler components of a ball joint (the shoulder gimbal)
have no anatomical range on their own and are absent. ``violations`` flags,
per coordinate, how far a trajectory leaves its range.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

LOWER_LIMB_RANGES_DEG: dict[str, tuple[float, float]] = {
    "hip_flexion": (-30.0, 120.0),
    "hip_adduction": (-50.0, 30.0),
    "hip_rotation": (-40.0, 40.0),
    "knee_angle": (-120.0, 10.0),
    "ankle_angle": (-40.0, 30.0),
    "subtalar_angle": (-20.0, 20.0),
    "mtp_angle": (-30.0, 30.0),
}
UPPER_RANGES_DEG: dict[str, tuple[float, float]] = {
    "SpineInputX": (-35.0, 35.0),
    "SpineInputY": (-45.0, 45.0),
    "TorsoInput": (-100.0, 100.0),
    "NeckInputX": (-45.0, 45.0),
    "NeckInputY": (-60.0, 60.0),
    "NeckInputZ": (-80.0, 80.0),
    "LScapInputX": (-10.0, 30.0),
    "RScapInputX": (-30.0, 10.0),
    "LScapInputY": (-40.0, 40.0),
    "RScapInputY": (-40.0, 40.0),
    "LEInput": (-150.0, 5.0),
    "REInput": (-150.0, 5.0),
    "LFInput": (-90.0, 90.0),
    "RFInput": (-90.0, 90.0),
    "LWInputX": (-40.0, 25.0),  # ulnar .. radial deviation from a neutral grip
    "RWInputX": (-40.0, 25.0),
    "LWInputY": (-70.0, 70.0),  # flexion .. extension
    "RWInputY": (-70.0, 70.0),
}
HUMAN_RANGES_DEG: dict[str, tuple[float, float]] = {
    **UPPER_RANGES_DEG,
    **{
        f"{joint}_{side}": bounds
        for joint, bounds in LOWER_LIMB_RANGES_DEG.items()
        for side in ("r", "l")
    },
}


@dataclass(frozen=True)
class Violation:
    coordinate: str
    max_excess_deg: float
    frames: int
    fraction: float


def violations(
    q: Array,
    coordinate_order: Sequence[str],
    ranges_deg: Mapping[str, tuple[float, float]] = HUMAN_RANGES_DEG,
    tolerance_deg: float = 0.5,
) -> dict[str, Violation]:
    """Coordinates that leave their range by more than ``tolerance_deg``.

    ``q`` is (frames, coordinates) in radians. Precondition: shapes agree.
    Postcondition: only coordinates with a range and at least one offending
    frame are returned, keyed by name.
    """
    values = np.asarray(q, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(coordinate_order):
        raise ValueError("q must be (frames, coordinates) matching the order")
    if tolerance_deg < 0:
        raise ValueError("Tolerance must be nonnegative")
    out: dict[str, Violation] = {}
    for index, name in enumerate(coordinate_order):
        if name not in ranges_deg:
            continue
        lo, hi = ranges_deg[name]
        # Continuous trajectories may carry whole turns (unwrapped Euler
        # angles); ranges are stated within a turn, so compare modulo 360.
        deg = (np.degrees(values[:, index]) + 180.0) % 360.0 - 180.0
        excess = np.maximum(lo - deg, deg - hi)
        bad = excess > tolerance_deg
        if bad.any():
            out[name] = Violation(
                name,
                float(excess.max()),
                int(bad.sum()),
                float(bad.mean()),
            )
    return out


def as_document(
    ranges_deg: Mapping[str, tuple[float, float]],
) -> dict[str, list[float]]:
    """JSON form of a range table."""
    return {name: [float(lo), float(hi)] for name, (lo, hi) in ranges_deg.items()}
