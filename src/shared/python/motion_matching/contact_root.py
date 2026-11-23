"""Refine an already discovered contact bracket without interpolating states.

The caller owns physical signed clearance, finite-face inclusion and material
point velocity, and must evaluate the same continuous integrated trajectory at
every time. This kernel neither discovers brackets nor certifies first contact.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq


@dataclass(frozen=True)
class ContactSample:
    """Instantaneous state, signed clearance, interior flag and approach speed.

    Clearance is positive before contact; approach speed is positive towards
    the ball. The flag must include front-side and finite-face checks.
    """

    q: NDArray[np.float64]
    qd: NDArray[np.float64]
    clearance_m: float
    on_face: bool
    approach_mps: float


@dataclass(frozen=True)
class ContactRoot:
    time_s: float
    sample: ContactSample


def locate_contact_root(
    evaluate: Callable[[float], ContactSample],
    start_s: float,
    end_s: float,
    *,
    time_tolerance_s: float = 1e-9,
    distance_tolerance_m: float = 1e-6,
    min_approach_mps: float = 1e-7,
) -> ContactRoot | None:
    """Return an accepted root, or None for an unresolved/ineligible bracket.

    A positive-to-nonpositive endpoint sign change is necessary but does not
    prove continuity, uniqueness, or absence of earlier contact. The residual
    check rejects a sign jump, and geometry and approach are checked at the
    re-evaluated root. A caller must independently establish scan coverage.
    """
    params = (start_s, end_s, time_tolerance_s, distance_tolerance_m, min_approach_mps)
    if not all(math.isfinite(v) for v in params):
        raise ValueError("times and tolerances must be finite")
    if not (start_s < end_s and 0.0 < time_tolerance_s < end_s - start_s):
        raise ValueError("invalid bracket or time tolerance")
    if distance_tolerance_m <= 0.0 or min_approach_mps < 0.0:
        raise ValueError("invalid distance or approach tolerance")

    shapes: tuple[tuple[int, ...], tuple[int, ...]] | None = None

    def checked(t: float) -> ContactSample:
        nonlocal shapes
        value = evaluate(t)
        q = np.asarray(value.q, dtype=float)
        qd = np.asarray(value.qd, dtype=float)
        if (
            q.ndim != 1
            or qd.ndim != 1
            or not q.size
            or not qd.size
            or not np.isfinite(q).all()
            or not np.isfinite(qd).all()
            or not math.isfinite(value.clearance_m)
            or not math.isfinite(value.approach_mps)
        ):
            raise ValueError(
                "contact samples must contain finite vector states and scalars"
            )
        if not isinstance(value.on_face, (bool, np.bool_)):
            raise ValueError("on_face must be a boolean")
        current_shapes = (q.shape, qd.shape)
        if shapes is None:
            shapes = current_shapes
        elif shapes != current_shapes:
            raise ValueError("contact sample state dimensions changed")
        return ContactSample(
            q.copy(),
            qd.copy(),
            float(value.clearance_m),
            bool(value.on_face),
            float(value.approach_mps),
        )

    left, right = checked(start_s), checked(end_s)
    if left.clearance_m <= 0.0 or right.clearance_m > 0.0:
        return None
    try:
        t = float(
            brentq(
                lambda x: checked(x).clearance_m, start_s, end_s, xtol=time_tolerance_s
            )
        )
    except RuntimeError:
        return None
    root = checked(t)
    if (
        abs(root.clearance_m) > distance_tolerance_m
        or not root.on_face
        or root.approach_mps <= min_approach_mps
    ):
        return None
    return ContactRoot(t, root)
