"""Which reference sample counts as impact (SHOW-C3c, #14559, supervisor D44).

The same-input bundle records no ball impact (``ball_impact`` is ``None``), so
the harness chooses the sample from the clubhead path itself:

* with a ball position: the first sample where the clubhead centre crosses the
  ball plane, i.e. where the signed distance along the target-normal axis
  changes sign (or reaches zero) for the first time;
* without one: the sample of minimum clubhead height (the first, on ties).

Speed, path and face are then read AT that sample, never as a maximum over a
window.  Samples 1324 to 1330 of the v2 reference are an impact-step transient
(speed 94 to 157 mph, path +15 to -22 deg within 6 ms); the receipt discloses it.

Design by Contract:
    Preconditions: ``head_pos`` is (T, 3) with T >= 2; a given ball position
    is crossed by the path.
    Postconditions: the index is in ``[0, T)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require

Array = NDArray[np.float64]

TRANSIENT_NOTE = (
    "reference samples 1324 to 1330 are an impact-step transient (clubhead "
    "speed 94 to 157 mph, path +15 to -22 deg within 6 ms); the impact sample "
    "is a point read, not a max, and is sensitive to that transient"
)


def impact_sample(
    head_pos: Array,
    ball_pos: Array | None = None,
    target_normal: Array | None = None,
) -> int:
    """Index of the impact sample of a (T, 3) clubhead-centre path."""
    pos = np.asarray(head_pos, dtype=float)
    require(
        pos.ndim == 2 and pos.shape[1] == 3 and pos.shape[0] >= 2,
        "head_pos must be (T >= 2, 3)",
        pos.shape,
    )
    if ball_pos is None or target_normal is None:
        idx = int(np.argmin(pos[:, 2]))  # first minimum on ties
    else:
        normal = np.asarray(target_normal, dtype=float)
        normal = normal / float(np.linalg.norm(normal))
        dist = (pos - np.asarray(ball_pos, dtype=float)) @ normal
        side = np.sign(dist[0]) or 1.0
        crossed = np.nonzero(side * dist <= 0.0)[0]
        require(crossed.size > 0, "the clubhead never reaches the ball plane")
        idx = int(crossed[0])
    ensure(0 <= idx < pos.shape[0], "impact index out of range", idx)
    return idx
