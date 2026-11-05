"""Which reference sample counts as impact (SHOW-C3c, #14559, supervisor D44).

The same-input bundle records no ball impact (``ball_impact`` is ``None``), so
the harness chooses the sample from the clubhead path itself:

* with a ball position: the first sample where the clubhead centre crosses the
  ball plane, i.e. where the signed distance along the target-normal axis
  changes sign (or reaches zero) for the first time;
* without one: the bundle's recorded impact event (D52), because world height
  is not height above the ball plane in these bundles (minimum height lands
  mid-downswing); with neither, the rule refuses instead of guessing.

Speed, path and face are then read AT that sample, never as a maximum over a
window.  Samples 1324 to 1330 of the v2 reference are an impact-step transient
(speed 94 to 157 mph, path +15 to -22 deg within 6 ms); the receipt discloses it.

Design by Contract:
    Preconditions: ``head_pos`` is (T, 3) with T >= 2; a given ball position
    is crossed by the path; without a ball an event index inside the path is
    given.
    Postconditions: the index is in ``[0, T)``.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    event_index: int | None = None,
) -> int:
    """Index of the impact sample of a (T, 3) clubhead-centre path."""
    pos = np.asarray(head_pos, dtype=float)
    require(
        pos.ndim == 2 and pos.shape[1] == 3 and pos.shape[0] >= 2,
        "head_pos must be (T >= 2, 3)",
        pos.shape,
    )
    if ball_pos is None or target_normal is None:
        require(
            event_index is not None,
            "no ball position and no recorded impact event: refusing to guess",
        )
        idx = int(event_index)  # type: ignore[arg-type]
        require(
            0 <= idx < pos.shape[0],
            "recorded impact event is outside the path",
            (idx, pos.shape[0]),
        )
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


def recorded_impact_frame(receipt_path: Path) -> int | None:
    """The ``event_frames.impact`` bundle step of a receipt, or None if absent."""
    if not receipt_path.is_file():
        return None
    frames = json.loads(receipt_path.read_text(encoding="utf-8")).get(
        "event_frames", {}
    )
    if "impact" not in frames:
        return None
    frame = frames["impact"]
    require(
        isinstance(frame, int) and not isinstance(frame, bool) and frame >= 0,
        "receipt event_frames.impact must be a non-negative integer",
        frame,
    )
    return int(frame)
