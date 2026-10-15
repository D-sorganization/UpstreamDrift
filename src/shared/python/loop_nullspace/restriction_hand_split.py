"""Hand-split envelope quantities under a restricted reaction set (NSX-26).

The split coordinates are the affine closing-hand map of NSX-16
(:func:`closing_hand_map`) evaluated on the closure multiplier.  The
multiplier ``lambda`` lives at the closure point ``P`` of the grip frame; the
hand grip points lie on the shaft at signed coordinates ``x_L, x_R`` (grip x
axis) from ``P``.  Statics fixes the transport:

* the closing hand's wrench on the club is ``-lambda``, so the hand wrench
  about its own grip point is ``-(f, t - x_c e_x x f)`` with ``x_c`` the
  closing hand's coordinate;
* the net wrench ``(F, M_P)`` about ``P`` moves to the grip midpoint
  ``o = (x_L + x_R)/2 e_x`` as ``(F, M_P - o x F)``.

A dropped closure row is a multiplier fixed at zero, so the restricted
quantity is the same affine functional on the embedded multiplier.  Nothing
here is evidence about the true grip mechanics.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.contracts import ClosedLoopSnapshot, ClosureKind
from src.shared.python.loop_nullspace.envelopes import EnvelopeQuantity
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.grip_split import (
    SPLIT_COLUMN_NAMES,
    SPLIT_UNITS,
    closing_hand_map,
)
from src.shared.python.loop_nullspace.restriction import restriction_rows

__all__ = ["HandSplitFrame", "hand_split_quantities", "hand_split_values"]

_EX = np.array([1.0, 0.0, 0.0])


@dataclass(frozen=True, eq=False)
class HandSplitFrame:
    """Grip geometry and net wrench of one frame (grip axes, shaft on ``e_x``).

    ``net_wrench`` is the net hand wrench ``(F, M_P)`` on the club about the
    closure point ``P``; ``closing_hand`` is the hand whose wrench is the
    closure multiplier.  Raises ``ValueError`` for coincident hands, a wrong
    ``net_wrench`` shape or an unknown hand.
    """

    x_left: float
    x_right: float
    net_wrench: ArrayLike
    closing_hand: str = "R"

    def __post_init__(self) -> None:
        net = np.asarray(self.net_wrench, dtype=float)
        if net.shape != (6,) or not np.all(np.isfinite(net)):
            raise ValueError("net_wrench must be a finite 6-vector")
        if self.closing_hand not in ("L", "R"):
            raise ValueError(
                f"closing_hand must be 'L' or 'R', got {self.closing_hand!r}"
            )
        if not (np.isfinite(self.x_left) and np.isfinite(self.x_right)):
            raise ValueError("hand coordinates must be finite")
        if abs(self.x_right - self.x_left) <= 1e-12:
            raise ValueError("the two hand grip points must be distinct")
        object.__setattr__(self, "net_wrench", net)


def _transport(frame: HandSplitFrame) -> tuple[np.ndarray, np.ndarray]:
    """Affine map ``s = A lambda_P + s0`` from the multiplier at ``P``."""
    x_c = frame.x_right if frame.closing_hand == "R" else frame.x_left
    skew = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -x_c], [0.0, x_c, 0.0]])
    shift = np.block([[np.eye(3), np.zeros((3, 3))], [-skew, np.eye(3)]])
    origin = 0.5 * (frame.x_left + frame.x_right) * _EX
    net = np.asarray(frame.net_wrench, dtype=float)
    net_o = np.concatenate([net[:3], net[3:] - np.cross(origin, net[:3])])
    half = 0.5 * (frame.x_right - frame.x_left) * _EX
    a_map, s0 = closing_hand_map(frame.closing_hand, half, net_o)
    return a_map @ shift, s0


def hand_split_values(frame: HandSplitFrame, lam_p: ArrayLike) -> np.ndarray:
    """Hand-split coordinates ``s`` for the full 6-row multiplier at ``P``."""
    lam = np.asarray(lam_p, dtype=float)
    if lam.shape != (6,):
        raise ValueError("lam_p must be the 6-row multiplier at the closure point")
    a_map, s0 = _transport(frame)
    return np.asarray(a_map @ lam + s0)


def hand_split_quantities(
    frame_of: Callable[[ClosedLoopSnapshot], HandSplitFrame],
) -> Callable[[FeasibleSet, ClosedLoopSnapshot], Sequence[EnvelopeQuantity]]:
    """Quantity builder of the six hand-split coordinates for each frame.

    ``frame_of(snapshot)`` returns the frame geometry.  The restricted
    snapshot's closure kind selects the embedded multiplier rows, so the same
    ``frame_of`` serves every kind.
    """

    def build(fs: FeasibleSet, snapshot: ClosedLoopSnapshot) -> list[EnvelopeQuantity]:
        dec = fs.decomposition
        if dec is None:
            raise ValueError("this FeasibleSet carries no decomposition")
        rows = list(restriction_rows(ClosureKind.WELD6, snapshot.topology.closure_kind))
        a_map, s0 = _transport(frame_of(snapshot))
        n_a = dec.n_actuated
        select = np.zeros((6, dec.dim_x))
        select[:, [n_a + k for k in range(len(rows))]] = a_map[:, rows]
        scales = (dec.scales.force,) * 3 + (dec.scales.moment,) * 3
        return [
            EnvelopeQuantity(
                SPLIT_COLUMN_NAMES[k],
                select[k] @ dec.N,
                float(select[k] @ dec.x_p + s0[k]),
                SPLIT_UNITS[k],
                scales[k],
            )
            for k in range(6)
        ]

    return build
