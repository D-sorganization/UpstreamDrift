"""Pelvis, upper-trunk and shoulder-girdle segment rotation metrics (#12042).

Event-value view of ``swing_comparison.turn`` for a ``SwingMotion``; re-exported
from ``swing_comparison.metrics`` so existing imports keep working.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.swing_comparison.events import SwingEvents
from src.shared.python.swing_comparison.motion import SwingMotion
from src.shared.python.swing_comparison.turn import (
    LineTurn,
    TurnMetrics,
    marker_turn_lines,
)


@dataclass(frozen=True)
class SegmentRotationMetrics:
    """Pelvis, upper-trunk and shoulder-girdle turn, and X-factor separation.

    All angles in degrees, relative to the same line at address, positive in the
    backswing (see ``swing_comparison.turn`` for the frame convention).  Values
    are NaN where the markers are unavailable; unavailable never means zero.

    The former "thorax yaw" was the shoulder-marker line, which rides the
    scapula.  It is now the ``shoulder_girdle`` line; the BackLeft/BackRight
    ``upper_trunk`` line is separate.  ``thorax_yaw*`` remain as deprecated
    aliases of the shoulder-girdle line.

    Attributes:
        pelvis_yaw / shoulder_girdle_yaw / upper_trunk_yaw: Turn time series (N,).
        x_factor: Upper-trunk minus pelvis turn (N,).
        x_factor_shoulder_girdle: Shoulder-girdle minus pelvis turn (N,).
        *_address / *_top / *_impact: Values at the three swing events.
        x_factor_stretch: Maximum absolute X-factor during the swing (NaN if none).
        turn: The full ``TurnMetrics`` (status, reasons, max backswing).
    """

    pelvis_yaw: np.ndarray
    shoulder_girdle_yaw: np.ndarray
    upper_trunk_yaw: np.ndarray
    x_factor: np.ndarray
    x_factor_shoulder_girdle: np.ndarray
    pelvis_yaw_address: float
    pelvis_yaw_top: float
    pelvis_yaw_impact: float
    shoulder_girdle_yaw_address: float
    shoulder_girdle_yaw_top: float
    shoulder_girdle_yaw_impact: float
    upper_trunk_yaw_address: float
    upper_trunk_yaw_top: float
    upper_trunk_yaw_impact: float
    x_factor_address: float
    x_factor_top: float
    x_factor_impact: float
    x_factor_stretch: float
    turn: TurnMetrics | None = None

    def _deprecated(self, name: str, replacement: str) -> None:
        warnings.warn(
            f"SegmentRotationMetrics.{name} is deprecated: it was the shoulder-"
            f"marker (scapular) line, now named {replacement}; the rib-cage line is "
            "upper_trunk_yaw",
            DeprecationWarning,
            stacklevel=3,
        )

    @property
    def thorax_yaw(self) -> np.ndarray:
        """Deprecated alias of ``shoulder_girdle_yaw``."""
        self._deprecated("thorax_yaw", "shoulder_girdle_yaw")
        return self.shoulder_girdle_yaw

    @property
    def thorax_yaw_address(self) -> float:
        """Deprecated alias of ``shoulder_girdle_yaw_address``."""
        self._deprecated("thorax_yaw_address", "shoulder_girdle_yaw_address")
        return self.shoulder_girdle_yaw_address

    @property
    def thorax_yaw_top(self) -> float:
        """Deprecated alias of ``shoulder_girdle_yaw_top``."""
        self._deprecated("thorax_yaw_top", "shoulder_girdle_yaw_top")
        return self.shoulder_girdle_yaw_top

    @property
    def thorax_yaw_impact(self) -> float:
        """Deprecated alias of ``shoulder_girdle_yaw_impact``."""
        self._deprecated("thorax_yaw_impact", "shoulder_girdle_yaw_impact")
        return self.shoulder_girdle_yaw_impact

    def to_dict(self) -> dict[str, float]:
        gird = {
            f"shoulder_girdle_yaw_{k}_deg": float(
                getattr(self, f"shoulder_girdle_yaw_{k}")
            )
            for k in ("address", "top", "impact")
        }
        return {
            "pelvis_yaw_address_deg": float(self.pelvis_yaw_address),
            "pelvis_yaw_top_deg": float(self.pelvis_yaw_top),
            "pelvis_yaw_impact_deg": float(self.pelvis_yaw_impact),
            **gird,
            "upper_trunk_yaw_address_deg": float(self.upper_trunk_yaw_address),
            "upper_trunk_yaw_top_deg": float(self.upper_trunk_yaw_top),
            "upper_trunk_yaw_impact_deg": float(self.upper_trunk_yaw_impact),
            # Deprecated keys kept for existing consumers; shoulder-girdle values.
            "thorax_yaw_address_deg": gird["shoulder_girdle_yaw_address_deg"],
            "thorax_yaw_top_deg": gird["shoulder_girdle_yaw_top_deg"],
            "thorax_yaw_impact_deg": gird["shoulder_girdle_yaw_impact_deg"],
            "x_factor_address_deg": float(self.x_factor_address),
            "x_factor_top_deg": float(self.x_factor_top),
            "x_factor_impact_deg": float(self.x_factor_impact),
            "x_factor_stretch_deg": float(self.x_factor_stretch),
        }


def compute_segment_rotations(
    motion: SwingMotion,
    events: SwingEvents,
) -> SegmentRotationMetrics:
    """Compute pelvis, upper-trunk and shoulder-girdle turn and X-factor.

    Definitions (see ``swing_comparison.turn``; Z up, golfer faces -X, target -Y):
    - Pelvis: WaistLeft/WaistRight line turn relative to address.
    - Upper trunk: BackLeft/BackRight line turn relative to address.
    - Shoulder girdle: ShoulderBack line turn (rides the scapula) relative to address.
    - X-factor: upper-trunk turn minus pelvis turn; the shoulder-girdle variant is
      ``x_factor_shoulder_girdle``.
    - X-factor stretch: maximum absolute X-factor during the swing.
    A pair with a missing marker or a long gap yields NaN with a reason in
    ``result.turn``, never zero.

    Preconditions:
        - len(motion.t) >= 4
        - 0 <= events.address_idx <= events.top_idx <= events.impact_idx < len(motion.t)

    Returns:
        SegmentRotationMetrics with angle time series and event values.
    """
    n = len(motion.t)
    require(
        0 <= events.address_idx <= events.top_idx <= events.impact_idx < n,
        "Events must satisfy 0 <= address <= top <= impact < n",
    )
    turn = marker_turn_lines(motion.markers, motion.t, events)

    def at(line: LineTurn, idx: int) -> float:
        return float(line.turn_deg[idx])

    xf = turn.x_factor.turn_deg
    stretch = float(np.nanmax(np.abs(xf))) if np.isfinite(xf).any() else float("nan")
    a, tp, im = events.address_idx, events.top_idx, events.impact_idx
    return SegmentRotationMetrics(
        pelvis_yaw=turn.pelvis.turn_deg,
        shoulder_girdle_yaw=turn.shoulder_girdle.turn_deg,
        upper_trunk_yaw=turn.upper_trunk.turn_deg,
        x_factor=xf,
        x_factor_shoulder_girdle=turn.x_factor_shoulder_girdle.turn_deg,
        pelvis_yaw_address=at(turn.pelvis, a),
        pelvis_yaw_top=at(turn.pelvis, tp),
        pelvis_yaw_impact=at(turn.pelvis, im),
        shoulder_girdle_yaw_address=at(turn.shoulder_girdle, a),
        shoulder_girdle_yaw_top=at(turn.shoulder_girdle, tp),
        shoulder_girdle_yaw_impact=at(turn.shoulder_girdle, im),
        upper_trunk_yaw_address=at(turn.upper_trunk, a),
        upper_trunk_yaw_top=at(turn.upper_trunk, tp),
        upper_trunk_yaw_impact=at(turn.upper_trunk, im),
        x_factor_address=at(turn.x_factor, a),
        x_factor_top=at(turn.x_factor, tp),
        x_factor_impact=at(turn.x_factor, im),
        x_factor_stretch=stretch,
        turn=turn,
    )
