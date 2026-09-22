"""Model hand-frame offsets and golfer handedness for club-only IK (CO-03 #10607).

Hand offsets are reviewed model geometry, not copied prose. For a right-handed
golfer the lead (lower) hand is the left hand tip; left-handed swaps frames
while preserving the same lead/trail offset magnitudes along the grip axis.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from src.shared.python.contracts import postcondition, precondition

# Defaults matching DualHandIKSolverSettings grip-Z offsets (metres).
_DEFAULT_LEAD_OFFSET_M = 0.04
_DEFAULT_TRAIL_OFFSET_M = -0.04
_LEFT_TIP = "hand_left_tip"
_RIGHT_TIP = "hand_right_tip"


class GolferHandedness(str, Enum):
    """Golfer handedness used to bind lead/trail hands to model frames."""

    RIGHT = "right"
    LEFT = "left"


@dataclass(frozen=True)
class ModelHandFrameOffsets:
    """Actual model hand-frame names and grip-axis offsets for one model."""

    model_id: str
    handedness: GolferHandedness
    lead_frame_name: str
    trail_frame_name: str
    lead_hand_offset_m: float
    trail_hand_offset_m: float
    geometry_hash: str

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not self.lead_frame_name or not self.trail_frame_name:
            raise ValueError("hand frame names must be non-empty")
        if self.lead_frame_name == self.trail_frame_name:
            raise ValueError("lead and trail frames must differ")
        for name, value in (
            ("lead_hand_offset_m", self.lead_hand_offset_m),
            ("trail_hand_offset_m", self.trail_hand_offset_m),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.lead_hand_offset_m == self.trail_hand_offset_m:
            raise ValueError("lead and trail offsets must differ")
        if not self.geometry_hash:
            raise ValueError("geometry_hash must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "handedness": self.handedness.value,
            "lead_frame_name": self.lead_frame_name,
            "trail_frame_name": self.trail_frame_name,
            "lead_hand_offset_m": self.lead_hand_offset_m,
            "trail_hand_offset_m": self.trail_hand_offset_m,
            "geometry_hash": self.geometry_hash,
        }


def _offset_geometry_hash(
    *,
    model_id: str,
    handedness: GolferHandedness,
    lead_offset_m: float,
    trail_offset_m: float,
    lead_frame: str,
    trail_frame: str,
) -> str:
    payload = {
        "model_id": model_id,
        "handedness": handedness.value,
        "lead_hand_offset_m": lead_offset_m,
        "trail_hand_offset_m": trail_offset_m,
        "lead_frame_name": lead_frame,
        "trail_frame_name": trail_frame,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@precondition(
    lambda model_id, handedness, lead_hand_offset_m=None, trail_hand_offset_m=None: (
        isinstance(model_id, str)
        and bool(model_id)
        and isinstance(handedness, GolferHandedness)
    ),
    "model_id and GolferHandedness required",
)
@postcondition(
    lambda result: isinstance(result, ModelHandFrameOffsets),
    "must return ModelHandFrameOffsets",
)
def resolve_hand_frame_offsets(
    *,
    model_id: str,
    handedness: GolferHandedness,
    lead_hand_offset_m: float | None = None,
    trail_hand_offset_m: float | None = None,
) -> ModelHandFrameOffsets:
    """Bind lead/trail grip offsets to the correct model hand frames.

    Right-handed: lead = left tip, trail = right tip.
    Left-handed: lead = right tip, trail = left tip.
    Offset magnitudes stay on lead/trail roles; only frame names swap.
    """
    lead_offset = (
        _DEFAULT_LEAD_OFFSET_M
        if lead_hand_offset_m is None
        else float(lead_hand_offset_m)
    )
    trail_offset = (
        _DEFAULT_TRAIL_OFFSET_M
        if trail_hand_offset_m is None
        else float(trail_hand_offset_m)
    )
    if handedness is GolferHandedness.RIGHT:
        lead_frame, trail_frame = _LEFT_TIP, _RIGHT_TIP
    elif handedness is GolferHandedness.LEFT:
        lead_frame, trail_frame = _RIGHT_TIP, _LEFT_TIP
    else:
        raise ValueError(f"unsupported handedness: {handedness!r}")

    return ModelHandFrameOffsets(
        model_id=model_id,
        handedness=handedness,
        lead_frame_name=lead_frame,
        trail_frame_name=trail_frame,
        lead_hand_offset_m=lead_offset,
        trail_hand_offset_m=trail_offset,
        geometry_hash=_offset_geometry_hash(
            model_id=model_id,
            handedness=handedness,
            lead_offset_m=lead_offset,
            trail_offset_m=trail_offset,
            lead_frame=lead_frame,
            trail_frame=trail_frame,
        ),
    )
