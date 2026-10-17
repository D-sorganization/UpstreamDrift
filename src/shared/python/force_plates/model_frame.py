"""Plate placement from a model's declared target frame (OF-03, #13083).

The repository has more than one world convention (ADR-0041's native world has
the target along ``-Y``; the full-body spec world has it along ``+Y``), so the
target direction is never assumed: it comes from the model's
:class:`~src.shared.python.impact_parameters.target_frame.TargetFrame`.

open-fp's stance frame has ``+x`` from plate L toward plate R and ``+y`` toward
the ball (the golfer faces ``+y``).  The target therefore lies along stance
``-x`` for a right-handed golfer and ``+x`` for a left-handed one.  The
placement yaw is the unique rotation about ``+Z`` that turns that stance
direction onto ``target_dir``: a proper rotation in both cases, never a
reflection, so anatomical left and right stay left and right.  A trial whose
handedness disagrees with the target frame is refused, never mirrored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast, get_args

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.force_plates.contracts import Handedness
from src.shared.python.force_plates.placement import ROTATION_ATOL, PlatePlacement
from src.shared.python.impact_parameters.target_frame import TargetFrame

__all__ = ["TargetedPlacement", "placement_for_target"]

#: Stance-frame direction of the target for each handedness.
_STANCE_TARGET: dict[str, tuple[float, float]] = {"RH": (-1.0, 0.0), "LH": (1.0, 0.0)}
#: Trial handedness -> :class:`TargetFrame` handedness.
_FRAME_HANDEDNESS: dict[str, str] = {"RH": "right", "LH": "left"}
_ZHAT = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True, eq=False)
class TargetedPlacement:
    """A plate placement together with the target frame it was built for.

    ``frame_id`` is the :attr:`TargetFrame.frame_id`, so a later comparison can
    refuse to mix worlds; ``handedness`` is the trial's ``"RH"``/``"LH"``.
    """

    placement: PlatePlacement
    frame_id: str
    handedness: Handedness


def _check_handedness(handedness: object, target: TargetFrame) -> Handedness:
    if handedness not in get_args(Handedness):
        raise ValueError(f"handedness must be 'RH' or 'LH', got {handedness!r}")
    expected = _FRAME_HANDEDNESS[str(handedness)]
    if target.handedness != expected:
        raise ValueError(
            f"trial handedness {handedness!r} does not match the target frame's "
            f"{target.handedness!r}; a swing is never mirrored silently"
        )
    return cast(Handedness, handedness)


def _yaw(stance_target: tuple[float, float], target_dir: np.ndarray) -> float:
    """Yaw about ``+Z`` turning ``stance_target`` onto horizontal ``target_dir``."""
    stance = math.atan2(stance_target[1], stance_target[0])
    world = math.atan2(float(target_dir[1]), float(target_dir[0]))
    return world - stance


def placement_for_target(
    handedness: Handedness, target: TargetFrame, *, origin_m: ArrayLike
) -> TargetedPlacement:
    """Placement of the stance frame in the world described by ``target``.

    ``origin_m`` is the world point of the stance origin (midway between the
    plate centres) on the plate top surface.  It is required: where the plates
    sit in a capture volume is calibration data and is never assumed.

    Raises ``TypeError`` when ``target`` is not a :class:`TargetFrame` and
    ``ValueError`` when the handedness is unknown or disagrees with
    ``target.handedness``, when ``target.up`` is not world ``+Z`` (ground
    reaction is Z-up), or when ``origin_m`` is not a finite 3-vector.
    Postcondition: the rotation is a proper yaw that maps the stance target
    direction onto ``target.target_dir``.
    """
    if not isinstance(target, TargetFrame):
        raise TypeError("target must be a TargetFrame")
    side = _check_handedness(handedness, target)
    if not np.allclose(target.z_t, _ZHAT, atol=ROTATION_ATOL):
        raise ValueError(f"target.up must be world +Z, got {target.up}")
    yaw = _yaw(_STANCE_TARGET[side], target.x_t)
    placement = PlatePlacement.from_yaw(yaw, origin_m=origin_m)
    return TargetedPlacement(placement, target.frame_id, side)
