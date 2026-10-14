"""Back-squat bar geometry from the parity standard (#12347).

The same-input full-body spec is the golf humanoid with its dual-grip club
welded to both hands, 0.08 m apart in front of the body.  A back squat carries
the bar on the upper back (trapezius shelf), the palm centres a standard grip
width apart and the bar perpendicular to the sagittal plane.  This module holds
the part of that change that needs no physics engine:

* :class:`BackBarGeometry` - grip width, bar axis and the start-pose bar
  position, all read from the vendored standard through :mod:`.lift_spec`
  (never copied here), the position scaled to the lifter height;
* :func:`apply_grip_width` - a derived full-body spec whose left hand sits one
  grip width from the right along the bar (the closure keeps the right hand);
* :class:`BarShelf` - the bar attachment frame on the torso: a fixed offset in
  a torso spec frame, fitted so the bar sits at the standard's position when
  the lifter stands.

Positions use the standard's pose frame: origin on the floor at mid-foot,
``forward`` along the stance, ``left`` lateral, ``up`` vertical.  The IK that
poses the arms on the bar is :func:`.bundles.solve_back_bar_standing`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import lift_spec
from .sagittal_retarget import SagittalFrame

Array = NDArray[np.float64]

BACK_SQUAT = "squat"
#: Spec frames on the club body: the left palm, the right palm, and the bar
#: centre (its x axis is the bar's long axis, midway between the palms).
LEFT_HAND_FRAME = "LW"
RIGHT_HAND_FRAME = "RW"
BAR_CENTRE_FRAME = "Grip"
#: Torso spec frame the shelf is measured in (the shoulder-girdle centre).
SHELF_FRAME = "Hub"
#: Largest plausible shelf offset from the shoulder-girdle centre, metres.
MAX_SHELF_OFFSET_M = 0.25
#: Club-body solids that move with the left hand.
_LEFT_HAND_SOLIDS = (
    "GolfSwing3D_Kinetic/Grip/LHand",
    "GolfSwing3D_Kinetic/Grip/LHandStandoff",
)
#: The bar's long axis in the bar-centre frame (the frame's x axis).
BAR_LONG_AXIS_LOCAL = (1.0, 0.0, 0.0)
_AXIS_TOLERANCE = 1e-6


def _finite_triple(values: Any, name: str) -> Array:
    array = np.asarray(values, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return array


@dataclass(frozen=True)
class BackBarGeometry:
    """What the standard asks of a back-squat bar, for one lifter height.

    Attributes:
        lift: Standard lift key (``"squat"``).
        height_m: Lifter height the position is scaled to.
        grip_width_m: Palm-centre separation along the bar.
        bar_axis: Bar long axis in the standard's frame (``+Y``).
        start_bar_position_m: Bar centre at the ``start`` via-pose, standard
            pose frame, scaled by ``height_m / standard height``.
    """

    lift: str
    height_m: float
    grip_width_m: float
    bar_axis: tuple[float, float, float]
    start_bar_position_m: tuple[float, float, float]

    @classmethod
    def from_standard(cls, lift: str, *, height_m: float) -> BackBarGeometry:
        """Read the geometry of *lift* from the parity standard.

        Raises:
            ValueError: If *lift* is not the back squat or *height_m* is not
                positive and finite.
        """
        if lift != BACK_SQUAT:
            raise ValueError(
                f"back-bar geometry is defined for {BACK_SQUAT!r}: {lift!r}"
            )
        if not (math.isfinite(height_m) and height_m > 0.0):
            raise ValueError(f"height_m must be positive and finite: {height_m!r}")
        scale = height_m / lift_spec.standard_height_m()
        x, y, z = (
            scale * v for v in lift_spec.lift_via_pose(lift, "start").bar_position_m
        )
        return cls(
            lift=lift,
            height_m=float(height_m),
            grip_width_m=lift_spec.lift_grip(lift).width_m,
            bar_axis=lift_spec.bar_axis(),
            start_bar_position_m=(x, y, z),
        )

    def to_record(self) -> dict[str, Any]:
        """JSON-serialisable provenance of the geometry."""
        return {
            "lift": self.lift,
            "standard_version": lift_spec.standard_version(),
            "height_m": self.height_m,
            "grip_width_m": self.grip_width_m,
            "bar_axis": list(self.bar_axis),
            "start_bar_position_m": list(self.start_bar_position_m),
        }


def standard_bar_world(
    geometry: BackBarGeometry, mid_foot_m: Array, frame: SagittalFrame
) -> Array:
    """World position of the standard's start-pose bar centre.

    Args:
        geometry: Standard back-bar geometry.
        mid_foot_m: World point on the floor at mid-foot (pose-frame origin).
        frame: Lifter's sagittal frame (``forward``/``up``; ``left`` derived).
    """
    origin = _finite_triple(mid_foot_m, "mid_foot_m")
    x, y, z = geometry.start_bar_position_m
    return (
        origin
        + x * np.asarray(frame.forward)
        + y * frame.left
        + z * np.asarray(frame.up)
    )


def bar_axis_world(geometry: BackBarGeometry, frame: SagittalFrame) -> Array:
    """The standard's bar axis (``+Y``: lifter's left) in world coordinates."""
    ax, ay, az = geometry.bar_axis
    return ax * np.asarray(frame.forward) + ay * frame.left + az * np.asarray(frame.up)


def _placement_y(placement: list[Any]) -> float:
    row = placement[1]
    return float(row[3] if not isinstance(row, str) else row.split()[3])


def _with_placement_y(placement: list[Any], y: float) -> list[Any]:
    """Copy of a 4x4 placement (rows of numbers or of strings) with new y."""
    out = list(placement)
    if isinstance(out[1], str):
        cells = out[1].split()
        cells[3] = repr(float(y))
        out[1] = " ".join(cells)
    else:
        out[1] = [*out[1][:3], float(y)]
    return out


def _frame(spec: dict[str, Any], name: str) -> dict[str, Any]:
    for frame in spec.get("frames", []):
        if frame["name"] == name:
            found: dict[str, Any] = frame
            return found
    raise ValueError(f"spec has no frame {name!r}")


def apply_grip_width(spec_bytes: bytes, width_m: float) -> bytes:
    """Full-body spec whose palms are ``width_m`` apart along the club axis.

    The closure keeps the right hand where it is; the left-hand frame, its hand
    solids (mass and inertia travel with the hand) and the bar-centre frame
    move along the bar's long axis so the bar centre stays midway between the
    palms.  Only ``frames`` and ``bodies`` change; the result depends on
    ``width_m`` alone (applying it twice equals applying the last width).

    Raises:
        TypeError: If *spec_bytes* is not bytes.
        ValueError: If *width_m* is not positive and finite, or the spec lacks
            the hand frames, the closure or a bar axis along the club body y.
    """
    if not isinstance(spec_bytes, bytes | bytearray):
        raise TypeError("spec_bytes must be the raw full-body spec bytes")
    if not (math.isfinite(width_m) and width_m > 0.0):
        raise ValueError(f"width_m must be positive and finite: {width_m!r}")
    spec = json.loads(bytes(spec_bytes))
    left, centre = _frame(spec, LEFT_HAND_FRAME), _frame(spec, BAR_CENTRE_FRAME)
    rotation = np.array([row[:3] for row in centre["placement"][:3]], dtype=float)
    if not np.allclose(np.abs(rotation[:, 0]), (0.0, 1.0, 0.0), atol=_AXIS_TOLERANCE):
        raise ValueError(f"frame {BAR_CENTRE_FRAME!r} x axis must be the club body y")
    try:
        right_y = _placement_y(spec["closure"]["placement_b"])
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("spec closure has no right-hand placement") from error
    old_y = _placement_y(left["placement"])
    direction = 1.0 if right_y >= old_y else -1.0
    new_y = right_y - direction * width_m
    shift = new_y - old_y
    left["placement"] = _with_placement_y(left["placement"], new_y)
    centre["placement"] = _with_placement_y(
        centre["placement"], 0.5 * (right_y + new_y)
    )
    for body in spec["bodies"]:
        for solid in body["solids"]:
            if solid["name"] in _LEFT_HAND_SOLIDS:
                solid["placement"] = _with_placement_y(
                    solid["placement"], _placement_y(solid["placement"]) + shift
                )
    return json.dumps(spec, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class BarShelf:
    """Bar attachment frame on the torso (upper-trapezius shelf).

    A fixed ``offset_m`` in the torso spec frame ``frame``; the bar centre sits
    at :meth:`world_position` whatever the trunk does.
    """

    frame: str
    offset_m: tuple[float, float, float]

    def world_position(self, rotation: Array, origin: Array) -> Array:
        """Shelf point for a torso frame of orientation/position given."""
        return np.asarray(origin, dtype=float) + np.asarray(
            rotation, dtype=float
        ) @ np.asarray(self.offset_m)

    def to_record(self) -> dict[str, Any]:
        return {"frame": self.frame, "offset_m": list(self.offset_m)}


def fit_bar_shelf(
    frame: str, rotation: Array, origin: Array, bar_world: Array
) -> BarShelf:
    """Shelf offset in torso frame ``frame`` that puts the bar at ``bar_world``.

    Postcondition: ``shelf.world_position(rotation, origin) == bar_world``.

    Raises:
        ValueError: If an input is not finite, or the bar is more than
            :data:`MAX_SHELF_OFFSET_M` from the torso frame origin (not a point
            of the upper back).
    """
    rot = np.asarray(rotation, dtype=float)
    if rot.shape != (3, 3) or not np.isfinite(rot).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    offset = rot.T @ (
        _finite_triple(bar_world, "bar_world") - _finite_triple(origin, "origin")
    )
    if float(np.linalg.norm(offset)) > MAX_SHELF_OFFSET_M:
        raise ValueError(
            f"bar is {np.linalg.norm(offset):.3f} m from the {frame!r} frame; the "
            f"torso shelf must lie within {MAX_SHELF_OFFSET_M} m of it"
        )
    x, y, z = (float(v) for v in offset)
    return BarShelf(frame=frame, offset_m=(x, y, z))


@dataclass(frozen=True)
class BackBarSetup:
    """Geometry-derived inputs of a back-bar lift.

    ``spec_bytes`` is the derived spec (grip width applied); ``shelf`` the torso
    attachment; ``bar_target_m`` the standing bar position the shelf was fitted
    to and ``mid_foot_m`` the pose-frame origin it was measured from.
    """

    spec_bytes: bytes
    geometry: BackBarGeometry
    shelf: BarShelf
    bar_target_m: Array
    mid_foot_m: Array

    def to_record(self) -> dict[str, Any]:
        """JSON-serialisable provenance (the spec itself is the bundle's)."""
        return {
            "geometry": self.geometry.to_record(),
            "shelf": self.shelf.to_record(),
            "bar_target_m": self.bar_target_m.tolist(),
            "mid_foot_m": self.mid_foot_m.tolist(),
        }


@dataclass(frozen=True)
class BackBarStart:
    """Standing start pose with the arms on the back bar.

    ``q`` is in spec coordinate order and refers to ``setup.spec_bytes``.
    """

    setup: BackBarSetup
    q: Array
    frame: SagittalFrame

    @property
    def spec_bytes(self) -> bytes:
        return self.setup.spec_bytes
