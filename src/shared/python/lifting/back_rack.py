"""Back-squat rack positions on the lift body's trunk landmarks (OSL-17, #13131).

Until #13131 the back-squat bar sat on a torso shelf fitted to the parity
standard's start-pose bar position.  That position lies 0.20 m above the
shoulder centres, so on the lift body the bar ended up in front of the face,
not on the back.  A back squat carries the bar on the upper back:

* **High bar**: "across the top of the trapezius, just below the process of
  the C7 vertebra" (Glassbrook et al. 2019, citing Wretenberg et al. 1996).
* **Low bar**: "on the lower trapezius, just over the posterior deltoid and
  along the spine of the scapula" (same sources).

The lift body (:mod:`.lift_body`) carries the C7 landmark as its ``BackTop``
marker (the C7 spinous-process surface proxy of the AN-1 marker set, on the
trunk body).  :data:`HIGH_BAR` puts the bar's long axis one shaft radius
behind that skin point, along the trunk's posterior axis, at the landmark's
own level on the sagittal midline: the bar lies on the skin just below C7.
The lift body has no spine-of-scapula landmark below that level (its
posterior shoulder markers sit at the C7 marker's height), so no low-bar
rack is defined here.

Everything is read from the spec document; no landmark offset is copied.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = [
    "BackRack",
    "HIGH_BAR",
    "HIGH_BAR_GRIP_WIDTH_M",
    "RACK_SOURCES",
    "RACKS",
    "rack_bar_world",
    "rack_offset",
]

#: Sources of the rack definitions (landmark wording quoted in the docstring).
RACK_SOURCES: tuple[str, ...] = (
    "Glassbrook DJ, Brown SR, Helms ER, Duncan JS, Storey AG. The high-bar and "
    "low-bar back-squats: a biomechanical analysis. J Strength Cond Res "
    "2019;33(S1):S1-S18. doi:10.1519/JSC.0000000000001836",
    "Wretenberg P, Feng Y, Arborelius UP. High- and low-bar squatting "
    "techniques during weight-training. Med Sci Sports Exerc "
    "1996;28(2):218-224",
)
#: Posterior direction of the lift body's trunk frames (``BackTop`` and the
#: posterior shoulder markers sit at negative x, the face markers at positive).
TRUNK_POSTERIOR_LOCAL = (-1.0, 0.0, 0.0)
#: Men's bar shaft radius (28 mm shaft), :class:`~.barbell_bushing.Barbell`.
SHAFT_RADIUS_M = 0.014
#: High-bar grip width, metres: 0.75 m. Chosen from the reachability sweep
#: inside the parity standard's grip range [0.45, 0.90] m (OSL-19 #13158,
#: OSL-20 #13159; see docs/development/lifting/evidence/back_squat/README.md).
#: Note: Glassbrook et al. 2019 and Wretenberg et al. 1996 describe the bar
#: placement on the trapezius, not a grip width.
HIGH_BAR_GRIP_WIDTH_M = 0.75


@dataclass(frozen=True)
class BackRack:
    """A bar rack position: one trunk landmark and the bar's clearance off it.

    Attributes:
        name: Rack key (``"high_bar"``).
        landmark: Spec marker the bar rests on (``marker_attachments`` key).
        description: Landmark definition, as worded by :data:`RACK_SOURCES`.
        clearance_m: Distance from the skin landmark to the bar's long axis
            along the trunk's posterior axis (the shaft radius).
        grip_width_m: Grip width for this rack position, metres
            (OSL-19, #13158; OSL-20, #13159). If ``None``, defaults to the
            parity standard's grip width.
    """

    name: str
    landmark: str
    description: str
    clearance_m: float = SHAFT_RADIUS_M
    grip_width_m: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.landmark:
            raise ValueError("a rack needs a name and a landmark")
        if not (math.isfinite(self.clearance_m) and self.clearance_m >= 0.0):
            raise ValueError(f"clearance_m must be finite and >= 0: {self.clearance_m}")
        if self.grip_width_m is not None and not (
            math.isfinite(self.grip_width_m) and self.grip_width_m > 0.0
        ):
            raise ValueError(
                f"grip_width_m must be finite and > 0: {self.grip_width_m}"
            )

    def to_record(self) -> dict[str, Any]:
        """JSON-serialisable provenance of the rack definition."""
        return {
            "name": self.name,
            "landmark": self.landmark,
            "description": self.description,
            "clearance_m": self.clearance_m,
            "grip_width_m": self.grip_width_m,
            "posterior_axis_local": list(TRUNK_POSTERIOR_LOCAL),
            "sources": list(RACK_SOURCES),
        }


HIGH_BAR = BackRack(
    name="high_bar",
    landmark="BackTop",
    description="across the top of the trapezius, just below the process of "
    "the C7 vertebra (lift-body marker BackTop: C7 spinous-process surface)",
    grip_width_m=HIGH_BAR_GRIP_WIDTH_M,
)
RACKS: Mapping[str, BackRack] = {HIGH_BAR.name: HIGH_BAR}


def _document(spec: bytes | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(spec, bytes | bytearray):
        loaded: Mapping[str, Any] = json.loads(bytes(spec))
        return loaded
    if isinstance(spec, Mapping):
        return spec
    raise TypeError("spec must be the spec bytes or its parsed document")


def rack_offset(
    rack: BackRack, spec: bytes | Mapping[str, Any]
) -> tuple[str, tuple[float, float, float]]:
    """Spec frame and offset in it of the rack's bar centre.

    The landmark's own ``marker_attachments`` offset plus
    ``rack.clearance_m`` along :data:`TRUNK_POSTERIOR_LOCAL`.

    Raises:
        TypeError: If *rack* is not a :class:`BackRack`.
        ValueError: If the spec has no such marker or no finite offset for it.
    """
    if not isinstance(rack, BackRack):
        raise TypeError("rack must be a BackRack")
    marker = dict(_document(spec).get("marker_attachments", {})).get(rack.landmark)
    offset = None if marker is None else marker.get("offset_m")
    if marker is None or offset is None:
        raise ValueError(f"spec has no marker {rack.landmark!r} with an offset")
    local = np.asarray(offset, dtype=float) + rack.clearance_m * np.asarray(
        TRUNK_POSTERIOR_LOCAL
    )
    if local.shape != (3,) or not np.isfinite(local).all():
        raise ValueError(f"marker {rack.landmark!r} offset must be a finite 3-vector")
    x, y, z = (float(v) for v in local)
    return str(marker["body"]), (x, y, z)


def rack_bar_world(
    rack: BackRack, spec: bytes | Mapping[str, Any], kin: Any, q: Array
) -> Array:
    """World bar centre of *rack* at pose *q*.

    ``kin`` is the spec's marker kinematics
    (:class:`~src.engines.physics_engines.mujoco.python.full_body_ik.\
FullBodyMarkerKinematics`), whose ``body_poses`` places the landmark's frame.

    Postcondition: the point rides the landmark's frame rigidly.
    """
    frame, offset = rack_offset(rack, spec)
    rotation, origin = kin.body_poses(np.asarray(q, dtype=float), [frame])[frame]
    return np.asarray(origin, dtype=float) + np.asarray(rotation) @ np.asarray(offset)
