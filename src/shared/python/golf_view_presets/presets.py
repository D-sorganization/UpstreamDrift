"""Golf camera view presets in the specification frame (NV-1, #11674).

Convention (matches the full-body specification and every engine export):
the world is Z-up, the golfer faces -X and, for a right-handed golfer, the
target line runs toward -Y. A camera is described by the heading and
elevation of its *viewing direction* using the MuJoCo convention::

    view = (cos(el) cos(az), cos(el) sin(az), sin(el))

so ``azimuth 0`` looks along +X (a camera in front of the golfer), and a
negative elevation looks downward. Presets are pure data; the per-engine
adapters in :mod:`.adapters` translate them. The legacy Qt viewpoint model in
``gui_pkg/viewpoint_controls.py`` uses a different azimuth origin and is not
used for exports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
_WORLD_UP = np.array([0.0, 0.0, 1.0])


def check_point3(value: Sequence[float] | Array, name: str) -> Array:
    """Return ``value`` as a finite float 3-vector or raise ``ValueError``."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite 3-vector, got {value!r}")
    return arr


@dataclass(frozen=True)
class ViewPreset:
    """A named golf camera direction.

    Postconditions: :meth:`view_direction` is a unit vector with a strictly
    negative Z component; :meth:`image_right` and :meth:`image_up` complete a
    right-handed screen frame with ``image_up`` having positive Z.
    """

    name: str
    label: str
    azimuth_deg: float
    elevation_deg: float
    default_distance_m: float
    #: Moving target the camera follows (``"grip_midpoint"``), or ``None`` for a
    #: fixed look-at point (GCV-10, #11716).
    tracks: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.label:
            raise ValueError("name and label must be non-empty")
        if not (math.isfinite(self.azimuth_deg) and math.isfinite(self.elevation_deg)):
            raise ValueError("azimuth_deg and elevation_deg must be finite")
        if not -89.999 <= self.elevation_deg <= 89.999:
            raise ValueError("elevation_deg must lie strictly inside (-90, 90)")
        if not (math.isfinite(self.default_distance_m) and self.default_distance_m > 0):
            raise ValueError("default_distance_m must be positive and finite")

    def view_direction(self) -> Array:
        """Unit vector from the camera toward the look-at point (world frame)."""
        a, e = math.radians(self.azimuth_deg), math.radians(self.elevation_deg)
        return np.array(
            [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]
        )

    def image_right(self) -> Array:
        """Unit world vector pointing to the right of the image."""
        right = np.cross(self.view_direction(), _WORLD_UP)
        return right / np.linalg.norm(right)

    def image_up(self) -> Array:
        """Unit world vector pointing up in the image."""
        return np.asarray(
            np.cross(self.image_right(), self.view_direction()), dtype=np.float64
        )

    def camera_position(
        self, lookat_m: Sequence[float] | Array, distance_m: float
    ) -> Array:
        """World position of a camera ``distance_m`` before ``lookat_m``."""
        look = check_point3(lookat_m, "lookat_m")
        if not (math.isfinite(distance_m) and distance_m > 0.0):
            raise ValueError(f"distance_m must be positive and finite: {distance_m}")
        return look - float(distance_m) * self.view_direction()


VIEW_ORDER: tuple[str, ...] = ("face_on", "down_the_line", "overhead", "oblique")

VIEW_PRESETS = MappingProxyType(
    {
        "face_on": ViewPreset(
            "face_on", "Face-on (target to image right)", 0.0, -6.0, 3.4
        ),
        "down_the_line": ViewPreset(
            "down_the_line", "Down-the-line (behind, on target line)", -90.0, -8.0, 3.75
        ),
        "overhead": ViewPreset(
            "overhead", "Overhead (target to image right)", 0.0, -89.0, 2.28
        ),
        "oblique": ViewPreset(
            "oblique", "Oblique (rear, target side)", 135.0, -14.0, 3.95
        ),
        # Not part of VIEW_ORDER (the 2x2 grid stays four views): request it by name.
        "hands_closeup": ViewPreset(
            "hands_closeup",
            "Hands close-up (follows the grip midpoint)",
            -35.0,
            -18.0,
            0.7,
            tracks="grip_midpoint",
        ),
    }
)


def get_view_preset(name: str) -> ViewPreset:
    """Return the preset called ``name``; raise ``ValueError`` if unknown."""
    try:
        return VIEW_PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown view preset {name!r}; expected one of {list(VIEW_PRESETS)}"
        ) from None


def tracked_lookats(
    preset: ViewPreset,
    static_lookat_m: Sequence[float] | Array,
    focus_m: Sequence[Sequence[float] | None],
) -> list[tuple[float, float, float]]:
    """Look-at point per frame for ``preset``.

    A tracking preset follows ``focus_m`` (for ``hands_closeup``, the grip
    midpoint per frame); a sample that is ``None`` or not finite holds the last
    known point, and before any is known the static look-at is used, never the
    origin.  A fixed preset returns ``static_lookat_m`` for every frame.

    Raises:
        ValueError: if ``static_lookat_m`` is not a finite 3-vector.
    """
    checked = check_point3(static_lookat_m, "static_lookat_m")
    static = (float(checked[0]), float(checked[1]), float(checked[2]))
    out: list[tuple[float, float, float]] = []
    held = static
    for point in focus_m:
        if preset.tracks is not None and point is not None:
            arr = np.asarray(point, dtype=float)
            if arr.shape == (3,) and np.isfinite(arr).all():
                held = (float(arr[0]), float(arr[1]), float(arr[2]))
        out.append(held if preset.tracks is not None else static)
    return out
