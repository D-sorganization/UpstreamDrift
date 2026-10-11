"""Decorative golf ball placement for native-viewer exports (GCV-13, #11719).

The ball is computed once per swing and handed to every engine backend, so all
viewers draw the same ball at the same place. It is visual only: no backend
adds it to a dynamics model. The ball is visible from the start until impact
(``hide_after_s``); post-impact flight is a later slice.

Placement policy, in order:

1. the bundle's recorded ball impact (``bundle.ball_impact().ball_centre_m``) --
   the ball every engine replay actually strikes, used verbatim;
2. the address geometry: MuJoCo forward kinematics of the first state, the
   club face centre and normal in world, then
   :func:`model_appearance.ball.ball_position_at_address`.

``address_ball`` returns a reason string instead of a placement when neither is
available -- it never invents a zero position.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import logging
import math
from typing import Any

import numpy as np

from src.shared.python.model_appearance.ball import (
    BALL_RADIUS_M,
    resolve_ball_visual,
)
from src.tools.native_viewer_export.core import (
    ExportSettings,
    SwingInput,
    impact_time_s as resolve_impact_time_s,
)

logger = logging.getLogger(__name__)

SOURCE_BUNDLE = "bundle:ball_impact"
SOURCE_ADDRESS = "address_geometry"
BALL_RGBA = (0.97, 0.97, 0.95, 1.0)
#: A clubhead centre this far above the ground is not an address stance.
MAX_ADDRESS_HEIGHT_M = 0.5


@dataclass(frozen=True)
class BallPlacement:
    """World position of the decorative ball and when it disappears.

    Preconditions (``ValueError``/``TypeError`` on construction): the position
    is a finite 3-vector, the radius is positive and finite, ``hide_after_s``
    is ``None`` or finite. The ball is visible while ``t <= hide_after_s``.
    """

    position_m: tuple[float, float, float]
    radius_m: float
    source: str
    hide_after_s: float | None = None

    def __post_init__(self) -> None:
        try:
            pos = tuple(float(v) for v in self.position_m)
        except TypeError as exc:
            raise TypeError("position_m must be a sequence of three numbers") from exc
        if len(pos) != 3 or not all(math.isfinite(v) for v in pos):
            raise ValueError("position_m must be a finite 3-vector")
        object.__setattr__(self, "position_m", pos)
        radius = float(self.radius_m)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius_m must be positive and finite")
        object.__setattr__(self, "radius_m", radius)
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string")
        if self.hide_after_s is not None:
            hide = float(self.hide_after_s)
            if not math.isfinite(hide):
                raise ValueError("hide_after_s must be finite or None")
            object.__setattr__(self, "hide_after_s", hide)

    def visible_at(self, t_s: float) -> bool:
        """Whether the ball is drawn at swing time ``t_s`` (until impact)."""
        return self.hide_after_s is None or t_s <= self.hide_after_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_m": list(self.position_m),
            "radius_m": self.radius_m,
            "source": self.source,
            "hide_after_s": self.hide_after_s,
        }

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> BallPlacement:
        return cls(
            tuple(doc["position_m"]),  # type: ignore[arg-type]
            doc["radius_m"],
            doc["source"],
            doc.get("hide_after_s"),
        )


def _bundle_ball_centre(swing: SwingInput) -> np.ndarray | None:
    impact = swing.bundle.ball_impact()
    centre = getattr(impact, "ball_centre_m", None)
    if centre is None:
        return None
    arr = np.asarray(centre, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        return None
    return arr


def _address_face(swing: SwingInput) -> tuple[np.ndarray, np.ndarray, float] | str:
    """World face centre, face normal and ground height at ``swing.q[0]``."""
    try:
        from src.engines.physics_engines.mujoco.python.overlay_source import (
            MujocoOverlaySource,
        )
    except ImportError as exc:
        return f"mujoco is not available for the address ball: {exc}"
    from src.shared.python.model_appearance.club_assembly import (
        assembly_from_spec,
        clubface_centre,
        clubface_vector,
    )

    spec = json.loads(swing.bundle.spec_bytes)
    club = assembly_from_spec(spec)
    frame = next((f for f in spec["frames"] if f["name"] == "Clubhead"), None)
    if club is None or frame is None:
        return "spec has no club body"
    source = MujocoOverlaySource(swing.bundle.spec_bytes)
    names = swing.bundle.coordinate_order
    coords = dict(zip(names, map(float, swing.q[0]), strict=True))
    # world pose of the spec club-body frame = site pose @ inverse(frame placement)
    body = source.frame_poses(coords)["Clubhead"] @ np.linalg.inv(
        np.asarray(frame["placement"], dtype=float)
    )
    centre = body[:3, :3] @ clubface_centre(club) + body[:3, 3]
    normal = body[:3, :3] @ clubface_vector(club)
    ground = float(spec["contact"]["ground"].get("height_m") or 0.0)
    return centre, normal, ground


def address_ball(swing: SwingInput, impact_time_s: float | None) -> BallPlacement | str:
    """The decorative ball for ``swing``, or the reason it is unavailable.

    Postcondition: a returned placement has a finite position resting on the
    ground (address geometry) or at the bundle's recorded ball centre.
    """
    if not isinstance(swing, SwingInput):
        raise TypeError(f"swing must be a SwingInput, got {type(swing).__name__}")
    if impact_time_s is not None and not math.isfinite(impact_time_s):
        raise ValueError("impact_time_s must be finite or None")
    recorded = _bundle_ball_centre(swing)
    if recorded is not None:
        pos, source = recorded, SOURCE_BUNDLE
    else:
        try:
            face = _address_face(swing)
        except (KeyError, IndexError, ValueError, TypeError, RuntimeError) as exc:
            return f"address geometry unavailable: {exc!r}"
        if isinstance(face, str):
            return face
        centre, normal, ground = face
        height = abs(float(centre[2]) - ground)
        if height > MAX_ADDRESS_HEIGHT_M:
            return (
                f"first state is not an address stance (face centre {height:.3f} m "
                "above ground) and the bundle records no ball impact"
            )
        resolved = resolve_ball_visual(
            enabled=True,
            position_m=None,
            source=SOURCE_ADDRESS,
            face_centre_m=centre,
            face_normal=normal,
            ground_height_m=ground,
        )
        if resolved is None:  # pragma: no cover - enabled=True always resolves
            return "ball placement unavailable"
        pos, source = resolved
    return BallPlacement(
        (float(pos[0]), float(pos[1]), float(pos[2])),
        BALL_RADIUS_M,
        source,
        impact_time_s,
    )


def attach_ball(
    swing: SwingInput, settings: ExportSettings
) -> tuple[SwingInput, dict[str, Any] | None]:
    """``swing`` carrying its ball placement, plus the manifest record.

    The record is ``None`` when ``settings.ball`` is off, the placement dict
    when available, else ``{"unavailable": reason}`` (also logged). A ball
    already on ``swing`` is kept so a caller can supply a measured one.
    """
    if not settings.ball:
        return replace(swing, ball=None), None
    placed = swing.ball
    if placed is None:
        placed = address_ball(swing, resolve_impact_time_s(swing, settings))
    if isinstance(placed, str):
        logger.warning("no golf ball drawn: %s", placed)
        return replace(swing, ball=None), {"unavailable": placed}
    return replace(swing, ball=placed), placed.to_dict()
