"""Behind-the-golfer views sit midway between the ball and the feet (#15150).

The down-the-line views look along the target line from behind the golfer. The
camera's lateral position is the point a ``behind_fraction`` of the way from the
address ball to the midpoint of the feet (default 0.5), read per bundle from the
replayed model; the look-at moves with it so the target line runs straight away
from the camera. Height and distance keep their existing settings
(``camera_framing_settings.json`` height, preset distance). An explicit
``--camera-lookat`` wins; without the feet the preset look-at stays, with a
warning. The one look-at per view feeds both the engine cameras and the
``pinhole_for_view`` overlays, so they cannot disagree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import logging
from typing import Any

import numpy as np

from src.shared.python.golf_view_presets import get_view_preset
from src.tools.native_viewer_export.camera_framing import SETTINGS_PATH

logger = logging.getLogger(__name__)
Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class BehindSettings:
    """``behind_*`` and ``foot_bodies`` of ``camera_framing_settings.json``."""

    views: tuple[str, ...]
    fraction: float
    foot_bodies: tuple[str, ...]


def check_fraction(value: float) -> float:
    """``value`` in [0, 1] (0 = ball, 1 = feet); raises ``ValueError`` otherwise."""
    if not (isinstance(value, (int, float)) and 0.0 <= value <= 1.0):
        raise ValueError(f"behind fraction must be in [0, 1], got {value!r}")
    return float(value)


def load_behind_settings() -> BehindSettings:
    """The behind-camera settings; ``ValueError`` if the fraction is not in [0, 1]."""
    raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    return BehindSettings(
        tuple(raw["behind_views"]),
        check_fraction(raw["behind_fraction"]),
        tuple(raw["foot_bodies"]),
    )


def anchor_point(
    ball_m: Sequence[float], feet_m: Sequence[float], fraction: float
) -> Vec3:
    """The point ``fraction`` of the way from ``ball_m`` to ``feet_m``."""
    f = check_fraction(fraction)
    ball, feet = np.asarray(ball_m, float), np.asarray(feet_m, float)
    p = ball + f * (feet - ball)
    return (float(p[0]), float(p[1]), float(p[2]))


def feet_midpoint(
    model: Any, coords: Mapping[str, float], names: Sequence[str]
) -> Vec3:
    """Mean world position of the foot bodies ``names`` at ``coords``."""
    found = model.body_positions(coords, names)
    mean = np.mean([found[n] for n in names], axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


def anchor_from(
    ball_m: Sequence[float], model: Any, coords: Mapping[str, float], s: BehindSettings
) -> Vec3 | None:
    """The lateral anchor, or ``None`` (with a warning) if the feet are missing."""
    try:
        feet = feet_midpoint(model, coords, s.foot_bodies)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("feet not found, keeping the preset camera: %r", exc)
        return None
    return anchor_point(ball_m, feet, s.fraction)


def address_anchor(swing: Any) -> Vec3 | None:
    """Anchor for ``swing`` from its address ball and feet; ``None`` if unavailable."""
    from src.tools.native_viewer_export.ball import BallPlacement, address_ball
    from src.tools.native_viewer_export.club_pose import ClubPoseModel

    ball = address_ball(swing, None)
    if not isinstance(ball, BallPlacement):
        logger.warning("no address ball, keeping the preset camera: %s", ball)
        return None
    try:
        model = ClubPoseModel(swing.bundle.spec_bytes)
    except (ImportError, ValueError) as exc:
        logger.warning("feet not found, keeping the preset camera: %r", exc)
        return None
    names = swing.bundle.coordinate_order
    coords = dict(zip(names, map(float, swing.q[0]), strict=True))
    return anchor_from(ball.position_m, model, coords, load_behind_settings())


def behind_views(views: Sequence[str]) -> bool:
    """Whether any of ``views`` is a behind-the-golfer view."""
    return bool(set(views) & set(load_behind_settings().views))


def view_lookat(view: str, settings: Any) -> Vec3:
    """Look-at of ``view``: ``settings.lookat_m``, shifted sideways for behind views.

    Postcondition: an explicit look-at or no anchor leaves it untouched; else the
    camera of a behind view lies on the line through ``settings.behind_anchor_m``
    along the view direction, with the same height and depth as before.
    """
    look = tuple(settings.lookat_m)
    anchor = settings.behind_anchor_m
    if settings.lookat_explicit or anchor is None:
        return look  # type: ignore[return-value]
    if view not in load_behind_settings().views:
        return look  # type: ignore[return-value]
    right = get_view_preset(view).image_right()
    shift = float(np.dot(np.asarray(anchor, float) - np.asarray(look, float), right))
    moved = np.asarray(look, float) + shift * right
    return (float(moved[0]), float(moved[1]), float(moved[2]))


def look_map(settings: Any) -> dict[str, Vec3]:
    """``view_lookat`` for every view of ``settings`` (what the overlays project)."""
    return {v: view_lookat(v, settings) for v in settings.views}


def lookat_of(lookat: Any, view: str) -> Sequence[float]:
    """The look-at of ``view`` from a per-view mapping or one shared point."""
    return lookat[view] if isinstance(lookat, Mapping) else lookat
