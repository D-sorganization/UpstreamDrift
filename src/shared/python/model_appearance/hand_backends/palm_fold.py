"""A fold across the palm, so the palm of one hand cups the thumb of the other.

The palm of the source hand is one plate: its skin follows the wrist bone, and
only the thumb's base moves on it. A real palm folds across its width when it
closes on something narrow: the thenar side, with the thumb's base on it, turns
toward the little-finger side about a line along the palm. This module adds
that fold as a joint of the rig (issue #15270, epic #15150).

The joint is a hinge along a line in the rest hand. Skin of the palm bones on
the fold side of the line turns about it, by nothing at the line and by the
whole angle one blend width further out; the bones carried on that side (the
thumb) turn with the skin at their base. The fold is applied to the rest hand,
which is the wrist's own frame, so the folded hand is still skinned by plain
linear blending and every other joint works on it as before.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.hand_backends.skinning import (
    SkinnedHandAsset,
    asset_hand_length_m,
    rotation_about_axis,
)

Array = NDArray[np.float64]


@dataclass(frozen=True)
class PalmFoldSettings:
    """How the palm folds round the other hand's thumb (``pair.palm_fold``).

    ``max_deg`` is the furthest the fold closes (zero: no fold) and
    ``step_deg`` the step it closes in. The hinge lies ``offset_m`` to the fold
    side of the thumb's axis and ``below_thumb_m`` under it; skin turns by the
    whole angle ``blend_m`` beyond the hinge. Toward the knuckles the fold
    fades out between ``fade`` (fractions of the hand's length from the wrist),
    so the fingers stay where they hold the club. ``press_m`` is how much
    further than the unfolded palm the folded pad may press into what it
    closes on (the skin correctives lay pressed skin on the surface).
    ``bones`` is the skin that may fold, ``carried`` the bones that ride on
    the fold and ``thumb_bones`` the bones of the other hand's thumb whose
    skin gives the line to fold along.
    """

    max_deg: float
    step_deg: float
    offset_m: float
    blend_m: float
    below_thumb_m: float
    fade: tuple[float, float]
    press_m: float
    bones: tuple[str, ...]
    carried: tuple[str, ...]
    thumb_bones: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_deg < 90.0 or not self.step_deg > 0.0:
            raise ValueError("palm fold max_deg must be in [0, 90), step_deg positive")
        if self.offset_m < 0.0 or not self.blend_m > 0.0 or self.press_m < 0.0:
            raise ValueError("palm fold offset, blend and press must not be negative")
        if not math.isfinite(self.below_thumb_m):
            raise ValueError("palm fold below_thumb_m must be finite")
        low, high = self.fade
        if not 0.0 <= low < high:
            raise ValueError("palm fold fade must be two increasing fractions")
        if not (self.bones and self.carried and self.thumb_bones):
            raise ValueError("palm fold bones, carried and thumb_bones need a bone")

    @classmethod
    def from_mapping(cls, raw: Any) -> PalmFoldSettings:
        low, high = raw["fade"]
        return cls(
            max_deg=float(raw["max_deg"]),
            step_deg=float(raw["step_deg"]),
            offset_m=float(raw["offset_m"]),
            blend_m=float(raw["blend_m"]),
            below_thumb_m=float(raw["below_thumb_m"]),
            fade=(float(low), float(high)),
            press_m=float(raw["press_m"]),
            bones=tuple(str(bone) for bone in raw["bones"]),
            carried=tuple(str(bone) for bone in raw["carried"]),
            thumb_bones=tuple(str(bone) for bone in raw["thumb_bones"]),
        )


@dataclass(frozen=True)
class FoldLine:
    """The hinge in the rest hand: a point on it, its unit direction (toward
    the fingers, in the plane of the palm) and the unit direction across it
    toward the side that folds."""

    point: Array
    direction: Array
    side: Array


def _smoothstep(x: Array) -> Array:
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fold_line(
    asset: SkinnedHandAsset, thumb_points: Array, settings: PalmFoldSettings
) -> FoldLine:
    """The hinge for a thumb lying on the palm.

    ``thumb_points`` is skin of the other hand's thumb in ``asset``'s rest
    frame (``z`` palmar). The hinge runs along the thumb in the plane of the
    palm, beside it on the side of the first carried bone and under it.
    """
    points = np.asarray(thumb_points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("fold line needs at least three finite thumb points")
    centre = points.mean(axis=0)
    direction = np.linalg.svd(points - centre)[2][0]
    direction[2] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError("the thumb stands on the palm; there is no line to fold on")
    direction = direction / norm * (1.0 if direction[1] >= 0.0 else -1.0)
    side = np.array([direction[1], -direction[0], 0.0])
    anchor = asset.bone_heads[asset.bone_index(settings.carried[0])]
    if (anchor - centre) @ side < 0.0:
        side = -side
    point = centre + settings.offset_m * side
    point[2] = centre[2] - settings.below_thumb_m
    return FoldLine(point, direction, side)


def _unknown(asset: SkinnedHandAsset, settings: PalmFoldSettings) -> None:
    unknown = {*settings.bones, *settings.carried} - set(asset.bone_names)
    if unknown:
        raise ValueError(f"pair.palm_fold names no bone: {sorted(unknown)}")


def _share(
    asset: SkinnedHandAsset, line: FoldLine, settings: PalmFoldSettings, at: Array
) -> Array:
    """Share of the fold angle that rest points ``at`` turn by, in ``[0, 1]``:
    none at the hinge, all of it ``blend_m`` out, fading toward the fingers."""
    across = _smoothstep(((at - line.point) @ line.side) / settings.blend_m)
    low, high = settings.fade
    reach = at[..., 1] / asset_hand_length_m(asset)
    return np.asarray(across * (1.0 - _smoothstep((reach - low) / (high - low))))


def _carried(asset: SkinnedHandAsset, settings: PalmFoldSettings) -> list[int]:
    return sorted({i for bone in settings.carried for i in asset.descendants(bone)})


def fold_weights(
    asset: SkinnedHandAsset, line: FoldLine, settings: PalmFoldSettings
) -> Array:
    """Share of the fold angle each rest vertex turns by, in ``[0, 1]``.

    Skin of ``bones`` turns by its own place across the hinge; skin of the
    carried bones turns with those bones, so it stays on them.
    """
    _unknown(asset, settings)
    dense = asset.dense_weights()
    carried = _carried(asset, settings)
    own = [asset.bone_index(b) for b in settings.bones if b not in settings.carried]
    own = sorted(set(own) - set(carried))
    anchor = asset.bone_heads[asset.bone_index(settings.carried[0])]
    rigid_share = float(_share(asset, line, settings, anchor))
    return np.asarray(
        dense[:, own].sum(axis=1) * _share(asset, line, settings, asset.rest_vertices)
        + dense[:, carried].sum(axis=1) * rigid_share
    )


def _turn_sign(line: FoldLine) -> float:
    # Turning the fold side about the hinge brings it palmar-ward (+z) for
    # one sign only; which one depends on which side folds.
    return 1.0 if float(np.cross(line.direction, line.side)[2]) > 0.0 else -1.0


def folded(
    asset: SkinnedHandAsset,
    line: FoldLine,
    angle_rad: float,
    settings: PalmFoldSettings,
) -> SkinnedHandAsset:
    """``asset`` with its palm folded by ``angle_rad`` about ``line``.

    Postconditions: same faces, bones and weights; a vertex with fold weight
    ``w`` has moved ``w`` of the way to where the whole turn about the hinge
    takes it, the fold side toward the palmar side; the carried bones and
    everything below them have turned rigidly, by the share of the angle at
    the first carried bone's head. Zero angle returns ``asset`` itself.
    """
    if not math.isfinite(angle_rad):
        raise ValueError("fold angle must be finite")
    if angle_rad == 0.0:
        return asset
    weights = fold_weights(asset, line, settings)
    signed = _turn_sign(line) * angle_rad
    rest = asset.rest_vertices
    whole = rotation_about_axis(line.direction, signed)
    turned = (rest - line.point) @ whole.T + line.point
    out = rest + weights[:, None] * (turned - rest)
    heads, tails = asset.bone_heads.copy(), asset.bone_tails.copy()
    axes = asset.flex_axes.copy()
    anchor = asset.bone_heads[asset.bone_index(settings.carried[0])]
    share = float(_share(asset, line, settings, anchor))
    turn = rotation_about_axis(line.direction, signed * share)
    # The base joint goes where the skin round it goes; the bones turn on it.
    moved = anchor + share * ((anchor - line.point) @ whole.T + line.point - anchor)
    for i in _carried(asset, settings):
        heads[i] = moved + turn @ (asset.bone_heads[i] - anchor)
        tails[i] = moved + turn @ (asset.bone_tails[i] - anchor)
        axes[i] = turn @ axes[i]
    return replace(
        asset, rest_vertices=out, bone_heads=heads, bone_tails=tails, flex_axes=axes
    )


def close_fold(settings: PalmFoldSettings, gap_at: Callable[[float], float]) -> float:
    """The furthest the fold closes, in radians.

    ``gap_at(angle_rad)`` is the least signed distance from the folding skin
    to what it closes on. The fold closes in steps of ``step_deg`` up to
    ``max_deg`` and stops before the step at which the skin would press in
    more than ``press_m`` further than the unfolded palm does.
    """
    floor = min(gap_at(0.0), 0.0) - settings.press_m
    step = math.radians(settings.step_deg)
    angle = 0.0
    for k in range(1, int(settings.max_deg / settings.step_deg + 1e-9) + 1):
        if gap_at(k * step) < floor:
            break
        angle = k * step
    return angle


def folding_ids(
    asset: SkinnedHandAsset, line: FoldLine, settings: PalmFoldSettings
) -> NDArray[np.int64]:
    """Vertices that the fold moves and that must stop on what the palm
    closes on: skin of ``bones`` and of the ``carried`` bones themselves (the
    pad at the thumb's base), not of the bones below those (the thumb, which
    is laid on the grip afterwards)."""
    moved = np.flatnonzero(fold_weights(asset, line, settings) > 0.0)
    pads = asset.dominant_vertex_ids((*settings.bones, *settings.carried))
    return np.intersect1d(moved, pads)


def _refit(hand: Any, asset: SkinnedHandAsset) -> Any:
    """``hand`` as it is placed and closed, on ``asset``."""
    new = type(hand)(asset, hand.settings)
    new.placement, new.press_axis = hand.placement, hand.press_axis
    new.fixed.update(hand.fixed)
    new.offsets.update(hand.offsets)
    for bone, swing in hand.side.items():
        new.swing_sideways(bone, swing)
    return new


def fold_palm(
    hand: Any,
    skin: SkinnedHandAsset,
    angles: Mapping[str, float],
    thumb_points: Array | None,
    obstacle: Callable[[Array], Array],
) -> tuple[Any, SkinnedHandAsset]:
    """Fold the palm of a posed ``hand`` round the thumb it lies on.

    ``hand`` is the solver's hand (its asset, settings, placement and
    ``points``), ``skin`` the asset it is shown with, ``thumb_points`` the
    other hand's thumb skin in the build frame and ``obstacle`` the signed
    distance to what the palm closes on. Returns the hand and the skin asset
    folded as far as :func:`close_fold` allows: the same objects if
    ``pair.palm_fold.max_deg`` is zero, there is no thumb, or no step closes.
    """
    settings: PalmFoldSettings = hand.settings.pair_palm_fold
    if thumb_points is None or settings.max_deg == 0.0:
        return hand, skin
    asset: SkinnedHandAsset = hand.asset
    wrist = hand.transforms(angles)[asset.bone_index("wrist")]
    rest_from_build = np.linalg.inv(hand.placement @ wrist)
    thumb = thumb_points @ rest_from_build[:3, :3].T + rest_from_build[:3, 3]
    line = fold_line(asset, thumb, settings)
    ids = folding_ids(asset, line, settings)
    if len(ids) == 0:
        return hand, skin

    def at(angle: float) -> Any:
        return _refit(hand, folded(asset, line, angle, settings))

    angle = close_fold(
        settings, lambda a: float(obstacle(at(a).points(angles, ids)).min())
    )
    if angle == 0.0:
        return hand, skin
    return at(angle), folded(skin, line, angle, settings)


__all__ = [
    "FoldLine",
    "PalmFoldSettings",
    "close_fold",
    "fold_line",
    "fold_palm",
    "fold_weights",
    "folded",
    "folding_ids",
]
