"""A set arch of the metacarpals: the rays of the palm turned at their bases.

The palm of the source hand cups only as its knuckles flex
(``curl.arch_gain``), and a finger's curl plane is fixed to its ray. A real
hand also moves each metacarpal on the wrist bones a little: it flexes toward
the palm and leans toward the thumb or the little finger, most at the little
finger and least at the middle one. This module adds that movement as a
setting per hand and per ray (``metacarpal_arch``, issue #15329, epic #15150).

A set ray turns about its own base, by one rotation whose axis lies across the
ray, so the ray never turns about its own length and the finger keeps its
place on it. The skin follows by the hand's own skin weights. A ray that is set
gives up the arch coupled to its knuckle, so the set angles are the whole of
its movement. Nothing is set as shipped, so every hand is drawn as before.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.hand_backends.joint_angles import FINGERS
from src.shared.python.model_appearance.hand_backends.skinning import (
    SkinnedHandAsset,
    rotation_about_axis,
)

Array = NDArray[np.float64]
SIDES = ("lead", "trail")
_ANGLES = ("flexion_deg", "ulnar_deviation_deg")


class _ArchedHand(Protocol):
    asset: SkinnedHandAsset
    fixed: dict[str, Array]

    @property
    def settings(self) -> Any: ...


def _numbers(raw: Mapping[str, Any], keys: tuple[str, str], name: str) -> list[Any]:
    if set(raw) != set(keys):
        raise ValueError(f"metacarpal_arch {name} needs exactly {keys}, got {set(raw)}")
    return [raw[key] for key in keys]


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"metacarpal_arch {name} must be finite")
    return number


Range = tuple[float, float]


@dataclass(frozen=True)
class MetacarpalArchSettings:
    """The set arch of each hand (``metacarpal_arch`` of the hand settings).

    ``limits_deg`` gives, per ray, the range its flexion (toward the palm
    positive) and its lean toward the little finger may be set in;
    ``max_step_deg`` the most two neighbouring rays may differ by in each, so
    the skin between their knuckles is not torn; ``hands``, per hand and per
    set ray, those two angles in degrees. A ray that is left out, or set to
    nothing, is not moved.
    """

    limits_deg: tuple[tuple[str, tuple[Range, Range]], ...]
    max_step_deg: tuple[float, float]
    hands: tuple[tuple[str, tuple[tuple[str, tuple[float, float]], ...]], ...]

    def __post_init__(self) -> None:
        limits = dict(self.limits_deg)
        if tuple(limits) != FINGERS:
            raise ValueError(f"metacarpal_arch limits_deg needs the rays {FINGERS}")
        if any(
            not low <= 0.0 <= high for pair in limits.values() for low, high in pair
        ):
            raise ValueError("metacarpal_arch limits_deg ranges must hold zero")
        if any(step < 0.0 for step in self.max_step_deg):
            raise ValueError("metacarpal_arch max_neighbour_step_deg is negative")
        if tuple(side for side, _ in self.hands) != SIDES:
            raise ValueError(f"metacarpal_arch needs the hands {SIDES}")
        for side, rays in self.hands:
            for ray, angles in rays:
                if ray not in limits:
                    raise ValueError(f"metacarpal_arch.{side} names no ray: {ray!r}")
                if any(
                    not lo <= a <= hi
                    for a, (lo, hi) in zip(angles, limits[ray], strict=True)
                ):
                    raise ValueError(
                        f"metacarpal_arch.{side}.{ray} {angles} is outside the "
                        f"joint's range {limits[ray]}"
                    )
            self._check_steps(side, dict(rays))

    def _check_steps(self, side: str, rays: Mapping[str, tuple[float, float]]) -> None:
        still = (0.0, 0.0)
        for ray, neighbour in zip(FINGERS, FINGERS[1:], strict=False):
            steps = np.subtract(rays.get(ray, still), rays.get(neighbour, still))
            if (np.abs(steps) > np.asarray(self.max_step_deg) + 1e-9).any():
                raise ValueError(
                    f"metacarpal_arch.{side}: {ray} and {neighbour} differ by more "
                    f"than max_neighbour_step_deg {self.max_step_deg}"
                )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> MetacarpalArchSettings:
        """The ``metacarpal_arch`` section. Preconditions: ``limits_deg`` has a
        ``flexion`` and an ``ulnar_deviation`` range for every ray, in the
        order of the fingers; each hand's rays have ``flexion_deg`` and
        ``ulnar_deviation_deg`` inside them (``ValueError`` otherwise)."""
        kinds = ("flexion", "ulnar_deviation")
        limits = []
        for ray, limit in raw["limits_deg"].items():
            ranges = _numbers(limit, kinds, f"limits_deg.{ray}")
            pair = tuple(
                (_finite(low, "limits_deg"), _finite(high, "limits_deg"))
                for low, high in ranges
            )
            limits.append((str(ray), pair))
        steps = _numbers(raw["max_neighbour_step_deg"], kinds, "max_neighbour_step_deg")
        hands = tuple(
            (
                side,
                tuple(
                    (
                        str(ray),
                        tuple(
                            _finite(angle, f"{side}.{ray}")
                            for angle in _numbers(angles, _ANGLES, f"{side}.{ray}")
                        ),
                    )
                    for ray, angles in raw[side].items()
                ),
            )
            for side in SIDES
        )
        return cls(
            tuple(limits),  # type: ignore[arg-type]
            (_finite(steps[0], "step"), _finite(steps[1], "step")),
            hands,  # type: ignore[arg-type]
        )

    def of(self, side: str) -> dict[str, tuple[float, float]]:
        """The rays of hand ``side`` that are moved, with their angles.

        Precondition: ``side`` is ``"lead"`` or ``"trail"``."""
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        return {ray: a for ray, a in dict(self.hands)[side] if a != (0.0, 0.0)}


def arch_rotation(
    asset: SkinnedHandAsset, ray: str, flexion_rad: float, deviation_rad: float
) -> Array:
    """Rotation of metacarpal ``ray`` about its base, in rest-frame axes.

    Preconditions: ``ray`` is a finger of ``asset`` whose metacarpal flexes;
    finite angles. Postcondition: one rotation whose axis lies across the
    ray, so it has no part about the ray's own length; ``flexion_rad`` turns
    it toward the palm and ``deviation_rad`` toward the little finger.
    """
    if ray not in FINGERS:
        raise ValueError(f"no metacarpal ray named {ray!r}")
    if not (math.isfinite(flexion_rad) and math.isfinite(deviation_rad)):
        raise ValueError("arch angles must be finite")
    i = asset.bone_index(f"meta_{ray}")
    along = asset.bone_tails[i] - asset.bone_heads[i]
    along = along / np.linalg.norm(along)
    across = asset.flex_axes[i] - (asset.flex_axes[i] @ along) * along
    if float(np.linalg.norm(across)) < 1e-9:
        raise ValueError(f"metacarpal {ray!r} has no axis to flex about")
    across = across / np.linalg.norm(across)
    turn = flexion_rad * across + deviation_rad * np.cross(across, along)
    angle = float(np.linalg.norm(turn))
    return np.eye(3) if angle == 0.0 else rotation_about_axis(turn, angle)


def set_metacarpal_arch(hand: _ArchedHand, side: str) -> None:
    """Turn the set rays of ``hand`` (the hand ``side`` of its settings).

    Postconditions: with nothing set for ``side`` the hand is untouched;
    otherwise each set ray carries its rotation before any other of its own.
    """
    arch = hand.settings.metacarpal_arch
    for ray, angles in arch.of(side).items():
        flexion, deviation = (math.radians(angle) for angle in angles)
        turn = arch_rotation(hand.asset, ray, flexion, deviation)
        bone = f"meta_{ray}"
        hand.fixed[bone] = turn @ hand.fixed[bone] if bone in hand.fixed else turn


def edge_stretch(
    before: Array, after: Array, faces: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Length of every triangle edge in ``after`` over its length in ``before``.

    Precondition: two vertex arrays of one shape. Postcondition: ``(F, 3)``.
    """
    if before.shape != after.shape:
        raise ValueError("edge_stretch needs two poses of one mesh")
    ends = (faces, np.roll(faces, -1, axis=1))
    was, now = (
        np.linalg.norm(points[ends[0]] - points[ends[1]], axis=2)
        for points in (before, after)
    )
    return np.asarray(now / was)


__all__ = [
    "MetacarpalArchSettings",
    "arch_rotation",
    "edge_stretch",
    "set_metacarpal_arch",
]
