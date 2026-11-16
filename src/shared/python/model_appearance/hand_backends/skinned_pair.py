"""Both skinned hands on one club (epic #15150, #15148).

The lead hand is posed on the bare grip. The trail hand is then posed around
the grip *and* the lead hand, whose digits stand in as capsules, so the two
hands rest on each other instead of passing through. This is a visual model
only; contact physics stays on the capsule hand.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from numpy.typing import NDArray

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends.joint_angles import FINGERS
from src.shared.python.model_appearance.hand_backends.posing import (
    ObstacleDistance,
    capsule_union_distance,
    cylinder_distance,
    inset_distance,
)
from src.shared.python.model_appearance.hand_backends.skinned_human import (
    _SEGMENTS,
    _THUMB_BONES,
    SkinnedHumanSettings,
    _bone_skin_radius,
    _check_inputs,
    _club_from_build,
    _result,
    _solve_right_hand,
    _SolvedHand,
    _targets,
    default_settings,
)
from src.shared.python.model_appearance.hand_backends.types import HandVisualResult

Array = NDArray[np.float64]


def _bone_capsules(
    solved: _SolvedHand, settings: SkinnedHumanSettings, radius_scale: float
) -> tuple[Array, Array, Array]:
    """The solved hand's digits and metacarpals as capsules (its build frame),
    ``radius_scale`` times as thick as the skin around each bone."""
    hand = solved.hand
    bones = [f"{f}_{s}" for f in FINGERS for s in _SEGMENTS]
    bones += [f"meta_{f}" for f in FINGERS] + list(_THUMB_BONES)
    starts = [hand.joint_position(b, solved.angles) for b in bones]
    ends = [hand.joint_position(b, solved.angles, tail=True) for b in bones]
    radii = [
        _bone_skin_radius(hand.asset, b)
        * radius_scale
        * (settings.pair_thumb_radius_scale if b in _THUMB_BONES else 1.0)
        + settings.pair_capsule_margin_m
        for b in bones
    ]
    return np.array(starts), np.array(ends), np.array(radii)


class _LeadObstacle:
    """The grip and the lead hand, seen from the trail hand's build frame."""

    def __init__(
        self, spec: GripSpec, grip_radius_m: float, settings: SkinnedHumanSettings
    ) -> None:
        self.spec, self.settings = spec, settings
        self.grip = cylinder_distance(grip_radius_m)
        self.lead = _solve_right_hand(spec, "lead", grip_radius_m, settings, None)
        scale = settings.pair_capsule_radius_scale
        self._capsules = _bone_capsules(self.lead, settings, scale)
        self._palm_radii = _bone_capsules(
            self.lead, settings, settings.pair_palm_capsule_radius_scale
        )[2]
        lead_targets = self.lead.targets
        self._club_from_lead = _club_from_build(
            spec, "lead", lead_targets.mirrored, 0.0
        )
        self._trail_mirrored = _targets(spec, "trail", settings).mirrored

    def at(
        self, spacing_m: float, inset_m: float = 0.0, *, palm: bool = False
    ) -> ObstacleDistance:
        """Grip plus lead-hand capsules with the trail hand ``spacing_m`` away.

        ``inset_m`` shrinks the grip (not the lead hand) by a squeeze depth.
        ``palm`` selects the capsules the trail palm must clear, which are
        scaled by ``pair.palm_capsule_radius_scale`` instead.
        """
        club_from_trail = _club_from_build(
            self.spec, "trail", self._trail_mirrored, spacing_m
        )
        move = np.linalg.inv(club_from_trail) @ self._club_from_lead
        starts, ends, radii = self._capsules
        radii = self._palm_radii if palm else radii
        return capsule_union_distance(
            inset_distance(self.grip, inset_m),
            starts @ move[:3, :3].T + move[:3, 3],
            ends @ move[:3, :3].T + move[:3, 3],
            radii,
        )


def _settled_spacing_m(scene: _LeadObstacle, grip_radius_m: float) -> float:
    """Spacing at which the trail hand's grip fingers find the grip free.

    Starting from the spec's ``hand_spacing_m``, the trail hand slides toward
    the club head by the least amount (at most ``pair.max_spacing_adjust_m``)
    that takes its ``pair.grip_fingers``, curled on the bare grip, off the
    lead hand. The fingers not listed (the little finger of an overlap grip)
    are left to rest on the lead hand.
    """
    spec, settings = scene.spec, scene.settings
    limit = settings.pair_max_spacing_adjust_m
    if limit == 0.0 or not settings.pair_grip_fingers:
        return spec.hand_spacing_m
    alone = _solve_right_hand(spec, "trail", grip_radius_m, settings, None)
    bones = [f"{f}_{s}" for f in settings.pair_grip_fingers for s in _SEGMENTS]
    hand = alone.hand
    skin = hand.points(alone.angles, hand.asset.dominant_vertex_ids(bones))
    allowance = settings.curl.penetration_allowance_m
    squeeze = settings.squeeze_finger_depth_m

    def free(extra: float) -> bool:
        distance = scene.at(spec.hand_spacing_m + extra, squeeze)
        return float(distance(skin).min()) >= -allowance

    if free(0.0):
        return spec.hand_spacing_m
    low, high = 0.0, limit
    for _ in range(settings.seat_iterations):
        mid = 0.5 * (low + high)
        low, high = (low, mid) if free(mid) else (mid, high)
    return spec.hand_spacing_m + high


def build_grip_hands(
    grip_spec: GripSpec,
    *,
    grip_radius_m: float,
    settings: SkinnedHumanSettings | None = None,
) -> dict[str, HandVisualResult]:
    """Both hands on one club, in the lead hand's grip frame.

    The lead hand is posed on the bare grip. The trail hand sits toward the
    club head (see :func:`_settled_spacing_m`) and is posed around the grip
    *and* the lead hand's digits, so it rests on them instead of passing
    through. Its ``contact_gaps_m`` are clearances to that combined obstacle.

    Preconditions: as :func:`skinned_human.build_hand_visual`. Postconditions:
    the lead hand equals the single-hand build; the trail hand's frame is at
    least ``hand_spacing_m`` toward the head.
    """
    _check_inputs(grip_spec, "lead", grip_radius_m)
    settings = settings or default_settings()
    scene = _LeadObstacle(grip_spec, grip_radius_m, settings)
    spacing = _settled_spacing_m(scene, grip_radius_m)
    trail = _solve_right_hand(
        grip_spec, "trail", grip_radius_m, settings, None, partial(scene.at, spacing)
    )
    return {
        "lead": _result(scene.lead, grip_spec, "lead", grip_radius_m, settings, 0.0),
        "trail": _result(trail, grip_spec, "trail", grip_radius_m, settings, spacing),
    }
