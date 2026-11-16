"""Both skinned hands on one club (epic #15150, #15148).

The lead hand is posed on the bare grip. The trail hand is then slid along the
grip until its fingers meet the lead forefinger and posed around the grip
*and* the lead hand (its skin, or capsules standing in for its digits), so
the two hands rest on each other instead of passing through. In an overlap
grip the trail little finger lies in the groove between the lead forefinger
and middle finger. This is a visual model only; contact physics stays on the
capsule hand.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
from numpy.typing import NDArray

from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends.joint_angles import FINGERS
from src.shared.python.model_appearance.hand_backends.posing import (
    ObstacleDistance,
    capsule_union_distance,
    cylinder_distance,
    inset_distance,
    skin_distance,
)
from src.shared.python.model_appearance.hand_backends.skinned_human import (
    _SEGMENTS,
    _THUMB_BONES,
    SkinnedHumanSettings,
    _bone_skin_radius,
    _check_inputs,
    _club_from_build,
    _result,
    _Scene,
    _solve_right_hand,
    _SolvedHand,
    _targets,
    default_settings,
)
from src.shared.python.model_appearance.hand_backends.skinning import vertex_normals
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


def _nothing(points: Array) -> Array:
    """The distance to no obstacle at all."""
    return np.full(len(points), np.inf)


class _LeadObstacle:
    """The grip and the lead hand, seen from the trail hand's build frame."""

    def __init__(
        self, spec: GripSpec, grip_radius_m: float, settings: SkinnedHumanSettings
    ) -> None:
        self.spec, self.settings = spec, settings
        self.grip = cylinder_distance(grip_radius_m)
        self.lead = _solve_right_hand(spec, "lead", grip_radius_m, settings, None)
        #: The lead hand as it is shown, in the club frame.
        self.lead_result = _result(
            self.lead, spec, "lead", grip_radius_m, settings, 0.0
        )
        shown = self.lead_result.mesh
        #: The lead hand's skin vertices, in the club frame.
        self._lead_skin: Array = shown.vertices
        self._skin = skin_distance(
            self._lead_skin,
            vertex_normals(shown.vertices, shown.faces),
            settings.pair_skin_reach_m,
        )
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

    def hand(self, spacing_m: float, *, palm: bool = False) -> ObstacleDistance:
        """The lead hand alone with the trail hand ``spacing_m`` away.

        It is its skin, ``pair.skin_margin_m`` thicker, or, with
        ``pair.obstacle`` set to ``capsules``, its digits as capsules;
        ``palm`` then selects the capsules the trail palm must clear, which
        are scaled by ``pair.palm_capsule_radius_scale`` instead.
        """
        club_from_trail = _club_from_build(
            self.spec, "trail", self._trail_mirrored, spacing_m
        )
        if self.settings.pair_obstacle == "skin":
            turn, shift = club_from_trail[:3, :3].T, club_from_trail[:3, 3]
            margin = self.settings.pair_skin_margin_m
            return lambda points: self._skin(points @ turn + shift) - margin
        move = np.linalg.inv(club_from_trail) @ self._club_from_lead
        starts, ends, radii = self._capsules
        return capsule_union_distance(
            _nothing,
            starts @ move[:3, :3].T + move[:3, 3],
            ends @ move[:3, :3].T + move[:3, 3],
            self._palm_radii if palm else radii,
        )

    def at(
        self, spacing_m: float, inset_m: float = 0.0, *, palm: bool = False
    ) -> ObstacleDistance:
        """Grip plus lead hand (see :meth:`hand`) for a trail hand ``spacing_m``
        away. ``inset_m`` shrinks the grip, not the lead hand, by a squeeze."""
        grip = inset_distance(self.grip, inset_m)
        lead = self.hand(spacing_m, palm=palm)
        return lambda points: np.minimum(grip(points), lead(points))

    def skin_points(self, spacing_m: float) -> Array:
        """The lead hand's skin vertices in the trail hand's build frame."""
        club_from_trail = _club_from_build(
            self.spec, "trail", self._trail_mirrored, spacing_m
        )
        turn, shift = club_from_trail[:3, :3], club_from_trail[:3, 3]
        return (self._lead_skin - shift) @ np.linalg.inv(turn).T

    def groove(self, spacing_m: float) -> Callable[[Array], Array] | None:
        """Where the trail little finger overlaps, in the trail build frame.

        Signed: the distance to the lead forefinger's bones less that to the
        lead middle finger's, so zero along the groove between them;
        ``pair.overlap_bias_m`` moves that zero toward the middle finger.
        ``None`` unless the grip style is overlap.
        """
        if self.spec.style is not GripStyle.OVERLAP:
            return None
        hand, angles = self.lead.hand, self.lead.angles
        turn, shift = self._club_from_lead[:3, :3].T, self._club_from_lead[:3, 3]

        def bones(finger: str) -> ObstacleDistance:
            names = [f"{finger}_{s}" for s in _SEGMENTS]
            ends = [
                np.array([hand.joint_position(b, angles, tail=t) for b in names]) @ turn
                + shift
                for t in (False, True)
            ]
            return capsule_union_distance(_nothing, ends[0], ends[1], np.zeros(3))

        index, middle = bones("index"), bones("middle")
        bias = self.settings.pair_overlap_bias_m
        club_from_trail = _club_from_build(
            self.spec, "trail", self._trail_mirrored, spacing_m
        )
        to_club, station = club_from_trail[:3, :3].T, club_from_trail[:3, 3]

        def offset(points: Array) -> Array:
            in_club = points @ to_club + station
            return index(in_club) - middle(in_club) - bias

        return offset


def _settled_spacing_m(scene: _LeadObstacle, trail: _SolvedHand, at_m: float) -> float:
    """Spacing at which the trail hand's grip fingers just meet the lead hand.

    ``trail`` is the trail hand as posed ``at_m`` from the lead hand. Held in
    that pose it slides along the grip until its ``pair.grip_fingers`` stand
    ``pair.finger_gap_m`` off the lead hand, staying within
    ``pair.max_spacing_close_m`` toward the lead hand and
    ``pair.max_spacing_adjust_m`` toward the club head of the spec's
    ``hand_spacing_m``. The fingers not listed (the little finger of an
    overlap grip) are left to rest on the lead hand.
    """
    spec, settings = scene.spec, scene.settings
    nearest = spec.hand_spacing_m - min(
        settings.pair_max_spacing_close_m, 0.5 * spec.hand_spacing_m
    )
    farthest = spec.hand_spacing_m + settings.pair_max_spacing_adjust_m
    bones = [f"{f}_{s}" for f in settings.pair_grip_fingers for s in _SEGMENTS]
    hand = trail.hand
    skin = hand.points(trail.angles, hand.asset.dominant_vertex_ids(bones))
    need = settings.pair_finger_gap_m
    if settings.pair_obstacle != "skin":  # slimmed capsules: let the skin in
        need = -settings.curl.penetration_allowance_m

    def free(spacing_m: float) -> bool:
        return float(scene.hand(spacing_m)(skin).min()) >= need

    low, high = (nearest, at_m) if free(at_m) else (at_m, farthest)
    if low >= high or free(low):
        return min(low, high)
    for _ in range(settings.seat_iterations):
        mid = 0.5 * (low + high)
        low, high = (low, mid) if free(mid) else (mid, high)
    return high


def build_grip_hands(
    grip_spec: GripSpec,
    *,
    grip_radius_m: float,
    settings: SkinnedHumanSettings | None = None,
) -> dict[str, HandVisualResult]:
    """Both hands on one club, in the lead hand's grip frame.

    The lead hand is posed on the bare grip. The trail hand sits toward the
    club head, up against the lead forefinger (see :func:`_settled_spacing_m`),
    and is posed around the grip *and* the lead hand, so it rests on it
    instead of passing through. Its ``contact_gaps_m`` are clearances to that
    combined obstacle.

    Preconditions: as :func:`skinned_human.build_hand_visual`. Postconditions:
    the lead hand equals the single-hand build; the trail hand's frame is
    toward the head, within ``pair.max_spacing_close_m`` and
    ``pair.max_spacing_adjust_m`` of ``hand_spacing_m``.
    """
    _check_inputs(grip_spec, "lead", grip_radius_m)
    settings = settings or default_settings()
    scene = _LeadObstacle(grip_spec, grip_radius_m, settings)

    def posed(spacing_m: float) -> _SolvedHand:
        around = _Scene(
            partial(scene.at, spacing_m),
            scene.groove(spacing_m),
            scene.skin_points(spacing_m),
        )
        return _solve_right_hand(
            grip_spec, "trail", grip_radius_m, settings, None, around
        )

    spacing = grip_spec.hand_spacing_m
    trail = posed(spacing)
    slides = settings.pair_max_spacing_close_m or settings.pair_max_spacing_adjust_m
    for _ in range(settings.pair_spacing_passes if slides else 0):
        if not settings.pair_grip_fingers:
            break
        spacing = _settled_spacing_m(scene, trail, spacing)
        trail = posed(spacing)
    return {
        "lead": scene.lead_result,
        "trail": _result(trail, grip_spec, "trail", grip_radius_m, settings, spacing),
    }
