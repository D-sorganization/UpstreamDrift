"""Skinned human hands on a club, in a frame squared to the clubface (#15251).

:mod:`grip_hands` turns the capsule hands from the wrist followers, which left
the top of the grip under the hands 46 degrees round the shaft from the top of
the grip the clubface defines. Here the posed pair of
:mod:`hand_backends.skinned_pair` is placed in a frame taken from the clubface
alone: ``+x`` is the top of the grip (at right angles to the face normal),
``+z`` runs from butt to head and the origin is on the grip axis, the lead
hand's distance below the butt. Visual only: no force or contact model reads it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.club_assembly import ClubAssembly
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_hands import (
    HAND_PARTS,
    face_square_axes,
)
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.golf_grip import load_golf_grip
from src.shared.python.model_appearance.hand_backends.qa import (
    load_hand_visual_quality,
    wrist_joint_m,
)
from src.shared.python.model_appearance.hand_backends.skinned_pair import (
    build_grip_hands,
)
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    SkinnedHumanSettings,
    default_settings,
    load_settings_mapping,
    settings_from_mapping,
)
from src.shared.python.model_appearance.hand_backends.types import HandVisualResult


Array = NDArray[np.float64]

_GRIP_AXIS = np.array([0.0, 1.0, 0.0])  # butt -> head in the club-body frame
#: Half length of the band of hand skin about the wrist joint that is read.
_BAND_M = 0.004
#: Distance from the cut plane within which a vertex is on the cut rim.
_RIM_M = 1e-9


@dataclass(frozen=True)
class WristBlend:
    """Widening of a drawn hand's thin wrist to a human girth (#15299): the cut
    rim is drawn ``rim_scale`` times its size about its section's centre, brought
    within ``max_centre_shift_m`` of the joint, easing out over ``length_m``."""

    rim_scale: float
    length_m: float
    max_centre_shift_m: float

    def centred(self, centre: Array) -> Array:
        """``centre`` (off the wrist joint) pulled within the allowed shift."""
        return centre * min(1.0, self.max_centre_shift_m / np.linalg.norm(centre))

    def __post_init__(self) -> None:
        if not 1.0 <= self.rim_scale <= 1.5:
            raise ValueError(f"rim_scale must be in [1, 1.5], got {self.rim_scale}")
        if not (math.isfinite(self.length_m) and self.length_m > 0.0):
            raise ValueError(f"hand_blend_m must be positive, got {self.length_m}")
        if not self.max_centre_shift_m > 0.0:
            raise ValueError("max_centre_shift_m must be positive")


def wrist_blend() -> WristBlend:
    """The widening of the drawn hands' wrists, from the quality settings."""
    quality: Mapping[str, Any] = load_hand_visual_quality()
    shape = quality["forearm_wrist_shape"]
    margin, cover = float(quality["forearm_wrist_margin"]), float(shape["cover"])
    if not 0.9 <= cover <= margin:
        raise ValueError(f"cover must be in [0.9, forearm_wrist_margin], got {cover}")
    return WristBlend(
        margin / cover, float(shape["hand_blend_m"]), float(shape["max_centre_shift_m"])
    )


@dataclass(frozen=True)
class WristBall:
    """Wrist range (degrees either way) that the drawn hand's wrist ball
    covers, and the poses taken per half range (#15299)."""

    flexion_deg: float
    deviation_deg: float
    steps: int

    def __post_init__(self) -> None:
        angles = (self.flexion_deg, self.deviation_deg)
        if not (all(0.0 < a <= 90.0 for a in angles) and 1 <= self.steps <= 12):
            raise ValueError(
                f"ball needs angles in (0, 90] and 1 <= steps <= 12, got {self}"
            )


def wrist_ball() -> WristBall:
    """The drawn hands' wrist ball, from the quality settings."""
    quality: Mapping[str, Any] = load_hand_visual_quality()
    ball = quality["forearm_wrist_shape"]["ball"]
    return WristBall(
        float(ball["flexion_deg"]), float(ball["deviation_deg"]), int(ball["steps"])
    )


def _ball(rim: Array, along: Array, wide: Array, ball: WristBall) -> Mesh:
    """Convex hull of a hand's cut rim swept through the wrist's range.

    ``rim`` are the rim's points from the wrist joint, ``along`` the unit
    direction up the forearm and ``wide`` the wrist's width axis. Only what
    lies on or above the rim's plane is kept, so the hand below is untouched;
    the solid fills the wrist under a forearm that ends at the joint, at any
    bend within the range. Postcondition: closed, wound outward.
    """
    grid = np.linspace(-1.0, 1.0, 2 * ball.steps + 1)
    flex = np.radians(ball.flexion_deg) * wide
    deviate = np.radians(ball.deviation_deg) * np.cross(along, wide)
    turns = [f * flex + d * deviate for f in grid for d in grid if f * f + d * d <= 1.0]
    swept = np.einsum(
        "kij,nj->kni", Rotation.from_rotvec(turns).as_matrix(), rim
    ).reshape(-1, 3)
    swept = swept[swept @ along >= (rim @ along).min() - _RIM_M]
    hull = ConvexHull(swept)
    points, faces = swept[hull.vertices], np.searchsorted(hull.vertices, hull.simplices)
    a, b, c = (points[faces[:, k]] for k in range(3))
    inward = np.einsum("ij,ij->i", np.cross(b - a, c - a), hull.equations[:, :3]) < 0
    faces[inward] = faces[inward][:, ::-1]
    return Mesh(points, faces)


@dataclass(frozen=True)
class TrailSeat:
    """A named seat of the trail hand on the grip (#15150): how far off square
    the club lies across the hand, the bend of its drawn wrist and how far the
    palm's centre is rolled round the grip from where the grip puts it, degrees;
    and the set arch of its metacarpals (#15329), per ray its flexion and its
    lean toward the little finger, as ``metacarpal_arch`` of the hand settings;
    and the knuckles it abducts toward the little finger and holds in their
    sideways range, as ``side_close.knuckle_abduction_deg`` of the same."""

    name: str
    club_cross_angle_deg: float
    wrist_extension_deg: float
    wrist_ulnar_deviation_deg: float
    palm_centre_wrap_deg: float
    metacarpal_arch: tuple[tuple[str, tuple[float, float]], ...] = ()
    knuckle_abduction_deg: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        angles = (
            self.club_cross_angle_deg,
            self.wrist_extension_deg,
            self.wrist_ulnar_deviation_deg,
            self.palm_centre_wrap_deg,
        )
        if not all(math.isfinite(angle) for angle in angles):
            raise ValueError(f"trail seat {self.name!r} needs finite angles")
        if not 0.0 <= self.club_cross_angle_deg < 90.0:
            raise ValueError("the club lies 0 to 90 degrees across the trail hand")


def trail_seat(name: str | None = None) -> TrailSeat:
    """The trail hand's seat ``name`` of ``owner_trail_seats``.

    ``None`` is the seat named by ``owner_trail_seat``. What a seat leaves out
    is ``owner_trail_cross_angle_deg``, and ``wrist.trail``,
    ``placement.palm_centre_wrap_deg.trail``, ``metacarpal_arch.trail`` and
    ``side_close.knuckle_abduction_deg.trail`` of the hand settings.
    Preconditions: ``name`` is a string in the table (``TypeError`` /
    ``ValueError`` otherwise). Postcondition: finite angles, validated.
    """
    quality: Mapping[str, Any] = load_hand_visual_quality()
    if name is None:
        name = str(quality["owner_trail_seat"])
    if not isinstance(name, str):
        raise TypeError("a trail seat is named by a string")
    seats: Mapping[str, Mapping[str, Any]] = quality["owner_trail_seats"]
    if name not in seats:
        raise ValueError(f"no trail seat named {name!r}; have {sorted(seats)}")
    seat, hand = seats[name], load_settings_mapping()
    wrist = hand["wrist"]["trail"]
    wrap = hand["placement"]["palm_centre_wrap_deg"]["trail"]
    cross = seat.get("club_cross_angle_deg", quality["owner_trail_cross_angle_deg"])
    arch = seat.get("metacarpal_arch", hand["metacarpal_arch"]["trail"])
    usual_swings = hand["side_close"]["knuckle_abduction_deg"]["trail"]
    swings = seat.get("knuckle_abduction_deg", usual_swings)
    return TrailSeat(
        name=name,
        club_cross_angle_deg=float(cross),
        wrist_extension_deg=float(
            seat.get("wrist_extension_deg", wrist["extension_deg"])
        ),
        wrist_ulnar_deviation_deg=float(
            seat.get("wrist_ulnar_deviation_deg", wrist["ulnar_deviation_deg"])
        ),
        palm_centre_wrap_deg=float(seat.get("palm_centre_wrap_deg", wrap)),
        metacarpal_arch=tuple(
            (str(ray), (float(a["flexion_deg"]), float(a["ulnar_deviation_deg"])))
            for ray, a in arch.items()
        ),
        knuckle_abduction_deg=tuple((str(f), float(d)) for f, d in swings.items()),
    )


def seat_settings(seat: TrailSeat) -> SkinnedHumanSettings:
    """The hand settings with the trail wrist and palm as ``seat`` has them.

    Precondition: ``seat`` is a :class:`TrailSeat` whose arch the hand
    settings accept (``ValueError`` otherwise). Postcondition: the default
    settings themselves when the seat has what they have; else a copy that
    differs in the trail wrist's bend, the trail palm's roll and the trail
    hand's set metacarpal arch and abducted knuckles alone.
    """
    if not isinstance(seat, TrailSeat):
        raise TypeError("seat_settings needs a TrailSeat")
    bend = (seat.wrist_extension_deg, seat.wrist_ulnar_deviation_deg)
    usual = default_settings()
    same_wrap = dict(usual.palm_centre_wrap_deg)["trail"] == seat.palm_centre_wrap_deg
    same_arch = dict(usual.metacarpal_arch.hands)["trail"] == seat.metacarpal_arch
    swings = dict(usual.side_close_knuckle_abduction_deg)["trail"]
    same_arch = same_arch and swings == seat.knuckle_abduction_deg
    if dict(usual.wrist_deg)["trail"] == bend and same_wrap and same_arch:
        return usual
    raw = load_settings_mapping()
    raw["side_close"]["knuckle_abduction_deg"]["trail"] = dict(
        seat.knuckle_abduction_deg
    )
    raw["metacarpal_arch"] = {
        **raw["metacarpal_arch"],
        "trail": {
            ray: {"flexion_deg": flexion, "ulnar_deviation_deg": lean}
            for ray, (flexion, lean) in seat.metacarpal_arch
        },
    }
    raw["wrist"]["trail"] = {
        **raw["wrist"]["trail"],
        "extension_deg": bend[0],
        "ulnar_deviation_deg": bend[1],
    }
    wraps = raw["placement"]["palm_centre_wrap_deg"]
    raw["placement"]["palm_centre_wrap_deg"] = {
        **wraps,
        "trail": seat.palm_centre_wrap_deg,
    }
    return settings_from_mapping(raw)


def owner_grip_spec(roll_deg: float = 0.0, seat: str | None = None) -> GripSpec:
    """The legacy grip with the owner's settings from ``hand_visual_quality``.

    The lead thumb sits ``owner_thumb_azimuth_deg.lead`` to the away-from-target
    side of the top of the grip and the trail thumb ``.trail`` to the target
    side; the club leaves the lead palm ``owner_lead_heel_pad_depth_m`` below
    the little-finger knuckle; it lies as far off square across the trail hand
    as the trail seat ``seat`` has it (:func:`trail_seat`; #15265, #15150); the
    lead hand is ``owner_lead_below_cap_edge_m``
    below the edge of the butt cap. ``roll_deg`` turns both hands about the shaft.
    """
    base = GripSpec.legacy()
    quality: Mapping[str, Any] = load_hand_visual_quality()
    thumbs = quality["owner_thumb_azimuth_deg"]
    heel = float(quality["owner_lead_heel_pad_depth_m"])
    below = float(quality["owner_lead_below_cap_edge_m"])
    return replace(
        base,
        lead=replace(
            base.lead,
            pinky_side_palm_depth_m=heel,
            shaft_roll_deg=float(roll_deg),
            thumb_azimuth_deg=float(thumbs["lead"]),
        ),
        trail=replace(
            base.trail,
            shaft_roll_deg=float(roll_deg),
            thumb_azimuth_deg=float(thumbs["trail"]),
            club_cross_angle_deg=trail_seat(seat).club_cross_angle_deg,
        ),
        lead_below_butt_m=below + load_golf_grip().cap_height_m,
    )


def skinned_hand_frame(club: ClubAssembly) -> Array:
    """4x4 pose of the skinned pair's grip frame in the club-body frame.

    Postcondition: a right-handed rotation whose ``z`` is the grip axis and
    whose ``x`` is at right angles to the clubface normal.
    """
    top, side = face_square_axes(club)
    frame = np.eye(4)
    frame[:3, 0], frame[:3, 1], frame[:3, 2] = top, side, _GRIP_AXIS
    butt = np.array([0.0, -club.length_m, club.axis_offset_m])
    frame[:3, 3] = butt + owner_grip_spec().lead_below_butt_m * _GRIP_AXIS
    return frame


def build_seated_pair(seat: str | None = None) -> dict[str, HandVisualResult]:
    """The posed pair with the trail hand in seat ``seat``, in its grip frame.

    Precondition: as :func:`trail_seat`. Postcondition: as
    :func:`skinned_pair.build_grip_hands`; solving it takes seconds.
    """
    quality: Mapping[str, Any] = load_hand_visual_quality()
    return build_grip_hands(
        owner_grip_spec(seat=seat),
        grip_radius_m=float(quality["grip_radius_m"]),
        settings=seat_settings(trail_seat(seat)),
    )


@lru_cache(maxsize=1)
def _pair() -> dict[str, HandVisualResult]:
    """The posed pair of the configured seat (``owner_trail_seat``)."""
    return build_seated_pair()


def forearm_cuff_m() -> float:
    """Length of forearm skin kept on a hand drawn on the club (a setting)."""
    quality: Mapping[str, Any] = load_hand_visual_quality()
    return float(quality["club_hand_forearm_cuff_m"])


def _joint_m(result: HandVisualResult, bone: str) -> Array:
    """Posed head of ``bone`` in the pair's grip frame."""
    asset, transforms = result.rest_asset, result.bone_transforms
    if asset is None or transforms is None or result.frame_from_rest is None:
        raise ValueError("result carries no skinning audit data")
    i = asset.bone_index(bone)
    posed = result.frame_from_rest @ transforms[i] @ np.append(asset.bone_heads[i], 1.0)
    return np.asarray(posed[:3], dtype=float)


def forearm_reach_m(club: ClubAssembly) -> dict[str, tuple[Array, Array]]:
    """Per hand: (wrist joint, unit direction from wrist to elbow), club frame."""
    frame, pair = skinned_hand_frame(club), _pair()
    reach = {}
    for side, result in pair.items():
        wrist = wrist_joint_m(result)
        along = _joint_m(result, "forearm") - wrist
        along = frame[:3, :3] @ (along / np.linalg.norm(along))
        reach[side] = (frame[:3, :3] @ wrist + frame[:3, 3], along)
    return reach


def _capped(vertices: Array, faces: NDArray[np.int64]) -> tuple[Array, NDArray]:
    """Close every open edge loop of a mesh with a fan about the loop's centre."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    paired = {(int(a), int(b)) for a, b in edges}
    following = {a: b for a, b in paired if (b, a) not in paired}
    points, fans = [vertices], []
    count = len(vertices)
    while following:
        start, nxt = following.popitem()
        loop = [start]
        while nxt != start:
            loop.append(nxt)
            nxt = following.pop(nxt)
        points.append(vertices[loop].mean(axis=0, keepdims=True))
        fans += [(loop[(k + 1) % len(loop)], loop[k], count) for k in range(len(loop))]
        count += 1
    if not fans:
        return vertices, faces
    return np.concatenate(points), np.concatenate([faces, np.array(fans)])


def _flat_rim(
    vertices: Array, faces: NDArray[np.int64], point: Array, normal: Array
) -> Array:
    """``vertices`` with every open-edge vertex moved along ``normal`` onto the
    plane through ``point``, so a cut that dropped whole triangles ends in one
    flat ring instead of a ragged edge."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    paired = {(int(a), int(b)) for a, b in edges}
    rim = sorted({a for a, b in paired if (b, a) not in paired})
    out = vertices.copy()
    out[rim] += np.outer((point - out[rim]) @ normal, normal)
    return out


def wrist_section(rel: Array, along: Array) -> tuple[float, float, Array, Array]:
    """Half width, half depth, width axis and centre of a hand's wrist skin:
    ``rel`` are skin points from the wrist joint, ``along`` the unit direction
    up the forearm; points within ``_BAND_M`` of the joint are read."""
    band = rel[np.abs(rel @ along) < _BAND_M]
    flat = band - np.outer(band @ along, along)
    _, axes = np.linalg.eigh(np.cov(flat.T))
    wide, deep = axes[:, 2], axes[:, 1]
    (w0, w1), (d0, d1) = ((np.min(flat @ k), np.max(flat @ k)) for k in (wide, deep))
    centre = 0.5 * (w0 + w1) * wide + 0.5 * (d0 + d1) * deep
    return float(0.5 * (w1 - w0)), float(0.5 * (d1 - d0)), wide, centre


def _widened_wrist(rel: Array, along: Array, blend: WristBlend) -> Array:
    """Skin points ``rel`` (from the wrist joint) grown about the wrist's axis.

    The cut rim lands on the ellipse of the hand's own wrist section,
    ``blend.rim_scale`` times its size; the growth eases to none
    ``blend.length_m`` down the hand. Postcondition: no point moves along the arm.
    """
    height = rel @ along
    half_width, half_depth, wide, centre = wrist_section(rel, along)
    flat = rel - np.outer(height, along) - centre
    x, y = flat @ wide / half_width, flat @ np.cross(along, wide) / half_depth
    angle, reach = np.arctan2(y, x), np.hypot(x, y)  # reach 1: on the ellipse
    rim = np.flatnonzero(np.abs(height - forearm_cuff_m()) < _RIM_M)
    rim = rim[np.argsort(angle[rim])]
    rim_reach = np.interp(angle, angle[rim], reach[rim], period=2.0 * np.pi)
    t = np.clip((forearm_cuff_m() - height) / blend.length_m, 0.0, 1.0)
    ease = (1.0 - t * t * (3.0 - 2.0 * t))[:, None]
    grown = (blend.rim_scale / rim_reach - 1.0)[:, None] * flat
    return rel + ease * (grown + blend.centred(centre) - centre)


def skinned_hand_meshes(
    club: ClubAssembly,
    *,
    cut_at_wrist: bool = True,
    widen_wrist: bool = True,
    with_ball: bool = False,
) -> dict[str, Mesh]:
    """``{"lead_hand", "trail_hand"}`` skinned meshes in the club-body frame.

    The hands are posed on the standard golf grip of ``golf_grip.json``, not on
    the club's drawn grip tube, which is up to about a millimetre fatter. With
    ``cut_at_wrist`` every triangle that reaches more than
    :func:`forearm_cuff_m` up the forearm from the wrist joint is dropped, so a
    model's own forearm continues from the hand; the cut edge is moved onto
    that plane and closed flat. With ``widen_wrist`` as well the cut hand's
    wrist is widened by :func:`wrist_blend` to a human girth (#15299).
    ``with_ball`` adds the :func:`wrist_ball` solid above the cut, for drawing.
    """
    frame, pair = skinned_hand_frame(club), _pair()
    reach = forearm_reach_m(club)
    meshes = {}
    for hand, label in HAND_PARTS.items():
        mesh = pair[hand.value].mesh
        vertices = mesh.vertices @ frame[:3, :3].T + frame[:3, 3]
        faces = mesh.faces
        if cut_at_wrist:
            wrist, along = reach[hand.value]
            keep = (vertices - wrist) @ along <= forearm_cuff_m()
            faces = faces[keep[faces].all(axis=1)]
            rim = wrist + forearm_cuff_m() * along
            vertices = _flat_rim(vertices, faces, rim, along)
            if widen_wrist:
                drawn = np.unique(faces)
                vertices[drawn] = wrist + _widened_wrist(
                    vertices[drawn] - wrist, along, wrist_blend()
                )
            rel = vertices[np.unique(faces)] - wrist
            rim = rel[np.abs(rel @ along - forearm_cuff_m()) < _RIM_M]
            vertices, faces = _capped(vertices, faces)
            if with_ball:
                ball = _ball(rim, along, wrist_section(rel, along)[2], wrist_ball())
                faces = np.concatenate([faces, ball.faces + len(vertices)])
                vertices = np.concatenate([vertices, wrist + ball.vertices])
        meshes[label] = Mesh(vertices, faces)
    return meshes


def wrist_joints_m(club: ClubAssembly) -> dict[str, Array]:
    """Wrist joint centre of each hand (``lead``, ``trail``) in the club-body frame.

    This is where a forearm attaches to the posed hand.
    """
    frame, pair = skinned_hand_frame(club), _pair()
    return {
        hand.value: frame[:3, :3] @ wrist_joint_m(pair[hand.value]) + frame[:3, 3]
        for hand in (Hand.LEAD, Hand.TRAIL)
    }
