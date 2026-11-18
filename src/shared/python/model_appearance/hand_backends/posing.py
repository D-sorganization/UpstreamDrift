"""Shared curl-to-contact hand posing (epic #15150).

Every visual hand backend can pose itself with this module; none of it knows
about a particular mesh. It follows Dieter's grip spec:

1. **Flat hand beside the club.** The club axis crosses the base of the
   fingers at a slight angle: it runs under the first segment of the index
   finger and, on the other side, just below the base of the little finger in
   the palm (:func:`club_axis_in_hand`, :func:`flat_hand_rotation`,
   :func:`seat_on_obstacle`).
2. **Curl.** Each digit flexes proximal to distal (MCP, then PIP, then DIP).
   A joint closes until the skin of its own segment touches the grip, or until
   anything it moves would go deeper than the penetration allowance, or until
   its limit (:func:`curl_to_contact`). Joints may drive coupled joints, which
   is how the metacarpal arch follows the knuckle so the palm cups.

A hand is posed **only by joint angles**. The backend supplies skin points for
a set of hinge angles through the :class:`PosableHand` protocol; this module
only chooses angles and one rigid transform. Contact physics is not modelled
here: this is the visual model, and the obstacle is any signed distance field
(the grip cylinder by default, optionally with the other hand added).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]
#: Signed distance (metres) from points ``(N, 3)`` to the obstacle; > 0 outside.
ObstacleDistance = Callable[[Array], Array]


def _finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _unit(vector: Array, name: str) -> Array:
    v = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(v))
    if v.shape != (3,) or not np.isfinite(v).all() or norm < 1e-12:
        raise ValueError(f"{name} must be a finite non-zero 3-vector")
    return v / norm


@dataclass(frozen=True)
class CoupledJoint:
    """A joint that follows its driver: ``angle = min(limit, gain * driver)``."""

    bone: str
    gain: float
    limit_rad: float

    def __post_init__(self) -> None:
        if _finite(self.gain, "gain") < 0.0 or _finite(self.limit_rad, "limit") < 0.0:
            raise ValueError("coupled joint gain and limit must be non-negative")

    def angle(self, driver_rad: float) -> float:
        return min(self.limit_rad, max(0.0, self.gain * driver_rad))


@dataclass(frozen=True)
class CurlJoint:
    """One hinge of a digit. Angles are flexion in radians, positive = curl."""

    bone: str
    lower_rad: float
    upper_rad: float
    start_rad: float
    coupled: tuple[CoupledJoint, ...] = ()

    def __post_init__(self) -> None:
        for name in ("lower_rad", "upper_rad", "start_rad"):
            _finite(getattr(self, name), name)
        if self.lower_rad > self.upper_rad:
            raise ValueError(f"{self.bone}: lower limit above upper limit")
        if not self.lower_rad <= self.start_rad <= self.upper_rad:
            raise ValueError(f"{self.bone}: start angle outside its limits")


@dataclass(frozen=True)
class CurlChain:
    """The joints of one digit, proximal first."""

    name: str
    joints: tuple[CurlJoint, ...]
    #: Segments that should end up touching (empty = every joint's segment).
    must_touch: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.joints:
            raise ValueError(f"chain {self.name!r} has no joints")
        bones = {joint.bone for joint in self.joints}
        if not set(self.must_touch) <= bones:
            raise ValueError(f"chain {self.name!r}: must_touch names an unknown joint")


@dataclass(frozen=True)
class CurlSettings:
    """Numerical settings of the curl search (all from configuration)."""

    step_rad: float
    bisection_steps: int
    contact_tolerance_m: float
    penetration_allowance_m: float

    def __post_init__(self) -> None:
        if not 0.0 < _finite(self.step_rad, "step_rad") <= math.pi / 4:
            raise ValueError("step_rad must be in (0, pi/4]")
        if self.bisection_steps < 1:
            raise ValueError("bisection_steps must be at least 1")
        if _finite(self.contact_tolerance_m, "contact_tolerance_m") <= 0.0:
            raise ValueError("contact_tolerance_m must be positive")
        if _finite(self.penetration_allowance_m, "penetration_allowance_m") < 0.0:
            raise ValueError("penetration_allowance_m must be non-negative")


class PosableHand(Protocol):
    """What a backend offers so that this module can pose it."""

    def points(self, angles: Mapping[str, float], ids: IntArray | None = None) -> Array:
        """Skin points ``(N, 3)`` in the obstacle frame for these hinge angles;
        only the points ``ids`` (in that order) when given."""

    def segment_ids(self, bone: str) -> IntArray:
        """Indices of the skin points that belong to ``bone``'s own segment."""

    def moved_ids(self, bone: str) -> IntArray:
        """Indices of every skin point that moves when ``bone`` rotates."""


@dataclass(frozen=True)
class CurlResult:
    """Solved hinge angles, the gap left under each segment, and solve order."""

    angles: dict[str, float]
    gaps: dict[str, float]
    order: tuple[str, ...]


def cylinder_distance(radius_m: float, taper: float = 0.0) -> ObstacleDistance:
    """Distance to the grip about ``z``: ``radius_m`` at ``z = 0``, changing
    by ``taper`` per metre of ``z`` (zero: an infinite cylinder; a golf grip
    narrows toward the head by about 0.015). It is the true distance to the
    cone, measured square to its surface, so a step of that length along the
    gradient lands on it."""
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("cylinder radius must be positive and finite")
    if not math.isfinite(taper):
        raise ValueError("grip taper must be finite")

    slant = math.hypot(1.0, taper)

    def distance(points: Array) -> Array:
        radius = radius_m + taper * points[:, 2]
        return (np.hypot(points[:, 0], points[:, 1]) - radius) / slant

    return distance


def inset_distance(distance: ObstacleDistance, depth_m: float) -> ObstacleDistance:
    """``distance`` to the obstacle shrunk by ``depth_m`` all round.

    A hand that closes on the shrunk obstacle presses its soft pads
    ``depth_m`` into the real one. ``depth_m == 0`` returns ``distance``.
    """
    depth = _finite(depth_m, "depth_m")
    if depth < 0.0:
        raise ValueError("inset depth must not be negative")
    if depth == 0.0:
        return distance
    return lambda points: distance(points) + depth


def capsule_union_distance(
    base: ObstacleDistance, starts: Array, ends: Array, radii: Array
) -> ObstacleDistance:
    """``base`` united with capsules (segment ``starts[i]``-``ends[i]``, ``radii[i]``).

    Used to pose one hand around the grip *and* the other hand's digits.
    """
    a = np.asarray(starts, dtype=float).reshape(-1, 3)
    b = np.asarray(ends, dtype=float).reshape(-1, 3)
    r = np.asarray(radii, dtype=float).reshape(-1)
    if (
        not (len(a) == len(b) == len(r))
        or not np.isfinite(np.c_[a, b, r[:, None]]).all()
    ):
        raise ValueError("capsules need matching finite starts, ends and radii")
    ab = b - a
    denom = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18)

    def distance(points: Array) -> Array:
        rel = points[:, None, :] - a[None, :, :]
        t = np.clip(np.einsum("nkj,kj->nk", rel, ab) / denom, 0.0, 1.0)
        off = rel - t[:, :, None] * ab[None, :, :]
        capsules = (np.linalg.norm(off, axis=2) - r).min(axis=1)
        return np.minimum(base(points), capsules)

    return distance


def skin_distance(
    vertices: Array, normals: Array, reach_m: float = math.inf
) -> ObstacleDistance:
    """Signed distance to a closed skin given by ``vertices`` and outward ``normals``.

    Near the skin the distance is the height above the tangent plane of the
    nearest vertex, which is exact to second order in the vertex spacing; far
    from it, the distance to that vertex. The sign is that of the height, so
    points under the skin are negative. Points farther than ``reach_m`` from
    every vertex are simply reported ``reach_m`` away, which keeps the search
    short. Used to pose one hand around the other hand's actual skin.
    """
    from scipy.spatial import cKDTree

    v = np.asarray(vertices, dtype=float).reshape(-1, 3)
    n = np.asarray(normals, dtype=float).reshape(-1, 3)
    if len(v) != len(n) or len(v) < 4 or not np.isfinite(np.c_[v, n]).all():
        raise ValueError("a skin needs matching finite vertices and normals")
    if not reach_m > 0.0:
        raise ValueError("skin reach_m must be positive")
    tree = cKDTree(v)
    spacing = float(np.median(np.asarray(tree.query(v, k=2)[0])[:, 1]))

    def distance(points: Array) -> Array:
        found = tree.query(points, distance_upper_bound=reach_m)
        reach, nearest = np.asarray(found[0]), np.asarray(found[1])
        near = np.isfinite(reach)
        nearest = nearest[near]
        height = np.einsum("ij,ij->i", points[near] - v[nearest], n[nearest])
        size = np.maximum(np.abs(height), reach[near] - spacing)
        out = np.full(len(points), reach_m)
        out[near] = np.where(height < 0.0, -size, size)
        return out

    return distance


def _apply(
    joint: CurlJoint, angle: float, angles: dict[str, float]
) -> dict[str, float]:
    out = dict(angles)
    out[joint.bone] = angle
    for coupled in joint.coupled:
        out[coupled.bone] = coupled.angle(angle)
    return out


def _curl_joint(
    hand: PosableHand,
    joint: CurlJoint,
    obstacle: ObstacleDistance,
    settings: CurlSettings,
    angles: dict[str, float],
) -> tuple[dict[str, float], float]:
    """Close one joint until its segment touches or something would penetrate."""
    # Only the skin this joint moves can change its clearance.
    watched = np.union1d(hand.moved_ids(joint.bone), hand.segment_ids(joint.bone))
    segment = np.searchsorted(watched, hand.segment_ids(joint.bone))
    moved = np.searchsorted(watched, hand.moved_ids(joint.bone))

    def clearances(angle: float) -> tuple[float, float]:
        d = obstacle(hand.points(_apply(joint, angle, angles), watched))
        return float(d[segment].min()), float(d[moved].min())

    def free(angle: float) -> bool:
        return clearances(angle)[1] >= -settings.penetration_allowance_m

    # A joint that an earlier coupling has already curled carries on from there.
    angle = min(
        joint.upper_rad, max(joint.start_rad, angles.get(joint.bone, joint.start_rad))
    )
    if not free(angle):
        # Already inside the obstacle: open the joint until it is clear.
        while angle > joint.lower_rad and not free(angle):
            angle = max(joint.lower_rad, angle - settings.step_rad)
        return _apply(joint, angle, angles), clearances(angle)[0]
    low = high = angle
    blocked = False
    while high < joint.upper_rad:
        high = min(joint.upper_rad, low + settings.step_rad)
        if not free(high):
            blocked = True
            break
        low = high
    if blocked:
        for _ in range(settings.bisection_steps):
            mid = 0.5 * (low + high)
            if free(mid):
                low = mid
            else:
                high = mid
    return _apply(joint, low, angles), clearances(low)[0]


def curl_to_contact(
    hand: PosableHand,
    chains: Sequence[CurlChain],
    obstacle: ObstacleDistance,
    settings: CurlSettings,
    initial_angles: Mapping[str, float],
) -> CurlResult:
    """Curl every chain, proximal joint first, until the skin meets the obstacle.

    Preconditions: ``chains`` non-empty; every joint bone is known to ``hand``.
    Postconditions: every solved angle lies inside its joint limits; no point
    that a solved joint moves is deeper in the obstacle than
    ``penetration_allowance_m`` (unless it already was at the lower limit).
    ``gaps[bone]`` is the clearance left under that bone's own segment.
    """
    if not chains:
        raise ValueError("curl_to_contact needs at least one chain")
    angles = {
        str(k): _finite(v, f"initial angle {k}") for k, v in initial_angles.items()
    }
    gaps: dict[str, float] = {}
    order: list[str] = []
    for chain in chains:
        for joint in chain.joints:
            angles, gaps[joint.bone] = _curl_joint(
                hand, joint, obstacle, settings, angles
            )
            order.append(joint.bone)
    # A later, more distal joint moves the skin near the joints before it, so
    # report every gap again in the final pose.
    final = obstacle(hand.points(angles))
    for bone in order:
        gaps[bone] = float(final[hand.segment_ids(bone)].min())
    for chain in chains:
        for joint in chain.joints:
            if (
                not joint.lower_rad - 1e-12
                <= angles[joint.bone]
                <= joint.upper_rad + 1e-12
            ):
                raise AssertionError(f"{joint.bone} left its joint limits")
    return CurlResult(angles=angles, gaps=gaps, order=tuple(order))


def _chain_cost(
    hand: PosableHand,
    chain: CurlChain,
    obstacle: ObstacleDistance,
    settings: CurlSettings,
    angles: Mapping[str, float],
) -> tuple[float, float]:
    """(contact shortfall in metres, largest fraction of a joint's range used)."""
    distance = obstacle(hand.points(angles))
    wanted = chain.must_touch or tuple(joint.bone for joint in chain.joints)
    shortfall = sum(
        max(
            0.0,
            float(distance[hand.segment_ids(b)].min()) - settings.contact_tolerance_m,
        )
        for b in wanted
    )
    used = max(
        (angles[j.bone] - j.lower_rad) / max(j.upper_rad - j.lower_rad, 1e-12)
        for j in chain.joints
    )
    return shortfall, used


def wrap_to_contact(
    hand: PosableHand,
    chains: Sequence[CurlChain],
    obstacle: ObstacleDistance,
    settings: CurlSettings,
    initial_angles: Mapping[str, float],
    *,
    samples: int,
    target: ObstacleDistance | None = None,
) -> CurlResult:
    """Curl each chain, then look for a better wrap than the greedy curl.

    The greedy curl (:func:`curl_to_contact`) closes the first joint as far
    as it goes, so a digit whose tip lands first is left hooked, its middle
    segment off the obstacle. Here the chain's first joint is also tried at
    ``samples`` angles across its range, the joints after it are closed to
    contact in order, and the pose is kept that leaves the chain's
    ``must_touch`` segments closest to the obstacle; among equally good poses
    the one that uses the smallest share of any joint's range wins. No pose
    that penetrates beyond the allowance is considered. ``target``, if given,
    is what the ``must_touch`` segments should lie on, when that is only part
    of ``obstacle`` (the grip, with the other hand also in the way).

    Preconditions and postconditions are those of :func:`curl_to_contact`;
    ``samples`` must be at least 2.
    """
    if samples < 2:
        raise ValueError("samples must be at least 2")
    angles = {str(k): float(v) for k, v in initial_angles.items()}
    order: list[str] = []
    wanted = target or obstacle
    for chain in chains:
        first, rest = chain.joints[0], chain.joints[1:]
        moved = hand.moved_ids(first.bone)
        best_angles = curl_to_contact(hand, (chain,), obstacle, settings, angles).angles
        best_cost = _chain_cost(hand, chain, wanted, settings, best_angles)
        span = first.upper_rad - first.lower_rad
        # A coarse pass over the whole range, then a fine pass around the best.
        for width in (span, 2.0 * span / (samples - 1)):
            centre = best_angles[first.bone] if width < span else first.lower_rad
            low = max(first.lower_rad, centre - 0.5 * width) if width < span else centre
            high = min(first.upper_rad, low + width)
            for angle in np.linspace(low, high, samples):
                candidate = _apply(first, float(angle), angles)
                for joint in rest:
                    candidate, _ = _curl_joint(
                        hand, joint, obstacle, settings, candidate
                    )
                clear = float(obstacle(hand.points(candidate))[moved].min())
                if clear < -settings.penetration_allowance_m:
                    continue
                cost = _chain_cost(hand, chain, wanted, settings, candidate)
                closer = cost[0] < best_cost[0] - 1e-9
                level = cost[0] <= best_cost[0] + 1e-9
                if closer or (level and cost[1] < best_cost[1]):
                    best_cost, best_angles = cost, candidate
        angles = best_angles
        order.extend(joint.bone for joint in chain.joints)
    final = obstacle(hand.points(angles))
    gaps = {bone: float(final[hand.segment_ids(bone)].min()) for bone in order}
    return CurlResult(angles=angles, gaps=gaps, order=tuple(order))


@dataclass(frozen=True)
class FlatHandLandmarks:
    """Points and axes of a flat hand, in the hand's own rest frame.

    ``radial`` points from the little finger toward the index finger,
    ``distal`` from the wrist toward the fingertips and ``palmar`` out of the
    palm; for a right hand ``radial x distal = palmar``.
    """

    index_mcp: Array
    index_pip: Array
    little_mcp: Array
    radial: Array
    distal: Array
    palmar: Array


def club_axis_in_hand(
    marks: FlatHandLandmarks,
    *,
    cross_angle_deg: float,
    index_station: float,
    pinky_palm_depth_m: float,
) -> tuple[Array, Array]:
    """Club axis (a point and the butt-to-head unit direction) on a flat hand.

    The axis passes under the index finger's first segment at ``index_station``
    (0 = knuckle, 1 = first interphalangeal joint). Its direction is
    ``cross_angle_deg`` from the radial axis toward the fingertips, so that on
    the little-finger side it lies in the palm below the finger base. A
    positive ``pinky_palm_depth_m`` instead pins the axis that far below the
    little-finger knuckle and overrides the cross angle.
    """
    if not 0.0 <= _finite(index_station, "index_station") <= 1.0:
        raise ValueError("index_station must be in [0, 1]")
    if not 0.0 <= _finite(cross_angle_deg, "cross_angle_deg") < 90.0:
        raise ValueError("cross_angle_deg must be in [0, 90)")
    if _finite(pinky_palm_depth_m, "pinky_palm_depth_m") < 0.0:
        raise ValueError("pinky_palm_depth_m must be non-negative")
    radial = _unit(marks.radial, "radial")
    distal = _unit(marks.distal, "distal")
    palmar = _unit(marks.palmar, "palmar")
    point = marks.index_mcp + index_station * (marks.index_pip - marks.index_mcp)
    if pinky_palm_depth_m > 0.0:
        target = marks.little_mcp - pinky_palm_depth_m * distal
        chord = point - target
        direction = _unit(chord - (chord @ palmar) * palmar, "club axis")
    else:
        angle = math.radians(cross_angle_deg)
        direction = math.cos(angle) * radial + math.sin(angle) * distal
    return np.asarray(point, dtype=float), direction


def flat_hand_rotation(
    marks: FlatHandLandmarks, club_direction: Array, palm_azimuth_rad: float
) -> Array:
    """Rotation taking a flat **right** hand into the grip frame.

    The club direction goes to ``+z`` (butt to head) and the palm faces the
    grip axis from azimuth ``palm_azimuth_rad``; the fingers then point toward
    increasing azimuth, the way they will curl.
    """
    shaft = _unit(club_direction, "club_direction")
    palmar = _unit(marks.palmar, "palmar")
    if abs(float(shaft @ palmar)) > 1e-9:
        raise ValueError("club direction must lie in the palm plane")
    across = np.cross(palmar, shaft)  # in-plane, toward the fingertips
    c, s = (
        math.cos(_finite(palm_azimuth_rad, "palm_azimuth_rad")),
        math.sin(palm_azimuth_rad),
    )
    radial_out = np.array([c, s, 0.0])
    tangent = np.array([-s, c, 0.0])
    target = np.stack([np.array([0.0, 0.0, 1.0]), tangent, -radial_out])
    source = np.stack([shaft, across, palmar])
    return target.T @ source


def rigid(rotation: Array, translation: Array) -> Array:
    """4x4 transform from a rotation and a translation."""
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


@dataclass(frozen=True)
class SeatLimits:
    """Search limits of :func:`seat_on_obstacle` (all from configuration)."""

    max_tilt_rad: float
    max_travel_m: float
    iterations: int

    def __post_init__(self) -> None:
        if _finite(self.max_tilt_rad, "max_tilt_rad") < 0.0:
            raise ValueError("max_tilt_rad must be non-negative")
        if _finite(self.max_travel_m, "max_travel_m") <= 0.0:
            raise ValueError("max_travel_m must be positive")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")


def seat_on_obstacle(
    points: Array,
    obstacle: ObstacleDistance,
    *,
    approach: Array,
    tilt_axis: Array,
    tilt_pivot: Array,
    split_coordinate: Array,
    limits: SeatLimits,
) -> tuple[Array, float, float]:
    """Rest rigid ``points`` on the obstacle so that both halves touch it.

    The points are pushed back along ``-approach`` until none is inside the
    obstacle, and tilted about ``tilt_axis`` (through ``tilt_pivot``) until
    the half with ``split_coordinate > 0`` and the half with
    ``split_coordinate <= 0`` are equally close. Returns the 4x4 rigid
    correction and the two remaining gaps (positive halves first).

    This is a rigid seat of the whole hand, not a deformation.
    """
    approach_u = _unit(approach, "approach")
    axis_u = _unit(tilt_axis, "tilt_axis")
    pts = np.asarray(points, dtype=float)
    upper = np.asarray(split_coordinate) > 0.0
    if not upper.any() or upper.all():
        raise ValueError("both halves of the hand need contact points")
    max_tilt_rad, max_travel_m = limits.max_tilt_rad, limits.max_travel_m
    iterations = limits.iterations

    def tilt_matrix(angle: float) -> Array:
        x, y, z = axis_u
        k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)

    def seated(angle: float) -> tuple[Array, float, float]:
        rot = tilt_matrix(angle)
        tilted = (pts - tilt_pivot) @ rot.T + tilt_pivot
        low, high = -max_travel_m, max_travel_m
        for _ in range(iterations):
            mid = 0.5 * (low + high)
            if float(obstacle(tilted - mid * approach_u).min()) >= 0.0:
                high = mid
            else:
                low = mid
        d = obstacle(tilted - high * approach_u)
        correction = rigid(rot, tilt_pivot - rot @ tilt_pivot - high * approach_u)
        return correction, float(d[upper].min()), float(d[~upper].min())

    low, high = -max_tilt_rad, max_tilt_rad
    sign_low = seated(low)
    balance_low = sign_low[1] - sign_low[2]
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        _, gap_up, gap_down = seated(mid)
        if (gap_up - gap_down) * balance_low > 0.0:
            low = mid
        else:
            high = mid
    return seated(0.5 * (low + high))
