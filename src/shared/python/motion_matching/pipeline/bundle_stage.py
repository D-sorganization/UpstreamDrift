"""Bundle adjustment of a lane's marker attachments (#15296).

Turns a pipeline lane (capture, ground, stance, bounds) and its plant into a
``BundleProblem`` for ``bundle_adjustment.solve_bundle`` and turns the result
back into attachments and a receipt block. Forward kinematics and the pose
Jacobian come from the plant's public marker linearisation of four probe points
per body, so nothing here names an engine; the grip weld and the ground spheres
come from its public constraint linearisation.
"""

from __future__ import annotations

import argparse
import copy
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.shared.python.motion_matching.club_face_target import require_club_assembly
from src.shared.python.motion_matching.constraint_kinematics import ConstraintOptions
from src.shared.python.motion_matching.pipeline.bundle_adjustment import (
    BundleProblem,
    BundleResult,
    BundleSettings,
    solve_bundle,
)
from src.shared.python.motion_matching.pipeline.constants import (
    PRIOR,
    SPIN_COORDINATES,
)

LOG = logging.getLogger(__name__)

Attachments = Mapping[str, tuple[str, Sequence[float]]]
Point = tuple[float, float, float]

#: Prior sigma of a marker's offset along its body's x, y, z axes, metres, by
#: label suffix (or prefix for the head). Values are those of the experiment in
#: ``FIT_PATH.md`` (placement review of the capture's marker set: how far each
#: landmark can plausibly sit from its seed), except the head. The experiment
#: gave head markers 25/20/25 mm and a tenth of the data weight, which let the
#: prior outweigh the data; here every marker has full weight and the head
#: 40 mm (upper-body study of #15296). A label not listed (the club markers) is
#: held at its seed.
OFFSET_SIGMAS_M: dict[str, tuple[float, float, float]] = {
    "Head": (0.040, 0.040, 0.040),
    "BackTop": (0.020, 0.015, 0.030),
    "BackLeft": (0.020, 0.015, 0.030),
    "BackRight": (0.020, 0.015, 0.030),
    "WaistLeft": (0.020, 0.015, 0.025),
    "WaistRight": (0.020, 0.015, 0.025),
    "WaistLBack": (0.020, 0.015, 0.025),
    "WaistRBack": (0.020, 0.015, 0.025),
    "ShoulderTop": (0.020, 0.020, 0.020),
    "ShoulderBack": (0.020, 0.020, 0.020),
    "UArmHigh": (0.020, 0.020, 0.040),
    "ElbowOut": (0.012, 0.012, 0.012),
    "WristTop": (0.012, 0.012, 0.008),
    "KneeOut": (0.015, 0.015, 0.015),
    "AnkleOut": (0.015, 0.015, 0.015),
    "ToeIn": (0.015, 0.015, 0.015),
    "ToeOut": (0.015, 0.015, 0.015),
}
#: Probe points of a body frame: its origin and the tips of its unit axes.
_PROBES = np.vstack([np.zeros(3), np.eye(3)])


@dataclass(frozen=True)
class StageSettings:
    """Settings of the stage; *experiment* marks values of the ported fit."""

    #: Every ``frame_stride``-th capture frame is adjusted (*experiment*: 20 at
    #: 360 Hz), plus ``anchor_frames`` (events such as top and impact).
    frame_stride: int = 20
    anchor_frames: tuple[int, ...] = ()
    #: Grip weld and ground-sphere residual scales (*experiment*).
    weld_sigma_m: float = 0.0005
    weld_sigma_rad: float = 0.005
    ground_sigma_m: float = 0.005
    #: Labels left out of the adjustment and scored afterwards (held-out check).
    held_out: tuple[str, ...] = ()
    #: The address frame enters this many times (*experiment*: 8), so the
    #: adjustment cannot buy the swing with the address.
    address_repeats: int = 1
    #: Start poses: ``strided`` solves only the selected frames, ``consecutive``
    #: solves every capture frame in order and picks the selected ones.
    start: str = "strided"
    #: Residual per radian a coordinate (a spin coordinate) leaves its start
    #: pose (*experiment*: 0.1 and 3.0); ``None`` keeps the lane's own prior.
    pose_prior_weight: float | None = None
    spin_prior_weight: float | None = None
    #: Farthest a body marker may sit from its seed along one axis, metres. A
    #: skin or garment marker placed on a named landmark is within a few
    #: centimetres of it; a larger fitted offset is the adjustment absorbing a
    #: model or pose error, not a placement. ``None``: only ``bound_sigmas``.
    #: Head markers are exempt: they sit on a cap, not on the skull landmarks.
    offset_reach_m: float | None = None
    #: Labels containing one of these stay at their seed. The leg markers are
    #: already calibrated under the lane's own stance model, which the swing
    #: solve uses again; an adjustment made without it can undo that.
    hold: tuple[str, ...] = ()
    #: Weigh each marker as the lane's own inverse kinematics does (head cap
    #: markers at a tenth, the shoulder-girdle split), so the offsets are fitted
    #: under the weights the swing solve then applies to them.
    lane_weights: bool = False
    #: Residual scale of a stance sphere sliding from where the first frame's
    #: start pose puts it, metres, while it stays in stance: the swing solve
    #: plants the feet, so offsets fitted with sliding feet do not fit it.
    #: ``None`` leaves the feet free to slide.
    plant_sigma_m: float | None = None
    #: Pose targets of the address frame: (coordinate, target deg, sigma deg).
    #: Markers alone leave the elbow flexion and the humeral axial rotation of
    #: an address pose weakly decided, so the adjustment can buy marker fit
    #: with a bent lead arm (*experiment*: ``LEInput`` -5 +/- 5, ``REInput``
    #: -20 +/- 10, ``LSInputZ`` and ``RSInputZ`` 0 +/- 15). Empty: none.
    address_pose: tuple[tuple[str, float, float], ...] = ()
    #: The stage stops when a step gains under a millionth of the cost: the
    #: core's default asks for more digits than marker data carry.
    solver: BundleSettings = field(
        default_factory=lambda: BundleSettings(tolerance=1e-6, max_iterations=100)
    )

    def sigma(self, label: str) -> tuple[float, float, float]:
        """Prior sigma of ``label``: ``offset_sigma`` unless it is held."""
        if any(part in label for part in self.hold):
            return (0.0, 0.0, 0.0)
        return offset_sigma(label)


def offset_sigma(label: str) -> tuple[float, float, float]:
    """Prior sigma of ``label`` from ``OFFSET_SIGMAS_M``; zeros hold it."""
    for key, sigma in OFFSET_SIGMAS_M.items():
        if label.endswith(key) or label.startswith(key):
            return sigma
    return (0.0, 0.0, 0.0)


def select_frames(n_frames: int, settings: StageSettings) -> tuple[int, ...]:
    """Every ``frame_stride``-th frame plus the anchors, sorted and unique."""
    if settings.frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    for anchor in settings.anchor_frames:
        if not 0 <= anchor < n_frames:
            raise ValueError(f"anchor frame {anchor} is outside 0..{n_frames - 1}")
    chosen = {*range(0, n_frames, settings.frame_stride), *settings.anchor_frames}
    return tuple(sorted(chosen))


class BodyKinematics:
    """Body frames and twists of a plant, through probe-point markers only."""

    def __init__(self, plant: Any, bodies: Sequence[str]) -> None:
        """Initialize the instance."""
        self.bodies = tuple(bodies)
        probe_rows: list[tuple[float, float, float]] = [
            (float(x), float(y), float(z)) for x, y, z in np.asarray(_PROBES).tolist()
        ]
        probes: dict[str, tuple[str, tuple[float, float, float]]] = {
            f"{body}#{i}": (body, point)
            for body in self.bodies
            for i, point in enumerate(probe_rows)
        }
        self.kin = plant.create_ik(probes)
        self._pose: Any = None
        self._linear: Any = None

    def _linearise(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linearise the bundle stage about the current pose."""
        if self._pose is None or not np.array_equal(pose, self._pose):
            linear = self.kin.marker_linearization(pose)
            shape = (len(self.bodies), 4, 3)
            points = np.reshape(linear.positions, shape)
            slopes = np.reshape(linear.jacobian, (*shape, len(pose)))
            self._pose, self._linear = np.array(pose), (points, slopes)
        return self._linear

    def frames(self, pose: np.ndarray) -> np.ndarray:
        """World frame ``(n_bodies, 4, 4)`` of every body at ``pose``."""
        points, _ = self._linearise(pose)
        out = np.tile(np.eye(4), (len(self.bodies), 1, 1))
        out[:, :3, :3] = (points[:, 1:] - points[:, :1]).transpose(0, 2, 1)
        out[:, :3, 3] = points[:, 0]
        return out

    def twists(self, pose: np.ndarray) -> np.ndarray:
        """``(n_bodies, 6, n_q)``: origin velocity, then angular velocity."""
        points, slopes = self._linearise(pose)
        axes = (points[:, 1:] - points[:, :1])[..., None]
        turning = np.cross(axes, slopes[:, 1:] - slopes[:, :1], axis=2).sum(axis=1)
        return np.concatenate([slopes[:, 0], 0.5 * turning], axis=1)


def _planted(
    lane: Any, bodies: BodyKinematics, frames: Sequence[int], start: np.ndarray
) -> tuple[dict[str, np.ndarray], list[frozenset[str]]]:
    """Where ``start`` (the first frame's pose) puts each stance sphere, and
    the spheres still planted at each of ``frames``: in stance at every
    capture frame up to it, as the swing solve drops a lifted sphere's plant."""
    home = bodies.kin.sphere_ground_points(start, lane.ground)
    still, kept = set(lane.stance[frames[0]]) & set(home), []
    for stance in lane.stance[frames[0] : max(frames) + 1]:
        still &= set(stance)
        kept.append(frozenset(still))
    return home, [kept[k - frames[0]] for k in frames]


def _constraint(
    lane: Any,
    bodies: BodyKinematics,
    frames: Sequence[int],
    settings: StageSettings,
    extras: tuple[tuple[str, Sequence[float]] | None, np.ndarray],
) -> Callable[[int, np.ndarray], np.ndarray]:
    """Weld and ground residuals of a frame; at the first frame also the
    height of the sole point ``extras[0]`` (body, point) above the ground,
    when one is given; with ``plant_sigma_m`` also the slide of every planted
    sphere from where ``extras[1]`` (the first frame's start pose) puts it."""
    sole, start = extras
    normal = np.asarray(lane.ground.normal, float)
    home, planted = (
        ({}, [])
        if settings.plant_sigma_m is None
        else _planted(lane, bodies, frames, start)
    )

    def slide(k: int, pose: np.ndarray) -> np.ndarray:
        """Slide the contact sample."""
        here = bodies.kin.sphere_ground_points(pose, lane.ground)
        rows = np.zeros((len(home), 3))
        for i, name in enumerate(sorted(home)):
            if name in planted[k]:
                gap = here[name] - home[name]
                rows[i] = (gap - (gap @ normal) * normal) / settings.plant_sigma_m
        return rows.ravel()

    def residual(k: int, pose: np.ndarray) -> np.ndarray:
        """Return the residual."""
        options = ConstraintOptions(
            ground=lane.ground,
            position_weight=1.0,
            rotation_weight=1.0,
            ground_weight=1.0,
            position_scale_m=settings.weld_sigma_m,
            rotation_scale_rad=settings.weld_sigma_rad,
            ground_scale_m=settings.ground_sigma_m,
            pinned_spheres=tuple(lane.stance[frames[k]]),
        )
        rows = np.asarray(
            bodies.kin.constraint_residual_jacobian(pose, options).residual
        )
        if sole is not None:
            frame = bodies.frames(pose)[bodies.bodies.index(sole[0])]
            point = frame[:3, 3] + frame[:3, :3] @ np.asarray(sole[1], float)
            height = point @ normal - lane.ground.height_m
            rows = np.append(rows, (k == 0) * height / settings.ground_sigma_m)
        return (
            rows if settings.plant_sigma_m is None else np.append(rows, slide(k, pose))
        )

    return residual


def _address_pose(
    settings: StageSettings, order: Sequence[str], frames: Sequence[int]
) -> tuple[list[int], list[int], np.ndarray]:
    """Rows (every copy of the first frame), columns and (target, sigma) in
    radians of ``settings.address_pose``. Raises ``ValueError`` on an unknown
    coordinate, a non-finite target or a sigma that is not positive."""
    for name, target, sigma in settings.address_pose:
        if name not in order or not (np.isfinite(target) and 0 < sigma < np.inf):
            raise ValueError(
                f"address pose needs a model coordinate, a finite target and a "
                f"positive sigma, got {(name, target, sigma)}"
            )
    rows = [row for row, frame in enumerate(frames) if frame == frames[0]]
    columns = [order.index(name) for name, _, _ in settings.address_pose]
    values = [entry[1:] for entry in settings.address_pose]
    return rows, columns, np.radians(np.reshape(values, (-1, 2)))


def address_pose_prior(settings: StageSettings) -> dict[str, tuple[float, float]]:
    """``settings.address_pose`` as the swing solve's posture prior:
    ``{coordinate: (target_rad, weight)}``, the weight being that of a squared
    residual in metres (the adjustment's marker sigma over the pose sigma)."""
    scale = settings.solver.marker_sigma_m
    return {
        name: (float(np.radians(target)), float((scale / np.radians(sigma)) ** 2))
        for name, target, sigma in settings.address_pose
    }


def _marker_weights(lane: Any, labels: Sequence[str]) -> np.ndarray:
    """The lane's inverse-kinematics weight of every label (one when unset)."""
    weights = {**lane.marker_weights, **lane.split_marker_weights}
    return np.array([float(weights.get(label, 1.0)) for label in labels])


def build_problem(
    lane: Any,
    attachments: Attachments,
    poses: np.ndarray,
    settings: StageSettings,
    sole: tuple[str, Sequence[float]] | None = None,
) -> tuple[BundleProblem, tuple[str, ...], tuple[int, ...]]:
    """The bundle problem of ``lane``: (problem, fitted labels, frames).

    ``poses`` are the start poses of the selected frames, in the plant's
    coordinate order; they are also the mean of the lane's joint prior.
    ``sole`` is the club's ground point as (body, point in the body frame);
    ``None`` adds no club-ground term (the body origin is never assumed).
    """
    if settings.address_repeats < 1:
        raise ValueError("address_repeats must be at least 1")
    frames = select_frames(len(lane.points), settings)
    frames = (frames[0],) * (settings.address_repeats - 1) + frames
    poses = np.asarray(poses, float)
    if len(poses) != len(frames):
        raise ValueError(f"poses must have one row per frame ({len(frames)})")
    labels = tuple(
        label
        for label in lane.labels
        if label in attachments and label not in settings.held_out
    )
    names = [attachments[label][0] for label in labels]
    bodies = BodyKinematics(lane.plant, sorted({*names, *([sole[0]] if sole else [])}))
    order = tuple(bodies.kin.coordinate_order)
    lower, upper = np.full(len(order), -np.inf), np.full(len(order), np.inf)
    for name, (low, high) in lane.bounds.items():
        lower[order.index(name)], upper[order.index(name)] = low, high
    for name in bodies.kin.shaft_rest_coordinates():
        lower[order.index(name)] = upper[order.index(name)] = 0.0
    sigmas = np.array([lane.prior_weights.get(name, PRIOR) ** -0.5 for name in order])
    if settings.pose_prior_weight is not None:
        sigmas = np.full(len(order), 1.0 / settings.pose_prior_weight)
    if settings.spin_prior_weight is not None:
        spin = [order.index(name) for name in SPIN_COORDINATES if name in order]
        sigmas[spin] = 1.0 / settings.spin_prior_weight
    mean, spread = poses.copy(), np.tile(sigmas, (len(frames), 1))
    rows, held, target = _address_pose(settings, order, frames)
    mean[np.ix_(rows, held)], spread[np.ix_(rows, held)] = target[:, 0], target[:, 1]
    reach = np.full((len(labels), 3), np.inf)
    if settings.offset_reach_m is not None:
        body = [not label.startswith("Head") for label in labels]
        reach[body] = settings.offset_reach_m
    columns = [lane.labels.index(label) for label in labels]
    problem = BundleProblem(
        fk=bodies.frames,
        marker_bodies=np.array([bodies.bodies.index(name) for name in names]),
        observations=np.asarray(lane.points)[list(frames)][:, columns],
        valid=np.asarray(lane.valid)[list(frames)][:, columns],
        offset_seeds=np.array([attachments[label][1] for label in labels], float),
        offset_sigmas=np.array([settings.sigma(label) for label in labels]),
        pose_start=poses,
        pose_bounds=(lower, upper),
        pose_prior=(mean, spread),
        constraint=_constraint(lane, bodies, frames, settings, (sole, poses[0])),
        body_jacobian=bodies.twists,
        offset_reach=reach,
        marker_weights=_marker_weights(lane, labels) if settings.lane_weights else None,
    )
    return problem, labels, frames


def adjusted_attachments(
    attachments: Attachments, labels: Sequence[str], result: BundleResult
) -> dict[str, tuple[str, Point]]:
    """``attachments`` with the fitted labels moved to the adjusted offsets."""
    out = {
        label: (body, (float(o[0]), float(o[1]), float(o[2])))
        for label, (body, o) in attachments.items()
    }
    for label, offset in zip(labels, result.offsets.tolist(), strict=True):
        out[label] = (out[label][0], (offset[0], offset[1], offset[2]))
    return out


def stage_report(
    result: BundleResult,
    labels: Sequence[str],
    frames: Sequence[int],
    settings: StageSettings,
) -> dict[str, Any]:
    """Receipt block of one adjustment. Held markers never appear as weak or
    on a bound; ``per_marker_rms_mm`` is over the adjusted frames only."""
    return {
        "frames": [int(frame) for frame in frames],
        "converged": result.converged,
        "rms_mm": 1e3 * result.rms_m,
        "per_marker_rms_mm": dict(
            zip(labels, (1e3 * result.per_marker_rms_m).tolist(), strict=True)
        ),
        "chi2": {
            "data": result.data_chi2,
            "constraint": result.constraint_chi2,
            "offset_prior": result.offset_prior_chi2,
            "pose_prior": result.pose_prior_chi2,
            "prior_share": result.prior_share,
        },
        "offsets_on_bound": [[labels[m], "xyz"[a]] for m, a in result.on_bound],
        "weakly_identified_markers": [labels[m] for m in result.weakly_identified],
        "weak_information_ratios": list(result.weak_information_ratios),
        "free_markers": [label for label in labels if any(settings.sigma(label))],
        "offset_sigmas_m": {key: list(value) for key, value in OFFSET_SIGMAS_M.items()},
        "settings": asdict(settings),
    }


def add_bundle_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the opt-in ``--bundle-adjust`` stage and its settings."""
    default = StageSettings()
    parser.add_argument(
        "--bundle-adjust",
        action="store_true",
        help="adjust the body-marker attachments over frames of the whole "
        "capture before the swing solve (#15296); off by default, and a run "
        "without it is unchanged",
    )
    parser.add_argument(
        "--bundle-stride",
        type=int,
        default=default.frame_stride,
        help="every this many capture frames enter the adjustment",
    )
    parser.add_argument(
        "--bundle-anchor-frames",
        type=int,
        nargs="*",
        default=[],
        help="further frames to include, such as the top and impact",
    )
    parser.add_argument(
        "--bundle-held-out",
        nargs="*",
        default=[],
        help="marker labels left out of the adjustment and scored afterwards",
    )
    parser.add_argument(
        "--bundle-hold",
        nargs="*",
        default=[],
        help="markers whose label contains one of these stay at their seed",
    )
    parser.add_argument(
        "--bundle-plant-mm",
        type=float,
        default=None,
        help="residual scale of a planted stance sphere sliding, mm (default: free)",
    )
    parser.add_argument(
        "--bundle-address-pose",
        nargs="*",
        default=[],
        metavar="COORDINATE:DEG:SIGMA_DEG",
        help="pose targets of the address frame in the adjustment and on the "
        "first frame of the swing solve, such as LEInput:-5:5 (default: none)",
    )
    parser.add_argument(
        "--bundle-lane-weights",
        action="store_true",
        help="weigh markers in the adjustment as the inverse kinematics does",
    )
    parser.add_argument(
        "--bundle-start",
        choices=("strided", "consecutive"),
        default=default.start,
        help="start poses: inverse kinematics of the selected frames only, or "
        "of every capture frame in order",
    )
    parser.add_argument(
        "--bundle-address-repeats",
        type=int,
        default=default.address_repeats,
        help="how many times the address frame enters the adjustment",
    )
    parser.add_argument(
        "--bundle-offset-reach-mm",
        type=float,
        default=None,
        help="farthest a body marker may move from its seed along one axis",
    )
    parser.add_argument(
        "--bundle-pose-prior",
        type=float,
        nargs=2,
        default=None,
        metavar=("WEIGHT", "SPIN_WEIGHT"),
        help="residual per radian a coordinate (a spin coordinate) leaves its "
        "start pose; default: the lane's own prior",
    )


def _pose_target(item: str) -> tuple[str, float, float]:
    """``COORDINATE:DEG:SIGMA_DEG`` as a tuple; ``ValueError`` otherwise."""
    parts = item.split(":")
    if len(parts) != 3:
        raise ValueError(f"address pose must be COORDINATE:DEG:SIGMA_DEG, got {item}")
    return parts[0], float(parts[1]), float(parts[2])


def stage_settings(args: argparse.Namespace) -> StageSettings | None:
    """Settings of the run ``args`` asks for; ``None`` without the flag."""
    if not getattr(args, "bundle_adjust", False):
        return None
    reach, prior = args.bundle_offset_reach_mm, args.bundle_pose_prior
    plant = args.bundle_plant_mm
    return StageSettings(
        frame_stride=args.bundle_stride,
        anchor_frames=tuple(args.bundle_anchor_frames),
        held_out=tuple(args.bundle_held_out),
        start=args.bundle_start,
        hold=tuple(args.bundle_hold),
        lane_weights=args.bundle_lane_weights,
        plant_sigma_m=None if plant is None else 1e-3 * plant,
        address_repeats=args.bundle_address_repeats,
        address_pose=tuple(
            _pose_target(item) for item in getattr(args, "bundle_address_pose", [])
        ),
        offset_reach_m=None if reach is None else 1e-3 * reach,
        pose_prior_weight=None if prior is None else prior[0],
        spin_prior_weight=None if prior is None else prior[1],
    )


def stated_sole(spec: Mapping[str, Any]) -> tuple[str, Point] | None:
    """The club's ground point as (club body, ``club.sole_point_m``) when the
    document states one; ``None`` otherwise, never the body origin."""
    point = (spec.get("club") or {}).get("sole_point_m")
    if point is None:
        return None
    x, y, z = (float(value) for value in point)
    return require_club_assembly(spec)[1], (x, y, z)


@dataclass(frozen=True)
class Adjustment:
    """Adjusted ``attachments``, the adjusted address pose and the receipt block."""

    attachments: dict[str, tuple[str, Point]]
    address_q: np.ndarray
    report: dict[str, Any]


def held_out_rms_mm(
    lane: Any, seeds: Attachments, frames: Sequence[int], poses: np.ndarray
) -> dict[str, float]:
    """RMS of each marker in ``seeds`` at ``poses`` of ``frames``, millimetres."""
    if not seeds:
        return {}
    kin = lane.plant.create_ik(dict(seeds))
    model = np.array([kin.marker_linearization(pose).positions for pose in poses])
    out = {}
    for column, label in enumerate(seeds):
        seen = lane.labels.index(label)
        valid = np.asarray(lane.valid)[list(frames), seen]
        gap = model[:, column] - np.asarray(lane.points)[list(frames), seen]
        out[label] = 1e3 * float(np.sqrt(np.mean(np.sum(gap[valid] ** 2, axis=1))))
    return out


def held_out_scatter_mm(
    lane: Any, seeds: Attachments, frames: Sequence[int], poses: np.ndarray
) -> dict[str, float]:
    """How far each marker in ``seeds`` wanders in its own body frame over
    ``poses`` of ``frames``, RMS about its mean, millimetres.

    A skin marker is nearly rigid on its segment, so better poses give less
    scatter. The seed takes no part (only its body does): the score stays fair
    when the adjustment moves the marker's neighbours in common mode.
    """
    if not seeds:
        return {}
    bodies = BodyKinematics(lane.plant, sorted({body for body, _ in seeds.values()}))
    world = np.array([bodies.frames(pose) for pose in poses])
    out = {}
    for label, (body, _) in seeds.items():
        seen = lane.labels.index(label)
        valid = np.asarray(lane.valid)[list(frames), seen]
        frame = world[valid, bodies.bodies.index(body)]
        gap = np.asarray(lane.points)[list(frames), seen][valid] - frame[:, :3, 3]
        local = np.einsum("kji,kj->ki", frame[:, :3, :3], gap)
        spread = np.sum((local - local.mean(axis=0)) ** 2, axis=1)
        out[label] = 1e3 * float(np.sqrt(np.mean(spread)))
    return out


def start_poses(
    lane: Any, kin: Any, q_start: np.ndarray, settings: StageSettings
) -> np.ndarray:
    """Start pose of every selected frame from the lane's inverse kinematics
    with the seed attachments. Held-out markers are hidden from it, so their
    score at these poses is a fair baseline."""
    if settings.start not in ("strided", "consecutive"):
        raise ValueError("start must be 'strided' or 'consecutive'")
    frames = list(select_frames(len(lane.points), settings))
    blind = copy.copy(lane)
    blind.valid = np.array(lane.valid, bool)
    blind.valid[:, [lane.labels.index(name) for name in settings.held_out]] = False
    if settings.start == "strided":
        return np.asarray(blind.trajectory(kin, q_start, frames=frames)[0])
    return np.asarray(blind.trajectory(kin, q_start)[0])[frames]


def adjust(
    lane: Any,
    kin: Any,
    attachments: Attachments,
    q_start: np.ndarray,
    settings: StageSettings,
    sole: tuple[str, Sequence[float]] | None = None,
) -> Adjustment:
    """Adjust ``attachments`` over the selected frames of ``lane``.

    Start poses are ``start_poses``. A held-out marker keeps its seed; it is
    scored at the start poses and at the adjusted poses, neither of which
    fitted it.
    """
    start = start_poses(lane, kin, q_start, settings)
    start = np.vstack([start[:1]] * (settings.address_repeats - 1) + [start])
    problem, labels, frames = build_problem(lane, attachments, start, settings, sole)
    LOG.info("bundle adjustment: %d frames, %d markers", len(frames), len(labels))
    result = solve_bundle(problem, settings.solver)
    LOG.info(
        "bundle adjustment: converged %s, RMS %.1f mm",
        result.converged,
        1e3 * result.rms_m,
    )
    fitted = adjusted_attachments(attachments, labels, result)
    lane.address_pose_prior = address_pose_prior(settings)  # swing solve, frame 0
    held = {name: attachments[name] for name in settings.held_out}
    scores = [held_out_rms_mm(lane, held, frames, q) for q in (start, result.poses)]
    loose = [held_out_scatter_mm(lane, held, frames, q) for q in (start, result.poses)]
    shift = np.linalg.norm(result.offsets - problem.offset_seeds, axis=1)
    report = stage_report(result, labels, frames, settings) | {
        "offset_shift_mm": dict(zip(labels, (1e3 * shift).tolist(), strict=True)),
        "per_frame_rms_mm": (1e3 * result.per_frame_rms_m).tolist(),
        "held_out_rms_mm": {
            name: {"start": scores[0][name], "adjusted": scores[1][name]}
            for name in held
        },
        "held_out_scatter_mm": {
            name: {"start": loose[0][name], "adjusted": loose[1][name]} for name in held
        },
        "club_sole_point": None if sole is None else [sole[0], list(sole[1])],
    }
    return Adjustment(fitted, result.poses[0], report)
