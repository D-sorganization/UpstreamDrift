"""Bundle adjustment of a lane's marker attachments (#15296).

Turns a pipeline lane (capture, ground, stance, bounds) and its plant into a
``BundleProblem`` for ``bundle_adjustment.solve_bundle`` and turns the result
back into attachments and a receipt block. Forward kinematics and the pose
Jacobian come from the plant's public marker linearisation of four probe points
per body, so nothing here names an engine; the grip weld and the ground spheres
come from its public constraint linearisation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.shared.python.motion_matching.constraint_kinematics import ConstraintOptions
from src.shared.python.motion_matching.pipeline.bundle_adjustment import (
    BundleProblem,
    BundleResult,
    BundleSettings,
)
from src.shared.python.motion_matching.pipeline.constants import PRIOR

Attachments = Mapping[str, tuple[str, Sequence[float]]]

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
    solver: BundleSettings = field(default_factory=BundleSettings)


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
        self.bodies = tuple(bodies)
        probes = {
            f"{body}#{i}": (body, tuple(point))
            for body in self.bodies
            for i, point in enumerate(_PROBES)
        }
        self.kin = plant.create_ik(probes)
        self._pose: Any = None
        self._linear: Any = None

    def _linearise(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def _constraint(
    lane: Any,
    bodies: BodyKinematics,
    frames: Sequence[int],
    settings: StageSettings,
    sole: tuple[str, Sequence[float]] | None,
) -> Callable[[int, np.ndarray], np.ndarray]:
    """Weld and ground residuals of a frame; at the first frame also the
    height of ``sole`` (body, point) above the ground, when one is given."""

    def residual(k: int, pose: np.ndarray) -> np.ndarray:
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
        rows = bodies.kin.constraint_residual_jacobian(pose, options).residual
        if sole is None:
            return np.asarray(rows)
        frame = bodies.frames(pose)[bodies.bodies.index(sole[0])]
        point = frame[:3, 3] + frame[:3, :3] @ np.asarray(sole[1], float)
        height = point @ np.asarray(lane.ground.normal) - lane.ground.height_m
        return np.append(rows, (k == 0) * height / settings.ground_sigma_m)

    return residual


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
    frames = select_frames(len(lane.points), settings)
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
    weights = np.array([lane.prior_weights.get(name, PRIOR) for name in order])
    columns = [lane.labels.index(label) for label in labels]
    problem = BundleProblem(
        fk=bodies.frames,
        marker_bodies=np.array([bodies.bodies.index(name) for name in names]),
        observations=np.asarray(lane.points)[list(frames)][:, columns],
        valid=np.asarray(lane.valid)[list(frames)][:, columns],
        offset_seeds=np.array([attachments[label][1] for label in labels], float),
        offset_sigmas=np.array([offset_sigma(label) for label in labels]),
        pose_start=poses,
        pose_bounds=(lower, upper),
        pose_prior=(poses, np.tile(weights**-0.5, (len(frames), 1))),
        constraint=_constraint(lane, bodies, frames, settings, sole),
        body_jacobian=bodies.twists,
    )
    return problem, labels, frames


def adjusted_attachments(
    attachments: Attachments, labels: Sequence[str], result: BundleResult
) -> dict[str, tuple[str, tuple[float, float, float]]]:
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
        "free_markers": [label for label in labels if any(offset_sigma(label))],
        "offset_sigmas_m": {key: list(value) for key, value in OFFSET_SIGMAS_M.items()},
        "settings": asdict(settings),
    }
