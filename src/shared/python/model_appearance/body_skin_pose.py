"""Pose the fitted CC0 body skin; the shoulder as one continuous surface (#15299).

Rigid per-body meshes cannot keep a shirt continuous over the shoulder when the
girdle moves: the sleeve cap rides on the girdle link and steps away from the
trunk. Here every vertex is a weighted blend (linear blend skinning) of the
thorax, the girdle link, a deltoid group that follows part of the upper arm's
motion, and the upper arm, so the surface stretches instead of stepping.

``smooth_shoulder_weights`` relaxes MakeHuman's weights around the shoulder
joints; ``shoulder_transforms`` turns the three body motions into group
transforms; ``pose_skin`` applies them; ``acromion_edges`` and
``edge_length_change`` are the measurable criterion (settings and sources in
``assets/body_skin_cc0/pose_settings.json``). Appearance only; opt-in.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.shared.python.model_appearance.body_skin_fit import (
    ASSET_DIR,
    SIDES,
    BodySkin,
    FittedSkin,
)

Array = NDArray[np.float64]
Edges = NDArray[np.int64]


@dataclass(frozen=True)
class Rigid:
    """``x -> rotation @ x + translation`` from the fitted rest to the pose."""

    rotation: Array
    translation: Array

    @classmethod
    def about(cls, rotation: Array, pivot: Array) -> Rigid:
        """The rotation applied about ``pivot``."""
        return cls(rotation, pivot - rotation @ pivot)

    def apply(self, points: Array) -> Array:
        return points @ self.rotation.T + self.translation


IDENTITY = Rigid(np.eye(3), np.zeros(3))


@dataclass(frozen=True)
class PoseSettings:
    """Named skinning settings; values and sources live in ``pose_settings.json``."""

    shoulder_follow: float
    smoothing_iterations: int
    smoothing_radius_m: float
    acromion_half_width_m: float
    acromion_inner_m: float
    acromion_reach_m: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.shoulder_follow <= 1.0:
            raise ValueError("shoulder_follow must lie in [0, 1]")
        if self.smoothing_iterations < 0:
            raise ValueError("smoothing_iterations must not be negative")
        for name in ("smoothing_radius_m", "acromion_half_width_m", "acromion_reach_m"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive")
        if self.acromion_inner_m < 0.0:
            raise ValueError("acromion_inner_m must not be negative")


def load_pose_settings(path: Path | None = None) -> PoseSettings:
    """Read ``pose_settings.json``; raises ``ValueError`` for invalid values."""
    raw = json.loads((path or ASSET_DIR / "pose_settings.json").read_text())
    return PoseSettings(**{f.name: raw[f.name] for f in fields(PoseSettings)})


def mesh_edges(faces: NDArray[np.int64]) -> Edges:
    """Unique undirected edges ``(E, 2)`` of a triangle mesh."""
    pairs = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(pairs, axis=1), axis=0)


def smooth_shoulder_weights(
    skin: BodySkin, fitted: FittedSkin, settings: PoseSettings
) -> Array:
    """Skin weights relaxed over the mesh near the shoulder joints.

    Each pass moves a vertex half way to the mean of its neighbours; vertices
    farther than ``smoothing_radius_m`` from both shoulder joints keep the
    stored weights. Postcondition: rows still sum to one.
    """
    edges = mesh_edges(skin.faces)
    reach = np.min(
        [
            np.linalg.norm(fitted.vertices - fitted.joints[f"shoulder_{side}"], axis=1)
            for side, _ in SIDES
        ],
        axis=0,
    )
    free = reach < settings.smoothing_radius_m
    count = np.bincount(edges.ravel(), minlength=len(skin.vertices))[:, None]
    weights = skin.weights.copy()
    for _ in range(settings.smoothing_iterations):
        total = np.zeros_like(weights)
        np.add.at(total, edges[:, 0], weights[edges[:, 1]])
        np.add.at(total, edges[:, 1], weights[edges[:, 0]])
        weights[free] = 0.5 * weights[free] + 0.5 * (total / count)[free]
    return weights / weights.sum(axis=1, keepdims=True)


def blend(first: Rigid, second: Rigid, share: float, pivot: Array) -> Rigid:
    """Between two motions: rotation slerped, ``pivot`` carried to the mixed image."""
    step = Rotation.from_matrix(first.rotation.T @ second.rotation).as_rotvec()
    rotation = first.rotation @ Rotation.from_rotvec(share * step).as_matrix()
    target = (1.0 - share) * first.apply(pivot) + share * second.apply(pivot)
    return Rigid(rotation, target - rotation @ pivot)


def shoulder_transforms(
    motions: Mapping[str, Rigid], fitted: FittedSkin, settings: PoseSettings
) -> dict[str, Rigid]:
    """Group transforms of trunk, girdle and arms from the body motions.

    ``motions`` holds ``thorax`` and, per side, ``girdle_l/r``, ``upper_arm_l/r``
    and optionally ``forearm_l/r`` (default: the upper arm's). The clavicle
    group rides the girdle link; the deltoid (``shoulder``) group turns
    ``shoulder_follow`` of the way from the girdle to the upper arm about the
    shoulder joint. Raises ``ValueError`` when a motion is missing.
    """
    try:
        out = dict.fromkeys(("pelvis", "spine05", "spine04", "spine03"), IDENTITY)
        out["thorax"] = motions["thorax"]
        for side, _ in SIDES:
            girdle, arm = motions[f"girdle_{side}"], motions[f"upper_arm_{side}"]
            joint = fitted.joints[f"shoulder_{side}"]
            out[f"clavicle_{side}"] = girdle
            out[f"shoulder_{side}"] = blend(
                girdle, arm, settings.shoulder_follow, joint
            )
            out[f"upperarm_proximal_{side}"] = out[f"upperarm_distal_{side}"] = arm
            fore = motions.get(f"forearm_{side}", arm)
            out[f"forearm_proximal_{side}"] = out[f"forearm_distal_{side}"] = fore
    except KeyError as missing:
        raise ValueError(f"missing body motion {missing}") from missing
    return out


def pose_skin(
    vertices: Array, weights: Array, groups: tuple[str, ...], moves: Mapping[str, Rigid]
) -> Array:
    """Linear blend skinning; groups without a transform stay at rest.

    Precondition: ``weights`` is ``(len(vertices), len(groups))`` with rows
    summing to one.
    """
    if weights.shape != (len(vertices), len(groups)):
        raise ValueError("weights must be (vertices, groups)")
    posed = np.zeros_like(vertices)
    for column, group in enumerate(groups):
        used = weights[:, column] > 0.0
        image = moves.get(group, IDENTITY).apply(vertices[used])
        posed[used] += weights[used, column, None] * image
    return posed


def acromion_edges(
    skin: BodySkin, fitted: FittedSkin, side: str, settings: PoseSettings
) -> Edges:
    """Mesh edges on the path from the neck base over the acromion to the sleeve cap.

    A frontal strip through the shoulder joint, ``acromion_half_width_m`` thick,
    from ``acromion_inner_m`` off the midline to ``acromion_reach_m`` beyond and
    below the joint: where a shirt meets its sleeve.
    """
    sign = dict(SIDES)[side]
    joint = fitted.joints[f"shoulder_{side}"]
    local = fitted.vertices - joint
    out = sign * fitted.vertices[:, 1]
    keep = (
        (np.abs(local[:, 0]) < settings.acromion_half_width_m)
        & (out > settings.acromion_inner_m)
        & (sign * local[:, 1] < settings.acromion_reach_m)
        & (local[:, 2] > -settings.acromion_reach_m)
    )
    edges = mesh_edges(skin.faces)
    return edges[keep[edges].all(axis=1)]


def edge_length_change(before: Array, after: Array, edges: Edges) -> Array:
    """Per-edge change of length in metres between two posed surfaces."""
    first = np.linalg.norm(before[edges[:, 0]] - before[edges[:, 1]], axis=1)
    second = np.linalg.norm(after[edges[:, 0]] - after[edges[:, 1]], axis=1)
    return np.abs(second - first)
