"""Render a known skeleton through known cameras into per-view observations.

Output uses the ``view-observations/1.0.0`` schema ingest writes, so any
reconstruction stage consumes synthetic and real bundles identically. The
truth (3-D joints per frame, camera records, bone lengths, and the exact set
of corrupted observations) is written beside it for the metrics in
:mod:`.metrics`. Corruptions are explicit and seeded: Gaussian pixel noise,
random occlusion (joint dropped, low confidence), and gross outliers (joint
moved by a large random offset while keeping a *high* confidence — the case a
robust fitter must catch).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from src.shared.python.core.contracts import require
from src.shared.python.pose_estimation.observations import (
    DetectorLayout,
    KeypointObservation,
)

from .cameras import PinholeCamera
from .skeleton import JOINT_NAMES, RigidSkeleton, swing_trajectory

Array = npt.NDArray[np.float64]
TRUTH_SCHEMA_VERSION = "synthetic-truth/1.0.0"
OBSERVATIONS_SCHEMA_VERSION = "view-observations/1.0.0"
LAYOUT = DetectorLayout(name="synthetic_15", keypoint_names=JOINT_NAMES)


@dataclass(frozen=True)
class RenderOptions:
    """Seeded corruption applied to the ideal projections."""

    noise_px: float = 1.0
    occlusion_rate: float = 0.05
    outlier_rate: float = 0.02
    outlier_offset_px: float = 120.0
    seed: int = 0

    def __post_init__(self) -> None:
        require(self.noise_px >= 0, "noise_px must be non-negative", self.noise_px)
        for name in ("occlusion_rate", "outlier_rate"):
            v = getattr(self, name)
            require(0.0 <= v < 1.0, f"{name} must be in [0, 1)", v)
        require(self.outlier_offset_px > 0, "outlier_offset_px must be positive")


class SyntheticTruth(BaseModel):
    """``truth.json``: what the renderer knew."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = TRUTH_SCHEMA_VERSION
    joint_names: tuple[str, ...]
    fps: float
    joints_3d_m: list[list[list[float]]]  # (T, K, 3)
    bone_lengths_m: dict[str, float]
    cameras: list[dict[str, Any]]  # CameraCalibration.to_dict()
    outliers: dict[str, list[list[int]]]  # view -> [[frame, joint], ...]
    occluded: dict[str, list[list[int]]]
    options: dict[str, Any]


@dataclass(frozen=True)
class SyntheticScene:
    """Cameras, a skeleton and its motion, ready to render."""

    cameras: Sequence[PinholeCamera]
    skeleton: RigidSkeleton = field(default_factory=RigidSkeleton)
    fps: float = 60.0
    n_frames: int = 120

    def __post_init__(self) -> None:
        require(len(self.cameras) >= 1, "at least one camera")
        ids = [c.camera_id for c in self.cameras]
        require(len(set(ids)) == len(ids), "camera ids must be unique", ids)
        require(self.fps > 0 and self.n_frames > 0, "fps and n_frames positive")

    def joints_3d(self) -> Array:
        """Ground-truth joint positions ``(T, K, 3)`` from the swing motion."""
        roots, rotations = swing_trajectory(self.n_frames, self.fps)
        return np.stack(
            [
                self.skeleton.forward(roots[k], rotations[k])
                for k in range(self.n_frames)
            ]
        )

    def render(
        self, options: RenderOptions = RenderOptions()
    ) -> tuple[dict[str, dict[str, Any]], SyntheticTruth]:
        """Per-view observation payloads and the truth.

        Postcondition: every view has ``n_frames`` rows; every corrupted
        observation is listed in the truth by ``[frame, joint]``.
        """
        rng = np.random.default_rng(options.seed)
        joints = self.joints_3d()
        views: dict[str, dict[str, Any]] = {}
        outliers: dict[str, list[list[int]]] = {}
        occluded: dict[str, list[list[int]]] = {}
        for cam in self.cameras:
            rows, out_ids, occ_ids = self._render_view(cam, joints, options, rng)
            views[cam.camera_id] = {
                "schema_version": OBSERVATIONS_SCHEMA_VERSION,
                "view": cam.camera_id,
                "identity": cam.camera_id,
                "camera_id": cam.camera_id,
                "fps": self.fps,
                "width": cam.image_size_px[0],
                "height": cam.image_size_px[1],
                "frames_total": self.n_frames,
                "frames_with_pose": len(rows),
                "detector_layout": LAYOUT.to_dict(),
                "frames": rows,
                "provenance": {"estimator": "synthetic", "seed": options.seed},
            }
            outliers[cam.camera_id] = out_ids
            occluded[cam.camera_id] = occ_ids
        truth = SyntheticTruth(
            joint_names=JOINT_NAMES,
            fps=self.fps,
            joints_3d_m=joints.tolist(),
            bone_lengths_m=self.skeleton.bone_lengths(),
            cameras=[c.to_calibration().to_dict() for c in self.cameras],
            outliers=outliers,
            occluded=occluded,
            options=vars(options).copy(),
        )
        return views, truth

    def _render_view(
        self,
        cam: PinholeCamera,
        joints: Array,
        options: RenderOptions,
        rng: np.random.Generator,
    ) -> tuple[list[dict[str, Any]], list[list[int]], list[list[int]]]:
        rows: list[dict[str, Any]] = []
        out_ids: list[list[int]] = []
        occ_ids: list[list[int]] = []
        n_joints = len(JOINT_NAMES)
        for k in range(joints.shape[0]):
            px, in_front = cam.project(joints[k])
            visible: npt.NDArray[np.bool_] = in_front & cam.in_image(px)
            conf = np.where(visible, 0.95, 0.0)
            px = np.where(visible[:, None], px, 0.0)
            px = px + rng.normal(0.0, options.noise_px, px.shape) * visible[:, None]
            drop = rng.random(n_joints) < options.occlusion_rate
            gross = (rng.random(n_joints) < options.outlier_rate) & visible & ~drop
            conf[drop] = 0.05
            px[gross] += (
                rng.uniform(-1, 1, (int(gross.sum()), 2)) * options.outlier_offset_px
            )
            for j in np.flatnonzero(drop & visible):
                occ_ids.append([k, int(j)])
            for j in np.flatnonzero(gross):
                out_ids.append([k, int(j)])
            obs = KeypointObservation(
                camera_id=cam.camera_id,
                time_s=k / self.fps,
                keypoints_px=px,
                confidence=np.clip(conf, 0.0, 1.0),
            )
            rows.append(obs.to_dict())
        return rows, out_ids, occ_ids


def write_synthetic_bundle(
    out_dir: Path,
    views: Mapping[str, Mapping[str, Any]],
    truth: SyntheticTruth,
) -> Path:
    """``<out>/observations/<view>.json`` + ``<out>/truth.json``; returns ``out``."""
    obs_dir = out_dir / "observations"
    obs_dir.mkdir(parents=True, exist_ok=True)
    for view, payload in views.items():
        (obs_dir / f"{view}.json").write_text(
            json.dumps(payload, indent=1), encoding="utf-8"
        )
    (out_dir / "truth.json").write_text(
        truth.model_dump_json(indent=1), encoding="utf-8"
    )
    return out_dir


def load_truth(path: Path) -> SyntheticTruth:
    require(path.is_file(), "truth file must exist", str(path))
    return SyntheticTruth.model_validate_json(path.read_text(encoding="utf-8"))
