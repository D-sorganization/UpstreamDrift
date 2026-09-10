"""Project a variant's 3-D results back onto any camera view (#9795).

Two kinds of track per variant: the reconstructed joints (``reconstruct/
joints_3d_m.npy``, edges from the skeleton) and, when a model was fitted,
its landmarks (``model/landmarks_fit.npy``, edges through the landmark map).
A view the variant never used is still renderable ("held out"): its camera
comes from the variant's own reconstruction when present, else from the
variant the cameras were borrowed from, else from the session's default
variant. Qt-free; extended calibrated lenses load OpenCV only for projection.
Drawing lives in the tool package.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import require
from src.shared.python.pose_estimation.observations import CameraCalibration

from ..variants import get_variant, variant_dir
from .cameras import PinholeCamera
from .fit import RECONSTRUCTION_FILE
from .skeleton import JOINT_NAMES, PARENTS

Array: TypeAlias = npt.NDArray[np.float64]
Mask: TypeAlias = npt.NDArray[np.bool_]
Edge = tuple[int, int]
Colour = tuple[int, int, int]

#: Six distinct BGR colours; variant ``i`` draws in ``PALETTE[i % 6]``.
PALETTE: tuple[Colour, ...] = (
    (60, 255, 60),
    (255, 160, 40),
    (40, 80, 255),
    (255, 60, 220),
    (0, 220, 255),
    (200, 200, 200),
)
MODEL_DIM = 0.6  # the model track is drawn a shade darker than the joints


def dim(colour: Colour, factor: float = MODEL_DIM) -> Colour:
    return (int(colour[0] * factor), int(colour[1] * factor), int(colour[2] * factor))


def project_track(points_world: Array, camera: PinholeCamera) -> tuple[Array, Mask]:
    """``(px (T, K, 2), visible (T, K))`` for world points ``(T, K, 3)``.

    Not visible: behind the camera, outside the image, non-finite, or the
    all-zero rows the reconstruction uses for unobservable joints.
    """
    pts = np.asarray(points_world, dtype=float)
    require(pts.ndim == 3 and pts.shape[2] == 3, "points must be (T, K, 3)", pts.shape)
    t, k = pts.shape[:2]
    flat = pts.reshape(-1, 3)
    finite = np.isfinite(flat).all(axis=1) & (np.abs(flat).sum(axis=1) > 0)
    if camera.distortion is not None and np.any(camera.distortion):
        from ..reference.registration import project_reference_to_camera

        px, visible = project_reference_to_camera(np.nan_to_num(flat), finite, camera)
        return px.reshape(t, k, 2), visible.reshape(t, k)
    px, in_front = camera.project(np.nan_to_num(flat))
    visible = in_front & finite & camera.in_image(px)
    return px.reshape(t, k, 2), visible.reshape(t, k)


def skeleton_edges(names: Sequence[str] = JOINT_NAMES) -> tuple[Edge, ...]:
    index = {n: i for i, n in enumerate(names)}
    return tuple(
        (index[child], index[parent])
        for child, parent in PARENTS.items()
        if parent is not None and child in index and parent in index
    )


def edges_from_landmark_map(
    landmark_names: Sequence[str], landmark_map: Mapping[str, Any]
) -> tuple[Edge, ...]:
    """Edges between model landmarks whose reconstruct joints are parented."""
    joint_of = {name: src for name, src in landmark_map.items() if isinstance(src, str)}
    landmark_of_joint = {j: n for n, j in joint_of.items()}
    index = {n: i for i, n in enumerate(landmark_names)}
    out = []
    for child, parent in PARENTS.items():
        if parent is None:
            continue
        a, b = landmark_of_joint.get(child), landmark_of_joint.get(parent)
        if a is not None and b is not None and a in index and b in index:
            out.append((index[a], index[b]))
    return tuple(out)


@dataclass(frozen=True)
class Track:
    """One projected series on one view."""

    variant: str
    kind: str  # "joints" | "model"
    label: str
    colour: Colour
    px: Array  # (T, K, 2)
    visible: Mask  # (T, K)
    edges: tuple[Edge, ...]
    held_out: bool
    names: tuple[str, ...]

    @property
    def frames(self) -> int:
        return int(self.px.shape[0])

    def at(self, frame: int) -> tuple[Array, Array] | None:
        """``(keypoints (K, 2), confidence (K,))`` or ``None`` past the end."""
        if not 0 <= frame < self.frames:
            return None
        return self.px[frame], self.visible[frame].astype(float)


def _cameras_of(root: Path) -> dict[str, CameraCalibration]:
    path = root / "reconstruct" / RECONSTRUCTION_FILE
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    cameras = (CameraCalibration.from_dict(row) for row in payload["cameras"])
    return {camera.camera_id: camera for camera in cameras}


def camera_for_view(session: Path, variant: str, view: str) -> PinholeCamera:
    """Existing ideal-pinhole projection interface; camera selection stays shared."""
    return PinholeCamera.from_calibration(calibration_for_view(session, variant, view))


def calibration_for_view(session: Path, variant: str, view: str) -> CameraCalibration:
    """The view's camera: own reconstruction, then ``cameras_from``, then default.

    Precondition: some variant of the session has a camera for ``view``.
    """
    candidates = [variant]
    record = get_variant(session, variant)
    if record is not None and record.source.get("cameras_from") is not None:
        candidates.append(str(record.source["cameras_from"]))
    candidates.append("")
    for name in candidates:
        camera = _cameras_of(variant_dir(session, name)).get(view)
        if camera is not None:
            return camera
    raise ValueError(
        f"no camera for view {view!r} in variant {variant!r} or its sources"
    )


def variant_views(session: Path, variant: str) -> tuple[str, ...]:
    record = get_variant(session, variant)
    if record is not None:
        return record.views
    summary = (
        variant_dir(session, variant) / "reconstruct" / "session_reconstruction.json"
    )
    if summary.is_file():
        return tuple(json.loads(summary.read_text(encoding="utf-8")).get("views", ()))
    return ()


def variant_tracks(
    session: Path, variant: str, view: str, colour: Colour
) -> list[Track]:
    """Joint and model tracks of ``variant`` projected onto ``view``.

    Postcondition: at least one track, or ``ValueError`` when the variant
    has neither joints nor a model fit.
    """
    root = variant_dir(session, variant)
    camera = camera_for_view(session, variant, view)
    held_out = view not in variant_views(session, variant)
    label = (variant or "default") + (" (held out)" if held_out else "")
    tracks: list[Track] = []
    joints_file = root / "reconstruct" / "joints_3d_m.npy"
    if joints_file.is_file():
        px, visible = project_track(np.load(joints_file), camera)
        tracks.append(
            Track(
                variant,
                "joints",
                f"{label} joints",
                colour,
                px,
                visible,
                skeleton_edges(),
                held_out,
                JOINT_NAMES,
            )
        )
    angles = root / "model" / "joint_angles.json"
    landmarks_file = root / "model" / "landmarks_fit.npy"
    if angles.is_file() and landmarks_file.is_file():
        payload = json.loads(angles.read_text(encoding="utf-8"))
        names = tuple(_landmark_names(payload))
        px, visible = project_track(np.load(landmarks_file), camera)
        tracks.append(
            Track(
                variant,
                "model",
                f"{label} model {payload.get('model', '')}",
                dim(colour),
                px,
                visible,
                edges_from_landmark_map(names, payload.get("landmark_map", {})),
                held_out,
                names,
            )
        )
    require(bool(tracks), "variant has no reconstruction or model fit", variant)
    return tracks


def _landmark_names(payload: Mapping[str, Any]) -> list[str]:
    """Model landmark order: from the registry when the model is known."""
    from .model.kinematics import ArticulatedModel
    from .model.registry import MODELS

    for registered in MODELS.values():
        if registered.spec.name == payload.get("model"):
            return list(ArticulatedModel(registered.spec).landmark_names)
    return list(payload.get("landmark_map", {}))


def reprojection_rms_px(
    track: Track, keypoints_px: Array, confidence: Array
) -> float | None:
    """RMS pixel distance between a track and observed keypoints ``(T, K, 2)``.

    Only frames/points visible in the track and confident in the observation
    count; ``None`` when nothing overlaps. Precondition: same ``K``.
    """
    require(keypoints_px.shape[1] == track.px.shape[1], "keypoint count differs")
    t = min(track.frames, keypoints_px.shape[0])
    mask = track.visible[:t] & (confidence[:t] > 0)
    if not mask.any():
        return None
    diff = track.px[:t] - keypoints_px[:t]
    d = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))[
        mask
    ]  # ⚡ Bolt: np.sqrt(np.einsum) is ~10x faster than np.linalg.norm(..., axis=2)
    return float(np.sqrt(np.mean(d**2)))


def reference_track(
    registration: Any,
    motion: Any,
    camera: PinholeCamera | CameraCalibration,
    scene_times: Array,
    colour: Colour,
    *,
    label: str | None = None,
) -> Track:
    """Project a registered ReferenceMotion onto a view at requested scene times (#9865).

    Missing joints across interpolation gaps and behind-camera/out-of-frame points
    are marked invisible. The track kind is "reference".
    """
    from src.motion_capture.reference.registration import (
        project_reference_to_camera,
        sample_reference_motion,
    )

    t_eval = np.asarray(scene_times, dtype=float)
    require(t_eval.ndim == 1, "scene_times must be 1-D")
    pts_world, valid_mask = sample_reference_motion(motion, registration, t_eval)
    px, visible = project_reference_to_camera(pts_world, valid_mask, camera)

    track_label = label or f"reference: {getattr(motion, 'title', 'motion')}"
    return Track(
        variant=getattr(registration, "reference_id", "reference"),
        kind="reference",
        label=track_label,
        colour=colour,
        px=px,
        visible=visible,
        edges=tuple(tuple(e) for e in getattr(motion, "edges", ())),
        held_out=False,
        names=tuple(getattr(motion, "joint_names", ())),
    )
