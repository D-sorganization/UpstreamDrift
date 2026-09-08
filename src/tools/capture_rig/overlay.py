"""Pose overlays for playback: observations drawn on the frame they came from.

A :class:`PoseTrack` is one view's ``observations/<view>.json`` indexed by
frame number (ingest writes ``time_s = frame_index / fps`` and skips frames
without a pose, so the index is recovered from the time). Skeleton edges come
from the estimator registry entry named in the file's provenance, so MediaPipe
and BODY_25 tracks draw their own bones without this module knowing either
layout.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import require

Array = npt.NDArray[np.float64]
Edge = tuple[int, int]

POINT_COLOUR = (0, 220, 255)  # BGR
EDGE_COLOUR = (60, 255, 60)
LOW_COLOUR = (0, 0, 255)


def skeleton_edges(
    names: Sequence[str], skeleton: Sequence[dict[str, Any]]
) -> tuple[Edge, ...]:
    """``(child, parent)`` index pairs for every parented joint present in ``names``."""
    index = {n: i for i, n in enumerate(names)}
    out = []
    for joint in skeleton:
        parent = joint.get("parent")
        if parent is not None and joint["name"] in index and parent in index:
            out.append((index[joint["name"]], index[parent]))
    return tuple(out)


def _registry_skeleton(estimator: str | None) -> tuple[dict[str, Any], ...]:
    if not estimator:
        return ()
    from src.shared.python.pose_estimation.registry import get_estimator_info

    try:
        return get_estimator_info(estimator).skeleton
    except KeyError:
        return ()


@dataclass(frozen=True)
class PoseTrack:
    """One view's detections keyed by frame index."""

    fps: float
    names: tuple[str, ...]
    edges: tuple[Edge, ...]
    frames: dict[int, tuple[Array, Array]]
    estimator: str | None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PoseTrack:
        fps = float(payload["fps"])
        require(fps > 0, "observations need a positive fps", fps)
        names = tuple(payload["detector_layout"]["keypoint_names"])
        estimator = payload.get("provenance", {}).get("estimator")
        frames: dict[int, tuple[Array, Array]] = {}
        for row in payload["frames"]:
            index = int(round(float(row["time_s"]) * fps))
            px = np.asarray(row["keypoints_px"], dtype=float)
            conf = np.asarray(row["confidence"], dtype=float)
            require(px.shape == (len(names), 2), "keypoint shape", px.shape)
            frames[index] = (px, conf)
        edges = skeleton_edges(names, _registry_skeleton(estimator))
        return cls(fps, names, edges, frames, estimator)

    @classmethod
    def load(cls, path: Path) -> PoseTrack:
        require(path.is_file(), "observations file must exist", str(path))
        return cls.from_payload(json.loads(path.read_text(encoding="utf-8")))

    def at(self, frame_index: int) -> tuple[Array, Array] | None:
        return self.frames.get(frame_index)

    @property
    def coverage(self) -> int:
        return len(self.frames)


def draw_pose(
    image_bgr: npt.NDArray[np.uint8],
    keypoints_px: Array,
    confidence: Array,
    edges: Sequence[Edge] = (),
    *,
    min_confidence: float = 0.5,
    point_colour: tuple[int, int, int] = POINT_COLOUR,
    edge_colour: tuple[int, int, int] = EDGE_COLOUR,
    thickness: int | None = None,
) -> npt.NDArray[np.uint8]:
    """A copy of the frame with confident joints and bones drawn on it.

    Joints below ``min_confidence`` are drawn small and red rather than
    hidden, so a coach sees where the detector was unsure. Colours are
    parameters so several tracks can share a frame (#9795). Precondition:
    one confidence per keypoint.
    """
    import cv2

    require(keypoints_px.shape[0] == confidence.shape[0], "one confidence per joint")
    out = image_bgr.copy()
    scale = thickness or max(1, int(round(min(out.shape[:2]) / 400)))
    ok = confidence >= min_confidence
    for a, b in edges:
        if ok[a] and ok[b] and np.isfinite(keypoints_px[[a, b]]).all():
            pa, pb = keypoints_px[a], keypoints_px[b]
            cv2.line(
                out,
                (int(pa[0]), int(pa[1])),
                (int(pb[0]), int(pb[1])),
                edge_colour,
                scale,
            )
    for i, (x, y) in enumerate(keypoints_px):
        if not np.isfinite([x, y]).all():
            continue
        colour = point_colour if ok[i] else LOW_COLOUR
        radius = 3 * scale if ok[i] else 2 * scale
        cv2.circle(out, (int(x), int(y)), radius, colour, -1)
    return out
