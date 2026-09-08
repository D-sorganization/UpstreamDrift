"""Render variants' 3-D results on the recordings (#9795).

``render_frame`` draws every :class:`Track` of every requested variant on
one frame with a legend; ``export_overlay`` writes a clip for one view (or
a grid of all views) plus a JSON sidecar with the colours and, when the
view has observations, each track's reprojection RMS on it. Reuses the
tool's :func:`draw_pose`, :class:`VideoReader` and the clip writer.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.motion_capture.provenance import write_stamped
from src.motion_capture.reconstruct.overlay3d import (
    PALETTE,
    Colour,
    Track,
    reprojection_rms_px,
    variant_tracks,
)
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES
from src.shared.python.core.contracts import require

from .clips import _stamp, _writer
from .overlay import PoseTrack, draw_pose
from .player import VideoReader
from .session import load_session

OVERLAY_SCHEMA = "overlay-clip/1.0.0"
LEGEND_LINE_PX = 22


@dataclass(frozen=True)
class OverlaySpec:
    """What to draw: the tracks of several variants, with their colours."""

    session: Path
    view: str
    variants: tuple[str, ...]
    tracks: tuple[Track, ...]
    colours: dict[str, Colour]

    @classmethod
    def build(cls, session: Path, view: str, variants: Sequence[str]) -> OverlaySpec:
        """Precondition: at least one variant; each has joints or a model fit."""
        require(len(variants) >= 1, "at least one variant to overlay")
        colours = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(variants)}
        tracks: list[Track] = []
        for name in variants:
            tracks.extend(variant_tracks(session, name, view, colours[name]))
        return cls(session, view, tuple(variants), tuple(tracks), colours)

    @property
    def frames(self) -> int:
        return max((t.frames for t in self.tracks), default=0)


def draw_legend(
    frame: npt.NDArray[np.uint8], tracks: Sequence[Track]
) -> npt.NDArray[np.uint8]:
    """Colour swatches and labels in the top-right corner."""
    import cv2

    out = np.ascontiguousarray(frame)
    scale = max(out.shape[0] / 720.0, 0.5)
    line = int(LEGEND_LINE_PX * scale)
    x0 = out.shape[1] - int(320 * scale)
    y = int(20 * scale)
    for track in tracks:
        cv2.rectangle(
            out, (x0, y - line // 2), (x0 + line, y + line // 2), track.colour, -1
        )
        cv2.putText(
            out,
            track.label,
            (x0 + line + int(8 * scale), y + line // 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * scale,
            (255, 255, 255),
            max(1, int(scale)),
            cv2.LINE_AA,
        )
        y += line + int(4 * scale)
    return out


def render_frame(
    frame_bgr: npt.NDArray[np.uint8],
    tracks: Sequence[Track],
    frame_index: int,
    *,
    legend: bool = True,
) -> npt.NDArray[np.uint8]:
    """Every track's visible points and bones on a copy of the frame."""
    out = frame_bgr
    for track in tracks:
        pose = track.at(frame_index)
        if pose is None:
            continue
        out = draw_pose(
            out,
            pose[0],
            pose[1],
            track.edges,
            min_confidence=0.5,
            point_colour=track.colour,
            edge_colour=track.colour,
            thickness=2 if track.kind == "joints" else 1,
        )
    return draw_legend(out, tracks) if legend else np.ascontiguousarray(out)


def observed_track(session: Path, view: str, observation_set: str) -> PoseTrack | None:
    media = load_session(session)
    view_media = media.view(view)
    sets = view_media.observation_sets or {}
    path = sets.get(observation_set)
    return PoseTrack.load(path) if path else None


def track_metrics(
    spec: OverlaySpec, observed: PoseTrack | None
) -> list[dict[str, Any]]:
    """Per track: label, colour, held-out flag and reprojection RMS on the view."""
    rows = []
    for track in spec.tracks:
        rms = None
        if observed is not None and track.kind == "joints":
            kp, conf = _observed_arrays(observed, track.frames)
            rms = reprojection_rms_px(track, kp, conf)
        rows.append(
            {
                "variant": track.variant,
                "kind": track.kind,
                "label": track.label,
                "colour_bgr": list(track.colour),
                "held_out": track.held_out,
                "frames": track.frames,
                "reprojection_rms_px": rms,
            }
        )
    return rows


def _observed_arrays(observed: PoseTrack, frames: int) -> tuple[Any, Any]:
    """Observed keypoints in the reconstruct layout, ``(T, 15, 2)`` and ``(T, 15)``."""
    k = len(JOINT_NAMES)
    kp = np.zeros((frames, k, 2))
    conf = np.zeros((frames, k))
    names = list(observed.names)
    cols = [names.index(n) for n in JOINT_NAMES if n in names]
    rows = [i for i, n in enumerate(JOINT_NAMES) if n in names]
    for f in range(frames):
        pose = observed.at(f)
        if pose is None:
            continue
        kp[f, rows] = pose[0][cols]
        conf[f, rows] = pose[1][cols]
    return kp, conf


@dataclass(frozen=True)
class ClipRange:
    """Frames to render and the playback speed factor."""

    start: int = 0
    stop: int | None = None
    speed: float = 1.0

    def __post_init__(self) -> None:
        require(self.speed > 0, "speed must be positive", self.speed)
        require(self.start >= 0, "start must be >= 0", self.start)


def export_overlay(
    session: Path,
    view: str,
    variants: Sequence[str],
    out: Path,
    *,
    clip: ClipRange | None = None,
    observation_set: str = "observations",
    legend: bool = True,
) -> dict[str, Any]:
    """Write ``out`` (mp4) and ``out.with_suffix(".json")``; returns the sidecar.

    Preconditions: the view has a playable recording; ``clip.start`` inside
    the recording and ``<= stop``. Postcondition: the sidecar lists every
    track drawn.
    """
    clip = clip or ClipRange()
    start, stop, speed = clip.start, clip.stop, clip.speed
    media = load_session(session)
    playable = media.view(view).playable
    require(playable is not None, "view has no playable recording", view)
    assert playable is not None
    spec = OverlaySpec.build(session, view, variants)
    observed = observed_track(session, view, observation_set)
    with VideoReader(playable) as reader:
        last = (
            reader.frame_count - 1
            if stop is None
            else min(stop, reader.frame_count - 1)
        )
        require(0 <= start <= last, "start must be <= stop within the recording")
        fps = (reader.fps or 30.0) * speed
        writer = _writer(out, fps, (reader.width, reader.height))
        written = 0
        try:
            for index in range(start, last + 1):
                frame = reader.read(index)
                if frame is None:
                    break
                image = render_frame(frame, spec.tracks, index, legend=legend)
                writer.write(_stamp(image, f"{view} f{index}"))
                written += 1
        finally:
            writer.release()
    sidecar = {
        "view": view,
        "variants": list(variants),
        "frames": written,
        "start": start,
        "speed": speed,
        "video": str(out),
        "tracks": track_metrics(spec, observed),
    }
    write_stamped(
        out.with_suffix(".json"),
        sidecar,
        schema_version=OVERLAY_SCHEMA,
        module=__name__,
        inputs=[playable],
        parameters={"variants": list(variants), "view": view},
        base=session,
    )
    return sidecar
