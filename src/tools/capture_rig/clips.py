"""Coaching clips: trimmed, annotated, slowed; two takes side by side (#9680, #9681).

The workflow needs three editing operations, not an editor: cut a swing out
of a take, draw the pose and the frame clock on it, slow it down; and put
two takes next to each other aligned on the same event. Both reuse the
overlay renderer and the frame reader the tile plays with, so what the coach
exports is what the tile showed.

Frame selection is by event name (``address``, ``top``, ``peak``, ``finish``)
resolved from a session's 2-D analysis or 3-D swing summary, or by frame
number. Slow motion is a playback-rate change: every source frame is kept,
the output fps is ``fps * speed``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.motion_capture.rig.edits import CropRect
from src.shared.python.core.contracts import require

from .overlay import PoseTrack, draw_pose
from .player import VideoReader, clamp_index
from .session import SessionMedia, ViewMedia, flatten_numbers, load_session

EVENT_KEYS = {
    "address": "address_frame",
    "top": "top_frame",
    "peak": "peak_speed_frame",
    "finish": "finish_frame",
}
MIN_OUT_FPS = 1.0
TEXT_COLOUR = (255, 255, 255)


@dataclass(frozen=True)
class ClipRendering:
    """Optional crop and cancellable export; defaults retain coaching clips."""

    crop: CropRect | None = None
    clock: bool = True
    strict: bool = False
    cancelled: Callable[[], bool] = lambda: False
    progress: Callable[[int, int], None] = lambda done, total: None

    def size(self, width: int, height: int) -> tuple[int, int]:
        if self.crop:
            self.crop.validate_size(width, height)
            # Video encoders require even dimensions: pad, never cut source pixels.
            width, height = self.crop.width, self.crop.height
            return width + width % 2, height + height % 2
        return width, height

    def image(self, frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        if self.crop is None:
            return frame
        crop = self.crop
        cropped = frame[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width]
        return np.pad(
            cropped,
            ((0, crop.height % 2), (0, crop.width % 2), (0, 0)),
            mode="edge",
        )


@dataclass(frozen=True)
class ClipRange:
    """Inclusive frame range of one view."""

    first: int
    last: int

    def __post_init__(self) -> None:
        require(self.first >= 0, "first frame must be >= 0", self.first)
        require(
            self.last >= self.first, "last frame before first", (self.first, self.last)
        )

    @property
    def frames(self) -> int:
        return self.last - self.first + 1


def events_for(media: SessionMedia, view: str) -> dict[str, int]:
    """``{event: frame}`` from the 2-D analysis of ``view``, else the 3-D summary."""
    payload: dict[str, Any] | None = None
    if media.analysis_2d and view in media.analysis_2d:
        payload = media.analysis_2d[view]
    elif media.swing_summary:
        payload = media.swing_summary
    if payload is None:
        return {}
    events = payload.get("events", {})
    return {name: int(events[key]) for name, key in EVENT_KEYS.items() if key in events}


def resolve_frame(spec: str, events: dict[str, int], *, offset: int = 0) -> int:
    """``"peak"``, ``"top-10"``, ``"address+5"`` or a frame number → frame index."""
    text = spec.strip().lower()
    for sign in ("+", "-"):
        if sign in text and not text.lstrip("-").isdigit():
            name, _, delta = text.partition(sign)
            base = resolve_frame(name, events)
            return max(base + int(sign + delta), 0)
    if text.lstrip("-").isdigit():
        return max(int(text) + offset, 0)
    require(text in events, "unknown event or no analysis for it", spec)
    return max(events[text] + offset, 0)


def _writer(path: Path, fps: float, size: tuple[int, int]) -> Any:
    import cv2

    fourcc = cv2.VideoWriter.fourcc(*("avc1" if path.suffix == ".mp4" else "MJPG"))
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():  # avc1 may be unavailable; mp4v always is
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
    require(writer.isOpened(), "could not open a video writer", str(path))
    return writer


def _stamp(frame: npt.NDArray[np.uint8], text: str) -> npt.NDArray[np.uint8]:
    import cv2

    scale = max(frame.shape[0] / 720.0, 0.5)
    cv2.putText(
        frame,
        text,
        (int(12 * scale), int(28 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7 * scale,
        (0, 0, 0),
        int(3 * scale),
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (int(12 * scale), int(28 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7 * scale,
        TEXT_COLOUR,
        int(1 * scale),
        cv2.LINE_AA,
    )
    return frame


def _rendered(
    reader: VideoReader,
    index: int,
    track: PoseTrack | None,
    *,
    min_confidence: float,
    label: str,
    rendering: ClipRendering = ClipRendering(),
) -> npt.NDArray[np.uint8] | None:
    frame = reader.read(index)
    if frame is None:
        return None
    pose = track.at(index) if track else None
    if pose is not None:
        frame = draw_pose(
            frame,
            pose[0],
            pose[1],
            track.edges if track else (),
            min_confidence=min_confidence,
        )
    frame = np.ascontiguousarray(rendering.image(frame))
    fps = reader.fps or 30.0
    return (
        _stamp(frame, f"{label} f{index} t={index / fps:.3f}s")
        if rendering.clock
        else frame
    )


def export_clip(
    view: ViewMedia,
    clip: ClipRange,
    out: Path,
    *,
    speed: float = 1.0,
    observations: Path | None = None,
    min_confidence: float = 0.5,
    label: str | None = None,
    rendering: ClipRendering = ClipRendering(),
) -> dict[str, Any]:
    """Write ``out`` with frames ``clip`` of ``view`` at ``fps * speed``.

    Preconditions: a playable view, ``0 < speed <= 1`` (slow motion or real
    time; nothing is dropped). Postcondition: every frame of the range that
    decodes is written, with the overlay when a track is given.
    """
    require(view.playable is not None, "view has no playable recording", view.view)
    require(0.0 < speed <= 1.0, "speed must be in (0, 1]", speed)
    assert view.playable is not None
    if out.resolve() in {p.resolve() for p in (view.recording, view.proxy) if p}:
        raise ValueError("Export must not overwrite source media")
    track_path = observations or view.observations
    track = PoseTrack.load(track_path) if track_path else None
    with VideoReader(view.playable) as reader:
        if rendering.strict and clip.last >= reader.frame_count:
            raise ValueError("Selection exceeds the decodable recording")
        last = clamp_index(clip.last, reader.frame_count)
        rate = max((reader.fps or view.fps or 30.0) * speed, MIN_OUT_FPS)
        writer = _writer(out, rate, rendering.size(reader.width, reader.height))
        written = 0
        try:
            for index in range(clip.first, last + 1):
                if rendering.cancelled():
                    raise InterruptedError("Swing export cancelled")
                frame = _rendered(
                    reader,
                    index,
                    track,
                    min_confidence=min_confidence,
                    label=label or view.view,
                    rendering=rendering,
                )
                if frame is None:
                    if rendering.strict:
                        raise ValueError(f"Could not decode source frame {index}")
                    break
                writer.write(frame)
                written += 1
                rendering.progress(written, clip.frames)
        finally:
            writer.release()
    require(written > 0, "no frame of the range decoded", (clip.first, last))
    return {
        "file": str(out),
        "view": view.view,
        "first": clip.first,
        "last": last,
        "frames": written,
        "fps": rate,
        "speed": speed,
        "overlay": str(track_path) if track_path else None,
    }


def _resize_to_height(
    frame: npt.NDArray[np.uint8], height: int
) -> npt.NDArray[np.uint8]:
    import cv2

    if frame.shape[0] == height:
        return frame
    width = int(round(frame.shape[1] * height / frame.shape[0]))
    return np.asarray(cv2.resize(frame, (width, height)), dtype=np.uint8)


def _take_producer(
    reader: VideoReader,
    start: int,
    track: PoseTrack | None,
    *,
    ratio: float,
    height: int,
    label: str,
    min_confidence: float,
) -> Callable[[int], npt.NDArray[np.uint8] | None]:
    """Output index ``k`` -> this take's annotated frame ``start + k * ratio``.

    Indices before the take starts show its first frame (held), as the
    compositor holds a source that returns ``None``.
    """

    def at(k: int) -> npt.NDArray[np.uint8] | None:
        index = max(start + int(round(k * ratio)), 0)
        frame = _rendered(
            reader, index, track, min_confidence=min_confidence, label=label
        )
        return None if frame is None else _resize_to_height(frame, height)

    return at


def compare_takes(
    left: tuple[ViewMedia, dict[str, int]],
    right: tuple[ViewMedia, dict[str, int]],
    out: Path,
    *,
    align: str = "top",
    before_s: float = 1.0,
    after_s: float = 1.0,
    speed: float = 0.5,
    min_confidence: float = 0.5,
) -> dict[str, Any]:
    """Side-by-side video of two views aligned so ``align`` happens on the same frame.

    Each side runs at its own fps; the output uses the left take's rate. A
    side that runs out of frames holds its last frame. The stitching is the
    layout compositor's (``mosaic.write_composite`` through a 1x2
    ``LayoutSpec``), so this is the same picture the multiview export makes.
    Precondition: both takes have the ``align`` event.
    """
    from .layout_model import Cell, LayoutSpec, SourceRef, Tile
    from .mosaic import write_composite

    (view_l, ev_l), (view_r, ev_r) = left, right
    require(align in ev_l and align in ev_r, "both takes need the event", align)
    require(
        view_l.playable is not None and view_r.playable is not None, "playable views"
    )
    assert view_l.playable is not None and view_r.playable is not None
    track_l = PoseTrack.load(view_l.observations) if view_l.observations else None
    track_r = PoseTrack.load(view_r.observations) if view_r.observations else None
    with VideoReader(view_l.playable) as rl, VideoReader(view_r.playable) as rr:
        fps_l, fps_r = rl.fps or 30.0, rr.fps or 30.0
        n = int(round((before_s + after_s) * fps_l))
        start_l = ev_l[align] - int(round(before_s * fps_l))
        start_r = ev_r[align] - int(round(before_s * fps_r))
        height = min(rl.height, rr.height)
        require(
            rl.read(max(start_l, 0)) is not None
            and rr.read(max(start_r, 0)) is not None,
            "takes must decode",
        )
        widths = [int(round(r.width * height / r.height)) for r in (rl, rr)]
        spec = LayoutSpec(
            name="compare_takes",
            rows=1,
            cols=2,
            tiles=(
                Tile(SourceRef("recorded", "A"), Cell(0, 0), show_label=False),
                Tile(SourceRef("recorded", "B"), Cell(0, 1), show_label=False),
            ),
            canvas=(sum(widths), height),
        )
        producers = {
            "recorded:A": _take_producer(
                rl,
                start_l,
                track_l,
                ratio=1.0,
                height=height,
                label="A",
                min_confidence=min_confidence,
            ),
            "recorded:B": _take_producer(
                rr,
                start_r,
                track_r,
                ratio=fps_r / fps_l,
                height=height,
                label="B",
                min_confidence=min_confidence,
            ),
        }
        written = write_composite(
            producers,
            spec,
            out,
            fps=max(fps_l * speed, MIN_OUT_FPS),
            size=spec.canvas,
            first=0,
            last=n - 1,
        )
    return {
        "file": str(out),
        "align": align,
        "frames": written,
        "left": {"view": view_l.view, "event_frame": ev_l[align]},
        "right": {"view": view_r.view, "event_frame": ev_r[align]},
        "speed": speed,
    }


def metric_deltas(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    """``{metric: {"a": x, "b": y, "delta": y - x}}`` over shared numeric leaves."""
    rows_a = dict(flatten_numbers(a or {}))
    rows_b = dict(flatten_numbers(b or {}))
    out: dict[str, Any] = {}
    for key in rows_a.keys() & rows_b.keys():
        try:
            x, y = float(rows_a[key]), float(rows_b[key])
        except ValueError:
            continue
        out[key] = {"a": x, "b": y, "delta": y - x}
    return out


def clip_from_session(
    session: Path,
    view: str,
    *,
    start: str,
    end: str,
    out: Path,
    speed: float = 1.0,
    observation_set: str | None = None,
) -> dict[str, Any]:
    """CLI entry: resolve events for ``view`` in ``session`` and export."""
    media = load_session(session)
    vm = media.view(view)
    events = events_for(media, view)
    clip = ClipRange(resolve_frame(start, events), resolve_frame(end, events))
    obs = None
    if observation_set:
        require(
            vm.observation_sets is not None and observation_set in vm.observation_sets,
            "unknown observation set",
            observation_set,
        )
        assert vm.observation_sets is not None
        obs = vm.observation_sets[observation_set]
    result = export_clip(vm, clip, out, speed=speed, observations=obs)
    result["events"] = events
    return result


def compare_from_sessions(
    session_a: Path,
    view_a: str,
    session_b: Path,
    view_b: str,
    *,
    out: Path,
    align: str = "top",
    speed: float = 0.5,
) -> dict[str, Any]:
    """CLI entry: side-by-side of two sessions' views plus metric deltas."""
    ma, mb = load_session(session_a), load_session(session_b)
    result = compare_takes(
        (ma.view(view_a), events_for(ma, view_a)),
        (mb.view(view_b), events_for(mb, view_b)),
        out,
        align=align,
        speed=speed,
    )
    summary_a = (ma.analysis_2d or {}).get(view_a) or ma.swing_summary
    summary_b = (mb.analysis_2d or {}).get(view_b) or mb.swing_summary
    result["deltas"] = metric_deltas(summary_a, summary_b)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
