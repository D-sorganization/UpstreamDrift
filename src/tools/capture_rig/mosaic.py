"""Composite (multipicture) video export: several streams in one file (#9815).

One :class:`~.layout_model.LayoutSpec` says which view goes where; this
module walks the sources frame by frame, synchronised by frame index (plus
the per-view offsets the session's strobe alignment recorded, when it has
them), draws each frame through :func:`~.layout_model.compose` and writes
the result with the tool's clip writer. A ``recorded`` (or ``live``, which
an export can only read back as its recording) tile shows the view's
playable file; an ``overlay`` tile draws the observed 2-D pose and the
requested variants' 3-D tracks on it, exactly as the player and ``rig
overlay`` do. A source that runs out of frames holds its last one, so the
longest source sets the length.

Slow motion is a playback-rate change, as in :mod:`clips` and
:mod:`overlay_render`: every frame is kept and the output fps is
``fps * speed``. Beside the video, ``out.with_suffix(".json")`` records the
layout used, the sources with their files and offsets, the frame range, the
speed and the standard provenance block (tool version, git SHA, hashed
inputs). :func:`write_composite` is the one stitching loop; ``clips.compare_takes``
runs through it too.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.motion_capture.provenance import relative_to, write_stamped
from src.motion_capture.rig.alignment import view_timing
from src.motion_capture.rig.bundle import load_bundle
from src.shared.python.core.contracts import require
from src.shared.python.logging_pkg.logging_config import get_logger

from .layout_model import (
    DEFAULT_PALETTE,
    PRESET_NAMES,
    Frame,
    LayoutSpec,
    Palette,
    SourceRef,
    compose,
    preset,
)
from .layout_presets import SESSION, USER, LayoutStore, LayoutStoreError
from .overlay import PoseTrack, draw_pose
from .overlay_render import ClipRange, OverlaySpec, render_frame
from .player import VideoReader
from .session import SessionMedia, ViewMedia, load_session

logger = get_logger(__name__)

MOSAIC_SCHEMA = "mosaic-clip/1.0.0"
MIN_OUT_FPS = 1.0
DEFAULT_FPS = 30.0
NS_PER_S = 1e9
_SIZE_RE = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")
_PLAYABLE_KINDS = ("recorded", "live", "overlay")

#: Gives the frame a source shows at output index ``k`` (``None`` = none).
FrameFn = Callable[[int], Frame | None]


def parse_size(text: str) -> tuple[int, int]:
    """``"1280x720"`` -> ``(1280, 720)``; raises ``ValueError`` mentioning ``WxH``."""
    match = _SIZE_RE.match(text or "")
    if match is None:
        raise ValueError(f"size must be WxH (got {text!r})")
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError(f"size must be a positive WxH (got {text!r})")
    return width, height


def default_sources(media: SessionMedia) -> tuple[SourceRef, ...]:
    """One ``recorded`` source per playable view, in plan order."""
    return tuple(
        SourceRef("recorded", v.view) for v in media.views if v.playable is not None
    )


def resolve_layout(
    text: str,
    session: Path | None,
    sources: Sequence[SourceRef] = (),
    *,
    user_root: Path | None = None,
) -> LayoutSpec:
    """The layout named by ``--layout``: a preset, a saved name or a JSON file.

    Order: a path to an existing ``.json`` file; a built-in preset (filled
    with ``sources``); a layout saved in the session scope; one saved in the
    user scope. Raises ``ValueError`` naming the layout when nothing matches
    or the file is malformed.
    """
    require(isinstance(text, str) and text.strip() != "", "layout must be named")
    name = text.strip()
    candidate = Path(name)
    if candidate.suffix.lower() == ".json" or candidate.is_file():
        return _layout_file(candidate)
    if name in PRESET_NAMES:
        return preset(name, sources)
    store = LayoutStore(user_root=user_root, session=session)
    for scope in (SESSION, USER):
        if store.exists(name, scope):
            return store.load(name, scope)
    raise ValueError(
        f"unknown layout {name!r}: not a preset {PRESET_NAMES}, "
        "a saved layout or a JSON file"
    )


def _layout_file(path: Path) -> LayoutSpec:
    if not path.is_file():
        raise ValueError(f"layout file not found: {path}")
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"layout file {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"layout file {path.name}: top level must be an object")
    return LayoutSpec.from_dict(payload)


def frame_offsets(session: Path, fps_by_view: Mapping[str, float]) -> dict[str, int]:
    """Per-view frame offsets from the manifest's strobe ``timing`` block.

    A view whose arrival clock stamps ``offset_ns`` later than the reference
    shows the same instant ``offset_ns * fps`` frames later, so its frame
    for output index ``k`` is ``k + offset``. Views without an available
    offset (or a zero one) are absent from the result.
    """
    _, _, manifest = load_bundle(session)
    timing = dict(manifest.timing)
    out: dict[str, int] = {}
    for view, fps in fps_by_view.items():
        entry = view_timing(timing, view)
        if not entry or entry.get("status") != "available":
            continue
        offset_ns = entry.get("offset_ns")
        if offset_ns is None:
            continue
        frames = int(round(int(offset_ns) / NS_PER_S * fps))
        if frames:
            out[view] = frames
    return out


# ---------------------------------------------------------------- sources


@dataclass
class _Source:
    """One tile's stream: a reader plus how to decorate and re-index frames."""

    ref: SourceRef
    view: ViewMedia
    reader: VideoReader
    offset: int = 0
    observed: PoseTrack | None = None
    overlay: OverlaySpec | None = None
    file: Path = field(init=False)

    def __post_init__(self) -> None:
        self.file = self.reader.path

    @property
    def fps(self) -> float:
        return self.reader.fps or self.view.fps or DEFAULT_FPS

    @property
    def last_index(self) -> int:
        """The last output index this source still has its own frame for."""
        return self.reader.frame_count - 1 - self.offset

    def at(self, index: int) -> Frame | None:
        own = index + self.offset
        if own < 0:
            return None
        frame = self.reader.read(own)
        if frame is None or self.ref.kind != "overlay":
            return frame
        return self._decorate(frame, own)

    def _decorate(self, frame: Frame, own: int) -> Frame:
        out = frame
        pose = self.observed.at(own) if self.observed is not None else None
        if pose is not None:
            edges = self.observed.edges if self.observed is not None else ()
            out = draw_pose(out, pose[0], pose[1], edges)
        if self.overlay is not None:
            out = render_frame(out, self.overlay.tracks, own, legend=True)
        return out

    def record(self, session: Path) -> dict[str, Any]:
        return {
            "key": self.ref.key,
            "kind": self.ref.kind,
            "view": self.ref.view,
            "variants": list(self.ref.variants),
            "file": relative_to(self.file, session),
            "fps": self.fps,
            "frames": self.reader.frame_count,
            "offset_frames": self.offset,
            "observation_set": (
                None if self.observed is None else self.observed.estimator
            ),
        }

    def close(self) -> None:
        self.reader.close()


def _observed(view: ViewMedia, observation_set: str) -> PoseTrack | None:
    sets = view.observation_sets or {}
    path = sets.get(observation_set)
    return PoseTrack.load(path) if path else None


def _overlay_spec(
    session: Path, ref: SourceRef, variants: Sequence[str]
) -> OverlaySpec | None:
    names = tuple(ref.variants) or tuple(variants)
    if not names:
        return None
    return OverlaySpec.build(session, ref.view, names)


def _open_sources(
    session: Path,
    media: SessionMedia,
    spec: LayoutSpec,
    *,
    variants: Sequence[str],
    observation_set: str,
) -> list[_Source]:
    """A reader per distinct non-empty source of ``spec``; all-or-nothing."""
    refs: list[SourceRef] = []
    for tile in spec.tiles:
        if not tile.source.is_empty and tile.source.key not in {r.key for r in refs}:
            refs.append(tile.source)
    require(len(refs) > 0, "layout has no source to export", spec.name)
    sources: list[_Source] = []
    try:
        for ref in refs:
            require(ref.kind in _PLAYABLE_KINDS, "unknown source kind", ref.kind)
            view = media.view(ref.view)
            require(view.playable is not None, "view has no playable recording", ref)
            assert view.playable is not None
            is_overlay = ref.kind == "overlay"
            sources.append(
                _Source(
                    ref=ref,
                    view=view,
                    reader=VideoReader(view.playable),
                    observed=_observed(view, observation_set) if is_overlay else None,
                    overlay=(
                        _overlay_spec(session, ref, variants) if is_overlay else None
                    ),
                )
            )
    except BaseException:
        for source in sources:
            source.close()
        raise
    offsets = frame_offsets(session, {s.ref.view: s.fps for s in sources})
    for source in sources:
        source.offset = offsets.get(source.ref.view, 0)
    return sources


# ------------------------------------------------------------------ writer


def write_composite(
    producers: Mapping[str, FrameFn],
    spec: LayoutSpec,
    out: Path,
    *,
    fps: float,
    size: tuple[int, int],
    first: int,
    last: int,
    palette: Palette = DEFAULT_PALETTE,
) -> int:
    """Compose output frames ``first..last`` from ``producers`` into ``out``.

    Each producer maps an output index to its own frame (or ``None``); a
    producer that returns ``None`` holds its last frame, and a tile whose
    producer never produced shows the placeholder. Preconditions:
    ``0 <= first <= last``, positive ``size``, ``fps >= MIN_OUT_FPS``.
    Postcondition: returns the number of frames written (``last - first + 1``).
    """
    from .clips import _writer

    require(0 <= first <= last, "frame range must be 0 <= first <= last", (first, last))
    require(fps >= MIN_OUT_FPS, "output fps too low", fps)
    require(size[0] > 0 and size[1] > 0, "canvas size must be positive", size)
    held: dict[str, Frame] = {}
    writer = _writer(out, fps, size)
    written = 0
    try:
        for index in range(first, last + 1):
            for key, producer in producers.items():
                frame = producer(index)
                if frame is not None:
                    held[key] = frame
            writer.write(compose(held, spec, size, palette))
            written += 1
    finally:
        writer.release()
    return written


# ------------------------------------------------------------------ export


@dataclass(frozen=True)
class MosaicResult:
    """What :func:`export_multipicture` wrote."""

    video: Path
    sidecar: Path
    layout: str
    sources: tuple[str, ...]
    frames: int
    first: int
    last: int
    fps: float
    speed: float
    size: tuple[int, int]
    offsets: dict[str, int]


def _resolve_range(
    sources: Sequence[_Source], clip: ClipRange | None
) -> tuple[int, int]:
    longest = max(s.last_index for s in sources)
    require(longest >= 0, "no source has a frame after its alignment offset")
    start = clip.start if clip is not None else 0
    stop = clip.stop if clip is not None and clip.stop is not None else longest
    last = min(stop, longest)
    require(start <= last, "start must be <= stop within the recordings", (start, last))
    return start, last


def export_multipicture(
    session: Path,
    spec: LayoutSpec,
    out: Path,
    *,
    clip: ClipRange | None = None,
    variants: Sequence[str] = (),
    observation_set: str = "observations",
    size: tuple[int, int] | None = None,
    speed: float | None = None,
    palette: Palette = DEFAULT_PALETTE,
) -> MosaicResult:
    """Write ``out`` through ``spec`` plus a provenance sidecar beside it.

    ``clip`` gives the output frame range (``stop`` ``None`` = the longest
    source); ``speed`` overrides ``clip.speed`` (every frame kept, fps
    scaled). ``variants`` are drawn on overlay tiles that name none of their
    own; ``observation_set`` supplies the 2-D pose of overlay tiles.
    Preconditions: every non-empty tile names a view with a playable
    recording; positive ``size``; ``0 < speed``; the range starts inside the
    recordings. Postcondition: the video holds ``result.frames`` frames and
    ``result.sidecar`` exists with the layout, sources and provenance.
    """
    require(isinstance(spec, LayoutSpec), "spec must be a LayoutSpec", type(spec))
    rate = speed if speed is not None else (clip.speed if clip is not None else 1.0)
    require(rate > 0, "speed must be positive", rate)
    canvas = size or spec.canvas
    require(canvas[0] > 0 and canvas[1] > 0, "canvas size must be positive", canvas)
    media = load_session(session)
    sources = _open_sources(
        session, media, spec, variants=variants, observation_set=observation_set
    )
    try:
        first, last = _resolve_range(sources, clip)
        fps = max(sources[0].fps * rate, MIN_OUT_FPS)
        logger.info(
            "multipicture %s: %d sources, frames %d-%d at %.3g fps -> %s",
            spec.name,
            len(sources),
            first,
            last,
            fps,
            out,
        )
        written = write_composite(
            {s.ref.key: s.at for s in sources},
            spec,
            out,
            fps=fps,
            size=canvas,
            first=first,
            last=last,
            palette=palette,
        )
        records = [s.record(session) for s in sources]
        files = [s.file for s in sources]
    finally:
        for source in sources:
            source.close()
    offsets = {r["view"]: r["offset_frames"] for r in records if r["offset_frames"]}
    sidecar = out.with_suffix(".json")
    write_stamped(
        sidecar,
        {
            "video": str(out),
            "layout": spec.to_dict(),
            "sources": records,
            "first": first,
            "last": last,
            "frames": written,
            "fps": fps,
            "speed": rate,
            "size": list(canvas),
        },
        schema_version=MOSAIC_SCHEMA,
        module=__name__,
        inputs=files,
        parameters={
            "layout": spec.name,
            "speed": rate,
            "variants": list(variants),
            "observation_set": observation_set,
            "size": list(canvas),
        },
        base=session,
    )
    return MosaicResult(
        video=out,
        sidecar=sidecar,
        layout=spec.name,
        sources=tuple(r["key"] for r in records),
        frames=written,
        first=first,
        last=last,
        fps=fps,
        speed=rate,
        size=(canvas[0], canvas[1]),
        offsets=offsets,
    )


def export_from_session(
    session: Path,
    layout: str,
    out: Path,
    *,
    clip: ClipRange | None = None,
    variants: Sequence[str] = (),
    observation_set: str = "observations",
    size: tuple[int, int] | None = None,
    speed: float | None = None,
) -> MosaicResult:
    """CLI entry: resolve ``layout`` (preset, saved name or file) and export.

    A preset is filled with the session's playable views in plan order.
    """
    media = load_session(session)
    try:
        spec = resolve_layout(layout, session, default_sources(media))
    except LayoutStoreError as exc:
        raise ValueError(f"layout {layout!r}: {exc}") from exc
    return export_multipicture(
        session,
        spec,
        out,
        clip=clip,
        variants=variants,
        observation_set=observation_set,
        size=size,
        speed=speed,
    )
