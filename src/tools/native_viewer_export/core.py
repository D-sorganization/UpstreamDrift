"""Pure core of the native viewer video export tool (NV-5, #11678).

Everything here is engine-SDK-free: settings, swing inputs, the backend
contract, overlay feeds and the clip-writing loop. Engine backends live in
:mod:`.backends`; each reports ``unavailable_reason()`` so a missing optional
dependency skips cleanly instead of failing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
import functools
import math
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
import warnings

import numpy as np
from numpy.typing import NDArray

from src.shared.python.force_overlay.glyphs import ForceGlyphStyle, GlyphSet
from src.shared.python.force_overlay.renderers.meshcat_glyphs import legend_text
from src.shared.python.golf_view_presets import VIEW_ORDER, get_view_preset
from src.shared.python.motion_matching.same_input import InputBundle
from src.shared.python.video_timing.frame_schedule import (
    DEFAULT_FPS,
    MAX_SPEED,
    FrameSchedule,
    speed_suffix,
)
from src.tools.native_viewer_export.compositor import (
    HudInfo,
    compose_grid,
    draw_hud,
    draw_label,
)

Image8 = NDArray[np.uint8]
ENGINES = ("drake", "pinocchio", "opensim", "myosuite")
VIEWER_NAMES = {
    "drake": "Drake MeshCat",
    "pinocchio": "Pinocchio MeshcatVisualizer",
    "opensim": "OpenSim simbody-visualizer",
    "myosuite": "MyoSuite MJRenderer arena",
}
GRID_VIEW = "2x2"
DEFAULT_SPEEDS = (1.0, 0.5)
IMPACT_CLIP_SPEED = 0.1
IMPACT_SUFFIX = "_impact"
HQ_SIZE = (1280, 720)
PREVIEW_SIZE = (640, 544)
HQ_CRF = 18
PREVIEW_CRF = 23
MAX_CRF = 51


class BackendUnavailable(RuntimeError):
    """A backend's optional dependency or runtime is missing."""


@dataclass(frozen=True)
class ExportSettings:
    """Rendering options shared by every backend.

    Playback is time-based: video frame ``j`` shows the swing at
    ``t0 + j * speed / fps`` for each of ``speeds`` (one clip set per speed,
    suffixed ``_1x``, ``_0p5x``). The default is the 1280x720 high-quality
    preset (libx264, yuv420p, CRF 18); :meth:`preview` keeps 640x544.
    ``impact_window_s`` adds a clip of that many seconds centred on impact at
    ``impact_speed``. ``stride`` is a deprecated alias for one fixed index step.

    Preconditions are validated on construction (``ValueError``).
    """

    views: tuple[str, ...] = VIEW_ORDER
    width: int = HQ_SIZE[0]
    height: int = HQ_SIZE[1]
    fps: int = int(DEFAULT_FPS)
    speeds: tuple[float, ...] = DEFAULT_SPEEDS
    stride: int | None = None
    crf: int = HQ_CRF
    impact_time_s: float | None = None
    impact_window_s: float | None = None
    impact_speed: float = IMPACT_CLIP_SPEED
    overlays: bool = True
    multiview: bool = True
    lookat_m: tuple[float, float, float] = (1.0, 0.0, 0.9)
    distance_m: float | None = None

    @classmethod
    def preview(cls, **overrides: Any) -> ExportSettings:
        """The small 640x544 preview preset (CRF 23)."""
        base: dict[str, Any] = {
            "width": PREVIEW_SIZE[0],
            "height": PREVIEW_SIZE[1],
            "crf": PREVIEW_CRF,
        }
        return cls(**{**base, **overrides})

    def _validate_clips(self) -> None:
        if not self.speeds:
            raise ValueError("speeds must not be empty")
        for speed in (*self.speeds, self.impact_speed):
            if not (math.isfinite(speed) and 0.0 < speed <= MAX_SPEED):
                raise ValueError(f"speeds must lie in (0, {MAX_SPEED:g}], got {speed}")
        if len({speed_suffix(v) for v in self.speeds}) != len(self.speeds):
            raise ValueError("speeds must be unique")
        if not 0 <= self.crf <= MAX_CRF:
            raise ValueError(f"crf must lie in [0, {MAX_CRF}]")
        for name in ("impact_time_s", "impact_window_s"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.impact_window_s is not None and self.impact_window_s <= 0.0:
            raise ValueError("impact_window_s must be positive")

    def __post_init__(self) -> None:
        if not self.views:
            raise ValueError("views must not be empty")
        for name in self.views:
            get_view_preset(name)
        if len(set(self.views)) != len(self.views):
            raise ValueError("views must be unique")
        for field_name in ("width", "height", "fps"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.stride is not None:
            if not isinstance(self.stride, int) or self.stride < 1:
                raise ValueError("stride must be a positive integer")
            warnings.warn(
                "ExportSettings.stride is deprecated: playback is time-based now; "
                "use speeds and fps",
                DeprecationWarning,
                stacklevel=3,
            )
        self._validate_clips()
        if len(self.lookat_m) != 3 or not all(math.isfinite(v) for v in self.lookat_m):
            raise ValueError("lookat_m must be three finite numbers")
        if self.distance_m is not None and not (
            math.isfinite(self.distance_m) and self.distance_m > 0.0
        ):
            raise ValueError("distance_m must be positive and finite")
        if self.multiview and len(self.views) != 4:
            raise ValueError("the 2x2 grid needs exactly four views")

    def effective_speeds(self, dt_s: float) -> tuple[float, ...]:
        """Speeds to export; a deprecated ``stride`` maps to its equivalent speed."""
        if self.stride is None:
            return self.speeds
        speed = self.stride * dt_s * self.fps
        if not 0.0 < speed <= MAX_SPEED:
            raise ValueError(
                f"stride {self.stride} at {self.fps} fps is {speed:.2f}x playback; "
                f"speeds must lie in (0, {MAX_SPEED:g}]"
            )
        return (speed,)


@dataclass(frozen=True)
class SwingInput:
    """A same-input bundle plus the engine rollout states to display."""

    bundle: InputBundle
    q: NDArray[np.float64]
    swing: str
    club: str
    rollout_engine: str
    sample_times_s: tuple[float, ...] | None = None  # set on resampled swings

    @property
    def reference_shape(self) -> tuple[int, ...]:
        """Shape the rollout ``q`` must match (the bundle reference)."""
        reference = self.bundle.reference_q
        if self.sample_times_s is not None:
            return (len(self.sample_times_s), int(reference.shape[1]))
        return tuple(reference.shape)

    def __post_init__(self) -> None:
        expected = self.reference_shape
        if self.q.shape != expected:
            raise ValueError(f"rollout q has shape {self.q.shape}, expected {expected}")
        if not np.isfinite(self.q).all():
            raise ValueError("rollout q must be finite")

    @property
    def n_states(self) -> int:
        return int(self.q.shape[0])

    def time_s(self, index: int) -> float:
        if self.sample_times_s is not None:
            return float(self.sample_times_s[index])
        return float(index * self.bundle.dt_s)

    @property
    def source_times_s(self) -> NDArray[np.float64]:
        """Swing time of every state of the original (unresampled) rollout."""
        return np.arange(self._source_steps(), dtype=float) * self.bundle.dt_s

    def _source_steps(self) -> int:
        return int(self.bundle.reference_q.shape[0])

    def _provenance_value(self, key: str) -> Any:
        return self.bundle.provenance.get(key)

    @property
    def impact_time_s(self) -> float:
        """Bundle ``provenance['impact_time_s']`` when present, else the last sample."""
        value = self._provenance_value("impact_time_s")
        return float(value) if value is not None else float(self.source_times_s[-1])


def load_swing_input(
    bundle_path: Path,
    rollout_path: Path | None = None,
    *,
    swing: str,
    club: str,
    rollout_engine: str = "reference",
) -> SwingInput:
    """Load a bundle and optionally an engine rollout ``.npz`` holding ``q``."""
    bundle = InputBundle.load(Path(bundle_path))
    if rollout_path is None:
        q = bundle.reference_q
        rollout_engine = bundle.reference_engine
    else:
        with np.load(Path(rollout_path)) as data:
            if "q" not in data.files:
                raise ValueError(f"{rollout_path} has no 'q' array")
            q = np.asarray(data["q"], dtype=float)
    return SwingInput(bundle, q, swing, club, rollout_engine)


def load_receipt_rollout(
    receipt_path: Path, bundle: InputBundle
) -> tuple[NDArray[np.float64], str]:
    """Rollout states and engine of a same-input receipt (``.json`` + ``.npz``).

    The receipt must describe ``bundle`` (matching spec hash) and a sibling
    ``.npz`` with the same stem must hold ``q``. Raises ``ValueError`` /
    ``FileNotFoundError`` otherwise.
    """
    import json

    receipt_path = Path(receipt_path)
    doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not str(doc.get("schema", "")).startswith("same-input-"):
        raise ValueError(f"{receipt_path} is not a same-input receipt")
    if doc.get("bundle", {}).get("spec_sha256") != bundle.spec_sha256:
        raise ValueError(f"{receipt_path} was produced from a different specification")
    npz = receipt_path.with_suffix(".npz")
    with np.load(npz) as data:
        if "q" not in data.files:
            raise ValueError(f"{npz} has no 'q' array")
        q = np.asarray(data["q"], dtype=float)
    return q, str(doc.get("engine", "unknown"))


def frame_indices(n_states: int, stride: int) -> list[int]:
    """State indices to render: every ``stride``-th state, always ending on the last."""
    if n_states < 1 or stride < 1:
        raise ValueError("n_states and stride must be positive")
    idx = list(range(0, n_states, stride))
    if idx[-1] != n_states - 1 and n_states - 1 - idx[-1] >= stride // 2:
        idx.append(n_states - 1)
    return idx


def default_glyph_style() -> ForceGlyphStyle:
    """Golfer-scale glyph style (about 0.7 m per kN, 0.5 m radius per 250 N m)."""
    return ForceGlyphStyle(
        force_scale_m_per_n=0.7 / 1000.0,
        torque_scale_m_per_nm=0.5 / 250.0,
        max_length_m=0.9,
        min_length_m=0.04,
        shaft_radius_m=0.014,
        magnitude_floor_nm=8.0,
        magnitude_floor_n=20.0,
    )


@dataclass
class OverlayFeed:
    """Glyph sets by state index, built lazily from an overlay provider."""

    frame_at: Callable[[int], Any]
    style: ForceGlyphStyle = field(default_factory=default_glyph_style)
    build: Callable[[Any, ForceGlyphStyle], GlyphSet] | None = None

    def glyphs_at(self, index: int) -> GlyphSet:
        build = self.build
        if build is None:
            from src.shared.python.force_overlay.glyphs import build_glyphs

            build = build_glyphs
        return build(self.frame_at(index), self.style)

    @staticmethod
    def legend(glyphs: GlyphSet | None) -> str:
        return "" if glyphs is None else legend_text(glyphs)


@runtime_checkable
class NativeBackend(Protocol):
    """One engine's native viewer."""

    engine: str

    def unavailable_reason(self) -> str | None:
        """``None`` when ready, else why this backend cannot run here."""
        ...

    def render(
        self,
        swing: SwingInput,
        settings: ExportSettings,
        indices: Sequence[int],
        overlay: OverlayFeed | None,
    ) -> Iterator[dict[str, Image8]]:
        """Yield ``{view name: RGB frame}`` for every state index in order."""
        ...


class ClipWriter(Protocol):
    def append_data(self, frame: Image8) -> None: ...
    def close(self) -> None: ...


WriterFactory = Callable[[Path, int], ClipWriter]


def imageio_writer(path: Path, fps: int, crf: int = HQ_CRF) -> ClipWriter:
    """Default clip writer: libx264, ``yuv420p``, constant rate factor ``crf``."""
    import imageio.v2 as imageio

    return imageio.get_writer(  # type: ignore[no-any-return]
        str(path),
        fps=fps,
        codec="libx264",
        quality=None,
        pixelformat="yuv420p",
        macro_block_size=8,
        output_params=["-crf", str(crf), "-preset", "medium"],
    )


@dataclass(frozen=True)
class ExportResult:
    """Outcome of one engine export.

    ``paths`` is keyed ``<view><suffix>`` (``face_on_1x``, ``2x2_0p5x``);
    ``frames`` counts the first clip set and ``frames_by_suffix`` every set.
    """

    engine: str
    paths: dict[str, Path]
    frames: int
    skipped_reason: str | None = None
    glyph_counts: tuple[int, ...] = ()
    frames_by_suffix: dict[str, int] = field(default_factory=dict)

    @property
    def skipped(self) -> bool:
        return self.skipped_reason is not None


def clip_name(swing: SwingInput, engine: str, view: str, suffix: str = "") -> str:
    return f"{swing.swing}_{engine}_{view}{suffix}.mp4"


@dataclass(frozen=True)
class ClipPlan:
    """One clip set: playback ``speed`` over an optional swing ``window``."""

    suffix: str
    speed: float
    window: tuple[float, float] | None = None


def impact_time_s(swing: SwingInput, settings: ExportSettings) -> float:
    """Impact time: the explicit setting, else the bundle provenance, else the end."""
    if settings.impact_time_s is not None:
        return settings.impact_time_s
    return swing.impact_time_s


def clip_plans(swing: SwingInput, settings: ExportSettings) -> list[ClipPlan]:
    """Clip sets to render: one per speed, plus the optional impact-window clip."""
    plans = [
        ClipPlan(speed_suffix(v), v)
        for v in settings.effective_speeds(swing.bundle.dt_s)
    ]
    if settings.impact_window_s is not None:
        centre, half = impact_time_s(swing, settings), settings.impact_window_s / 2.0
        plans.append(
            ClipPlan(
                IMPACT_SUFFIX + speed_suffix(settings.impact_speed),
                settings.impact_speed,
                (centre - half, centre + half),
            )
        )
    return plans


def _resampled(
    swing: SwingInput,
    feed: OverlayFeed | None,
    settings: ExportSettings,
    plan: ClipPlan,
) -> tuple[SwingInput, OverlayFeed | None]:
    """Swing re-sampled at the plan's frame times, with a matching overlay feed.

    Poses are interpolated; glyphs come from the nearest source sample (forces
    are not interpolated).
    """
    schedule = FrameSchedule(
        swing.source_times_s, settings.fps, plan.speed, plan.window
    )
    times = schedule.sample_times_s
    nearest = schedule.nearest_indices()
    resampled = replace(
        swing,
        q=schedule.interpolate(swing.q),
        sample_times_s=tuple(float(t) for t in times),
    )
    if feed is None:
        return resampled, None
    source = feed.frame_at
    return resampled, replace(feed, frame_at=lambda i: source(int(nearest[i])))


def _export_clip(
    backend: NativeBackend,
    swing: SwingInput,
    settings: ExportSettings,
    plan: ClipPlan,
    out_dir: Path,
    feed: OverlayFeed | None,
    writer_factory: WriterFactory,
) -> tuple[dict[str, Path], int, list[int]]:
    engine = backend.engine
    shown, shown_feed = _resampled(swing, feed, settings, plan)
    indices = list(range(shown.n_states))
    names = [*settings.views, *([GRID_VIEW] if settings.multiview else [])]
    paths = {
        f"{n}{plan.suffix}": out_dir / clip_name(swing, engine, n, plan.suffix)
        for n in names
    }
    impact = impact_time_s(swing, settings)
    writers: dict[str, ClipWriter] = {}
    glyph_counts: list[int] = []
    viewer = VIEWER_NAMES.get(engine, engine)
    frames = 0
    try:
        for k, tiles in zip(
            indices,
            backend.render(shown, settings, indices, shown_feed),
            strict=False,
        ):
            glyphs = shown_feed.glyphs_at(k) if shown_feed is not None else None
            if glyphs is not None:
                glyph_counts.append(len(glyphs.arrows) + len(glyphs.torque_arcs))
            t = shown.time_s(k)
            hud = HudInfo(
                t,
                viewer,
                swing.club,
                OverlayFeed.legend(glyphs),
                speed=plan.speed,
                ms_from_impact=(t - impact) * 1000.0,
            )
            labelled = {
                v: draw_label(tiles[v], get_view_preset(v).label, (8, 6))
                for v in settings.views
            }
            outputs = {v: draw_hud(labelled[v], hud) for v in settings.views}
            if settings.multiview:
                outputs[GRID_VIEW] = draw_hud(
                    compose_grid([labelled[v] for v in settings.views]), hud
                )
            for name, frame in outputs.items():
                key = f"{name}{plan.suffix}"
                if key not in writers:
                    writers[key] = writer_factory(paths[key], settings.fps)
                writers[key].append_data(frame)
            frames += 1
    finally:
        for writer in writers.values():
            writer.close()
    if frames != len(indices):
        raise RuntimeError(f"backend yielded {frames} frames, expected {len(indices)}")
    return paths, frames, glyph_counts


def export_swing(
    backend: NativeBackend,
    swing: SwingInput,
    settings: ExportSettings,
    out_dir: Path,
    overlay: OverlayFeed | None = None,
    writer_factory: WriterFactory | None = None,
) -> ExportResult:
    """Render ``swing`` in ``backend`` and write one clip per view (and the 2x2).

    One clip set is written per playback speed (and one for the optional
    impact window), each sampled by a time-based :class:`FrameSchedule`.
    Returns a skipped result (nothing written) when the backend reports a
    reason it cannot run. Postconditions: within a clip set every clip has the
    same number of frames, shown at ``settings.fps``.
    """
    engine = backend.engine
    reason = backend.unavailable_reason()
    if reason is not None:
        return ExportResult(engine, {}, 0, skipped_reason=reason)
    out_dir = Path(out_dir)
    plans = clip_plans(swing, settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    factory = writer_factory or functools.partial(imageio_writer, crf=settings.crf)
    feed = overlay if settings.overlays else None
    paths: dict[str, Path] = {}
    counts: dict[str, int] = {}
    glyph_counts: list[int] = []
    for plan in plans:
        clip_paths, frames, glyphs = _export_clip(
            backend, swing, settings, plan, out_dir, feed, factory
        )
        paths.update(clip_paths)
        counts[plan.suffix] = frames
        glyph_counts.extend(glyphs)
    return ExportResult(
        engine,
        paths,
        counts[plans[0].suffix],
        glyph_counts=tuple(glyph_counts),
        frames_by_suffix=counts,
    )
