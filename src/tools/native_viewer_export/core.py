"""Pure core of the native viewer video export tool (NV-5, #11678).

Everything here is engine-SDK-free: settings, swing inputs, the backend
contract, overlay feeds and the clip-writing loop. Engine backends live in
:mod:`.backends`; each reports ``unavailable_reason()`` so a missing optional
dependency skips cleanly instead of failing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.shared.python.force_overlay.glyphs import ForceGlyphStyle, GlyphSet
from src.shared.python.force_overlay.renderers.meshcat_glyphs import legend_text
from src.shared.python.golf_view_presets import VIEW_ORDER, get_view_preset
from src.shared.python.motion_matching.same_input import InputBundle
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


class BackendUnavailable(RuntimeError):
    """A backend's optional dependency or runtime is missing."""


@dataclass(frozen=True)
class ExportSettings:
    """Rendering options shared by every backend.

    Preconditions are validated on construction (``ValueError``).
    """

    views: tuple[str, ...] = VIEW_ORDER
    width: int = 640
    height: int = 544
    fps: int = 20
    stride: int = 40
    overlays: bool = True
    multiview: bool = True
    lookat_m: tuple[float, float, float] = (1.0, 0.0, 0.9)
    distance_m: float | None = None

    def __post_init__(self) -> None:
        if not self.views:
            raise ValueError("views must not be empty")
        for name in self.views:
            get_view_preset(name)
        if len(set(self.views)) != len(self.views):
            raise ValueError("views must be unique")
        for field_name in ("width", "height", "fps", "stride"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if len(self.lookat_m) != 3 or not all(math.isfinite(v) for v in self.lookat_m):
            raise ValueError("lookat_m must be three finite numbers")
        if self.distance_m is not None and not (
            math.isfinite(self.distance_m) and self.distance_m > 0.0
        ):
            raise ValueError("distance_m must be positive and finite")
        if self.multiview and len(self.views) != 4:
            raise ValueError("the 2x2 grid needs exactly four views")


@dataclass(frozen=True)
class SwingInput:
    """A same-input bundle plus the engine rollout states to display."""

    bundle: InputBundle
    q: NDArray[np.float64]
    swing: str
    club: str
    rollout_engine: str

    def __post_init__(self) -> None:
        if self.q.shape != self.bundle.reference_q.shape:
            raise ValueError(
                f"rollout q has shape {self.q.shape}, "
                f"expected {self.bundle.reference_q.shape}"
            )
        if not np.isfinite(self.q).all():
            raise ValueError("rollout q must be finite")

    @property
    def n_states(self) -> int:
        return int(self.q.shape[0])

    def time_s(self, index: int) -> float:
        return float(index * self.bundle.dt_s)


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


def imageio_writer(path: Path, fps: int) -> ClipWriter:
    """Default clip writer (H.264 mp4 through imageio-ffmpeg)."""
    import imageio.v2 as imageio

    return imageio.get_writer(  # type: ignore[no-any-return]
        str(path), fps=fps, codec="libx264", quality=8, macro_block_size=8
    )


@dataclass(frozen=True)
class ExportResult:
    """Outcome of one engine export."""

    engine: str
    paths: dict[str, Path]
    frames: int
    skipped_reason: str | None = None
    glyph_counts: tuple[int, ...] = ()

    @property
    def skipped(self) -> bool:
        return self.skipped_reason is not None


def clip_name(swing: SwingInput, engine: str, view: str) -> str:
    return f"{swing.swing}_{engine}_{view}.mp4"


def export_swing(
    backend: NativeBackend,
    swing: SwingInput,
    settings: ExportSettings,
    out_dir: Path,
    overlay: OverlayFeed | None = None,
    writer_factory: WriterFactory = imageio_writer,
) -> ExportResult:
    """Render ``swing`` in ``backend`` and write one clip per view (and the 2x2).

    Returns a skipped result (nothing written) when the backend reports a
    reason it cannot run. Postconditions: every written clip has the same
    number of frames.
    """
    engine = backend.engine
    reason = backend.unavailable_reason()
    if reason is not None:
        return ExportResult(engine, {}, 0, skipped_reason=reason)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = frame_indices(swing.n_states, settings.stride)
    names = [*settings.views, *([GRID_VIEW] if settings.multiview else [])]
    paths = {n: out_dir / clip_name(swing, engine, n) for n in names}
    writers: dict[str, ClipWriter] = {}
    glyph_counts: list[int] = []
    feed = overlay if settings.overlays else None
    viewer = VIEWER_NAMES.get(engine, engine)
    frames = 0
    try:
        for k, tiles in zip(
            indices, backend.render(swing, settings, indices, feed), strict=False
        ):
            glyphs = feed.glyphs_at(k) if feed is not None else None
            if glyphs is not None:
                glyph_counts.append(len(glyphs.arrows) + len(glyphs.torque_arcs))
            hud = HudInfo(
                swing.time_s(k), viewer, swing.club, OverlayFeed.legend(glyphs)
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
                if name not in writers:
                    writers[name] = writer_factory(paths[name], settings.fps)
                writers[name].append_data(frame)
            frames += 1
    finally:
        for writer in writers.values():
            writer.close()
    if frames != len(indices):
        raise RuntimeError(f"backend yielded {frames} frames, expected {len(indices)}")
    return ExportResult(engine, paths, frames, glyph_counts=tuple(glyph_counts))
