"""Full-mesh MuJoCo diagnostic render contract (Epic #15808, WP-0b).

Null-space and replay diagnostic previews are routed exclusively through
``native_viewer_export``: the full 3D MuJoCo golfer, one shared body-weight
force-arrow scale, an admitted-interval banner on every frame, and a
``allocation-only`` vs ``forward-replayed`` badge. Stick-skeleton diagnostics
are a contract violation, never a fallback.

Design by contract: :class:`DiagnosticRenderRequest` validates on construction
(:class:`ContractViolation`); :func:`export_diagnostic` refuses a backend that
is not MuJoCo, a missing overlay feed, and settings that would hide evidence.
Frames outside the admitted interval are visibly marked, never silently kept.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.video_timing.frame_schedule import FrameSchedule
from src.tools.native_viewer_export.compositor import draw_label
from src.tools.native_viewer_export.core import (
    GRID_VIEW,
    ClipWriter,
    ExportResult,
    ExportSettings,
    NativeBackend,
    OverlayFeed,
    SwingInput,
    WriterFactory,
    clip_name,
    clip_plans,
    default_glyph_style,
    export_swing,
    imageio_writer,
)

RENDERER = "native_viewer_export"
ENGINE = "mujoco"
GEOMETRY = "full_mesh"
#: settings a diagnostic may not switch off (they carry the evidence)
REQUIRED_ON = ("overlays", "force_vectors", "torque_vectors", "hud")
_BANNER_OK = (255, 255, 255)
_BANNER_OUTSIDE = (255, 90, 60)
_BANNER_MARGIN_PX = 34


class ContractViolation(ValueError):
    """The diagnostic render contract was broken."""


class ReplayBadge(str, Enum):
    """What the evidence supports: allocation only, or forward-replayed."""

    ALLOCATION_ONLY = "allocation-only"
    FORWARD_REPLAYED = "forward-replayed"


@dataclass(frozen=True)
class AdmittedInterval:
    """Closed time interval ``[start_s, end_s]`` where the baseline is admitted."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start_s) and math.isfinite(self.end_s)):
            raise ContractViolation("interval bounds must be finite")
        if not self.end_s > self.start_s:
            raise ContractViolation("interval must satisfy start_s < end_s")

    def contains(self, time_s: float) -> bool:
        return self.start_s <= time_s <= self.end_s


@dataclass(frozen=True)
class DiagnosticRenderRequest:
    """A validated request to render a diagnostic preview."""

    badge: ReplayBadge
    interval: AdmittedInterval | None
    body_mass_kg: float
    renderer: str = RENDERER
    engine: str = ENGINE
    geometry: str = GEOMETRY

    def __post_init__(self) -> None:
        if self.renderer != RENDERER:
            raise ContractViolation(
                f"renderer must be {RENDERER!r}, got {self.renderer!r}"
            )
        if self.engine != ENGINE:
            raise ContractViolation(f"engine must be {ENGINE!r}, got {self.engine!r}")
        if self.geometry != GEOMETRY:
            raise ContractViolation(
                f"geometry must be {GEOMETRY!r} (stick skeletons are forbidden), "
                f"got {self.geometry!r}"
            )
        if not isinstance(self.badge, ReplayBadge):
            raise ContractViolation("badge must be a ReplayBadge")
        if not isinstance(self.interval, AdmittedInterval):
            raise ContractViolation("an AdmittedInterval is required (interval)")
        mass = self.body_mass_kg
        if not (isinstance(mass, (int, float)) and math.isfinite(mass) and mass > 0.0):
            raise ContractViolation("body_mass_kg must be finite and positive")


def frame_validity(
    times_s: Sequence[float] | NDArray[np.float64], interval: AdmittedInterval
) -> tuple[bool, ...]:
    """Per-frame flag: is the frame time inside the admitted interval."""
    return tuple(interval.contains(float(t)) for t in times_s)


def banner_text(request: DiagnosticRenderRequest, time_s: float) -> str:
    """Banner for the frame at ``time_s``: badge, interval, outside warning."""
    interval = request.interval
    assert interval is not None  # guaranteed by the request contract
    text = (
        f"{request.badge.value.upper()} | admitted "
        f"{interval.start_s:.3f}-{interval.end_s:.3f} s"
    )
    if not interval.contains(time_s):
        text = f"OUTSIDE ADMITTED INTERVAL - no claims | {text}"
    return text


def diagnostic_settings(
    request: DiagnosticRenderRequest, **overrides: Any
) -> ExportSettings:
    """Export settings for ``request``; evidence-carrying flags stay on."""
    for name in REQUIRED_ON:
        if name in overrides and overrides[name] is not True:
            raise ContractViolation(f"diagnostics require {name}=True")
    return ExportSettings(**{**dict.fromkeys(REQUIRED_ON, True), **overrides})


def _banner_writer(
    inner: ClipWriter, request: DiagnosticRenderRequest, times_s: Sequence[float]
) -> ClipWriter:
    """Writer that burns the interval banner onto each frame before writing."""

    class _Banner:
        def __init__(self) -> None:
            self.n = 0

        def append_data(self, frame: Any) -> None:
            t = float(times_s[self.n])
            self.n += 1
            outside = not request.interval.contains(t)  # type: ignore[union-attr]
            y = max(0, frame.shape[0] - _BANNER_MARGIN_PX)
            inner.append_data(
                draw_label(
                    frame,
                    banner_text(request, t),
                    (8, y),
                    fill=_BANNER_OUTSIDE if outside else _BANNER_OK,
                )
            )

        def close(self) -> None:
            inner.close()

    return _Banner()


@dataclass(frozen=True)
class DiagnosticExport:
    """Result of :func:`export_diagnostic`."""

    result: ExportResult
    manifest_path: Path
    #: per clip key (``face_on_1x``): validity of every frame
    frame_valid: dict[str, tuple[bool, ...]] = field(default_factory=dict)


def export_diagnostic(
    backend: NativeBackend,
    swing: SwingInput,
    request: DiagnosticRenderRequest,
    out_dir: Path,
    overlay: OverlayFeed | None,
    writer_factory: WriterFactory | None = None,
    **settings_overrides: Any,
) -> DiagnosticExport:
    """Render ``swing`` as a contract-conforming full-mesh diagnostic.

    Preconditions: ``backend`` is the MuJoCo native viewer and ``overlay`` is
    given. Postconditions: every clip frame carries the banner, the overlay
    uses the shared body-weight scale, and a manifest records the badge,
    interval and per-frame validity next to the clips.
    """
    if backend.engine != request.engine:
        raise ContractViolation(
            f"diagnostics render in the {request.engine!r} backend, "
            f"got {backend.engine!r}"
        )
    if overlay is None:
        raise ContractViolation("an overlay feed is required for diagnostics")
    settings = diagnostic_settings(request, **settings_overrides)
    style = default_glyph_style(request.body_mass_kg)
    overlay = replace(overlay, style=style)

    schedules: dict[str, list[float]] = {}
    for plan in clip_plans(swing, settings):
        times = FrameSchedule(
            swing.source_times_s, settings.fps, plan.speed, plan.window
        ).sample_times_s
        for view in (*settings.views, *([GRID_VIEW] if settings.multiview else [])):
            schedules[clip_name(swing, backend.engine, view, plan.suffix)] = [
                float(t) for t in times
            ]
    base = writer_factory or (lambda p, fps: imageio_writer(p, fps, settings.crf))

    def factory(path: Path, fps: int) -> ClipWriter:
        return _banner_writer(base(path, fps), request, schedules[Path(path).name])

    result = export_swing(
        backend, swing, settings, out_dir, overlay=overlay, writer_factory=factory
    )
    assert request.interval is not None
    valid = {
        key: frame_validity(schedules[path.name], request.interval)
        for key, path in result.paths.items()
    }
    manifest = Path(out_dir) / f"{swing.swing}_{backend.engine}_diagnostic.json"
    manifest.write_text(
        json.dumps(
            {
                "renderer": request.renderer,
                "engine": request.engine,
                "geometry": request.geometry,
                "badge": request.badge.value,
                "admitted_interval_s": [
                    request.interval.start_s,
                    request.interval.end_s,
                ],
                "force_scale_mode": style.scale_mode,
                "force_reference_n": style.reference_force_n,
                "frame_valid": {k: list(v) for k, v in valid.items()},
            },
            indent=2,
        )
    )
    return DiagnosticExport(result, manifest, valid)
