"""Orchestration: bundle + receipts to native-viewer clips for several engines."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import logging
from pathlib import Path

from src.shared.python.biomechanics.grip_plot_model import (
    build_grip_plot_series,
    plot_series_to_json,
)
from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export.backends.registry import make_backend
from src.tools.native_viewer_export.ball import AddressBall, resolve_address_ball
from src.tools.native_viewer_export.core import (
    ENGINES,
    BackendUnavailable,
    ExportResult,
    ExportSettings,
    NativeBackend,
    OverlayFeed,
    SwingInput,
    WriterFactory,
    export_swing,
    load_receipt_rollout,
)
from src.tools.native_viewer_export.overlay import (
    build_overlay_feed,
    detect_impact_time_s,
)

logger = logging.getLogger(__name__)
GRIP_PLOT_STRIDE = 4  # one grip sample per 4 bundle steps in the plot payload


def write_grip_json(
    swing: SwingInput,
    engine: str,
    feed: OverlayFeed,
    out_dir: Path,
    *,
    impact_time_s: float | None = None,
) -> Path:
    """Write ``<swing>_<engine>_grip_wrench.json`` (the ``/analysis/grip-wrench`` shape)."""
    if feed.grip_analyses is None:
        raise ValueError("overlay feed has no grip analyses")
    steps = swing.bundle.steps
    indices = list(range(0, steps + 1, GRIP_PLOT_STRIDE))
    analyses = feed.grip_analyses(indices)
    times = [k * swing.bundle.dt_s for k in indices]
    events = {"impact": impact_time_s} if impact_time_s is not None else None
    series = build_grip_plot_series(times, analyses, events=events)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{swing.swing}_{engine}_grip_wrench.json"
    path.write_text(plot_series_to_json(series), encoding="utf-8")
    return path


OverlayFactory = Callable[
    [SwingInput, str], tuple[OverlayFeed, tuple[float, float, float]]
]


@dataclass(frozen=True)
class ExportJob:
    """What to render: one swing, per-engine rollouts and an output directory."""

    bundle_path: Path
    out_dir: Path
    swing: str
    club: str
    engines: Sequence[str] = ENGINES
    receipts: Mapping[str, Path] | None = None


def _swing_for(job: ExportJob, bundle: InputBundle, engine: str) -> SwingInput:
    receipt = (job.receipts or {}).get(engine)
    if receipt is None:
        return SwingInput(
            bundle, bundle.reference_q, job.swing, job.club, bundle.reference_engine
        )
    q, rollout_engine = load_receipt_rollout(receipt, bundle)
    return SwingInput(bundle, q, job.swing, job.club, rollout_engine)


def run_export(
    job: ExportJob,
    settings: ExportSettings,
    backend_factory: Callable[[str], NativeBackend] = make_backend,
    overlay_factory: OverlayFactory = build_overlay_feed,
    writer_factory: WriterFactory | None = None,
    impact_detector: Callable[[SwingInput], float] = detect_impact_time_s,
    ball_resolver: Callable[[SwingInput], AddressBall] = resolve_address_ball,
) -> list[ExportResult]:
    """Render every requested engine; unavailable backends are skipped, not fatal.

    The decorative ball (GCV-13, #11719) is resolved once per engine from
    ``swing`` here, *before* any clip plan windows or resamples it for a
    particular speed or the impact window, then passed unchanged to every
    clip of that engine (``export_swing``). Resolving it per clip instead
    would read a windowed clip's own first frame as the "address" -- for the
    impact window that is mid-swing, not the address, and the ball would be
    wrongly reported unavailable.
    """
    for engine in job.engines:
        if engine not in ENGINES:
            raise ValueError(
                f"unknown engine {engine!r}; expected one of {list(ENGINES)}"
            )
    bundle = InputBundle.load(job.bundle_path)
    results: list[ExportResult] = []
    for engine in job.engines:
        backend = backend_factory(engine)
        reason = backend.unavailable_reason()
        if reason is not None:
            logger.warning("skipping %s: %s", engine, reason)
            results.append(ExportResult(engine, {}, 0, skipped_reason=reason))
            continue
        swing = _swing_for(job, bundle, engine)
        feed, per_settings = None, settings
        if settings.impact_time_s is None and not swing.has_impact_provenance:
            try:
                per_settings = replace(settings, impact_time_s=impact_detector(swing))
            except (BackendUnavailable, ValueError, KeyError, RuntimeError) as exc:
                logger.warning("impact time not detected for %s: %s", engine, exc)
        if settings.overlays:
            try:
                feed, lookat = overlay_factory(swing, engine)
                per_settings = replace(per_settings, lookat_m=lookat)
            except BackendUnavailable as exc:
                logger.warning("rendering %s without overlays: %s", engine, exc)
        ball = ball_resolver(swing) if settings.ball else None
        result = export_swing(
            backend, swing, per_settings, job.out_dir, feed, writer_factory, ball
        )
        if feed is not None and feed.grip_analyses is not None:
            result = replace(
                result,
                grip_json=write_grip_json(
                    swing,
                    engine,
                    feed,
                    job.out_dir,
                    impact_time_s=per_settings.impact_time_s,
                ),
            )
        results.append(result)
    return results
