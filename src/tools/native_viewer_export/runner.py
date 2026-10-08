"""Orchestration: bundle + receipts to native-viewer clips for several engines."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import logging
from pathlib import Path

from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export.backends.registry import make_backend
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
    imageio_writer,
    load_receipt_rollout,
)
from src.tools.native_viewer_export.overlay import build_overlay_feed

logger = logging.getLogger(__name__)
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
    writer_factory: WriterFactory = imageio_writer,
) -> list[ExportResult]:
    """Render every requested engine; unavailable backends are skipped, not fatal."""
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
        if settings.overlays:
            try:
                feed, lookat = overlay_factory(swing, engine)
                per_settings = replace(settings, lookat_m=lookat)
            except BackendUnavailable as exc:
                logger.warning("rendering %s without overlays: %s", engine, exc)
        results.append(
            export_swing(
                backend, swing, per_settings, job.out_dir, feed, writer_factory
            )
        )
    return results
