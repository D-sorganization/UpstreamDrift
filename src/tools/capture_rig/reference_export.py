"""Strict comparison export through the shared coaching clip compositor."""

from __future__ import annotations

from .clips import verify_frame_clip as _verify_encoded

from src.motion_capture.coaching.geometry_storage import geometry_path, load_geometry

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from src.motion_capture.coaching.storage import load_layer, layer_path
from src.motion_capture.coaching import DrawingLayer
from src.motion_capture.provenance import write_json
from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.shared.python.pose_estimation.observations import CameraCalibration
from src.motion_capture.reference.evidence import CameraSnapshot
from src.motion_capture.reference.scene import session_clock, session_camera
from src.motion_capture.reference.comparison import (
    ComparisonExportSidecarSpec,
    ComparisonLayer,
    build_comparison_sidecar,
)
from src.motion_capture.reference.model import Asset, ReferenceMotion, ReferenceVideo
from src.motion_capture.reference.registration import ReferenceRegistration
from src.motion_capture.rig.edits import ViewEdit, load_edits, EDITS_FILE

from .clips import ClipRange, export_clip
from .player import VideoReader
from .session import load_session
from .swing_export import publish_export, _digest
from .reference_rendering import ComparisonRenderContext, ComparisonRenderer


def draw_reference_overlay(
    frame: npt.NDArray[np.uint8],
    asset: Asset,
    t_scene: float,
    registration: ReferenceRegistration,
    camera: PinholeCamera | CameraCalibration | None,
    layer: ComparisonLayer,
) -> npt.NDArray[np.uint8]:
    """One-shot compatibility API; playback/export retain a ComparisonRenderer."""
    context = ComparisonRenderContext("preview", asset, registration, layer)
    with ComparisonRenderer(asset) as renderer:
        return renderer.overlay(frame, t_scene, context, camera)


@dataclass(frozen=True)
class ComparisonVideoExportOptions:
    """Camera evidence and slow-motion/cancellable output settings."""

    camera: PinholeCamera | CameraCalibration | None = None
    speed: float = 1.0
    cancelled: Callable[[], bool] = lambda: False
    progress: Callable[[int, int], None] = lambda done, total: None
    drawings: DrawingLayer | None = None


def _with_drawing_snapshot(
    context: ComparisonRenderContext, drawings: DrawingLayer | None
) -> ComparisonRenderContext:
    if drawings is None:
        return context
    original = context.drawings
    if original is None or original.with_shapes(()) != drawings.with_shapes(()):
        raise ValueError("Drawing snapshot belongs to another source")
    return replace(context, drawings=drawings)


def _camera_snapshot(
    root: Path,
    view: str,
    registration: ReferenceRegistration,
    supplied: PinholeCamera | CameraCalibration | None,
) -> CameraSnapshot | None:
    current = registration.camera
    if current and current.provenance.startswith("Session reconstruction"):
        # Re-read disk evidence: a dialog may have been open during recalibration.
        return session_camera(root, "", view)
    if supplied is None:
        return current
    record = (
        supplied.to_calibration() if isinstance(supplied, PinholeCamera) else supplied
    )
    return CameraSnapshot.from_calibration(
        record, provenance="Comparison export camera"
    )


def _input_hashes(
    paths: set[Path], cancelled: Callable[[], bool]
) -> dict[Path, str | None]:
    return {path: _digest(path, cancelled) if path.exists() else None for path in paths}


def _render_recipe(
    root: Path,
    view: str,
    asset: Asset,
    registration: ReferenceRegistration,
    layer: ComparisonLayer,
    reader: VideoReader,
) -> tuple[ComparisonRenderContext, ClipRange]:
    edit = load_edits(root).views.get(view, ViewEdit())
    clip = ClipRange(
        edit.first, edit.last if edit.last is not None else reader.frame_count - 1
    )
    if clip.last >= reader.frame_count:
        raise ValueError("Selection exceeds the decodable recording")
    if not np.isfinite(reader.fps) or reader.fps <= 0:
        raise ValueError("Source frame rate must be finite and positive")
    if edit.crop:
        edit.crop.validate_size(reader.width, reader.height)
    drawings = load_layer(root, view, reader.width, reader.height, reader.frame_count)
    geometry = load_geometry(root)
    return ComparisonRenderContext(
        view,
        asset,
        registration,
        layer,
        crop=edit.crop,
        drawings=drawings,
        geometry=geometry if geometry.planes or geometry.points else None,
        scene_id=geometry.scene_id,
    ), clip


def _export_metadata(
    out: Path,
    source: Path,
    ctx: ComparisonRenderContext,
    clip: ClipRange,
    result: dict[str, Any],
    fps: float,
) -> dict[str, Any]:
    registration = ctx.registration
    times = [registration.scene_time(i / fps) for i in range(clip.first, clip.last + 1)]
    metadata = build_comparison_sidecar(
        ComparisonExportSidecarSpec(
            video_out=out,
            source_media=source,
            reference_asset=ctx.asset,
            view=ctx.view,
            fps=result["fps"],
            frame_count=result["frames"],
            output_frame_times=times,
            registration=registration,
            time_mapping=registration.time_mapping,
            crop=ctx.crop,
            layer=ctx.layer,
        )
    )
    metadata.update(
        reference_asset=ctx.asset.model_dump(mode="json"),
        drawings=ctx.drawings.model_dump(mode="json") if ctx.drawings else None,
        geometry=ctx.geometry.model_dump(mode="json") if ctx.geometry else None,
        selection={"first": clip.first, "last": clip.last},
        render_recipe={
            "version": "comparison-compositor/1.0.0",
            "order": [
                "player",
                "detected_pose",
                "drawings",
                "scene_geometry",
                "reference",
                "crop",
                "edge_padding",
                "clock",
            ],
            "min_pose_confidence": 0.5,
            "speed": result["speed"],
        },
    )
    return metadata


def export_comparison_video(
    root: Path,
    view: str,
    asset: Asset,
    registration: ReferenceRegistration,
    layer: ComparisonLayer,
    out: Path,
    options: ComparisonVideoExportOptions | None = None,
) -> dict[str, Any]:
    """Stage all frames and metadata; publish neither file on input/decode/cancel errors."""
    opts = options or ComparisonVideoExportOptions()
    if opts.cancelled():
        raise InterruptedError("Comparison export cancelled")
    if not np.isfinite(opts.speed) or not 0 < opts.speed <= 1:
        raise ValueError("Comparison speed must be in (0, 1]")
    out = out.resolve()
    if out.suffix.lower() not in (".avi", ".mp4"):
        raise ValueError("Choose an AVI or MP4 output file")
    if out.exists() or out.with_suffix(".json").exists():
        raise FileExistsError("Choose a new filename; video or sidecar already exists")
    media = load_session(root)
    original = media.view(view)
    source = original.recording
    if source is None:
        raise ValueError("Original recording is unavailable")
    snapshot = _camera_snapshot(root, view, registration, opts.camera)
    registration = registration.bound(
        asset, snapshot, session_clock(media.timing, view)
    )
    camera = snapshot.record() if snapshot else None
    if (
        isinstance(asset, ReferenceMotion)
        and layer.visible
        and layer.opacity > 0
        and camera is None
    ):
        raise ValueError(
            "A usable camera is required to export the visible motion reference"
        )
    paths = {
        source,
        root / EDITS_FILE,
        layer_path(root, view),
        geometry_path(root),
        root / "recordings.json",
        root / "observations" / "observations.json",
    }
    if original.observations:
        paths.add(original.observations)
    if isinstance(asset, ReferenceVideo) or asset.source_path.exists():
        paths.add(asset.source_path)
    before = _input_hashes(paths, opts.cancelled)
    if asset.source_path in before and before[asset.source_path] != asset.source.sha256:
        raise ValueError("Reference source changed or is unavailable; import it again")
    with VideoReader(source) as reader:
        ctx, clip = _render_recipe(root, view, asset, registration, layer, reader)
        ctx = _with_drawing_snapshot(ctx, opts.drawings)
        fps, width, height = reader.fps, reader.width, reader.height
        if fps * opts.speed < 1:
            raise ValueError("Comparison speed needs an output rate of at least 1 fps")
        if snapshot and snapshot.image_size_px != (width, height):
            raise ValueError("Camera dimensions differ from the original recording")
    with TemporaryDirectory(prefix=".comparison-export-", dir=out.parent) as temp:
        staged = Path(temp) / out.name
        with ComparisonRenderer(asset) as renderer:
            rendering = renderer.rendering(
                ctx, camera, fps, cancelled=opts.cancelled, progress=opts.progress
            )
            result = export_clip(
                replace(original, proxy=None),
                clip,
                staged,
                speed=opts.speed,
                rendering=rendering,
            )
        metadata = _export_metadata(out, source, ctx, clip, result, fps)
        size = rendering.size(width, height)
        _verify_encoded(staged, clip.frames, size, opts.cancelled)
        metadata["output_size_px"] = size
        metadata["source_size_px"] = (width, height)
        metadata["padding"] = {
            "right": (ctx.crop.width if ctx.crop else width) % 2,
            "bottom": (ctx.crop.height if ctx.crop else height) % 2,
        }
        metadata["input_sha256"] = {str(path): value for path, value in before.items()}
        write_json(staged.with_suffix(".json"), metadata)
        current = load_session(root)
        registration.validate_binding(
            asset,
            _camera_snapshot(root, view, registration, opts.camera),
            session_clock(current.timing, view),
        )
        if before != _input_hashes(paths, opts.cancelled):
            raise ValueError("Comparison source or recipe changed during export; retry")
        if opts.cancelled():
            raise InterruptedError("Comparison export cancelled")
        publish_export(staged, out)
    return metadata


class ComparisonExportWorker(QThread):
    progress = pyqtSignal(int, int)

    def __init__(
        self,
        root: Path,
        view: str,
        asset: Asset,
        registration: ReferenceRegistration,
        layer: ComparisonLayer,
        out: Path,
        camera: PinholeCamera | CameraCalibration | None,
        parent: QObject,
    ) -> None:
        super().__init__(parent)
        self.root, self.view = root, view
        self.asset, self.registration, self.layer = asset, registration, layer
        self.out, self.camera = out, camera
        self.error: str = ""

    def run(self) -> None:
        try:
            export_comparison_video(
                self.root,
                self.view,
                self.asset,
                self.registration,
                self.layer,
                self.out,
                options=ComparisonVideoExportOptions(
                    camera=self.camera,
                    cancelled=self.isInterruptionRequested,
                    progress=self.progress.emit,
                ),
            )
        except (ValueError, OSError, RuntimeError, InterruptedError, cv2.error) as exc:
            self.error = str(exc)
