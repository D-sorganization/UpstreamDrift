"""Comparison-backed persistence and export for the common drawing editor."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.motion_capture.coaching import DrawingLayer, render_layer
from src.motion_capture.coaching.geometry_storage import load_geometry
from src.motion_capture.provenance import sha256_of
from src.motion_capture.reference.evidence import CameraSnapshot
from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.motion_capture.rig.edits import ViewEdit, load_edits

from .coaching_export import publish_still
from .coaching_source import CaptureCoachingSource
from .comparison_frame_source import ComparisonFrameSource
from .reference_export import ComparisonVideoExportOptions, export_comparison_video
from .reference_rendering import Camera, ComparisonRenderContext
from .session import load_session
from .swing_export_actions import ExportJob


class ComparisonCoachingSource:
    """Edit original-pixel drawings over an immutable comparison display recipe."""

    def __init__(
        self, root: Path, context: ComparisonRenderContext, camera: Camera
    ) -> None:
        self.root, self.context, self.camera = root, context, camera
        capture = CaptureCoachingSource(root, context.view)
        self._capture = capture
        self.drawings = capture.drawings
        try:
            geometry = load_geometry(root)
            if (geometry.planes or geometry.points) and context.geometry != geometry:
                raise ValueError("Comparison geometry differs from the saved scene")
            edit = load_edits(root).views.get(context.view, ViewEdit())
            if context.crop != edit.crop:
                raise ValueError("Comparison crop differs from the saved scene")
            self._saved_evidence = self._evidence()
            self.reader = ComparisonFrameSource(capture.reader, context, camera)
        except (ValueError, OSError):
            capture.reader.close()
            raise

    @property
    def dirty(self) -> bool:
        return False

    def time_at(self, index: int) -> float:
        """Use the comparison's registered scene clock in the shared editor."""
        return self.reader.time_at(index)

    def _validate(self, drawings: DrawingLayer) -> None:
        if drawings.with_shapes(()) != self.drawings.with_shapes(()):
            raise ValueError("Drawing snapshot belongs to another source")

    def _evidence(self) -> dict[str, Any]:
        context = self.context
        media = load_session(self.root).view(context.view)
        paths = (media.recording, media.observations, context.asset.source_path)
        hashes = {
            str(path): sha256_of(path) if path.is_file() else None
            for path in paths
            if path is not None
        }
        geometry = load_geometry(self.root)
        edit = load_edits(self.root).views.get(context.view, ViewEdit())
        return {
            "input_sha256": hashes,
            "geometry": geometry.model_dump(mode="json"),
            "edit": edit.model_dump(mode="json"),
        }

    def _verify_evidence(self) -> None:
        if self._evidence() != self._saved_evidence:
            raise ValueError(
                "Comparison source or scene changed; reopen the drawing editor"
            )

    def save(self, drawings: DrawingLayer) -> None:
        """Persist through the capture store after checking source and scene identity."""
        self._validate(drawings)
        self._verify_evidence()
        self._capture.save(drawings)
        self.drawings = drawings

    def still(self, drawings: DrawingLayer, index: int, out: Path) -> None:
        """Publish the original uncropped editor pixels and exact comparison recipe."""
        self._validate(drawings)
        self._verify_evidence()
        image = self.reader.read(index)
        if image is None:
            raise ValueError("Could not decode comparison frame")
        image = self.reader.finalize(render_layer(image, drawings, index), index)
        context, camera = self.context, self.camera
        snapshot = None
        if camera is not None:
            record = (
                camera.to_calibration() if isinstance(camera, PinholeCamera) else camera
            )
            snapshot = CameraSnapshot.from_calibration(
                record, provenance="Comparison drawing view"
            )
        metadata = {
            "schema_version": "comparison-drawing-still/1.0.0",
            "pixel_grid": "original_uncropped",
            "view": context.view,
            "frame": index,
            "scene_seconds": self.time_at(index),
            "reference_asset": context.asset.model_dump(mode="json"),
            "registration": context.registration.model_dump(mode="json"),
            "appearance": context.layer.model_dump(mode="json"),
            "camera": snapshot.model_dump(mode="json") if snapshot else None,
            "drawings": drawings.model_dump(mode="json"),
            "evidence": self._saved_evidence,
        }
        self._verify_evidence()
        publish_still(image, out, metadata)

    def export_job(self, drawings: DrawingLayer) -> ExportJob:
        """Freeze edits and retain the established strict comparison video publisher."""
        self._validate(drawings)
        context, camera, root = self.context, self.camera, self.root

        def run(
            out: Path,
            cancelled: Callable[[], bool],
            progress: Callable[[int, int], None],
        ) -> None:
            if cancelled():
                raise InterruptedError("Comparison export cancelled")
            self._verify_evidence()
            export_comparison_video(
                root,
                context.view,
                context.asset,
                context.registration,
                context.layer,
                out,
                options=ComparisonVideoExportOptions(
                    camera=camera,
                    drawings=drawings,
                    cancelled=cancelled,
                    progress=progress,
                ),
            )

        return run
