"""Annotated reference comparison video and reproducible sidecar export (#9866)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from src.motion_capture.coaching import DrawingLayer, render_layer
from src.motion_capture.coaching.storage import load_layer
from src.motion_capture.provenance import write_json
from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.motion_capture.reconstruct.overlay3d import reference_track
from src.motion_capture.reference.comparison import (
    ComparisonLayer,
    build_comparison_sidecar,
)
from src.motion_capture.reference.model import Asset, ReferenceMotion, ReferenceVideo
from src.motion_capture.reference.registration import ReferenceRegistration
from src.motion_capture.rig.edits import CropRect, ViewEdit, load_edits
from src.shared.python.core.contracts import require

from .clips import _stamp, _writer
from .overlay import PoseTrack, draw_pose
from .player import VideoReader
from .session import load_session
from .swing_export import publish_export


def draw_reference_overlay(
    frame: npt.NDArray[np.uint8],
    asset: Asset,
    t_scene: float,
    registration: ReferenceRegistration,
    camera: PinholeCamera | None,
    layer: ComparisonLayer,
) -> npt.NDArray[np.uint8]:
    """Render reference overlay (3D projection or 2D video frame) onto scene frame."""
    if not layer.visible:
        return frame

    colour_hex = layer.colour.lstrip("#")
    r, g, b = tuple(int(colour_hex[i : i + 2], 16) for i in (0, 2, 4))
    bgr_colour = (b, g, r)

    if isinstance(asset, ReferenceMotion):
        if camera is None:
            return frame
        track = reference_track(
            registration,
            asset,
            camera,
            np.array([t_scene], dtype=float),
            bgr_colour,
            label=asset.title,
        )
        if track.frames > 0 and track.visible[0].any():
            pts = track.px[0]
            vis = track.visible[0]
            for edge in track.edges:
                p1, p2 = edge
                if vis[p1] and vis[p2] and layer.draw_skeleton:
                    pt1 = (int(round(pts[p1, 0])), int(round(pts[p1, 1])))
                    pt2 = (int(round(pts[p2, 0])), int(round(pts[p2, 1])))
                    cv2.line(frame, pt1, pt2, bgr_colour, layer.line_width, cv2.LINE_AA)
            if layer.draw_joints:
                for k in range(len(pts)):
                    if vis[k]:
                        center = (int(round(pts[k, 0])), int(round(pts[k, 1])))
                        radius = max(2, layer.line_width + 1)
                        cv2.circle(frame, center, radius, bgr_colour, -1, cv2.LINE_AA)
    elif isinstance(asset, ReferenceVideo):
        t_ref = registration.time_mapping.scene_to_reference(t_scene)
        ref_frame_idx = int(round(t_ref * asset.fps))
        if 0 <= ref_frame_idx < asset.frames and Path(asset.source.path).is_file():
            with VideoReader(Path(asset.source.path)) as ref_reader:
                ref_img = ref_reader.read(ref_frame_idx)
                if ref_img is not None:
                    h_scene, w_scene = frame.shape[:2]
                    resized_ref = cv2.resize(ref_img, (w_scene, h_scene))
                    alpha = float(layer.opacity)
                    cv2.addWeighted(
                        resized_ref, alpha, frame, 1.0 - alpha, 0.0, dst=frame
                    )
    return frame


def export_comparison_video(
    root: Path,
    view: str,
    asset: Asset,
    registration: ReferenceRegistration,
    layer: ComparisonLayer,
    out: Path,
    *,
    camera: PinholeCamera | None = None,
    speed: float = 1.0,
    cancelled: Callable[[], bool] = lambda: False,
    progress: Callable[[int, int], None] = lambda done, total: None,
) -> dict[str, Any]:
    """Write an annotated comparison video and reproducible sidecar."""
    out = out.resolve()
    sidecar = out.with_suffix(".json")
    if out.suffix.lower() not in (".avi", ".mp4"):
        raise ValueError("Choose an AVI or MP4 output file")
    if out.exists() or sidecar.exists():
        raise FileExistsError("Choose a new filename; video or sidecar already exists")

    media = load_session(root)
    original = media.view(view)
    if original.recording is None:
        raise ValueError("Original recording is unavailable")

    edit = load_edits(root).views.get(view, ViewEdit())
    crop = edit.crop
    track = PoseTrack.load(original.observations) if original.observations else None

    with VideoReader(original.recording) as reader:
        fps = reader.fps or 30.0
        first_frame = edit.first
        last_frame = edit.last if edit.last is not None else reader.frame_count - 1
        total_frames = max(1, last_frame - first_frame + 1)
        w, h = reader.width, reader.height
        if crop:
            w, h = crop.width, crop.height
        size = (w + w % 2, h + h % 2)

        try:
            drawings: DrawingLayer | None = load_layer(
                root, view, reader.width, reader.height, reader.frame_count
            )
        except (ValueError, OSError):
            drawings = None

        with TemporaryDirectory(prefix=".comparison-export-", dir=out.parent) as temp_d:
            staged = Path(temp_d) / out.name
            writer = _writer(staged, fps * speed, size)
            written = 0
            frame_times: list[float] = []

            try:
                for idx in range(first_frame, last_frame + 1):
                    if cancelled():
                        raise InterruptedError("Comparison export cancelled")
                    img = reader.read(idx)
                    if img is None:
                        break
                    t_scene = idx / fps
                    frame_times.append(t_scene)

                    if track:
                        pose = track.at(idx)
                        if pose is not None:
                            img = draw_pose(
                                img,
                                pose[0],
                                pose[1],
                                track.edges,
                                min_confidence=0.5,
                            )
                    if drawings:
                        img = render_layer(img, drawings, idx)

                    img = draw_reference_overlay(
                        img, asset, t_scene, registration, camera, layer
                    )

                    if crop:
                        img = img[
                            crop.y : crop.y + crop.height, crop.x : crop.x + crop.width
                        ]
                        img = np.pad(
                            img,
                            ((0, crop.height % 2), (0, crop.width % 2), (0, 0)),
                            mode="edge",
                        )

                    lbl = f"{view} · ref:{asset.title} f{idx} t={t_scene:.3f}s"
                    img = _stamp(img, lbl)
                    writer.write(img)
                    written += 1
                    progress(written, total_frames)
            finally:
                writer.release()

            require(written > 0, "No frames written for comparison video")
            sidecar_dict = build_comparison_sidecar(
                video_out=out,
                source_media=original.recording,
                reference_asset=asset,
                view=view,
                fps=fps * speed,
                frame_count=written,
                output_frame_times=frame_times,
                registration=registration,
                time_mapping=registration.time_mapping,
                crop=crop,
                layer=layer,
            )
            write_json(staged.with_suffix(".json"), sidecar_dict)
            if cancelled():
                raise InterruptedError("Comparison export cancelled")
            publish_export(staged, out)

    return sidecar_dict


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
        camera: PinholeCamera | None,
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
                camera=self.camera,
                cancelled=self.isInterruptionRequested,
                progress=self.progress.emit,
            )
        except (ValueError, OSError, RuntimeError, InterruptedError, cv2.error) as exc:
            self.error = str(exc)
