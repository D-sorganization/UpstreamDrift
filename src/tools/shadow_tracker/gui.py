"""Shadow Tracker review and editing GUI workbench (ST-11, #10134).

Provides interactive frame scrubbing, mask inspection and correction, worst-frame navigation,
session bundle save/reopen, and canonical export. Gracefully degrades in headless environments.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
from pathlib import Path
from typing import Any, Literal

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.shadow_tracker.artifacts import ShadowTrackerBundle
from src.shared.python.shadow_tracker.contracts import FitRequest, FrameObservation
from src.shared.python.shadow_tracker.mask_records import MaskFrame
from src.shared.python.shadow_tracker.service import (
    DefaultShadowTrackerService,
    UnavailableBackendError,
    WorstFrameMetric,
    WorstFrameReport,
)

logger = get_logger(__name__)

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QWidget


def format_clock_evidence_text(obs: FrameObservation) -> str:
    """Return the viewport timing line, rendering unknown physical time safely.

    Imported observations may legitimately carry ``physical_time_s=None`` (no
    evidenced clock mapping); the viewport shows the unknown state and the
    recorded reason instead of formatting None.
    """
    if obs.physical_time_s is None:
        physical_time_text = "unknown"
    else:
        physical_time_text = f"{obs.physical_time_s:.4f}s"
    return (
        f"PTS: {obs.pts_ticks} | Physical Time: {physical_time_text} "
        f"({obs.physical_time_reason}) | "
        f"Clock Authority: {obs.timing_mode} ({obs.clock_evidence}) | "
        f"Exact: {obs.is_timing_exact}"
    )


class ShadowTrackerReviewModel:
    """Headless state and controller for the Shadow Tracker review session."""

    def __init__(self, service: DefaultShadowTrackerService | None = None) -> None:
        self.service: DefaultShadowTrackerService = (
            service if service is not None else DefaultShadowTrackerService()
        )
        self.current_frame_index: int = 0
        self.bundle_path: Path | None = None
        self._dirty: bool = False
        self.last_error_message: str | None = None

    @property
    def is_dirty(self) -> bool:
        """Return True if session has unsaved mask changes."""
        return self._dirty

    def mark_dirty(self) -> None:
        """Mark session as having unsaved changes."""
        self._dirty = True

    def mark_clean(self) -> None:
        """Mark session as clean."""
        self._dirty = False

    @property
    def is_cancelled(self) -> bool:
        """Return True if underlying service operation was cancelled."""
        return self.service.is_cancelled

    @property
    def frame_count(self) -> int:
        """Number of frames in the current review session."""
        return len(self.service.get_observations())

    def get_current_observation(self) -> FrameObservation | None:
        """Return observation for current frame index, if loaded."""
        obs = self.service.get_observations()
        if 0 <= self.current_frame_index < len(obs):
            return obs[self.current_frame_index]
        return None

    def get_current_mask(self) -> MaskFrame | None:
        """Return mask for current frame index, if available."""
        obs = self.get_current_observation()
        if obs is None:
            return None
        return self.service.get_mask(obs.frame_id, shot_id=obs.shot_id)

    def import_video(
        self,
        video_path: str | Path,
        *,
        asset_id: str | None = None,
        shot_id: str = "shot-001",
        swing_id: str = "swing-001",
        camera_id: str = "camera-001",
        timing_mapping: Any | None = None,
        cuts: Sequence[tuple[int, int]] = (),
        transforms: Sequence[str] = (),
        max_frames: int | None = None,
    ) -> tuple[FrameObservation, ...]:
        """Import a video media file into the review session."""
        imported = self.service.import_video(
            video_path,
            asset_id=asset_id,
            shot_id=shot_id,
            swing_id=swing_id,
            camera_id=camera_id,
            timing_mapping=timing_mapping,
            cuts=cuts,
            transforms=transforms,
            max_frames=max_frames,
        )
        self.current_frame_index = 0
        self.bundle_path = None
        self.mark_clean()
        logger.info("Imported video with %d frames from %s", len(imported), video_path)
        return imported

    def load_bundle(self, bundle_dir: str | Path) -> None:
        """Load an existing review bundle directory."""
        path = Path(bundle_dir)
        self.service.load_bundle(path)
        self.bundle_path = path
        self.current_frame_index = 0
        self.mark_clean()
        logger.info(
            "Loaded bundle with %d observations from %s", self.frame_count, path
        )

    def save_bundle(self, bundle_dir: str | Path | None = None) -> Path:
        """Save the review bundle to disk."""
        target_dir = Path(bundle_dir) if bundle_dir is not None else self.bundle_path
        if target_dir is None:
            raise ValueError(
                "Target bundle directory must be specified for initial save"
            )
        self.service.save_bundle(target_dir)
        self.bundle_path = target_dir
        self.mark_clean()
        logger.info("Saved review bundle to %s", target_dir)
        return target_dir

    def export_canonical(self, export_path: str | Path) -> dict[str, Any]:
        """Export canonical dataset package."""
        path = Path(export_path)
        return self.service.export_canonical(path)

    def step_frame(self, delta: int) -> int:
        """Step current frame index by delta, clamped to valid range."""
        total = self.frame_count
        if total == 0:
            self.current_frame_index = 0
            return 0
        self.current_frame_index = max(
            0, min(total - 1, self.current_frame_index + delta)
        )
        return self.current_frame_index

    def jump_to_frame(self, index: int) -> int:
        """Jump directly to a frame index."""
        total = self.frame_count
        if total == 0:
            self.current_frame_index = 0
            return 0
        self.current_frame_index = max(0, min(total - 1, index))
        return self.current_frame_index

    def jump_to_worst_frame(
        self, metric: WorstFrameMetric = "mask_coverage"
    ) -> WorstFrameReport | None:
        """Jump to the worst frame ranked by quality/coverage metric."""
        ranked = self.service.worst_frames(metric=metric)
        if not ranked:
            return None
        worst = ranked[0]
        obs = self.service.get_observations()
        for idx, item in enumerate(obs):
            if item.frame_id == worst.frame_id and (
                not worst.shot_id or item.shot_id == worst.shot_id
            ):
                self.current_frame_index = idx
                break
        return worst

    def update_mask(
        self,
        *,
        body: bytes,
        club: bytes,
        valid: bytes,
        parent_revision_id: str | None,
        producer_id: str = "reviewer-gui",
        correction_note: str = "",
    ) -> MaskFrame:
        """Update mask for the current frame, invalidating downstream fits."""
        obs = self.get_current_observation()
        if obs is None:
            raise RuntimeError("Cannot update mask: no active frame observation")
        new_mask = self.service.update_mask(
            frame_id=obs.frame_id,
            shot_id=obs.shot_id,
            body=body,
            club=club,
            valid=valid,
            parent_revision_id=parent_revision_id,
            producer_id=producer_id,
            correction_note=correction_note,
        )
        self.mark_dirty()
        return new_mask

    def cancel(self) -> None:
        """Cancel ongoing service operations."""
        self.service.cancel()

    def request_fit(self, request: FitRequest) -> None:
        """Trigger fitting backend with actionable error handling."""
        try:
            self.service.fit(request)
            self.last_error_message = None
        except UnavailableBackendError as exc:
            self.last_error_message = str(exc)
            logger.warning("Automated fitting unavailable: %s", exc)
            raise


class ShadowTrackerViewportWidget(QWidget):
    """Custom viewport rendering for Shadow Tracker review frames (MS-84)."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._observation: FrameObservation | None = None
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background-color: #1a1a1a; color: #e0e0e0;")

    @property
    def observation(self) -> FrameObservation | None:
        """Return the currently displayed frame observation."""
        return self._observation

    def set_observation(self, obs: FrameObservation | None) -> None:
        """Set active observation and trigger redraw."""
        self._observation = obs
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:  # noqa: N802
        """Render review frame, authority metadata, and silhouette outlines."""
        painter = QtGui.QPainter(self)
        try:
            rect = self.rect()
            painter.fillRect(rect, QtGui.QColor("#1a1a1a"))

            if self._observation is None:
                painter.setPen(QtGui.QColor("#888888"))
                painter.drawText(
                    rect,
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    "No Bundle Loaded\nUse Open Bundle... to load footage, masks, and forward kinematic tracking.",
                )
                return

            obs = self._observation
            painter.setPen(QtGui.QColor("#4CAF50"))
            painter.drawRect(rect.adjusted(10, 10, -10, -10))

            painter.setPen(QtGui.QColor("#e0e0e0"))
            font = painter.font()
            font.setPointSize(11)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                20,
                35,
                f"Frame: {obs.frame_id} (Shot: {obs.shot_id}, Cam: {obs.camera_id})",
            )

            font.setBold(False)
            font.setPointSize(9)
            painter.setFont(font)
            painter.drawText(20, 60, format_clock_evidence_text(obs))

            mask_info = f"Masks: Body: {obs.body_mask_ref} | Club: {obs.club_mask_ref} | Valid: {obs.valid_mask_ref}"
            painter.drawText(20, 80, mask_info)

            center_rect = QtCore.QRect(
                40, 100, max(50, rect.width() - 80), max(50, rect.height() - 140)
            )
            painter.setPen(
                QtGui.QPen(QtGui.QColor("#555555"), 1, QtCore.Qt.PenStyle.DashLine)
            )
            painter.drawRect(center_rect)

            painter.setPen(QtGui.QColor("#aaaaaa"))
            painter.drawText(
                center_rect,
                QtCore.Qt.AlignmentFlag.AlignCenter,
                f"Viewport Active Review\nBody Silhouette & Kinematics Overlay\n"
                f"Decoder: {obs.decoder_name} | Provenance: {obs.confidence_provenance}",
            )
        finally:
            painter.end()


class ShadowTrackerWidget(QWidget):
    """Review and editing workbench widget for Shadow Tracker."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.model: ShadowTrackerReviewModel = ShadowTrackerReviewModel()
        self._init_ui()

    def _init_ui(self) -> None:
        """Construct UI controls."""
        layout = QtWidgets.QVBoxLayout(self)

        # Header toolbar
        toolbar = QtWidgets.QHBoxLayout()
        self.btn_import = QtWidgets.QPushButton("Import Video...", self)
        self.btn_open = QtWidgets.QPushButton("Open Bundle...", self)
        self.btn_save = QtWidgets.QPushButton("Save Bundle", self)
        self.btn_export = QtWidgets.QPushButton("Export Canonical...", self)
        self.btn_fit = QtWidgets.QPushButton("Auto-Fit", self)
        self.btn_worst = QtWidgets.QPushButton("Jump to Worst Frame", self)
        toolbar.addWidget(self.btn_import)
        toolbar.addWidget(self.btn_open)
        toolbar.addWidget(self.btn_save)
        toolbar.addWidget(self.btn_export)
        toolbar.addWidget(self.btn_fit)
        toolbar.addStretch()
        toolbar.addWidget(self.btn_worst)
        layout.addLayout(toolbar)

        # Frame navigation row
        nav_layout = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("◀ Prev", self)
        self.btn_next = QtWidgets.QPushButton("Next ▶", self)
        self.slider_frame = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.lbl_frame_info = QtWidgets.QLabel("Frame: 0 / 0", self)

        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.slider_frame)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addWidget(self.lbl_frame_info)
        layout.addLayout(nav_layout)

        # Main viewport / display
        self.viewport = ShadowTrackerViewportWidget(self)
        layout.addWidget(self.viewport, stretch=1)

        # Status / Error panel
        self.lbl_status = QtWidgets.QLabel("Status: Ready (Review Mode)", self)
        layout.addWidget(self.lbl_status)

        # Connect signals
        self.btn_import.clicked.connect(self._on_import_video)
        self.btn_open.clicked.connect(self._on_open_bundle)
        self.btn_save.clicked.connect(self._on_save_bundle)
        self.btn_export.clicked.connect(self._on_export_canonical)
        self.btn_fit.clicked.connect(self._on_request_fit)
        self.btn_prev.clicked.connect(self._on_prev)
        self.btn_next.clicked.connect(self._on_next)
        self.slider_frame.valueChanged.connect(self._on_slider_changed)
        self.btn_worst.clicked.connect(self._on_worst)

    def _on_import_video(self) -> None:
        video_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Video File to Import",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if not video_file:
            return
        try:
            obs = self.import_video(video_file)
            if self.model.is_cancelled:
                self.lbl_status.setText(
                    f"Import cancelled: {len(obs)} frames decoded from {Path(video_file).name}"
                )
            else:
                self.lbl_status.setText(
                    f"Imported {len(obs)} frames from {Path(video_file).name}"
                )
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            logger.warning("Failed to import video: %s", exc)
            self.lbl_status.setText(f"Error importing video: {exc}")

    def _on_open_bundle(self) -> None:
        bundle_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Shadow Tracker Bundle Directory"
        )
        if not bundle_dir:
            return
        try:
            self.load_bundle(bundle_dir)
            self.lbl_status.setText(f"Loaded bundle from {Path(bundle_dir).name}")
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            logger.warning("Failed to open bundle: %s", exc)
            self.lbl_status.setText(f"Error opening bundle: {exc}")

    def _on_save_bundle(self) -> None:
        try:
            saved_path = self.save_bundle()
            self.lbl_status.setText(f"Saved bundle to {saved_path.name}")
        except ValueError:
            target_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Destination Bundle Directory"
            )
            if not target_dir:
                return
            try:
                saved_path = self.save_bundle(target_dir)
                self.lbl_status.setText(f"Saved bundle to {saved_path.name}")
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                logger.warning("Failed to save bundle: %s", exc)
                self.lbl_status.setText(f"Error saving bundle: {exc}")
        except (OSError, RuntimeError, KeyError) as exc:
            logger.warning("Failed to save bundle: %s", exc)
            self.lbl_status.setText(f"Error saving bundle: {exc}")

    def _on_export_canonical(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Canonical Dataset Package",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not filename:
            return
        try:
            self.export_canonical(filename)
            self.lbl_status.setText(
                f"Exported canonical package to {Path(filename).name}"
            )
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            logger.warning("Failed to export canonical package: %s", exc)
            self.lbl_status.setText(f"Error exporting canonical package: {exc}")

    def _on_request_fit(self) -> None:
        """Trigger automated forward fitting and display honest refusal if unqualified (MMR-12)."""
        obs = self.model.get_current_observation()
        if obs is None:
            self.lbl_status.setText("Auto-Fit Unavailable: No observation loaded")
            return
        req = FitRequest(
            schema_version="shadow-tracker/fit-request/1.0.0",
            request_id=f"fit-{obs.shot_id}-{obs.frame_id}",
            shot_id=obs.shot_id,
            model_hash="0" * 64,
            candidate_count=1,
            objective_profile="silhouette_iou",
            time_window_start_pts=obs.pts_ticks,
            time_window_end_pts=obs.pts_ticks + 100,
            budget_seconds=5.0,
            engine_capability_requirement=("forward_dynamics",),
        )
        try:
            self.model.request_fit(req)
            self.lbl_status.setText("Auto-Fit completed")
        except UnavailableBackendError as exc:
            logger.info("Auto-fit honestly unavailable: %s", exc)
            self.lbl_status.setText(
                f"Auto-Fit Unavailable: {exc} (Forward dynamics gates ST-07..ST-10 unqualified)"
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("Auto-fit failed: %s", exc)
            self.lbl_status.setText(f"Auto-Fit Error: {exc}")

    def _on_prev(self) -> None:
        self.model.step_frame(-1)
        self._update_display()

    def _on_next(self) -> None:
        self.model.step_frame(1)
        self._update_display()

    def _on_slider_changed(self, value: int) -> None:
        self.model.jump_to_frame(value)
        self._update_display()

    def _on_worst(self) -> None:
        report = self.model.jump_to_worst_frame()
        if report:
            self._update_display()
            self.lbl_status.setText(
                f"Triage: Worst frame {report.frame_id} (coverage score {report.score:.3f})"
            )

    def _update_display(self) -> None:
        total = self.model.frame_count
        curr = self.model.current_frame_index
        self.lbl_frame_info.setText(f"Frame: {curr + 1 if total > 0 else 0} / {total}")
        self.slider_frame.blockSignals(True)
        self.slider_frame.setRange(0, max(0, total - 1))
        self.slider_frame.setValue(curr)
        self.slider_frame.blockSignals(False)

        obs = self.model.get_current_observation()
        self.viewport.set_observation(obs)

    # Public review API methods
    def import_video(
        self, video_path: str | Path, **kwargs: Any
    ) -> tuple[FrameObservation, ...]:
        """Import video into review model and refresh display."""
        res = self.model.import_video(video_path, **kwargs)
        self._update_display()
        return res

    def load_bundle(self, bundle_dir: str | Path) -> None:
        self.model.load_bundle(bundle_dir)
        self._update_display()

    def save_bundle(self, bundle_dir: str | Path | None = None) -> Path:
        return self.model.save_bundle(bundle_dir)

    def export_canonical(self, export_path: str | Path) -> dict[str, Any]:
        return self.model.export_canonical(export_path)

    def is_dirty(self) -> bool:
        return self.model.is_dirty

    def update_mask(
        self,
        *,
        body: bytes,
        club: bytes,
        valid: bytes,
        parent_revision_id: str | None = None,
        producer_id: str = "reviewer-gui",
        correction_note: str = "",
    ) -> MaskFrame:
        """Update mask for current frame, invalidating downstream fits and marking session dirty (MMR-12)."""
        new_mask = self.model.update_mask(
            body=body,
            club=club,
            valid=valid,
            parent_revision_id=parent_revision_id,
            producer_id=producer_id,
            correction_note=correction_note,
        )
        self._update_display()
        self.lbl_status.setText(
            f"Updated mask for {new_mask.frame.frame_id} (rev: {new_mask.revision_id})"
        )
        return new_mask

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:  # noqa: N802
        """Handle keyboard scrubbing, triage shortcuts, and navigation (MMR-12)."""
        if event is None:
            return
        key = event.key()
        modifiers = event.modifiers()
        if key == QtCore.Qt.Key.Key_Right:
            self._on_next()
            event.accept()
        elif key == QtCore.Qt.Key.Key_Left:
            self._on_prev()
            event.accept()
        elif key == QtCore.Qt.Key.Key_Home:
            self.model.jump_to_frame(0)
            self._update_display()
            event.accept()
        elif key == QtCore.Qt.Key.Key_End:
            self.model.jump_to_frame(max(0, self.model.frame_count - 1))
            self._update_display()
            event.accept()
        elif key == QtCore.Qt.Key.Key_W and not (
            modifiers & QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self._on_worst()
            event.accept()
        elif key == QtCore.Qt.Key.Key_F and not (
            modifiers & QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self._on_request_fit()
            event.accept()
        elif key == QtCore.Qt.Key.Key_S and (
            modifiers & QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self._on_save_bundle()
            event.accept()
        elif key == QtCore.Qt.Key.Key_O and (
            modifiers & QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self._on_open_bundle()
            event.accept()
        elif key == QtCore.Qt.Key.Key_I and (
            modifiers & QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self._on_import_video()
            event.accept()
        else:
            super().keyPressEvent(event)

    def cleanup(self) -> None:
        """Idempotently release resources upon shutdown or unload."""
        model = self.model
        model.cancel()
        logger.debug("ShadowTrackerWidget cleanup completed")
