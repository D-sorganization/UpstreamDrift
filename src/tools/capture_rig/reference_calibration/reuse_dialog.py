"""Explicit, asynchronous review of a prior capture's camera layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.motion_capture.rig.capture_notes import read_notes
from src.motion_capture.rig.plan import RigPlan

from .client import ReferenceWorkerClient


def review_text(preview: dict[str, Any]) -> str:
    """Show declared settings and original evidence without implying a new solve."""
    result = preview["result"]
    lines = [
        f"Source Swing: {preview['source_title']}",
        f"Capture ID: {preview['source_capture_id']}",
        f"Original Review: {result['reviewed_utc']}",
        f"Scene: {result['scene_id']}",
        f"Anchor Offset (m): {result['anchor']['translation_m']}",
        "",
    ]
    for camera in result["profile_selections"]:
        setup = camera["setup"]
        lines.extend(
            [
                f"{camera['view']}: {setup['camera_identity']}",
                f"Lens: {setup['lens']} | Zoom: {setup['zoom']} | Focus: {setup['focus']}",
                f"Sensor: {setup['sensor_mode']} | Pixels: {setup['image_size_px']}",
                "",
            ]
        )
    lines.append("Original Fit / Validation Evidence")
    for residual in result.get("residuals", []):
        role = "Validation" if residual["held_out"] else "Fit"
        lines.append(
            f"{residual['camera_key']} / {residual['placement_id']} ({role}): "
            f"mean {residual['mean_error_px']:.3f} px, "
            f"maximum {residual['max_error_px']:.3f} px, "
            f"{residual['point_count']} points"
        )
    lines.extend(["", "Original Limitations", *result.get("limitations", [])])
    lines.extend(
        [
            "",
            "This reuses the original estimate. It does not recalculate camera positions.",
            "If a camera moved, zoom/focus changed, or the ball origin/target direction changed, "
            "close this review and use Paper / Ruler References to calibrate again.",
        ]
    )
    return "\n".join(lines)


class ReuseCalibrationDialog(QDialog):
    """Keep feedback visible and discard approvals whenever the selection changes."""

    def __init__(
        self,
        plan: RigPlan,
        root: Path,
        library_root: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reuse a Reviewed Camera Layout")
        self.resize(700, 620)
        self.root, self.library_root = root, library_root
        self.capture_id = read_notes(root).capture_id
        self.expected = {
            camera.view: (camera.identity, (camera.mode.width, camera.mode.height))
            for camera in plan.cameras
        }
        self.output_path: Path | None = None
        self._source = ""
        self._digest = ""
        self._action = ""
        self._closed = False
        self.client = ReferenceWorkerClient(self)
        self.client.completed.connect(self._completed)
        self.client.failed.connect(self._failed)
        self.client.busy_changed.connect(self._refresh)
        layout = QVBoxLayout(self)
        title = QLabel(
            f"New Swing: {read_notes(root).title}\nCapture ID: {self.capture_id}"
        )
        title.setWordWrap(True)
        layout.addWidget(title)
        self.browse = QPushButton("Choose a Reviewed Layout…")
        self.browse.setToolTip(
            "Choose an original reviewed layout from another capture"
        )
        self.browse.clicked.connect(self._browse)
        layout.addWidget(self.browse)
        self.evidence = QTextBrowser()
        self.evidence.setPlainText(
            "Select a reviewed-….json file in the source capture's reference_calibration/results "
            "folder. The same named views, physical cameras and recorded image sizes are required. "
            "The original evidence will be copied into this swing, so the original can later be archived."
        )
        layout.addWidget(self.evidence, 1)
        confirmation = QLabel(
            "Confirm these were unchanged when this swing was recorded, including "
            "sensor mode, ball origin and target direction."
        )
        confirmation.setWordWrap(True)
        layout.addWidget(confirmation)
        self.settings = QCheckBox("Lens, Zoom and Focus Match")
        self.scene = QCheckBox("Camera Positions and Ball Reference Match")
        for check in (self.settings, self.scene):
            check.toggled.connect(self._refresh)
            layout.addWidget(check)
        self.status = QLabel("Choose a reviewed layout to begin.")
        self.status.setWordWrap(True)
        self.status.setAccessibleName("Calibration Reuse Status")
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.use = QPushButton("Use for This Swing")
        buttons.addButton(self.use, QDialogButtonBox.ButtonRole.AcceptRole)
        self.use.clicked.connect(self._adopt)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._refresh()

    def _refresh(self, *_: Any) -> None:
        busy = self.client.busy
        self.browse.setEnabled(not busy)
        self.settings.setEnabled(not busy and bool(self._digest))
        self.scene.setEnabled(not busy and bool(self._digest))
        self.use.setEnabled(
            not busy
            and bool(self._digest)
            and self.settings.isChecked()
            and self.scene.isChecked()
        )

    def _browse(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Choose an Original Reviewed Camera Layout",
            str(self.library_root),
            "Reviewed Camera Layout (reviewed-*.json)",
        )
        if selected:
            self._source, self._digest = selected, ""
            self.settings.setChecked(False)
            self.scene.setChecked(False)
            self.evidence.clear()
            self._request("inspect_reuse")

    def _request(self, action: str) -> None:
        self._action = action
        self.status.setText(
            "Checking original evidence…"
            if action == "inspect_reuse"
            else "Copying and verifying calibration evidence for this swing…"
        )
        try:
            self.client.request(
                {
                    "action": action,
                    "workspace": str(self.root),
                    "capture_id": self.capture_id,
                    "parameters": {
                        "source_path": self._source,
                        "source_sha256": self._digest,
                        "expected_cameras": self.expected,
                        "settings_confirmed": self.settings.isChecked(),
                        "scene_confirmed": self.scene.isChecked(),
                    },
                }
            )
        except (ValueError, OSError) as exc:
            self._failed(str(exc))

    def _adopt(self) -> None:
        if self.use.isEnabled():
            self._request("adopt_layout")

    def _completed(self, result: dict[str, Any]) -> None:
        if self._closed:
            return
        if self._action == "inspect_reuse":
            self._digest = result["source_sha256"]
            self.evidence.setPlainText(review_text(result))
            self.status.setText(
                "Evidence verified. Review the source settings and confirm both conditions to continue."
            )
            self._refresh()
        else:
            self.output_path = self.root / result["result_path"]
            self.accept()

    def _failed(self, error: str) -> None:
        if not self._closed:
            self._digest = ""
            self.status.setText(
                f"Calibration was not applied: {error}. Choose the layout again after correcting it."
            )
            self._refresh()

    def reject(self) -> None:
        self._closed = True
        self.client.cancel()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.reject()
        event.accept()
