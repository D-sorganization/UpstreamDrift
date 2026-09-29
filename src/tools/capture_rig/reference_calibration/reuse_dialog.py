"""Explicit, asynchronous review of a prior capture's camera layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
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
        self._source_controls(layout)
        self.evidence = QTextBrowser()
        self.evidence.setPlainText(
            "Choose From Capture Library to find a saved calibration by swing name and date. "
            "Open Layout File also accepts an original reviewed estimate from another location. "
            "The same named views, physical cameras and recorded image sizes are required. "
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

    def _source_controls(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        self.library_button = QPushButton("From Capture Library…")
        self.library_button.clicked.connect(self._library)
        self.browse = QPushButton("Open Layout File…")
        self.browse.setToolTip(
            "Open an original reviewed estimate from another location"
        )
        self.browse.clicked.connect(self._browse)
        row.addWidget(self.library_button)
        row.addWidget(self.browse)
        layout.addLayout(row)
        self.saved_layouts = QComboBox()
        self.saved_layouts.setAccessibleName("Saved Calibrations")
        self.saved_layouts.setMinimumContentsLength(20)
        self.saved_layouts.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.saved_layouts.addItem(
            "Choose From Capture Library to Load Saved Calibrations"
        )
        self.saved_layouts.activated.connect(self._choose_saved)
        layout.addWidget(self.saved_layouts)

    def _refresh(self, *_: Any) -> None:
        busy = self.client.busy
        self.browse.setEnabled(not busy)
        self.library_button.setEnabled(not busy)
        self.saved_layouts.setEnabled(not busy and self.saved_layouts.count() > 1)
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
            self.saved_layouts.setCurrentIndex(0)
            self._select_source(selected)

    def _reset_review(self) -> None:
        self._source, self._digest = "", ""
        self.settings.setChecked(False)
        self.scene.setChecked(False)
        self.evidence.clear()

    def _library(self) -> None:
        self._reset_review()
        self.saved_layouts.clear()
        self.saved_layouts.addItem("Loading Saved Calibrations…")
        self._request("reuse_choices")

    def _choose_saved(self, index: int) -> None:
        selected = self.saved_layouts.itemData(index)
        if selected:
            self._select_source(selected)

    def _select_source(self, selected: str) -> None:
        self._reset_review()
        self._source = selected
        self._request("inspect_reuse")

    def _request(self, action: str) -> None:
        self._action = action
        self.status.setText(
            {
                "reuse_choices": "Reading saved calibration names from Capture Library…",
                "inspect_reuse": "Checking original evidence…",
                "adopt_layout": "Copying and verifying calibration evidence for this swing…",
            }[action]
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
                        "library_root": str(self.library_root),
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
        if self._action == "reuse_choices":
            self._show_choices(result)
        elif self._action == "inspect_reuse":
            self._digest = result["source_sha256"]
            self.evidence.setPlainText(review_text(result))
            self.status.setText(
                "Evidence verified. Review the source settings and confirm both conditions to continue."
            )
            self._refresh()
        else:
            self.output_path = self.root / result["result_path"]
            self.accept()

    def _show_choices(self, result: dict[str, Any]) -> None:
        self.saved_layouts.clear()
        self.saved_layouts.addItem("Select a Saved Calibration…")
        for choice in result["choices"]:
            self.saved_layouts.addItem(choice["label"], choice["path"])
        self.status.setText(
            "Choose a saved calibration above to review its evidence."
            if result["choices"]
            else "No reviewed calibrations found. Choose Open Layout File, or calibrate a capture first."
        )
        if result["problems"]:
            self.evidence.setPlainText(
                "Some Results Could Not Be Listed\n\n" + "\n".join(result["problems"])
            )
        self._refresh()

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

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self.reject()
        if event is not None:
            event.accept()
