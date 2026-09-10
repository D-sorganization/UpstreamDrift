"""Repeatable player reference sessions over standard Qt controls and Tools IPC."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.motion_capture.rig.capture_notes import read_notes
from src.motion_capture.rig.plan import RigPlan

from ..calibration_dialog import CameraProfilePanel
from ..calibration_profiles import CalibrationProfile
from .client import ReferenceWorkerClient
from .frame_loader import FrameLoader
from .frame_selector import ReferenceFrameSelector
from .placement_panel import PlacementPanel
from .point_editor import ReferencePointEditor
from .solve_panel import SolvePanel
from .guidance import GUIDE
from ..styling import help_document_style


class ReferenceCalibrationDialog(QDialog):
    """Save observations explicitly; never present incomplete geometry as solved."""

    def __init__(
        self,
        plan: RigPlan,
        root: Path,
        profiles_root: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        notes = read_notes(root)
        self.root, self.capture_id, self.capture_title = (
            root,
            notes.capture_id,
            notes.title,
        )
        self._session: dict[str, Any] | None = None
        self.output_path: Path | None = None
        self._catalog: list[dict[str, Any]] = []
        self._action = ""
        self._dirty = False
        self._closing = False
        self._syncing = False
        self._frame_context: dict[str, Any] = {}
        self.setWindowTitle("Common Reference Calibration")
        self.resize(850, 720)
        self.client = ReferenceWorkerClient(self)
        self.loader = FrameLoader(self)
        self.client.completed.connect(self._completed)
        self.client.failed.connect(self._failed)
        self.loader.ready.connect(self._edit_frame)
        self.loader.failed.connect(self._failed)
        self.finished.connect(self._closed)
        self._build(plan, profiles_root)
        QTimer.singleShot(0, lambda: self._request("catalog"))

    def _build(self, plan: RigPlan, profiles_root: Path) -> None:
        layout = QVBoxLayout(self)
        heading = QLabel(f"{self.capture_title}\nCapture: {self.capture_id}")
        heading.setTextFormat(Qt.TextFormat.PlainText)
        heading.setWordWrap(True)
        layout.addWidget(heading)
        self.status = QLabel("Loading reference choices…")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.retry = QPushButton("Reload Reference Choices")
        self.retry.clicked.connect(lambda: self._request("catalog"))
        layout.addWidget(self.retry)
        self.tabs = QTabWidget()
        setup = QWidget()
        setup_layout = QVBoxLayout(setup)
        hint = QLabel(
            "Review the settings used for this recording. Enter ‘unknown’ when necessary. "
            "Changing optical zoom, focus, sensor crop or camera position requires a new session. "
            "Paper and rulers do not replace lens calibration. Observations can be saved before lens profiles are available."
        )
        hint.setWordWrap(True)
        setup_layout.addWidget(hint)
        self.scene = QLineEdit("Cameras Near Practice Area")
        self.scene.setMaxLength(150)
        setup_layout.addWidget(QLabel("Camera Setup Name"))
        setup_layout.addWidget(self.scene)
        camera_tabs = QTabWidget()
        self.panels = []
        for binding in plan.cameras:
            panel = CameraProfilePanel(binding, profiles_root / "profiles.json")
            for field in panel.fields.values():
                field.setText("unknown")
                field.textChanged.connect(self._settings_edited)
            panel.profiles.currentIndexChanged.connect(self._settings_edited)
            self.panels.append(panel)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(panel)
            camera_tabs.addTab(scroll, binding.view)
        setup_layout.addWidget(camera_tabs, 1)
        self.start = QPushButton("Start a New Reference Session")
        self.start.clicked.connect(self._new_session)
        setup_layout.addWidget(self.start)
        self.tabs.addTab(self._scroll_page(setup), "Camera Setup")
        self.placements = PlacementPanel()
        self.placements.frame_requested.connect(self._choose_frame)
        self.placements.revision_requested.connect(self._revise)
        self.placements.target_requested.connect(
            lambda parameters: self._request("target", parameters=parameters)
        )
        self.tabs.addTab(self._scroll_page(self.placements), "Reference Placements")
        self.tabs.setTabEnabled(1, False)
        self.solve_panel = SolvePanel()
        self.solve_panel.restore_requested.connect(self._restore_result)
        self.solve_panel.solve_requested.connect(
            lambda parameters: self._request("solve", parameters=parameters)
        )
        self.solve_panel.accept_requested.connect(
            lambda parameters: self._request("accept", parameters=parameters)
        )
        self.tabs.addTab(self._scroll_page(self.solve_panel), "Camera Positions")
        self.tabs.setTabEnabled(2, False)
        help_view = QTextBrowser()
        document = help_view.document()
        if document is not None:
            document.setDefaultStyleSheet(help_document_style())
        help_view.setMarkdown(GUIDE)
        self.tabs.addTab(help_view, "Help")
        layout.addWidget(self.tabs, 1)
        layout.addLayout(self._footer())
        self._busy(False)

    @staticmethod
    def _scroll_page(widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def _footer(self) -> QGridLayout:
        row = QGridLayout()
        self.again = QPushButton("Calibrate Again")
        self.again.clicked.connect(self._new_session)
        self.again.setToolTip(
            "Start fresh observations for these settings; earlier saved revisions remain available."
        )
        self.restore = QPushButton("Open Saved Revision…")
        self.restore.clicked.connect(self._restore)
        self.save = QPushButton("Save New Revision")
        self.save.clicked.connect(lambda: self._request("save"))
        self.stop = QPushButton("Stop Operation")
        self.stop.clicked.connect(self.client.cancel)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        for index, button in enumerate(
            (self.again, self.restore, self.save, self.stop)
        ):
            row.addWidget(button, index // 2, index % 2)
        row.addWidget(close, 2, 1)
        return row

    def _busy(self, active: bool) -> None:
        self.tabs.setEnabled(not active)
        self.start.setEnabled(not active and bool(self._catalog))
        self.again.setEnabled(not active and bool(self._catalog))
        self.restore.setEnabled(not active)
        self.retry.setEnabled(not active)
        self.retry.setVisible(not self._catalog)
        self.save.setEnabled(not active and self._session is not None and self._dirty)
        self.stop.setEnabled(active and self.client.busy)

    def _settings_edited(self) -> None:
        if self._session is not None and not self._syncing:
            self.tabs.setTabEnabled(1, False)
            self.tabs.setTabEnabled(2, False)
            self.status.setText(
                "Camera settings changed. Start a new reference session before marking or solving. Earlier saved revisions remain available."
            )

    def _sync_setup(self, result: dict[str, Any]) -> None:
        self._syncing = True
        try:
            for panel in self.panels:
                camera = next(
                    item
                    for item in result["cameras"]
                    if item["view"] == panel.binding.view
                )
                profile_id = (camera["profile"] or {}).get("profile_id")
                panel.profiles.setCurrentIndex(0)
                for index in range(panel.profiles.count()):
                    profile = panel.profiles.itemData(index)
                    if (
                        isinstance(profile, CalibrationProfile)
                        and profile.profile_id == profile_id
                    ):
                        panel.profiles.setCurrentIndex(index)
                        break
                else:
                    if profile_id is not None:
                        profile = CalibrationProfile.model_validate(camera["profile"])
                        panel.profiles.addItem(
                            f"{profile.name} · Saved with Capture", profile
                        )
                        panel.profiles.setCurrentIndex(panel.profiles.count() - 1)
                for key, field in panel.fields.items():
                    field.setText(camera["setup"][key])
        finally:
            self._syncing = False

    def _request(self, action: str, **extra: Any) -> None:
        if self._closing:
            return
        self._action = action
        payload = {
            "action": action,
            "workspace": str(self.root),
            "capture_id": self.capture_id,
            **extra,
        }
        if self._session is not None and "session" not in payload:
            payload["session"] = self._session
        self.status.setText(
            {
                "catalog": "Loading available reference sizes…",
                "validate": "Checking the camera setup…",
                "frame": "Reading and preserving the original frame…",
                "mark": "Checking point identities…",
                "revise": "Updating this session…",
                "save": "Verifying frames and saving a new revision…",
                "load": "Opening the saved revision…",
                "solve": "Estimating camera positions and checking held-out observations…",
                "accept": "Rechecking source frames and reviewed camera settings…",
                "target": "Adding the measured reference dimensions…",
                "result": "Verifying the saved camera estimate and its observations…",
            }.get(action, "Working…")
        )
        try:
            self.client.request(payload)
            self._busy(True)
        except (ValueError, OSError) as exc:
            self._failed(str(exc))

    def _discard_changes(self) -> bool:
        if not self._dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved Reference Changes",
            "Discard the unsaved changes? Earlier saved revisions will remain available.",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Discard

    def _new_session(self) -> None:
        if not self._discard_changes():
            return
        try:
            cameras = []
            for panel in self.panels:
                profile = panel.profiles.currentData()
                cameras.append(
                    {
                        "view": panel.binding.view,
                        "setup": panel.setup().model_dump(mode="json"),
                        "profile": profile.model_dump(mode="json")
                        if isinstance(profile, CalibrationProfile)
                        else None,
                    }
                )
            payload = {
                "capture_id": self.capture_id,
                "title": self.capture_title,
                "scene_id": f"{self.scene.text().strip()} · {uuid4()}",
                "cameras": cameras,
                "targets": self._catalog,
            }
            self._request("validate", session=payload)
        except ValueError as exc:
            self._failed(f"Review the camera settings: {exc}")

    def _restore(self) -> None:
        if not self._discard_changes():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Reference Revision",
            str(self.root / "reference_calibration"),
            "Reference Revision (*.json)",
        )
        if not path:
            return
        try:
            selected = Path(path)
            revision = UUID(selected.stem)
            if (
                selected.parent.resolve()
                != (self.root / "reference_calibration").resolve()
            ):
                raise ValueError("Choose a saved revision belonging to this capture")
            self._request("load", revision_id=str(revision))
        except ValueError as exc:
            self._failed(str(exc))

    def _revise(self, changes: dict[str, Any]) -> None:
        self._request("revise", changes=changes)

    def _restore_result(self) -> None:
        folder = self.root / "reference_calibration" / "results"
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Camera Estimate for This Revision",
            str(folder),
            "Camera Estimate (*.json)",
        )
        if not path:
            return
        try:
            selected = Path(path)
            result_id = UUID(selected.stem)
            if selected.parent.resolve() != folder.resolve():
                raise ValueError(
                    "Choose an original camera estimate belonging to this capture"
                )
            self._request("result", parameters={"result_id": str(result_id)})
        except ValueError as exc:
            self._failed(str(exc))

    def _choose_frame(self, parameters: dict[str, Any]) -> None:
        from src.motion_capture.rig.bundle import load_bundle

        panel = self.placements
        placement = panel.marking_parameters()["placement_id"]
        if not placement:
            self._failed(
                "Name this physical placement before choosing its camera frame."
            )
            return
        try:
            _, recordings, _ = load_bundle(self.root)
            recording = next(
                (
                    entry
                    for entry in recordings.recordings
                    if entry.view == parameters["view"]
                ),
                None,
            )
            if recording is None:
                raise ValueError(
                    "This view has no original recording. Restore it in Capture Library"
                )
            selector = ReferenceFrameSelector(
                self.root / recording.file,
                context=f"{self.capture_title} · {parameters['view']} · {placement}",
                initial_index=parameters["frame_index"],
                parent=self,
            )
            if (
                selector.exec() == QDialog.DialogCode.Accepted
                and selector.selected_frame is not None
            ):
                panel.frame_number.setValue(selector.selected_frame)
                self._open_frame({**parameters, "frame_index": selector.selected_frame})
            else:
                self.status.setText(
                    "Frame selection cancelled. Existing observations remain unchanged."
                )
        except (ValueError, OSError) as exc:
            self._failed(str(exc))

    def _open_frame(self, parameters: dict[str, Any]) -> None:
        marking = self.placements.marking_parameters()
        if not marking["placement_id"]:
            self._failed(
                "Name this physical placement before marking its camera views."
            )
            return
        self._frame_context = {
            "marking": marking,
            "point_ids": self.placements.point_ids(),
        }
        self._request("frame", parameters=parameters)

    def _completed(self, result: dict[str, Any]) -> None:
        if self._closing:
            return
        action = self._action
        if action == "catalog":
            self._catalog = result["targets"]
            self.status.setText(
                "Review camera settings, then start a session or open a saved revision."
            )
        elif action == "frame":
            record = result["frame"]
            self._frame_context["record"] = record
            self.status.setText("Verifying the archived image before point marking…")
            self.loader.load(
                self.root, record["path"], record["sha256"], self.capture_id
            )
            return
        elif action in {"solve", "result"}:
            self._dirty = False
            self.solve_panel.show_result(result["result"], result["result_sha256"])
            self.status.setText(
                f"Camera estimate saved at {result['result_path']}. Review the errors and setup. Earlier revisions remain available."
            )
        elif action == "accept":
            self.output_path = self.root / result["result_path"]
            self._dirty = False
            self.accept()
            return
        else:
            self._session = result
            self._sync_setup(result)
            self._dirty = action not in {"save", "load"}
            self.placements.show_session(result)
            if action == "target":
                target = self.placements.target
                target.setCurrentIndex(target.count() - 1)
            self.tabs.setTabEnabled(1, True)
            self.tabs.setTabEnabled(2, True)
            self.solve_panel.show_session(result)
            self.tabs.setCurrentIndex(1)
            count = len(result["samples"])
            state = (
                "Saved"
                if action == "save"
                else "Opened"
                if action == "load"
                else "Unsaved changes"
            )
            self.status.setText(
                f"{state} · {count} observations · Revision {result['revision_id'][:8]}. "
                "Add another view or placement. Camera positions have not been solved."
            )
        self._busy(False)

    def _edit_frame(self, frame: Any) -> None:
        if self._closing:
            return
        record = self._frame_context["record"]
        marking = self._frame_context["marking"]
        points = {}
        for sample in (self._session or {}).get("samples", []):
            observation = sample["observation"]
            if (
                observation["placement_id"] == marking["placement_id"]
                and observation["camera_key"] == record["view"]
                and observation["frame_sequence"] == record["frame_index"]
                and observation["reference_id"] == marking["reference_id"]
                and sample["source_sha256"] == record["sha256"]
            ):
                points = dict(
                    zip(observation["point_ids"], observation["pixels_px"], strict=True)
                )
        editor = ReferencePointEditor(
            frame,
            self._frame_context["point_ids"],
            context=f"{self.capture_title} · {marking['placement_id']} · {record['view']} · Frame {record['frame_index']}",
            parent=self,
            points=points,
        )
        if editor.exec() == QDialog.DialogCode.Accepted:
            self._request(
                "mark",
                parameters={
                    **marking,
                    "frame_path": record["path"],
                    "frame_sha256": record["sha256"],
                    "points": editor.points,
                },
            )
        else:
            self.status.setText(
                "Point marking cancelled. Existing observations remain unchanged."
            )
            self._busy(False)

    def _failed(self, message: str) -> None:
        if not self._closing:
            self.status.setText(f"{message} Correct the selection and try again.")
            self._busy(False)

    def reject(self) -> None:
        if self._discard_changes():
            super().reject()

    def _closed(self, _result: int) -> None:
        self._closing = True
        self.client.cancel()
        self.loader.close()
