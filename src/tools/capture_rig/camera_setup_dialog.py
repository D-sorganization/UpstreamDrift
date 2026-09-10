"""Camera discovery and plan editing without requiring hand-written JSON."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from src.motion_capture.rig.plan import CameraBinding, CaptureMode, RigPlan, parse_mode
from src.motion_capture.rig.topology import CameraLocation

from .camera_setup import (
    MAX_CAMERAS,
    bind_camera,
    connection_status,
    create_plan,
    discover_cameras,
    load_plan,
    save_revision,
)
from .commands import MODE_PRESETS, mode_text


class CameraPlanRow:
    """Editable view/mode with a stable device identity retained across scans."""

    def __init__(self, view: str, binding: CameraBinding | None = None) -> None:
        self.view = QLineEdit(view)
        self.view.setMaxLength(48)
        self.view.setAccessibleName("View Name")
        self.device = QComboBox()
        self.device.setAccessibleName("Camera Identity")
        self.mode = QComboBox()
        self.mode.setAccessibleName("Capture Resolution and Frame Rate")
        self.mode.setEditable(True)
        self.mode.addItems([mode_text(mode) for mode in MODE_PRESETS])
        self.mode.setCurrentText(mode_text(binding.mode if binding else CaptureMode()))
        self.mode.setToolTip(
            "Width x height @ frames per second : format; use a mode your camera supports"
        )
        self.remove = QPushButton("Remove")
        self.remove.setAccessibleName(f"Remove View {view}")
        self.set_devices([], binding)

    def set_devices(
        self,
        cameras: Sequence[CameraLocation],
        saved: CameraBinding | None = None,
    ) -> None:
        current = saved if saved is not None else self.device.currentData()
        self.device.clear()
        self.device.addItem("Choose a Camera", None)
        selected = 0
        for camera in cameras:
            binding = bind_camera(self.view.text(), camera, CaptureMode())
            if (
                isinstance(current, CameraBinding)
                and current.identity == binding.identity
            ):
                binding = current
                selected = self.device.count()
            label = (
                f"Serial {camera.serial}"
                if camera.serial
                else f"USB Port {camera.identity}"
            )
            self.device.addItem(label, binding)
            self.device.setItemData(
                self.device.count() - 1, camera.camera, Qt.ItemDataRole.ToolTipRole
            )
        if isinstance(current, CameraBinding) and not selected:
            selected = self.device.count()
            self.device.addItem(f"Saved Identity: {current.identity}", current)
        self.device.setCurrentIndex(selected)

    def binding(self) -> CameraBinding:
        selected = self.device.currentData()
        if not isinstance(selected, CameraBinding):
            raise ValueError(
                f"Choose a camera for view {self.view.text() or '(unnamed)'}"
            )
        values = selected.model_dump()
        values.update(
            view=self.view.text().strip(), mode=parse_mode(self.mode.currentText())
        )
        return CameraBinding.model_validate(values)


class CameraSetupDialog(QDialog):
    """Save new plan revisions; scanning never opens a recording stream."""

    def __init__(
        self,
        library: Path,
        parent: QWidget | None = None,
        *,
        plan: RigPlan | None = None,
        discover: Callable[[Event], list[CameraLocation]] = discover_cameras,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Camera Setup")
        self.resize(760, 540)
        self.library = library
        self.saved_path: Path | None = None
        self._discover = discover
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="camera-setup"
        )
        self._future: Future[list[CameraLocation]] | None = None
        self._cancelled = Event()
        self.cameras: list[CameraLocation] = []
        self.rows: list[CameraPlanRow] = []
        self.name = QLineEdit("My Camera Setup")
        self.name.setMaxLength(120)
        self.notes = QPlainTextEdit()
        self.notes.setMaximumHeight(65)
        self.notes.setPlaceholderText("Camera placement or setup notes (optional)")
        self.status = QLabel(
            "Scan for supported USB rig cameras, or load a saved plan for offline editing. "
            "Live discovery currently supports the ELP rig on Windows."
        )
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setAccessibleName("Camera Setup Status")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["View Name", "Camera", "Resolution / FPS", ""]
        )
        horizontal_header = self.table.horizontalHeader()
        vertical_header = self.table.verticalHeader()
        assert horizontal_header is not None and vertical_header is not None
        horizontal_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        vertical_header.hide()
        self.table.setMinimumSize(0, 160)
        self.scan_button = QPushButton("&Scan for Cameras")
        self.cancel_scan_button = QPushButton("Cancel Scan")
        self.cancel_scan_button.setEnabled(False)
        self.add_button = QPushButton("Add &View")
        self.check_button = QPushButton("Check Connections")
        self.load_button = QPushButton("&Load Plan…")
        self.scan_button.clicked.connect(self.scan)
        self.cancel_scan_button.clicked.connect(self.cancel_scan)
        self.add_button.clicked.connect(lambda: self.add_view())
        self.check_button.clicked.connect(self.check_connections)
        self.load_button.clicked.connect(self.pick_plan)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        save_button = self.buttons.button(QDialogButtonBox.StandardButton.Save)
        assert save_button is not None
        save_button.setText("Save and Use")
        self.buttons.accepted.connect(self.save)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        instructions = QLabel(
            "1. Scan or load a setup.  2. Name each view and choose its camera and mode.  "
            "3. Save and Use, then run Plan Check and Preview. Every save creates a new "
            "revision; existing plans and swing captures are preserved."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        form = QFormLayout()
        form.addRow("Setup &Name", self.name)
        form.addRow("Notes", self.notes)
        layout.addLayout(form)
        tools = QHBoxLayout()
        for button in (self.scan_button, self.cancel_scan_button, self.load_button):
            tools.addWidget(button)
        layout.addLayout(tools)
        layout.addWidget(self.table, 1)
        editing = QHBoxLayout()
        editing.addWidget(self.add_button)
        editing.addWidget(self.check_button)
        layout.addLayout(editing)
        layout.addWidget(self.status)
        layout.addWidget(self.buttons)
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self.poll_scan)
        self.finished.connect(self._close_scan)
        if plan is not None:
            self.set_plan(plan)

    def add_view(self, binding: CameraBinding | None = None) -> None:
        if len(self.rows) >= MAX_CAMERAS:
            self.status.setText(f"Use at most {MAX_CAMERAS} views in one setup.")
            return
        used = {row.view.text() for row in self.rows}
        number = next(n for n in range(1, MAX_CAMERAS + 2) if f"camera_{n}" not in used)
        row = CameraPlanRow(binding.view if binding else f"camera_{number}", binding)
        row.set_devices(self.cameras)
        index = len(self.rows)
        self.rows.append(row)
        self.table.insertRow(index)
        for column, widget in enumerate((row.view, row.device, row.mode, row.remove)):
            self.table.setCellWidget(index, column, widget)
        row.remove.clicked.connect(lambda: self.remove_view(row))

    def remove_view(self, row: CameraPlanRow) -> None:
        index = self.rows.index(row)
        self.rows.pop(index)
        self.table.removeRow(index)

    def set_plan(self, plan: RigPlan) -> None:
        checked = create_plan(plan.name, plan.cameras, plan.notes)
        self.table.setRowCount(0)
        self.rows.clear()
        self.name.setText(checked.name)
        self.notes.setPlainText(checked.notes)
        for binding in checked.cameras:
            self.add_view(binding)

    def plan(self) -> RigPlan:
        return create_plan(
            self.name.text(),
            [row.binding() for row in self.rows],
            self.notes.toPlainText(),
        )

    def apply_devices(self, cameras: Sequence[CameraLocation]) -> None:
        self.cameras = list(cameras)
        if not self.rows:
            for index, camera in enumerate(self.cameras[:MAX_CAMERAS]):
                self.add_view(bind_camera(f"camera_{index + 1}", camera, CaptureMode()))
        else:
            for row in self.rows:
                row.set_devices(self.cameras)
        self.status.setText(
            f"Found {len(self.cameras)} supported camera(s). Review each assignment and mode."
            if self.cameras
            else "No supported cameras were reported. Check USB connections and camera permissions, "
            "then rescan. Existing choices are preserved; you can also load a plan or import video."
        )

    def scan(self) -> None:
        if self._future is not None:
            return
        self._cancelled = Event()
        self.status.setText(
            "Scanning camera identities and USB connections… You can continue editing."
        )
        self.scan_button.setEnabled(False)
        self.cancel_scan_button.setEnabled(True)
        self._future = self._executor.submit(self._discover, self._cancelled)
        self._timer.start()

    def cancel_scan(self) -> None:
        self._cancelled.set()
        self.cancel_scan_button.setEnabled(False)
        self.status.setText(
            "Scan cancelled; choices preserved. Rescan becomes available when the current device probe stops."
        )

    def poll_scan(self) -> None:
        if self._future is None or not self._future.done():
            return
        future, self._future = self._future, None
        self._timer.stop()
        self.scan_button.setEnabled(True)
        self.cancel_scan_button.setEnabled(False)
        if self._cancelled.is_set():
            self.status.setText(
                "Scan cancelled. Existing choices are unchanged; you can scan again."
            )
            return
        try:
            self.apply_devices(future.result())
        except subprocess.TimeoutExpired:
            self.status.setText(
                "Camera discovery timed out. Existing choices are unchanged. "
                "Check USB connections and camera permissions, then scan again."
            )
        except (
            ValueError,
            OSError,
            ImportError,
            RuntimeError,
        ) as exc:
            self.status.setText(
                f"Could not scan cameras: {exc}. Retry, load a saved plan, or import video."
            )

    def check_connections(self) -> None:
        try:
            _ready, message = connection_status(self.plan(), self.cameras)
            self.status.setText(message)
        except ValueError as exc:
            self.status.setText(str(exc))

    def pick_plan(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Load Camera Plan",
            str(self.library / "camera-plans"),
            "Camera Plans (*.json)",
        )
        if not selected:
            return
        try:
            self.set_plan(load_plan(Path(selected)))
            self.status.setText(
                "Plan loaded. Rescan to check current connections; saved camera controls are preserved."
            )
        except (ValueError, OSError) as exc:
            self.status.setText(
                f"Cannot load this plan: {exc}. Current choices are unchanged."
            )

    def save(self) -> None:
        try:
            self.saved_path = save_revision(self.plan(), self.library)
        except (ValueError, OSError) as exc:
            self.status.setText(f"Cannot save this setup: {exc}")
            return
        self.accept()

    def _close_scan(self, _result: int) -> None:
        self._cancelled.set()
        self._timer.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
