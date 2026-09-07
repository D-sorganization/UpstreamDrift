"""Capture Rig: camera controller, recorder and pose-overlay player in one tile.

Three panels over one session directory:

* **Capture** — plan file, capture mode, view subset, UVC controls, duration;
  runs ``plan-check`` and ``record`` through the rig CLI as a child process
  so the bundle written is exactly what the terminal would write.
* **Process** — proxies, ingest with any registered estimator (MediaPipe,
  OpenPose, BODY_25 DNN), chessboard intrinsics and the joint reconstruction.
* **Review** — frame-accurate playback of any view with the ingested pose
  drawn on it, and the swing summary as a table.

Every command is built by :mod:`.commands`; every file is found by
:mod:`.session`. This module only arranges widgets and forwards clicks.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QProcess, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.motion_capture.reconstruct.skeleton import PARENTS
from src.motion_capture.rig.plan import CameraControls, CaptureMode
from src.shared.python.core.contracts import require

from . import commands
from .commands import MODE_PRESETS, PlanSelection, mode_text
from .overlay import PoseTrack, draw_pose
from .player import VideoReader, clamp_index
from .session import SessionMedia, ViewMedia, flatten_numbers, load_session

logger = logging.getLogger(__name__)

TOOL_ID = "capture_rig"
PLAN_DEFAULT = "plan default"
AUTO_EXPOSURE_CHOICES = ("camera default", "on", "off")


def _optional_float(text: str) -> float | None:
    """A float from a line edit, or ``None`` when it is blank."""
    text = text.strip()
    return float(text) if text else None


class RigProcessRunner(QWidget):
    """Runs one rig command at a time as a child process, streaming its output."""

    output = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._process.setWorkingDirectory(str(commands.repo_root()))
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._drain)
        self._process.finished.connect(self._on_finished)

    @property
    def busy(self) -> bool:
        return self._process.state() != QProcess.ProcessState.NotRunning

    def run(self, argv: Sequence[str]) -> None:
        """Start ``argv``. Precondition: nothing is running."""
        require(not self.busy, "a rig command is already running")
        require(len(argv) >= 1, "argv must name a program")
        self.output.emit("$ " + " ".join(argv) + "\n")
        self._process.start(argv[0], list(argv[1:]))

    def stop(self) -> None:
        if self.busy:
            self._process.kill()

    def _drain(self) -> None:
        data = bytes(self._process.readAllStandardOutput().data())
        self.output.emit(data.decode("utf-8", errors="replace"))

    def _on_finished(self, code: int, _status: Any) -> None:
        self.output.emit(f"[exit {code}]\n")
        self.finished.emit(int(code))


class CapturePanel(QGroupBox):
    """Plan, mode, views, UVC controls and duration for the camera commands."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Capture", parent)
        self.plan_edit = QLineEdit()
        self.session_edit = QLineEdit(str(Path.cwd() / "session"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(PLAN_DEFAULT, None)
        for mode in MODE_PRESETS:
            self.mode_combo.addItem(mode_text(mode), mode)
        self.views_edit = QLineEdit()
        self.views_edit.setPlaceholderText("all plan views, or e.g. cam_b,cam_c")
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.5, 600.0)
        self.duration_spin.setValue(10.0)
        self.exposure_edit = QLineEdit()
        self.exposure_edit.setPlaceholderText("camera default")
        self.gain_edit = QLineEdit()
        self.gain_edit.setPlaceholderText("camera default")
        self.auto_exposure_combo = QComboBox()
        self.auto_exposure_combo.addItems(AUTO_EXPOSURE_CHOICES)
        self.dry_run_check = QCheckBox("dry run (write the bundle, record nothing)")
        form = QFormLayout(self)
        form.addRow("Plan file", self._with_browse(self.plan_edit, self._pick_plan))
        form.addRow(
            "Session folder", self._with_browse(self.session_edit, self._pick_session)
        )
        form.addRow("Mode", self.mode_combo)
        form.addRow("Views", self.views_edit)
        form.addRow("Duration (s)", self.duration_spin)
        form.addRow("Exposure", self.exposure_edit)
        form.addRow("Gain", self.gain_edit)
        form.addRow("Auto-exposure", self.auto_exposure_combo)
        form.addRow(self.dry_run_check)

    def _with_browse(self, edit: QLineEdit, on_click: Callable[[], None]) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("…")
        button.setFixedWidth(32)
        button.clicked.connect(on_click)
        layout.addWidget(edit)
        layout.addWidget(button)
        return row

    def _pick_plan(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Rig plan", "", "Plan (*.json)")
        if path:
            self.plan_edit.setText(path)

    def _pick_session(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Session folder")
        if path:
            self.session_edit.setText(path)

    def controls(self) -> CameraControls:
        auto = self.auto_exposure_combo.currentText()
        return CameraControls(
            exposure=_optional_float(self.exposure_edit.text()),
            gain=_optional_float(self.gain_edit.text()),
            auto_exposure=None if auto == AUTO_EXPOSURE_CHOICES[0] else auto == "on",
        )

    def selection(self) -> PlanSelection:
        """Precondition: a plan path has been entered."""
        require(self.plan_edit.text().strip() != "", "choose a plan file first")
        mode: CaptureMode | None = self.mode_combo.currentData()
        views = tuple(v.strip() for v in self.views_edit.text().split(",") if v.strip())
        return PlanSelection(
            plan=Path(self.plan_edit.text().strip()),
            mode=mode,
            views=views,
            controls=self.controls(),
        )

    def session_dir(self) -> Path:
        require(self.session_edit.text().strip() != "", "choose a session folder")
        return Path(self.session_edit.text().strip())

    def duration_s(self) -> float:
        return float(self.duration_spin.value())

    def dry_run(self) -> bool:
        return self.dry_run_check.isChecked()


class ProcessPanel(QGroupBox):
    """Estimator, anchor and calibration inputs for the offline commands."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Process", parent)
        self.estimator_combo = QComboBox()
        for choice in commands.estimator_choices():
            label = choice.display_name + ("" if choice.available else " (unavailable)")
            self.estimator_combo.addItem(label, choice.name)
            if not choice.available and choice.hint:
                index = self.estimator_combo.count() - 1
                self.estimator_combo.setItemData(
                    index, choice.hint, Qt.ItemDataRole.ToolTipRole
                )
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(0, 1_000_000)
        self.max_frames_spin.setSpecialValueText("all")
        self.anchor_combo = QComboBox()
        for child, parent_joint in PARENTS.items():
            if parent_joint is not None:
                self.anchor_combo.addItem(f"{child} ← {parent_joint}", child)
        self.anchor_combo.setCurrentIndex(self.anchor_combo.findData("neck"))
        self.anchor_spin = QDoubleSpinBox()
        self.anchor_spin.setRange(0.05, 2.0)
        self.anchor_spin.setDecimals(3)
        self.anchor_spin.setValue(0.53)
        self.start_edit = QLineEdit()
        self.start_edit.setPlaceholderText(
            "intrinsics.json (first take) or reconstruction.json"
        )
        self.board_edit = QLineEdit("9x6")
        self.square_spin = QDoubleSpinBox()
        self.square_spin.setRange(0.001, 1.0)
        self.square_spin.setDecimals(4)
        self.square_spin.setValue(0.025)
        form = QFormLayout(self)
        form.addRow("Estimator", self.estimator_combo)
        form.addRow("Max frames", self.max_frames_spin)
        form.addRow("Anchor segment", self.anchor_combo)
        form.addRow("Anchor length (m)", self.anchor_spin)
        form.addRow("Start cameras from", self.start_edit)
        form.addRow("Board (inner corners)", self.board_edit)
        form.addRow("Square (m)", self.square_spin)

    def estimator(self) -> str:
        return str(self.estimator_combo.currentData())

    def max_frames(self) -> int | None:
        value = int(self.max_frames_spin.value())
        return value or None

    def anchor(self) -> tuple[str, float]:
        return str(self.anchor_combo.currentData()), float(self.anchor_spin.value())

    def start_file(self) -> tuple[Path | None, Path | None]:
        """``(cameras, intrinsics)``: a ``reconstruction.json`` counts as cameras."""
        text = self.start_edit.text().strip()
        require(text != "", "name the intrinsics or reconstruction file to start from")
        path = Path(text)
        if path.name.startswith("intrinsics"):
            return None, path
        return path, None

    def board(self) -> tuple[str, float]:
        return self.board_edit.text().strip(), float(self.square_spin.value())


class PlaybackPanel(QWidget):
    """One view at a time, frame-accurate, with the pose drawn on it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._reader: VideoReader | None = None
        self._track: PoseTrack | None = None
        self._index = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.step)
        self.view_combo = QComboBox()
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        self.overlay_check = QCheckBox("pose overlay")
        self.overlay_check.setChecked(True)
        self.overlay_check.toggled.connect(lambda _: self.show_frame(self._index))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.5)
        self.confidence_spin.valueChanged.connect(
            lambda _: self.show_frame(self._index)
        )
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        self.image = QLabel("no session loaded")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(320, 200)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.show_frame)
        self.status = QLabel("")
        top = QHBoxLayout()
        top.addWidget(QLabel("View"))
        top.addWidget(self.view_combo, 1)
        top.addWidget(self.overlay_check)
        top.addWidget(QLabel("min conf"))
        top.addWidget(self.confidence_spin)
        top.addWidget(self.play_button)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.slider)
        layout.addWidget(self.status)

    @property
    def frame_index(self) -> int:
        return self._index

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    def load(self, media: SessionMedia) -> None:
        """Offer every playable view; select the first."""
        self.close_media()
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        for view in media.views:
            if view.playable is not None:
                self.view_combo.addItem(view.view, view)
        self.view_combo.blockSignals(False)
        if self.view_combo.count():
            self._on_view_changed(0)
        else:
            self.image.setText("no playable recording in this session")

    def _on_view_changed(self, index: int) -> None:
        view: ViewMedia | None = self.view_combo.itemData(index)
        if view is None or view.playable is None:
            return
        self.close_media()
        self._reader = VideoReader(view.playable)
        self._track = PoseTrack.load(view.observations) if view.observations else None
        self.slider.setRange(0, max(self._reader.frame_count - 1, 0))
        rate = self._reader.fps or view.fps or 30.0
        self._timer.setInterval(max(int(1000.0 / rate), 1))
        self.show_frame(0)

    def close_media(self) -> None:
        self._timer.stop()
        self.play_button.setText("Play")
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self._track = None

    def toggle_play(self) -> None:
        if self._reader is None:
            return
        if self.playing:
            self._timer.stop()
            self.play_button.setText("Play")
        else:
            self._timer.start()
            self.play_button.setText("Pause")

    def step(self) -> None:
        if self._reader is None:
            return
        nxt = self._index + 1
        if nxt >= self._reader.frame_count:
            self.toggle_play()
            return
        self.slider.setValue(nxt)

    def show_frame(self, index: int) -> None:
        """Draw frame ``index`` (clamped) with the overlay when one exists."""
        if self._reader is None:
            return
        self._index = clamp_index(index, self._reader.frame_count)
        frame = self._reader.read(self._index)
        if frame is None:
            return
        pose = self._track.at(self._index) if self._track else None
        if pose is not None and self.overlay_check.isChecked():
            frame = draw_pose(
                frame,
                pose[0],
                pose[1],
                self._track.edges if self._track else (),
                min_confidence=float(self.confidence_spin.value()),
            )
        self._blit(frame)
        self.slider.blockSignals(True)
        self.slider.setValue(self._index)
        self.slider.blockSignals(False)
        detected = "pose" if pose is not None else "no pose"
        total = self._reader.frame_count
        self.status.setText(f"frame {self._index + 1}/{total} · {detected}")

    def _blit(self, frame_bgr: np.ndarray) -> None:
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        image = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.image.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image.setPixmap(pixmap)


class ResultsTable(QTableWidget):
    """The swing summary as ``key | value`` rows."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["metric", "value"])
        header = self.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)

    def fill(self, payload: dict[str, Any] | None) -> None:
        rows = flatten_numbers(payload) if payload else []
        self.setRowCount(len(rows))
        for r, (key, value) in enumerate(rows):
            self.setItem(r, 0, QTableWidgetItem(key))
            self.setItem(r, 1, QTableWidgetItem(value))


class CaptureRigWidget(QWidget):
    """The whole tool; embeddable in the launcher or shown in its own window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.capture = CapturePanel()
        self.process = ProcessPanel()
        self.playback = PlaybackPanel()
        self.results = ResultsTable()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.runner = RigProcessRunner(self)
        self.runner.output.connect(self._append_log)
        self.runner.finished.connect(self._on_command_finished)
        self.session_label = QLabel("no session loaded")
        self.buttons = self._buttons()
        self._layout()

    # -- layout -------------------------------------------------------------
    _ACTIONS: tuple[tuple[str, str], ...] = (
        ("plan_check", "Plan check"),
        ("record", "Record"),
        ("proxy", "Proxies"),
        ("ingest", "Ingest"),
        ("calibrate", "Calibrate intrinsics"),
        ("reconstruct", "Reconstruct"),
        ("stop", "Stop"),
        ("load", "Load session"),
    )

    def _buttons(self) -> dict[str, QPushButton]:
        out = {}
        for action, label in self._ACTIONS:
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, a=action: self.trigger(a))
            out[action] = button
        return out

    def _layout(self) -> None:
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.capture)
        left_layout.addWidget(self.process)
        row = QHBoxLayout()
        for button in self.buttons.values():
            row.addWidget(button)
        left_layout.addLayout(row)
        left_layout.addWidget(self.log, 1)
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.playback)
        right.addWidget(self.results)
        right.setStretchFactor(0, 3)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 2)
        layout = QVBoxLayout(self)
        layout.addWidget(self.session_label)
        layout.addWidget(splitter, 1)

    # -- commands -----------------------------------------------------------
    def command_for(self, action: str) -> list[str]:
        """The argv an action would run; raises on missing inputs."""
        session = self.capture.session_dir()
        if action == "plan_check":
            return commands.plan_check_command(self.capture.selection())
        if action == "record":
            return commands.record_command(
                self.capture.selection(),
                session,
                duration_s=self.capture.duration_s(),
                dry_run=self.capture.dry_run(),
            )
        if action == "proxy":
            return commands.proxy_command(session)
        if action == "ingest":
            return commands.ingest_command(
                session,
                estimator=self.process.estimator(),
                max_frames=self.process.max_frames(),
            )
        if action == "calibrate":
            board, square = self.process.board()
            return commands.calibrate_command(session, board=board, square_m=square)
        if action == "reconstruct":
            segment, metres = self.process.anchor()
            cameras, intrinsics = self.process.start_file()
            return commands.reconstruct_command(
                session,
                anchor_segment=segment,
                anchor_m=metres,
                cameras=cameras,
                intrinsics=intrinsics,
            )
        raise ValueError(f"unknown action {action!r}")

    def trigger(self, action: str) -> None:
        if action == "stop":
            self.runner.stop()
            return
        if action == "load":
            self.refresh_session()
            return
        if self.runner.busy:
            self._append_log("a command is still running; press Stop first\n")
            return
        try:
            argv = self.command_for(action)
        except (ValueError, TypeError) as exc:
            self._append_log(f"cannot run {action}: {exc}\n")
            return
        self.runner.run(argv)

    def _on_command_finished(self, code: int) -> None:
        if code == 0:
            self.refresh_session()

    def _append_log(self, text: str) -> None:
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)

    # -- session ------------------------------------------------------------
    def refresh_session(self) -> SessionMedia | None:
        """Re-read the session folder and refresh playback and results."""
        try:
            media = load_session(self.capture.session_dir())
        except (ValueError, TypeError, OSError) as exc:
            self.session_label.setText(f"session not loadable: {exc}")
            return None
        self.session_label.setText(
            f"{media.root} · plan {media.plan_name} · {len(media.views)} views"
            + (" · ingested" if media.ingested else "")
            + (f" · problems: {'; '.join(media.problems)}" if media.problems else "")
        )
        self.playback.load(media)
        self.results.fill(media.swing_summary)
        return media

    @property
    def busy(self) -> bool:
        """True while a rig command runs; the launcher asks before closing."""
        return self.runner.busy

    def shutdown(self) -> None:
        """Release the decoder and kill any running command."""
        self.playback.close_media()
        self.runner.stop()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        self.shutdown()
        super().closeEvent(event)


class CaptureRigWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Capture Rig")
        self.widget = CaptureRigWidget(self)
        self.setCentralWidget(self.widget)
        self.resize(1400, 850)


def get_dockable_ui() -> CaptureRigWindow:
    """Return the main window instance for docking in the unified launcher."""
    return CaptureRigWindow()
