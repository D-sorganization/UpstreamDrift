"""Capture Rig: guided capture, detection, review and reconstruction in one tile.

Panels over one session directory, driven by the workflow step model
(:mod:`.workflow`):

* **Workflow** — the steps with their status (done / ready / blocked with
  the reason / skipped for this session), the current step's requirements
  and instructions, and only the applicable actions enabled.
* **Capture** — plan file, capture mode, view subset, UVC controls, duration,
  or an import of existing files; single or multi-camera alike.
* **Process** — estimator with its settings, comparison, reliability, joint
  exclusion, intrinsics and the joint reconstruction, 2-D analysis for a
  single view, export to the motion pipeline.
* **Review** — frame-accurate playback of any view and any observation set
  with the pose drawn on it, and the results as a table.

Every command is built by :mod:`.commands`; every file is found by
:mod:`.session`. This module only arranges widgets and forwards clicks.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.motion_capture.reconstruct.measurements import TAPE_GUIDE, parse_measurement
from src.motion_capture.rig.plan import CameraControls, CaptureMode
from src.shared.python.core.contracts import require

from . import commands, workflow
from .commands import ESTIMATOR_OPTIONS, MODE_PRESETS, OptionSpec, PlanSelection
from .commands import mode_text
from .annotate_widget import AnnotateDialog, BaseSet
from .match_panel import MatchPanel, fit_model_args, reconstruct_args
from .overlay import PoseTrack, draw_pose
from .overlay_box import VariantOverlayBox
from .overlay_render import render_frame
from .provenance_tab import ProvenanceTab, SourcedTable
from .player import VideoReader, clamp_index
from .session import SessionMedia, ViewMedia, flatten_numbers, load_session

logger = logging.getLogger(__name__)

TOOL_ID = "capture_rig"
PLAN_DEFAULT = "plan default"
AUTO_EXPOSURE_CHOICES = ("camera default", "on", "off")
STATUS_GLYPH = {
    workflow.Status.DONE: "✓",
    workflow.Status.READY: "▶",
    workflow.Status.BLOCKED: "○",
    workflow.Status.SKIPPED: "–",
}
ALWAYS_ENABLED = frozenset({"stop", "load"})
BUTTONS_PER_ROW = 5


def _optional_float(text: str) -> float | None:
    """A float from a line edit, or ``None`` when it is blank."""
    text = text.strip()
    return float(text) if text else None


def _csv(text: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in text.split(",") if v.strip())


class RigProcessRunner(QWidget):
    """Runs one rig command at a time as a child process, streaming its output."""

    output = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._process.setWorkingDirectory(str(commands.repo_root()))
        env = QProcessEnvironment()
        for key, value in commands.child_environment().items():
            env.insert(key, value)
        self._process.setProcessEnvironment(env)
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


class WorkflowPanel(QGroupBox):
    """The guided steps: status list plus the selected step's guidance."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Workflow", parent)
        self.steps = QListWidget()
        self.steps.currentRowChanged.connect(self._show)
        self.guidance = QTextBrowser()
        self.guidance.setOpenExternalLinks(False)
        self._states: tuple[workflow.StepState, ...] = ()
        layout = QVBoxLayout(self)
        layout.addWidget(self.steps, 1)
        layout.addWidget(self.guidance, 2)

    def refresh(self, states: tuple[workflow.StepState, ...]) -> None:
        """Repaint the list; select the current step."""
        self._states = states
        self.steps.blockSignals(True)
        self.steps.clear()
        for state in states:
            text = f"{STATUS_GLYPH[state.status]}  {state.step.title}"
            if state.reason:
                text += f"  ({state.reason})"
            self.steps.addItem(QListWidgetItem(text))
        self.steps.blockSignals(False)
        current = workflow.current(states)
        row = states.index(current) if current is not None else 0
        self.steps.setCurrentRow(row)
        self._show(row)

    def _show(self, row: int) -> None:
        if not (0 <= row < len(self._states)):
            self.guidance.clear()
            return
        state = self._states[row]
        step = state.step
        html = [f"<h3>{step.title}</h3><p>{step.purpose}</p>"]
        html.append(f"<p><b>Status:</b> {state.status.value}")
        if state.reason:
            html.append(f" &mdash; {state.reason}")
        html.append("</p><p><b>You need</b></p><ul>")
        html.extend(f"<li>{r}</li>" for r in step.requirements)
        html.append("</ul><p><b>Do</b></p><ol>")
        html.extend(f"<li>{s}</li>" for s in step.instructions)
        html.append("</ol>")
        self.guidance.setHtml("".join(html))

    def statuses(self) -> dict[str, workflow.Status]:
        return {s.step.key: s.status for s in self._states}


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
        self.pending_import: list[tuple[str, Path]] = []
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

    def choose_import_files(self) -> list[tuple[str, Path]]:
        """Ask for video files; views are named after the file stems."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Videos to import", "", "Video (*.mp4 *.mkv *.avi *.mov)"
        )
        self.pending_import = [(Path(p).stem, Path(p)) for p in paths]
        return self.pending_import

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
        return PlanSelection(
            plan=Path(self.plan_edit.text().strip()),
            mode=mode,
            views=_csv(self.views_edit.text()),
            controls=self.controls(),
        )

    def session_dir(self) -> Path:
        require(self.session_edit.text().strip() != "", "choose a session folder")
        return Path(self.session_edit.text().strip())

    def duration_s(self) -> float:
        return float(self.duration_spin.value())

    def dry_run(self) -> bool:
        return self.dry_run_check.isChecked()


class EstimatorSettings(QWidget):
    """One form per estimator built from :data:`ESTIMATOR_OPTIONS`."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fields: dict[str, dict[str, QWidget]] = {}
        self._forms: dict[str, QWidget] = {}
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        for name, specs in ESTIMATOR_OPTIONS.items():
            form_widget = QWidget()
            form = QFormLayout(form_widget)
            form.setContentsMargins(0, 0, 0, 0)
            self._fields[name] = {}
            for spec in specs:
                field = self._field(spec)
                self._fields[name][spec.name] = field
                form.addRow(f"{spec.name} ({spec.help})", field)
            form_widget.hide()
            self._forms[name] = form_widget
            self._layout.addWidget(form_widget)

    @staticmethod
    def _field(spec: OptionSpec) -> QWidget:
        if spec.kind == "bool":
            box = QCheckBox()
            box.setChecked(bool(spec.default))
            return box
        if spec.kind == "int":
            spin = QSpinBox()
            spin.setRange(int(spec.minimum or 0), int(spec.maximum or 10_000))
            spin.setValue(int(spec.default))
            return spin
        if spec.kind == "float":
            dspin = QDoubleSpinBox()
            dspin.setDecimals(3)
            dspin.setSingleStep(0.05)
            dspin.setRange(float(spec.minimum or 0.0), float(spec.maximum or 1e6))
            dspin.setValue(float(spec.default))
            return dspin
        return QLineEdit(str(spec.default))

    def show_for(self, estimator: str) -> None:
        for name, form in self._forms.items():
            form.setVisible(name == estimator)

    def values(self, estimator: str) -> dict[str, float | int | bool | str]:
        """The current settings for ``estimator`` (empty for one without a form)."""
        out: dict[str, float | int | bool | str] = {}
        for key, field in self._fields.get(estimator, {}).items():
            if isinstance(field, QCheckBox):
                out[key] = field.isChecked()
            elif isinstance(field, QSpinBox):
                out[key] = int(field.value())
            elif isinstance(field, QDoubleSpinBox):
                out[key] = float(field.value())
            elif isinstance(field, QLineEdit):
                out[key] = field.text().strip()
        return out


class ProcessPanel(QGroupBox):
    """Estimator, settings, joints, anchor and calibration inputs."""

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
        self.settings = EstimatorSettings()
        self.estimator_combo.currentIndexChanged.connect(self._on_estimator)
        self.separate_set_check = QCheckBox("write to observations_<estimator>")
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(0, 1_000_000)
        self.max_frames_spin.setSpecialValueText("all")
        self.exclude_edit = QLineEdit()
        self.exclude_edit.setPlaceholderText("joints to leave out of the fit")
        self.measurements_edit = QLineEdit()
        self.measurements_edit.setPlaceholderText(
            "shank=0.42, forearm=0.26, ...  (first sets the scale; both sides)"
        )
        self.measurements_edit.setToolTip(
            "\n".join(f"{name}: {how}" for name, how in TAPE_GUIDE.items())
        )
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
        form.addRow(self.settings)
        form.addRow(self.separate_set_check)
        form.addRow("Max frames", self.max_frames_spin)
        form.addRow("Exclude joints", self.exclude_edit)
        form.addRow("Measured segments (m)", self.measurements_edit)
        form.addRow("Start cameras from", self.start_edit)
        form.addRow("Board (inner corners)", self.board_edit)
        form.addRow("Square (m)", self.square_spin)
        self.clip_speed_spin = QDoubleSpinBox()
        self.clip_speed_spin.setRange(0.05, 1.0)
        self.clip_speed_spin.setSingleStep(0.05)
        self.clip_speed_spin.setValue(0.25)
        self.other_session_edit = QLineEdit()
        self.other_session_edit.setPlaceholderText("session folder to compare against")
        self.other_view_edit = QLineEdit()
        self.other_view_edit.setPlaceholderText("its view (default: same name)")
        self.align_combo = QComboBox()
        self.align_combo.addItems(["top", "address", "peak", "finish"])
        self.model_combo = QComboBox()
        for name, description in commands.model_choices():
            self.model_combo.addItem(name, name)
            self.model_combo.setItemData(
                self.model_combo.count() - 1, description, Qt.ItemDataRole.ToolTipRole
            )
        self.fit_lengths_check = QCheckBox("learn the model's segment lengths")
        self.body_mass_spin = QDoubleSpinBox()
        self.body_mass_spin.setRange(20.0, 250.0)
        self.body_mass_spin.setValue(80.0)
        form.addRow("Model", self.model_combo)
        form.addRow(self.fit_lengths_check)
        form.addRow("Body mass (kg)", self.body_mass_spin)
        form.addRow("Clip speed (1 = real time)", self.clip_speed_spin)
        form.addRow("Compare with session", self.other_session_edit)
        form.addRow("Compare view", self.other_view_edit)
        form.addRow("Align on", self.align_combo)
        self._on_estimator(self.estimator_combo.currentIndex())

    def _on_estimator(self, _index: int) -> None:
        self.settings.show_for(self.estimator())

    def estimator(self) -> str:
        return str(self.estimator_combo.currentData())

    def options(self) -> dict[str, float | int | bool | str]:
        return self.settings.values(self.estimator())

    def ingest_out(self, session: Path) -> Path | None:
        if self.separate_set_check.isChecked():
            return session / f"observations_{self.estimator()}"
        return None

    def max_frames(self) -> int | None:
        value = int(self.max_frames_spin.value())
        return value or None

    def exclude_joints(self) -> tuple[str, ...]:
        return _csv(self.exclude_edit.text())

    def suggest_exclusions(self, joints: Sequence[str]) -> None:
        if not self.exclude_edit.text().strip():
            self.exclude_edit.setText(",".join(joints))

    def measurements(self) -> tuple[str, ...]:
        """``NAME=METRES`` entries, validated through the measurement parser."""
        items = _csv(self.measurements_edit.text())
        require(bool(items), "enter at least one measured segment, e.g. shank=0.42")
        for item in items:
            parse_measurement(item)  # raises on an unknown name or bad length
        return items

    def start_file(self) -> tuple[Path | None, Path | None]:
        """``(cameras, intrinsics)``: a ``reconstruction.json`` counts as cameras."""
        text = self.start_edit.text().strip()
        require(text != "", "name the intrinsics or reconstruction file to start from")
        path = Path(text)
        if path.name.startswith("intrinsics"):
            return None, path
        return path, None

    def suggest_start(self, intrinsics: Path | None) -> None:
        if intrinsics is not None and not self.start_edit.text().strip():
            self.start_edit.setText(str(intrinsics))

    def board(self) -> tuple[str, float]:
        return self.board_edit.text().strip(), float(self.square_spin.value())

    def clip_speed(self) -> float:
        return float(self.clip_speed_spin.value())

    def model_name(self) -> str:
        return str(self.model_combo.currentData())

    def fit_lengths(self) -> bool:
        return self.fit_lengths_check.isChecked()

    def body_mass(self) -> float:
        return float(self.body_mass_spin.value())

    def other_session(self) -> Path:
        text = self.other_session_edit.text().strip()
        require(text != "", "name the session folder to compare against")
        return Path(text)

    def other_view(self) -> str | None:
        return self.other_view_edit.text().strip() or None

    def align_event(self) -> str:
        return self.align_combo.currentText()


class PlaybackPanel(QWidget):
    """One view and one observation set at a time, frame-accurate, with overlay."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._reader: VideoReader | None = None
        self._track: PoseTrack | None = None
        self._index = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.step)
        self.view_combo = QComboBox()
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        self.set_combo = QComboBox()
        self.set_combo.currentIndexChanged.connect(self._on_set_changed)
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
        self.variants = VariantOverlayBox()
        self.variants.changed.connect(lambda: self.show_frame(self._index))
        self.image = QLabel("no session loaded")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(320, 200)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.show_frame)
        self.status = QLabel("")
        top = QHBoxLayout()
        top.addWidget(QLabel("View"))
        top.addWidget(self.view_combo, 1)
        top.addWidget(QLabel("Set"))
        top.addWidget(self.set_combo, 1)
        top.addWidget(self.overlay_check)
        top.addWidget(QLabel("min conf"))
        top.addWidget(self.confidence_spin)
        top.addWidget(self.play_button)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.variants)
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
        self.variants.load(media)
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

    def _current_view(self) -> ViewMedia | None:
        return self.view_combo.currentData()

    def current_view_name(self) -> str | None:
        view = self._current_view()
        return None if view is None else view.view

    def current_set_name(self) -> str | None:
        text = self.set_combo.currentText()
        return text or None

    def _on_view_changed(self, index: int) -> None:
        view: ViewMedia | None = self.view_combo.itemData(index)
        if view is None or view.playable is None:
            return
        self.close_media()
        self._reader = VideoReader(view.playable)
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        for name, path in (view.observation_sets or {}).items():
            self.set_combo.addItem(name, path)
        if not self.set_combo.count() and view.observations is not None:
            self.set_combo.addItem("observations", view.observations)
        self.set_combo.blockSignals(False)
        self._load_track()
        self.slider.setRange(0, max(self._reader.frame_count - 1, 0))
        rate = self._reader.fps or view.fps or 30.0
        self._timer.setInterval(max(int(1000.0 / rate), 1))
        self.show_frame(0)

    def _on_set_changed(self, _index: int) -> None:
        self._load_track()
        self.show_frame(self._index)

    def _load_track(self) -> None:
        path: Path | None = self.set_combo.currentData()
        self._track = PoseTrack.load(path) if path is not None else None

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
        view_name = self.current_view_name()
        tracks = self.variants.tracks_for(view_name) if view_name else ()
        if tracks:
            frame = render_frame(frame, tracks, self._index)
        self._blit(frame)
        self.slider.blockSignals(True)
        self.slider.setValue(self._index)
        self.slider.blockSignals(False)
        detected = "pose" if pose is not None else "no pose"
        total = self._reader.frame_count
        extra = f" · {self.variants.error}" if self.variants.error else ""
        self.status.setText(f"frame {self._index + 1}/{total} · {detected}{extra}")

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
    """Any result payload as ``key | value`` rows."""

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


def _reliability_rows(payload: dict[str, Any]) -> dict[str, Any]:
    """``{joint: "grade score"}`` from a reliability report."""
    out: dict[str, Any] = {}
    for joint in payload.get("joints", []):
        score = joint.get("score")
        grade = "unknown" if score is None else _grade(score)
        out[joint["joint"]] = f"{grade} ({score:.2f})" if score is not None else grade
    if payload.get("recommended_exclusions"):
        out["recommended_exclusions"] = ",".join(payload["recommended_exclusions"])
    return out


def _kinetics_rows(media: SessionMedia) -> dict[str, Any] | None:
    """Replay check plus the model ranking, flattened for the results table."""
    out: dict[str, Any] = {}
    if media.kinetics:
        out["replay"] = {
            k: v
            for k, v in media.kinetics.items()
            if k in ("replay_max_abs_error", "replay_worst_dof", "body_mass_kg")
        }
    if media.model_comparison:
        out["ranking"] = {
            f"{i + 1}. {s['model']}": f"rms {s['rms_mm']:.1f} mm, score {s['score']:.3f}"
            for i, s in enumerate(media.model_comparison.get("ranking", []))
        }
    return out or None


def _grade(score: float) -> str:
    return "reliable" if score >= 0.75 else ("usable" if score >= 0.5 else "weak")


class CaptureRigWidget(QWidget):
    """The whole tool; embeddable in the launcher or shown in its own window."""

    _ACTIONS: tuple[tuple[str, str], ...] = (
        ("plan_check", "Plan check"),
        ("record", "Record"),
        ("import", "Import videos"),
        ("proxy", "Proxies"),
        ("ingest", "Ingest"),
        ("compare", "Compare"),
        ("reliability", "Reliability"),
        ("calibrate", "Calibrate intrinsics"),
        ("reconstruct", "Reconstruct"),
        ("fit_model", "Fit model"),
        ("kinetics", "Kinetics"),
        ("compare_models", "Compare models"),
        ("analyze", "Analyze 2-D"),
        ("export", "Export"),
        ("clip", "Export clip"),
        ("compare_takes", "Compare takes"),
        ("annotate", "Annotate / edit points"),
        ("stop", "Stop"),
        ("load", "Load session"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.workflow = WorkflowPanel()
        self.capture = CapturePanel()
        self.process = ProcessPanel()
        self.match = MatchPanel()
        self.playback = PlaybackPanel()
        self.results = QTabWidget()
        self.swing_table = SourcedTable()
        self.analysis_table = SourcedTable()
        self.reliability_table = SourcedTable()
        self.results.addTab(self.swing_table, "Swing (3-D)")
        self.results.addTab(self.analysis_table, "Analysis (2-D)")
        self.model_table = SourcedTable()
        self.results.addTab(self.reliability_table, "Reliability")
        self.kinetics_table = SourcedTable()
        self.results.addTab(self.model_table, "Model fit")
        self.results.addTab(self.kinetics_table, "Kinetics")
        self.provenance = ProvenanceTab()
        self.results.addTab(self.provenance, "Provenance")
        for table in (
            self.swing_table,
            self.analysis_table,
            self.reliability_table,
            self.model_table,
            self.kinetics_table,
        ):
            table.source_selected.connect(self._show_provenance)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.runner = RigProcessRunner(self)
        self.runner.output.connect(self._append_log)
        self.runner.finished.connect(self._on_command_finished)
        self.session_label = QLabel("no session loaded")
        self.buttons = self._buttons()
        self._layout()
        self.media: SessionMedia | None = None
        self._apply_workflow(None)

    # -- layout -------------------------------------------------------------
    def _buttons(self) -> dict[str, QPushButton]:
        out = {}
        for action, label in self._ACTIONS:
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, a=action: self.trigger(a))
            out[action] = button
        return out

    def _layout(self) -> None:
        inputs = QTabWidget()
        inputs.addTab(self.capture, "Capture")
        inputs.addTab(self.process, "Process")
        inputs.addTab(self.match, "Match")
        left = QSplitter(Qt.Orientation.Vertical)
        left.addWidget(self.workflow)
        left.addWidget(inputs)
        buttons = QWidget()
        grid = QGridLayout(buttons)
        grid.setContentsMargins(0, 0, 0, 0)
        for i, button in enumerate(self.buttons.values()):
            grid.addWidget(button, i // BUTTONS_PER_ROW, i % BUTTONS_PER_ROW)
        middle = QWidget()
        mid_layout = QVBoxLayout(middle)
        mid_layout.addWidget(buttons)
        mid_layout.addWidget(self.log, 1)
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.playback)
        right.addWidget(self.results)
        right.setStretchFactor(0, 3)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(middle)
        splitter.addWidget(right)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([420, 520, 660])
        layout = QVBoxLayout(self)
        layout.addWidget(self.session_label)
        layout.addWidget(splitter, 1)

    # -- commands -----------------------------------------------------------
    def command_for(self, action: str) -> list[str]:
        """The argv an action would run; raises on missing inputs."""
        session = self.capture.session_dir()
        builder = {
            "plan_check": lambda: commands.plan_check_command(self.capture.selection()),
            "record": lambda: commands.record_command(
                self.capture.selection(),
                session,
                duration_s=self.capture.duration_s(),
                dry_run=self.capture.dry_run(),
            ),
            "import": lambda: commands.import_command(
                session, self.capture.pending_import
            ),
            "proxy": lambda: commands.proxy_command(session),
            "ingest": lambda: commands.ingest_command(
                session,
                estimator=self.process.estimator(),
                max_frames=self.process.max_frames(),
                options=self.process.options(),
                out=self.process.ingest_out(session),
            ),
            "compare": lambda: commands.compare_command(
                session, max_frames=self.process.max_frames()
            ),
            "reliability": lambda: commands.reliability_command(session),
            "calibrate": lambda: self._calibrate(session),
            "reconstruct": lambda: self._reconstruct(session),
            "analyze": lambda: commands.analyze_command(session),
            "fit_model": lambda: fit_model_args(
                session,
                self.match.selection(),
                model=self.process.model_name(),
                fit_lengths=self.process.fit_lengths(),
            ),
            "kinetics": lambda: commands.kinetics_command(
                session,
                model=self.process.model_name(),
                body_mass_kg=self.process.body_mass(),
                variant=self.match.selection().name,
            ),
            "compare_models": lambda: commands.compare_models_command(
                session,
                fit_lengths=self.process.fit_lengths(),
                variant=self.match.selection().name,
            ),
            "export": lambda: commands.export_command(
                session, variant=self.match.selection().name
            ),
            "clip": lambda: self._clip(session),
            "compare_takes": lambda: self._compare_takes(session),
        }
        if action not in builder:
            raise ValueError(f"unknown action {action!r}")
        return builder[action]()

    def _calibrate(self, session: Path) -> list[str]:
        board, square = self.process.board()
        return commands.calibrate_command(session, board=board, square_m=square)

    def _reconstruct(self, session: Path) -> list[str]:
        cameras, intrinsics = self.process.start_file()
        return reconstruct_args(
            session,
            self.match.selection(),
            measurements=self.process.measurements(),
            cameras=cameras,
            intrinsics=intrinsics,
            exclude_joints=self.process.exclude_joints(),
        )

    def _show_provenance(self, source: object) -> None:
        if self.media is not None and isinstance(source, Path):
            self.provenance.show_path(self.media.root, source)
            self.results.setCurrentWidget(self.provenance)

    def annotate_dialog(self) -> AnnotateDialog | None:
        """The Annotate/edit dialog for the player's view and set (not shown)."""
        if self.media is None:
            return None
        name = self.playback.current_view_name()
        if name is None:
            return None
        view = self.media.view(name)
        if view.playable is None:
            return None
        set_name = self.playback.current_set_name()
        base_file = (view.observation_sets or {}).get(set_name) if set_name else None
        base = BaseSet(set_name, base_file) if set_name and base_file else None
        return AnnotateDialog(
            self.media.root, name, view.playable, base=base, parent=self
        )

    def _clip(self, session: Path) -> list[str]:
        view = self.playback.current_view_name()
        require(view is not None, "load a session and pick a view first")
        assert view is not None
        set_name = self.playback.current_set_name()
        out = session / f"clip_{view}_{set_name or 'raw'}.mp4"
        return commands.clip_command(
            session,
            view,
            out,
            speed=self.process.clip_speed(),
            observation_set=set_name,
        )

    def _compare_takes(self, session: Path) -> list[str]:
        view = self.playback.current_view_name()
        require(view is not None, "load a session and pick a view first")
        assert view is not None
        other = self.process.other_session()
        other_view = self.process.other_view() or view
        out = session / f"compare_{view}_vs_{other.name}_{other_view}.mp4"
        return commands.compare_takes_command(
            session, view, other, other_view, out, align=self.process.align_event()
        )

    def trigger(self, action: str) -> None:
        if action == "stop":
            self.runner.stop()
            return
        if action == "load":
            self.refresh_session()
            return
        if action == "annotate":
            dialog = self.annotate_dialog()
            if dialog is None:
                self._append_log("load a session and pick a playable view first\n")
                return
            dialog.exec()
            self.refresh_session()
            return
        if action == "import" and not self.capture.pending_import:
            self.capture.choose_import_files()
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
            self.capture.pending_import = []
            self.refresh_session()

    def _append_log(self, text: str) -> None:
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)

    # -- session ------------------------------------------------------------
    def refresh_session(self) -> SessionMedia | None:
        """Re-read the session folder and refresh every panel."""
        try:
            media = load_session(self.capture.session_dir())
        except (ValueError, TypeError, OSError) as exc:
            self.session_label.setText(f"session not loadable: {exc}")
            self._apply_workflow(None)
            return None
        self.media = media
        kind = "single-camera" if len(media.views) == 1 else "multi-camera"
        self.session_label.setText(
            f"{media.root} · plan {media.plan_name} · {len(media.views)} views ({kind})"
            + (" · ingested" if media.ingested else "")
            + (f" · problems: {'; '.join(media.problems)}" if media.problems else "")
        )
        self.playback.load(media)
        self.match.load(media)
        self.results.setCurrentIndex(1 if len(media.views) == 1 else 0)
        recon = media.root / "reconstruct"
        self.swing_table.fill(
            media.swing_summary, recon / "session_reconstruction.json"
        )
        self.analysis_table.fill(media.analysis_2d)
        self.model_table.fill(
            media.model_fit, media.root / "model" / "joint_angles.json"
        )
        self.kinetics_table.fill(
            _kinetics_rows(media), media.root / "model" / "kinetics.json"
        )
        self.reliability_table.fill(
            _reliability_rows(media.reliability) if media.reliability else None,
            media.root / "reliability.json",
        )
        if media.reliability:
            self.process.suggest_exclusions(
                media.reliability.get("recommended_exclusions", [])
            )
        self.process.suggest_start(media.intrinsics)
        self._apply_workflow(media)
        return media

    def _apply_workflow(self, media: SessionMedia | None) -> None:
        states = workflow.evaluate(media)
        self.workflow.refresh(states)
        enabled = workflow.enabled_actions(states) | ALWAYS_ENABLED
        for action, button in self.buttons.items():
            button.setEnabled(action in enabled)

    def enabled_actions(self) -> frozenset[str]:
        return frozenset(a for a, b in self.buttons.items() if b.isEnabled())

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
        self.resize(1600, 900)


def get_dockable_ui() -> CaptureRigWindow:
    """Return the main window instance for docking in the unified launcher."""
    return CaptureRigWindow()
