"""Capture Rig: guided capture, detection, review and reconstruction in one tile.

Panels over one session directory, driven by the workflow step model
(:mod:`.workflow`):

* **Workflow** — the :class:`~.step_rail.StepRail` naming every step, where
  the operator is and the one obvious action to press next, over
  :class:`WorkflowPanel` with the current step's purpose, requirements and
  instructions.
* **Capture** — plan file, capture mode, view subset, UVC controls, duration,
  or an import of existing files; single or multi-camera alike.
* **Process** — estimator with its settings, comparison, reliability, joint
  exclusion, intrinsics and the joint reconstruction, 2-D analysis for a
  single view, export to the motion pipeline.
* **Review** — frame-accurate playback of any view and any observation set
  with the pose drawn on it (:mod:`.playback`), and the results as a table.

The live view is the middle of the tile and every control is a dock around
it (:mod:`.panes`, #9846); the log is a drawer, shut until the **Log**
toggle in the header opens it. A themed header (:mod:`.header`) carries the
session line, a status strip (cameras bound, recorder state, last take),
that toggle and the pane layout bar; the action buttons sit in
:class:`~.action_grid.ActionGrid`, grouped by workflow step. Every colour
and style comes from :mod:`.styling`, which follows the application theme.

Every command is built by :mod:`.commands`; every file is found by
:mod:`.session`. This module only arranges widgets and forwards clicks.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QResizeEvent, QShowEvent
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
    QSizePolicy,
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
from src.shared.python.theme.layout_metrics import LayoutMetrics

from . import commands, styling, workflow
from .action_grid import ActionGrid
from .annotate_widget import AnnotateDialog, BaseSet
from .commands import (
    ESTIMATOR_OPTIONS,
    MODE_PRESETS,
    OptionSpec,
    PlanSelection,
    mode_text,
)
from .header import HeaderBar, StatusStrip
from .journey import JourneyPanel
from .journey_actions import JourneyActions
from .calibration_actions import CalibrationActions
from .library_actions import LibraryActions
from .equipment_actions import EquipmentActions
from .wizard_actions import WizardActions
from . import multiview
from .layout import LayoutBar, LayoutStore, PaneExtras
from .panes import LOG_KEY, TileParts, build_host
from .responsive import LayoutMode, apply_responsive_mode, resolve_layout_mode
from .match_panel import MatchPanel, fit_model_args, reconstruct_args
from .playback import PlaybackPanel as PlaybackPanel  # re-export (moved, #9816)
from .preview import PreviewPanel
from .process_runner import RigProcessRunner
from .provenance_tab import ProvenanceTab, SourcedTable
from .record_bar import Phase, RecordBar
from .step_rail import StepRail
from .session import SessionMedia, flatten_numbers, load_session

logger = logging.getLogger(__name__)

TOOL_ID = "capture_rig"
PLAN_DEFAULT = "plan default"
AUTO_EXPOSURE_CHOICES = ("camera default", "on", "off")
ALWAYS_ENABLED = frozenset({"stop", "load", "preview"})
DEFAULT_EXPORT_LAYOUT = "three_across"  # a sensible default for the lab rig
LAB_PLAN = Path("docs/motion_capture/plans/lab_three_view_sonnet.json")
WINDOW_SIZE = (1600, 900)


def _optional_float(text: str) -> float | None:
    """A float from a line edit, or ``None`` when it is blank."""
    text = text.strip()
    return float(text) if text else None


def _csv(text: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in text.split(",") if v.strip())


class WorkflowPanel(QGroupBox):
    """The current step's purpose, requirements and instructions.

    The navigating is the :class:`~.step_rail.StepRail`'s job (#9845): this
    panel sits under the rail in the same dock and spells out the step the
    rail marks as current, so the two never show the same thing twice.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Step details", parent)
        self.guidance = QTextBrowser()
        self.guidance.setOpenExternalLinks(False)
        self._states: tuple[workflow.StepState, ...] = ()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            LayoutMetrics.SPACING_SM,
            LayoutMetrics.SPACING_SM,
            LayoutMetrics.SPACING_SM,
            LayoutMetrics.SPACING_SM,
        )
        layout.addWidget(self.guidance, 1)

    def refresh(self, states: tuple[workflow.StepState, ...]) -> None:
        """Show the guidance for the current step of ``states``.

        Postcondition: :meth:`statuses` reports exactly ``states``.
        """
        self._states = states
        current = workflow.current(states)
        self._show(states.index(current) if current is not None else 0)

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


def live_dir(session: Path) -> Path:
    """Where the recorder drops live snapshots for the tile during a take."""
    return session / ".live"


def stop_file(session: Path) -> Path:
    """The file whose appearance ends a take early."""
    return session / ".stop"


def default_plan_path() -> Path | None:
    """The lab plan shipped in docs when it exists, so Record works out of the box."""
    candidate = commands.repo_root() / LAB_PLAN
    return candidate if candidate.is_file() else None


def default_session_dir() -> Path:
    """A fresh ``sessions/<timestamp>`` under the repo for the next take."""
    from datetime import datetime

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    return commands.repo_root() / "sessions" / f"{stamp}-take"


class CapturePanel(QGroupBox):
    """Plan, mode, views and UVC controls for the camera commands."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Capture", parent)
        self.plan_edit = QLineEdit(str(default_plan_path() or ""))
        self.session_edit = QLineEdit(str(default_session_dir()))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(PLAN_DEFAULT, None)
        for mode in MODE_PRESETS:
            self.mode_combo.addItem(mode_text(mode), mode)
        self.views_edit = QLineEdit()
        self.views_edit.setPlaceholderText("all plan views, or e.g. cam_b,cam_c")
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
        form.addRow("Exposure", self.exposure_edit)
        form.addRow("Gain", self.gain_edit)
        form.addRow("Auto-exposure", self.auto_exposure_combo)
        form.addRow(self.dry_run_check)

    def _with_browse(self, edit: QLineEdit, on_click: Callable[[], None]) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LayoutMetrics.SPACING_SM)
        button = QPushButton("…")
        button.setFixedWidth(LayoutMetrics.ICON_BUTTON_WIDTH)
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

    def set_session_path(self, path: Path | str) -> None:
        self.session_edit.setText(str(path))

    def session_dir(self) -> Path:
        require(self.session_edit.text().strip() != "", "choose a session folder")
        return Path(self.session_edit.text().strip())

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

    def use_start_file(self, path: Path) -> None:
        """Apply an explicitly reviewed calibration revision."""
        self.start_edit.setText(str(path))

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
        ("preview", "Preview cameras"),
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
        ("multipicture", "Export multiview"),
        ("compare_takes", "Compare takes"),
        ("annotate", "Annotate / edit points"),
        ("stop", "Stop"),
        ("load", "Load session"),
    )

    def __init__(
        self, parent: QWidget | None = None, *, settings: QSettings | None = None
    ) -> None:
        super().__init__(parent)
        self.layout_store = LayoutStore(settings)
        self.workflow = WorkflowPanel()
        self.rail = StepRail(labels=dict(self._ACTIONS))
        self.rail.action_triggered.connect(self.trigger)
        self.capture = CapturePanel()
        self.process = ProcessPanel()
        self.match = MatchPanel()
        self.preview = PreviewPanel()
        self.preview.state_changed.connect(self._on_preview_state)
        self._resume_preview = False
        self.record_bar = RecordBar()
        self.record_bar.start_requested.connect(self.start_take)
        self.record_bar.stop_requested.connect(self.stop_take)
        self.record_bar.badge_changed.connect(self.preview.set_badge)
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
        self._take_running = False
        self.buttons = self._buttons()
        self.media: SessionMedia | None = None
        self.library_actions = LibraryActions(
            self,
            current_session=lambda: self.media.root if self.media else None,
            open_capture=self._open_library_capture,
            import_videos=self._import_library_videos,
            busy=lambda: self.runner.busy or self.record_bar.phase is not Phase.IDLE,
            settings=settings,
        )
        self.equipment_actions = EquipmentActions(
            self,
            library=self.library_actions.library,
            current_session=lambda: self.media.root if self.media else None,
            busy=lambda: self.runner.busy or self.record_bar.phase is not Phase.IDLE,
        )
        self.calibration_actions = CalibrationActions(
            self,
            selection=self.capture.selection,
            session=lambda: self.media.root if self.media else None,
            library_root=lambda: self.library_actions.library().root,
            apply=self.process.use_start_file,
            recalibrate=self._recalibrate_board,
            busy=lambda: self.runner.busy or self.record_bar.phase is not Phase.IDLE,
        )
        self.journey = JourneyPanel()
        self.journey_actions = JourneyActions(self)
        self.wizard_actions = WizardActions(self)
        self.journey.action_requested.connect(self.trigger)
        self.journey.source_requested.connect(self._show_provenance)
        self.journey.step_requested.connect(self.journey_actions.show_step)
        self.rail.step_selected.connect(self.journey_actions.show_step)
        self._layout()
        self._layout_mode: LayoutMode = resolve_layout_mode(max(self.width(), 0))
        apply_responsive_mode(self.panes, self._layout_mode)
        self._apply_workflow(None)
        self.layout_bar.restore_last()
        styling.connect_theme_changed(self.restyle)

    # -- layout -------------------------------------------------------------
    def _buttons(self) -> dict[str, QPushButton]:
        out = {}
        for action, label in self._ACTIONS:
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, a=action: self.trigger(a))
            out[action] = button
        return out

    def _layout(self) -> None:
        """Assemble the tile: the live view central, every control a dock.

        Postcondition: ``self.panes`` holds the live pane as its central
        widget and the log drawer shut (:mod:`.panes` says why).
        """
        self.action_grid = ActionGrid(self.buttons)
        self.panes = build_host(
            TileParts(
                live=self._live_pane(),
                playback=self.playback,
                results=self.results,
                rail=self._workflow_pane(),
                inputs=self._input_tabs(),
                actions=self.action_grid,
                log=self.log,
            )
        )
        self.layout_bar = LayoutBar(
            self.panes,
            self.layout_store,
            extras=PaneExtras(read=self.pane_extras, apply=self.apply_pane_extras),
        )
        self.preview.canvas.fullscreen_requested.connect(
            lambda: self.panes.fullscreen("live")
        )
        self.playback.image.fullscreen_requested.connect(
            lambda: self.panes.fullscreen("playback")
        )
        self.log_toggle = self._log_toggle()
        self.header = HeaderBar(
            self.layout_bar,
            toggles=(
                self.library_actions.library_button,
                self.wizard_actions.button,
                self.library_actions.edit_button,
                self.equipment_actions.button,
                self.calibration_actions.button,
                self.log_toggle,
            ),
        )
        self.session_label: QLabel = self.header.session
        self.status_strip: StatusStrip = self.header.status
        self.record_bar.badge_changed.connect(self._on_badge)
        layout = QVBoxLayout(self)
        margin = LayoutMetrics.SPACING_SM
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(LayoutMetrics.SPACING_SM)
        layout.addWidget(self.header)
        layout.addWidget(self.journey)
        layout.addWidget(self.panes, 1)

    def _input_tabs(self) -> QTabWidget:
        """Capture, Process and Match settings as one tabbed dock."""
        inputs = QTabWidget()
        inputs.addTab(self.capture, "Capture")
        inputs.addTab(self.process, "Process")
        inputs.addTab(self.match, "Match")
        return inputs

    def _workflow_pane(self) -> QWidget:
        """The step rail over the current step's details, in one dock."""
        pane = QSplitter(Qt.Orientation.Vertical)
        pane.addWidget(self.rail)
        pane.addWidget(self.workflow)
        pane.setStretchFactor(0, 3)
        pane.setStretchFactor(1, 2)
        return pane

    def _log_toggle(self) -> QPushButton:
        """The header button that opens and shuts the log drawer (#9846)."""
        button = QPushButton("Log")
        button.setCheckable(True)
        button.setToolTip(
            "Show the command output. It is a drawer behind the Actions "
            "pane, so opening it never takes space from the video."
        )
        button.toggled.connect(self._show_log)
        drawer = self.panes.docks[LOG_KEY]
        drawer.visibilityChanged.connect(
            lambda _shown, b=button, d=drawer: self._sync_log_toggle(b, d)
        )
        return button

    @staticmethod
    def _sync_log_toggle(button: QPushButton, drawer: QWidget) -> None:
        """Keep the header toggle telling the truth when the dock is closed."""
        wanted = not drawer.isHidden()
        if button.isChecked() != wanted:
            button.blockSignals(True)
            button.setChecked(wanted)
            button.blockSignals(False)

    def _show_log(self, shown: bool) -> None:
        """Open (and raise) or shut the log drawer."""
        self.panes.set_visible(LOG_KEY, shown)
        if shown:
            self.panes.docks[LOG_KEY].raise_()

    def pane_extras(self) -> dict[str, str]:
        """The multiview layout each pane is drawn through (#9813/#9814).

        Saved with the dock arrangement, so both come back on restart.
        """
        return {
            "preview_layout": self.preview.layout_name(),
            "playback_layout": self.playback.layout_name(),
        }

    def apply_pane_extras(self, extras: Mapping[str, str]) -> None:
        """Put a saved arrangement's multiview layouts back on the panes."""
        self.preview.set_layout_name(extras.get("preview_layout", ""))
        self.playback.set_layout_name(extras.get("playback_layout", ""))

    def restyle(self, _theme: str = "") -> None:
        """Re-read every style from the theme (connected to ``themeChanged``)."""
        self.header.restyle()
        self.action_grid.restyle()
        self.rail.restyle()
        self.preview.restyle()
        self.record_bar.restyle()

    def _live_pane(self) -> QWidget:
        """Preview tiles with the transport controls underneath, like a camera app.

        This is the tile's central widget (#9846): it declares itself
        expanding so the spare width of the window goes to the video rather
        than to the control docks around it.
        """
        pane = QWidget()
        pane.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        column = QVBoxLayout(pane)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.preview, 1)
        column.addWidget(self.record_bar)
        return pane

    # -- commands -----------------------------------------------------------
    def command_for(self, action: str) -> list[str]:
        """The argv an action would run; raises on missing inputs."""
        session = self.capture.session_dir()
        builder = {
            "plan_check": lambda: commands.plan_check_command(self.capture.selection()),
            "record": lambda: commands.record_command(
                self.capture.selection(),
                session,
                duration_s=self.record_bar.duration_s(),
                dry_run=self.capture.dry_run(),
                live_preview=live_dir(session),
                stop_file=stop_file(session),
                cameras=self.preview.camera_ids() or None,
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
            "multipicture": lambda: self._multipicture(session),
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
            self.panes.set_visible("results", True)
            self.panes.docks["results"].raise_()
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

    def _multipicture(self, session: Path) -> list[str]:
        """Composite video of the current layout: one file from several views."""
        name = self.export_layout_name()
        set_name = self.playback.current_set_name()
        out = session / f"multiview_{name}_{set_name or 'raw'}.mp4"
        return commands.multipicture_command(
            session,
            name,
            out,
            commands.MultipictureArgs(
                observation_set=set_name, speed=self.process.clip_speed()
            ),
        )

    def export_layout_name(self) -> str:
        """The layout the composite export uses: the live pane's choice.

        An unsaved edit has no name the CLI could resolve, so the default
        stands in until the operator saves the layout (#9813).
        """
        name = self.preview.layout_name()
        editable = name and name != multiview.EDITED
        return name if editable else DEFAULT_EXPORT_LAYOUT

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
        self.journey_actions.trigger(action)

    def _on_command_finished(self, code: int) -> None:
        if self._take_running:
            self._take_running = False
            self.status_strip.set_last_take(code)
        if code == 0:
            self.capture.pending_import = []
            self.refresh_session()
        self.preview.stop_watching()
        self.record_bar.recording_finished()
        if self._resume_preview:
            self._resume_preview = False
            self.toggle_preview(on=True)
        self.library_actions.command_finished(code)
        self.journey_actions.complete(code)

    # -- recording ------------------------------------------------------------
    def start_take(self) -> None:
        """Launch the recorder; the preview switches to the recorder's snapshots.

        The cameras belong to ffmpeg during a take, so the direct preview is
        released first and resumed when the take is written.
        """
        if self.runner.busy:
            self._append_log("a command is still running; press Stop first\n")
            self.record_bar.recording_finished()
            return
        try:
            argv = self.command_for("record")
            session = self.capture.session_dir()
        except (ValueError, TypeError) as exc:
            self.journey.notice(f"Cannot record: {exc}", retry="record")
            self._append_log(f"cannot record: {exc}\n")
            self.record_bar.recording_finished()
            return
        views = self.preview.views() or tuple(self.capture.selection().views)
        self._resume_preview = self.preview.active or not views
        self.preview.stop()
        live_dir(session).mkdir(parents=True, exist_ok=True)
        stop_file(session).unlink(missing_ok=True)
        if views:
            self.preview.watch_snapshots(live_dir(session), views)
        self._take_running = True
        self.journey_actions.begin("record")
        self.runner.run(argv)
        self.record_bar.recording_started()

    def stop_take(self) -> None:
        """End the running take early (the recorder sees the stop file)."""
        try:
            stop_file(self.capture.session_dir()).write_text("stop", encoding="utf-8")
        except (ValueError, OSError) as exc:
            self._append_log(f"cannot stop the take: {exc}\n")

    def toggle_preview(self, on: bool | None = None) -> None:
        """Start (or stop) the live preview of the Capture panel's plan."""
        want = (not self.preview.active) if on is None else on
        if not want:
            self.preview.stop()
            return
        try:
            self.preview.start(self.capture.selection())
        except (ValueError, TypeError) as exc:
            self.journey.notice(f"Cannot preview: {exc}", retry="preview")
            self._append_log(f"cannot preview: {exc}\n")

    def _on_preview_state(self, active: bool) -> None:
        self.buttons["preview"].setText("Stop preview" if active else "Preview cameras")
        bound = len(self.preview.camera_ids()) if active else 0
        self.status_strip.set_cameras(bound)
        if self.journey_actions.active_action is None:
            self.journey.notice(self.preview.status_text())

    def _on_badge(self, readout: str) -> None:
        self.status_strip.set_recording(self.record_bar.phase, readout)
        self.library_actions.refresh()
        self.equipment_actions.refresh()
        self.calibration_actions.refresh()

    def _append_log(self, text: str) -> None:
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(text)

    # -- session ------------------------------------------------------------
    def _open_library_capture(self, root: Path) -> None:
        self.capture.set_session_path(root)
        self.refresh_session()

    def _import_library_videos(self, target: Path) -> None:
        self.capture.choose_import_files()
        if self.capture.pending_import:
            self.capture.set_session_path(target)
            self.trigger("import")

    def refresh_session(self) -> SessionMedia | None:
        """Re-read the session folder and refresh every panel."""
        try:
            media = load_session(self.capture.session_dir())
        except (ValueError, TypeError, OSError) as exc:
            self.media = None
            self.journey_actions.clear_capture(str(exc))
            self.session_label.setText(f"session not loadable: {exc}")
            self._apply_workflow(None)
            return None
        self.media = media
        self.journey.set_capture(media)
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

    def _recalibrate_board(self, path: Path) -> None:
        self._open_library_capture(path)
        if self.media is not None and self.media.root == path:
            self.trigger("calibrate")

    def _apply_workflow(self, media: SessionMedia | None) -> None:
        self.library_actions.refresh()
        self.equipment_actions.refresh()
        self.calibration_actions.refresh()
        states = workflow.evaluate(media)
        self.workflow.refresh(states)
        enabled = workflow.enabled_actions(states) | ALWAYS_ENABLED
        busy = self.journey_actions.active_action is not None
        if busy:
            enabled = frozenset({"stop"})
        self.journey.set_steps(states, busy=busy)
        hints = workflow.action_hints(states, ALWAYS_ENABLED)
        if busy:
            hints = {
                key: "Wait for the current action, or press Stop."
                for key, _ in self._ACTIONS
            }
        self.rail.set_states(states, enabled=enabled, hints=hints)
        for action, button in self.buttons.items():
            button.setEnabled(action in enabled)
            button.setToolTip(hints.get(action, ""))

    def enabled_actions(self) -> frozenset[str]:
        return frozenset(a for a, b in self.buttons.items() if b.isEnabled())

    @property
    def busy(self) -> bool:
        """True while a rig command runs; the launcher asks before closing."""
        return self.runner.busy

    def shutdown(self) -> None:
        """Release the cameras and the decoder; kill any running command."""
        self.layout_bar.save_last()
        self.panes.redock_all()
        self.preview.stop()
        self.playback.close_media()
        self.runner.stop()

    @property
    def layout_mode(self) -> LayoutMode:
        """The active layout mode (compact or roomy)."""
        return self._layout_mode

    def resizeEvent(self, a0: QResizeEvent | None) -> None:  # noqa: N802 - Qt override
        """Adapt layout density when the window crosses the width threshold."""
        super().resizeEvent(a0)
        if a0 is not None:
            new_mode = resolve_layout_mode(max(a0.size().width(), 0))
            if new_mode != self._layout_mode:
                self._layout_mode = new_mode
                apply_responsive_mode(self.panes, new_mode)

    def showEvent(self, a0: QShowEvent | None) -> None:  # noqa: N802 - Qt override
        """Ensure initial responsive density matches shown width."""
        super().showEvent(a0)
        mode = resolve_layout_mode(max(self.width(), 0))
        self._layout_mode = mode
        apply_responsive_mode(self.panes, mode)

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        self.shutdown()
        super().closeEvent(event)


class CaptureRigWindow(QMainWindow):
    """The tile in its own window (``python -m src.tools.capture_rig``).

    ``autostart_preview`` opens the planned cameras as soon as the window
    shows, so the operator sees the bay without pressing anything.
    """

    def __init__(
        self, parent: QWidget | None = None, *, autostart_preview: bool = False
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Capture Rig")
        self.widget = CaptureRigWidget(self)
        self.setCentralWidget(self.widget)
        self.resize(*WINDOW_SIZE)
        styling.apply_theme(self)
        self._autostart_preview = autostart_preview

    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if self._autostart_preview:
            self._autostart_preview = False
            self.widget.toggle_preview(on=True)


def get_dockable_ui() -> CaptureRigWindow:
    """Return the main window instance for docking in the unified launcher."""
    return CaptureRigWindow()
