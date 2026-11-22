"""Export Clips dialog for the PyQt6 Model Explorer (#15150 G11).

Collects the choices, submits through the shared ``ExportJobQueue`` and polls
its status with a ``QTimer`` so the GUI thread never waits on a render. All
mapping and validation lives in :mod:`dialog_core` (the pure-data layer).
"""

from __future__ import annotations

from typing import Any, Protocol

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.tools.native_viewer_export import dialog_core as core

__all__ = ["ExportClipsDialog"]


class _Queue(Protocol):
    def submit(self, argv: list[str], command_line: str) -> dict[str, Any]: ...

    def status(self, job_id: str) -> dict[str, Any]: ...


def _checks(labels: list[str], columns: int) -> tuple[QGroupBox, list[QCheckBox]]:
    box, grid = QGroupBox(), QGridLayout()
    boxes = [QCheckBox(text) for text in labels]
    for index, check in enumerate(boxes):
        check.setChecked(True)
        grid.addWidget(check, index // columns, index % columns)
    box.setLayout(grid)
    return box, boxes


def _combo(options: tuple[tuple[str, str], ...]) -> QComboBox:
    combo = QComboBox()
    for value, label in options:
        combo.addItem(label, value)
    return combo


class ExportClipsDialog(QDialog):
    """Pick engines, speed variants, impact clip and folder; render clips."""

    def __init__(self, queue: _Queue | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        if queue is None:
            from src.sim_run.native_export_jobs import shared_queue

            queue = shared_queue()
        self._queue, self._job_id = queue, ""
        settings = core.load_dialog_settings()
        self.setWindowTitle("Export Clips")
        self.resize(settings["dialog_width"], settings["dialog_height"])
        self.timer = QTimer(self)
        self.timer.setInterval(int(settings["poll_interval_ms"]))
        self.timer.timeout.connect(self.poll_once)
        self._build()
        self.set_choices(core.default_choices())

    def _build(self) -> None:
        self.bundle_edit, self.swing_edit = QLineEdit(), QLineEdit()
        self.out_edit = QLineEdit()
        self.bundle_edit.setPlaceholderText("same-input bundle .npz")
        self.swing_edit.setPlaceholderText("name used in the clip file names")
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("Bundle", self._with_browse(self.bundle_edit, False))
        form.addRow("Swing name", self.swing_edit)
        form.addRow("Output folder", self._with_browse(self.out_edit, True))
        engines, self.engine_checks = _checks(list(core.available_engines()), columns=2)
        engines.setTitle("Engines")
        speeds, self.speed_checks = _checks(
            [f"{s:g}x" for s in core.available_speeds()], columns=3
        )
        speeds.setTitle("Speed variants")
        self.distance_combo = _combo(core.distance_unit_options())
        self.speed_combo = _combo(core.speed_unit_options())
        form.addRow("Distance unit", self.distance_combo)
        form.addRow("Speed unit", self.speed_combo)
        self.impact_check = QCheckBox("Impact clip (slow motion around impact)")
        self.state_label = QLabel("State: idle")
        self.state_label.setWordWrap(True)
        self.result_view = QPlainTextEdit()
        self.result_view.setReadOnly(True)
        self.result_view.setPlaceholderText("Clip files appear here when done.")
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.start_export)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.export_button)
        buttons.addWidget(close)
        layout = QVBoxLayout(self)
        for item in (form, engines, speeds, self.impact_check, self.state_label):
            layout.addWidget(item) if isinstance(item, QWidget) else layout.addLayout(
                item
            )
        layout.addWidget(self.result_view, 1)
        layout.addLayout(buttons)

    def _with_browse(self, edit: QLineEdit, folder: bool) -> QWidget:
        row, button = QWidget(), QPushButton("Browse...")
        button.clicked.connect(lambda: self._browse(edit, folder))
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return row

    def _browse(self, edit: QLineEdit, folder: bool) -> None:
        if folder:
            path = QFileDialog.getExistingDirectory(self, "Output Folder")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Bundle", "", "Bundle (*.npz)")
        if path:
            edit.setText(path)

    def set_choices(self, choices: core.ClipChoices) -> None:
        """Show ``choices`` in the widgets."""
        self.bundle_edit.setText(choices.bundle)
        self.swing_edit.setText(choices.swing)
        self.out_edit.setText(choices.out)
        for check, engine in zip(
            self.engine_checks, core.available_engines(), strict=True
        ):
            check.setChecked(engine in choices.engines)
        for check, speed in zip(
            self.speed_checks, core.available_speeds(), strict=True
        ):
            check.setChecked(speed in choices.speeds)
        self.impact_check.setChecked(choices.impact_clip)
        for combo, value in (
            (self.distance_combo, choices.distance_unit),
            (self.speed_combo, choices.speed_unit),
        ):
            combo.setCurrentIndex(max(0, combo.findData(value)))

    def choices(self) -> core.ClipChoices:
        """The choices currently shown."""
        return core.ClipChoices(
            bundle=self.bundle_edit.text().strip(),
            swing=self.swing_edit.text().strip(),
            out=self.out_edit.text().strip(),
            engines=tuple(
                e
                for e, c in zip(
                    core.available_engines(), self.engine_checks, strict=True
                )
                if c.isChecked()
            ),
            speeds=tuple(
                s
                for s, c in zip(core.available_speeds(), self.speed_checks, strict=True)
                if c.isChecked()
            ),
            impact_clip=self.impact_check.isChecked(),
            distance_unit=self.distance_combo.currentData(),
            speed_unit=self.speed_combo.currentData(),
        )

    def start_export(self) -> None:
        """Validate, queue the job and start polling; never renders here."""
        try:
            checked = core.validate_choices(self.choices())
        except (TypeError, ValueError) as exc:
            self.state_label.setText(f"Invalid request: {exc}")
            return
        job = self._queue.submit(checked["argv"], checked["command_line"])
        self._job_id = job["job_id"]
        self.result_view.clear()
        self.export_button.setEnabled(False)
        self.state_label.setText(f"State: {job['state']}")
        self.timer.start()

    def poll_once(self) -> None:
        """Show the job state; on a finished job list clips and stop polling."""
        job = self._queue.status(self._job_id)
        self.state_label.setText(f"State: {job['state']}")
        if job["state"] not in ("succeeded", "failed"):
            return
        self.timer.stop()
        self.export_button.setEnabled(True)
        if job["state"] == "failed":
            self.result_view.setPlainText(str(job.get("error") or "export failed"))
        else:
            self.result_view.setPlainText(
                "\n".join(core.manifest_lines(job["manifest"]))
            )
