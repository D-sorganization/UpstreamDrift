"""Subject anthropometrics calibration dialog (issue #4820).

A modal Qt dialog that walks the user through producing a fully
validated :class:`SubjectAnthropometrics` from a single C3D mocap
file. The dialog wraps the existing building blocks already on
``main``:

* :func:`anthropometrics.read_c3d_subject_metadata` — auto-fills
  height / mass from ``SUBJECT_INFO`` / ``PROCESSING`` when present.
* :class:`anthropometrics.estimators.DeLevaEstimator` /
  :class:`DempsterEstimator` / :class:`ZatsiorskyEstimator` —
  produce the canonical record from subject scalars.
* :class:`anthropometrics.SegmentPropertiesPanel` — displays the
  currently-selected segment as the user iterates.
* :func:`anthropometrics.save_subject` — persists the chosen record
  to ``~/.golf_modeling_suite/subjects/<id>.json`` (default location).
* :func:`anthropometrics.pipeline.run_pipeline` — re-runs the full
  pipeline against the chosen physics engine.

Design by Contract
------------------
* Spinbox widgets are bounded to physically realistic ranges
  (height 0.5–2.5 m, mass 10–300 kg) and the **Compute** button is
  disabled whenever either spinbox is invalid.
* :meth:`SubjectCalibrationDialog.open_for_mocap` raises a clear
  ``TypeError`` when handed something other than a path-like.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import pipeline as _pipeline
from .._subject_anthropometrics import SubjectAnthropometrics
from .._types import Sex
from ..engine_adapters import ADAPTER_REGISTRY
from ..estimators.from_de_leva import DeLevaEstimator
from ..estimators.from_dempster import DempsterEstimator
from ..estimators.from_zatsiorsky import ZatsiorskyEstimator
from ..persistence import default_subjects_dir, save_subject
from ..readers.c3d_subject_info import read_c3d_subject_metadata
from .segment_properties_panel import SegmentPropertiesPanel

if TYPE_CHECKING:
    from ..contracts import Estimator

__all__ = ["SubjectCalibrationDialog"]


_ESTIMATORS: dict[str, type] = {
    "de_leva": DeLevaEstimator,
    "dempster": DempsterEstimator,
    "zatsiorsky": ZatsiorskyEstimator,
}

_HEIGHT_MIN_M: float = 0.5
_HEIGHT_MAX_M: float = 2.5
_HEIGHT_STEP_M: float = 0.01
_HEIGHT_DEFAULT_M: float = 1.75

_MASS_MIN_KG: float = 10.0
_MASS_MAX_KG: float = 300.0
_MASS_STEP_KG: float = 0.1
_MASS_DEFAULT_KG: float = 75.0


class SubjectCalibrationDialog(QDialog):
    """Modal dialog driving the subject-calibration workflow.

    Parameters
    ----------
    parent
        Optional Qt parent.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Subject Anthropometrics Calibration")
        self.setModal(True)

        self._record: SubjectAnthropometrics | None = None
        self._sex: str = Sex.UNSPECIFIED.value
        self._age_years: float | None = None

        self._build_ui()
        self._wire_signals()
        self._refresh_compute_enabled()

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def open_for_mocap(self, path: Path | str) -> int:
        """Pre-load *path* into the picker and exec the dialog modally.

        Args:
            path: Path-like pointing at a ``.c3d`` mocap file.

        Returns:
            The Qt ``exec()`` return code (``QDialog.Accepted`` or
            ``QDialog.Rejected``).

        Raises:
            TypeError: When *path* is not :class:`os.PathLike` /
                :class:`str`.
        """
        if not isinstance(path, (str, Path)):
            raise TypeError(f"path must be a str or Path, got {type(path).__name__}")
        self.set_mocap_path(Path(path))
        return self.exec()

    def result_record(self) -> SubjectAnthropometrics | None:
        """The last-computed :class:`SubjectAnthropometrics`, or ``None``."""
        return self._record

    def set_mocap_path(self, path: Path) -> None:
        """Public seam used by tests to drive the picker programmatically."""
        self._mocap_edit.setText(str(path))
        self._auto_fill_from_c3d(path)

    # ------------------------------------------------------------------ #
    # UI construction                                                    #
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # --- mocap picker --------------------------------------------- #
        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("Mocap (C3D):"))
        self._mocap_edit = QLineEdit()
        self._mocap_edit.setObjectName("mocap_path_edit")
        self._mocap_edit.setReadOnly(True)
        picker_row.addWidget(self._mocap_edit, stretch=1)
        self._browse_button = QPushButton("Browse…")
        self._browse_button.setObjectName("browse_button")
        picker_row.addWidget(self._browse_button)
        outer.addLayout(picker_row)

        # --- subject scalars ------------------------------------------ #
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._subject_id_edit = QLineEdit()
        self._subject_id_edit.setObjectName("subject_id_edit")
        self._subject_id_edit.setPlaceholderText("subject_001")
        form.addRow("Subject ID:", self._subject_id_edit)

        self._height_spin = QDoubleSpinBox()
        self._height_spin.setObjectName("height_spin")
        self._height_spin.setRange(_HEIGHT_MIN_M, _HEIGHT_MAX_M)
        self._height_spin.setSingleStep(_HEIGHT_STEP_M)
        self._height_spin.setDecimals(3)
        self._height_spin.setSuffix(" m")
        self._height_spin.setValue(_HEIGHT_DEFAULT_M)
        form.addRow("Height:", self._height_spin)

        self._mass_spin = QDoubleSpinBox()
        self._mass_spin.setObjectName("mass_spin")
        self._mass_spin.setRange(_MASS_MIN_KG, _MASS_MAX_KG)
        self._mass_spin.setSingleStep(_MASS_STEP_KG)
        self._mass_spin.setDecimals(2)
        self._mass_spin.setSuffix(" kg")
        self._mass_spin.setValue(_MASS_DEFAULT_KG)
        form.addRow("Mass:", self._mass_spin)

        self._estimator_combo = QComboBox()
        self._estimator_combo.setObjectName("estimator_combo")
        for name in _ESTIMATORS:
            self._estimator_combo.addItem(name)
        form.addRow("Estimator:", self._estimator_combo)

        self._engine_combo = QComboBox()
        self._engine_combo.setObjectName("engine_combo")
        for engine_name in ADAPTER_REGISTRY:
            self._engine_combo.addItem(engine_name)
        form.addRow("Engine:", self._engine_combo)

        outer.addLayout(form)

        # --- segment selector + properties panel ---------------------- #
        body = QHBoxLayout()
        self._segment_list = QListWidget()
        self._segment_list.setObjectName("segment_list")
        self._segment_list.setMinimumWidth(140)
        body.addWidget(self._segment_list)

        self._panel = SegmentPropertiesPanel()
        body.addWidget(self._panel, stretch=1)
        outer.addLayout(body, stretch=1)

        # --- action buttons ------------------------------------------- #
        buttons = QHBoxLayout()
        self._compute_button = QPushButton("Compute")
        self._compute_button.setObjectName("compute_button")
        buttons.addWidget(self._compute_button)

        self._save_button = QPushButton("Save subject record…")
        self._save_button.setObjectName("save_button")
        self._save_button.setEnabled(False)
        buttons.addWidget(self._save_button)

        self._export_button = QPushButton("Export to engine…")
        self._export_button.setObjectName("export_button")
        self._export_button.setEnabled(False)
        buttons.addWidget(self._export_button)

        buttons.addStretch(1)

        self._close_button = QPushButton("Close")
        self._close_button.setObjectName("close_button")
        buttons.addWidget(self._close_button)
        outer.addLayout(buttons)

    def _wire_signals(self) -> None:
        self._browse_button.clicked.connect(self._on_browse)
        self._compute_button.clicked.connect(self._on_compute)
        self._save_button.clicked.connect(self._on_save)
        self._export_button.clicked.connect(self._on_export)
        self._close_button.clicked.connect(self.reject)

        self._height_spin.valueChanged.connect(self._refresh_compute_enabled)
        self._mass_spin.valueChanged.connect(self._refresh_compute_enabled)
        self._subject_id_edit.textChanged.connect(self._refresh_compute_enabled)
        self._segment_list.currentRowChanged.connect(self._on_segment_changed)

    # ------------------------------------------------------------------ #
    # Slots — picker / validation                                        #
    # ------------------------------------------------------------------ #
    def _on_browse(self) -> None:
        """Open a file dialog and feed the selection into the picker."""
        # pragma: no cover — interactive only.
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Select mocap file", "", "C3D files (*.c3d);;All files (*)"
        )
        if chosen:
            self.set_mocap_path(Path(chosen))

    def _auto_fill_from_c3d(self, path: Path) -> None:
        """Read C3D subject metadata and update spinboxes / id when present."""
        try:
            meta = read_c3d_subject_metadata(path)
        except (FileNotFoundError, ImportError, OSError):
            return

        if meta.subject_id and not self._subject_id_edit.text().strip():
            self._subject_id_edit.setText(meta.subject_id)
        if meta.height_m is not None and self._in_range(
            meta.height_m, _HEIGHT_MIN_M, _HEIGHT_MAX_M
        ):
            self._height_spin.setValue(float(meta.height_m))
        if meta.mass_kg is not None and self._in_range(
            meta.mass_kg, _MASS_MIN_KG, _MASS_MAX_KG
        ):
            self._mass_spin.setValue(float(meta.mass_kg))
        self._sex = meta.sex.value
        self._age_years = meta.age_years

    @staticmethod
    def _in_range(value: float, low: float, high: float) -> bool:
        return low <= float(value) <= high

    def _refresh_compute_enabled(self) -> None:
        """Disable Compute when the spinboxes carry invalid values."""
        height_ok = self._in_range(
            self._height_spin.value(), _HEIGHT_MIN_M, _HEIGHT_MAX_M
        )
        mass_ok = self._in_range(self._mass_spin.value(), _MASS_MIN_KG, _MASS_MAX_KG)
        self._compute_button.setEnabled(height_ok and mass_ok)

    # ------------------------------------------------------------------ #
    # Slots — actions                                                    #
    # ------------------------------------------------------------------ #
    def _on_compute(self) -> None:
        """Run the chosen estimator and update the segment selector + panel."""
        estimator_cls = _ESTIMATORS[self._estimator_combo.currentText()]
        estimator: Estimator = estimator_cls()
        subject_id = self._subject_id_edit.text().strip() or "subject"
        try:
            record = estimator.estimate(
                subject_id=subject_id,
                height_m=float(self._height_spin.value()),
                mass_kg=float(self._mass_spin.value()),
                sex=self._sex,
                age_years=self._age_years,
            )
        except ValueError as error:  # pragma: no cover - covered by tests
            QMessageBox.critical(self, "Compute failed", str(error))
            return

        self._record = record
        self._populate_segment_list(record)
        self._save_button.setEnabled(True)
        self._export_button.setEnabled(True)

    def _populate_segment_list(self, record: SubjectAnthropometrics) -> None:
        """Refill the segment selector and re-display the first segment."""
        self._segment_list.clear()
        for name, _props in record.segments:
            QListWidgetItem(name, self._segment_list)
        if self._segment_list.count() > 0:
            self._segment_list.setCurrentRow(0)

    def _on_segment_changed(self, row: int) -> None:
        """Update the embedded panel when the user picks another segment."""
        if self._record is None or row < 0 or row >= len(self._record.segments):
            self._panel.set_segment(None)
            return
        _name, props = self._record.segments[row]
        self._panel.set_segment(props)

    def _on_save(self) -> None:
        """Persist the current record to the default subjects dir."""
        if self._record is None:  # pragma: no cover - guarded by enabled state
            return
        target = default_subjects_dir() / f"{self._record.subject_id}.json"
        save_subject(self._record, target)
        QMessageBox.information(
            self, "Subject saved", f"Subject record written to:\n{target}"
        )

    def _on_export(self) -> None:
        """Re-run the full pipeline against the chosen engine."""
        if self._record is None:  # pragma: no cover - guarded by enabled state
            return
        mocap_text = self._mocap_edit.text().strip()
        if not mocap_text:
            QMessageBox.warning(
                self, "Missing mocap", "Pick a C3D mocap file before exporting."
            )
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Choose export directory")
        if not out_dir:
            return

        engine = self._engine_combo.currentText()
        estimator = self._estimator_combo.currentText()
        try:
            _pipeline.run_pipeline(
                Path(mocap_text),
                subject_height_m=float(self._height_spin.value()),
                subject_mass_kg=float(self._mass_spin.value()),
                estimator=estimator,  # type: ignore[arg-type]
                target_engines=(engine,),
                output_dir=Path(out_dir),
            )
        except (FileNotFoundError, ValueError) as error:  # pragma: no cover
            QMessageBox.critical(self, "Export failed", str(error))
            return
        QMessageBox.information(
            self, "Export complete", f"Wrote engine outputs to:\n{out_dir}"
        )
