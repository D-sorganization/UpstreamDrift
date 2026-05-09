"""Subject-anthropometrics calibration dialog (issue #4820).

Modal Qt dialog that asks the user for the four scalars needed by the
anthropometrics regression estimators (height, mass, age, sex), lets them
pick an estimator (de Leva / Dempster / Zatsiorsky), and on accept builds
a :class:`SubjectAnthropometrics` and persists it as JSON via
:func:`anthropometrics.persistence.save_subject`.

Design-by-contract: every public input is validated at the dialog
boundary. Out-of-range values disable the OK button and surface a status
message; the dialog never propagates a partial / invalid record to the
estimator. The dialog delegates all biomechanical math to the estimators
(DRY) and never reaches into estimator internals (LoD).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.anthropometrics._subject_anthropometrics import (
    SubjectAnthropometrics,
)
from src.shared.python.anthropometrics.estimators.from_de_leva import DeLevaEstimator
from src.shared.python.anthropometrics.estimators.from_dempster import DempsterEstimator
from src.shared.python.anthropometrics.estimators.from_zatsiorsky import (
    ZatsiorskyEstimator,
)
from src.shared.python.anthropometrics.persistence import (
    default_subjects_dir,
    save_subject,
)

logger = logging.getLogger(__name__)

EstimatorName = Literal["de_leva", "dempster", "zatsiorsky"]

ESTIMATORS: dict[str, type] = {
    "de_leva": DeLevaEstimator,
    "dempster": DempsterEstimator,
    "zatsiorsky": ZatsiorskyEstimator,
}

SEX_VALUES: tuple[str, ...] = ("M", "F", "unspecified")
SEX_LABELS: dict[str, str] = {
    "M": "Male",
    "F": "Female",
    "unspecified": "Unspecified",
}

# Validation bounds — guard against absurd inputs at the boundary.
HEIGHT_MIN_M, HEIGHT_MAX_M = 0.5, 2.5
MASS_MIN_KG, MASS_MAX_KG = 10.0, 250.0
AGE_MIN, AGE_MAX = 1, 120


@dataclass(frozen=True)
class CalibrationResult:
    """Immutable handle returned after a successful calibration."""

    record: SubjectAnthropometrics
    saved_path: Path


def build_subject_record(
    *,
    subject_id: str,
    height_m: float,
    mass_kg: float,
    age_years: int,
    sex: str,
    estimator: str,
) -> SubjectAnthropometrics:
    """Validate inputs and delegate to the chosen estimator.

    Raises ``ValueError`` on any out-of-range / invalid argument.
    """
    if not subject_id or not subject_id.strip():
        raise ValueError("subject_id must be a non-empty string")
    if not (HEIGHT_MIN_M <= height_m <= HEIGHT_MAX_M):
        raise ValueError(
            f"height_m must be in [{HEIGHT_MIN_M}, {HEIGHT_MAX_M}], got {height_m!r}"
        )
    if not (MASS_MIN_KG <= mass_kg <= MASS_MAX_KG):
        raise ValueError(
            f"mass_kg must be in [{MASS_MIN_KG}, {MASS_MAX_KG}], got {mass_kg!r}"
        )
    if not (AGE_MIN <= age_years <= AGE_MAX):
        raise ValueError(
            f"age_years must be in [{AGE_MIN}, {AGE_MAX}], got {age_years!r}"
        )
    if sex not in SEX_VALUES:
        raise ValueError(f"sex must be one of {SEX_VALUES}, got {sex!r}")
    if estimator not in ESTIMATORS:
        raise ValueError(
            f"estimator must be one of {tuple(ESTIMATORS)}, got {estimator!r}"
        )

    estimator_obj = ESTIMATORS[estimator]()
    return estimator_obj.estimate(
        subject_id=subject_id.strip(),
        height_m=float(height_m),
        mass_kg=float(mass_kg),
        sex=sex,
        age_years=float(age_years),
    )


class CalibrationDialog(QDialog):
    """Modal subject-calibration dialog.

    Emits :pyattr:`calibrated` with the produced
    :class:`SubjectAnthropometrics` whenever the user accepts a valid form.
    """

    calibrated = pyqtSignal(object)  # SubjectAnthropometrics

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        subjects_dir: Path | None = None,
        default_subject_id: str = "subject_001",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Subject Anthropometrics Calibration")
        self.setModal(True)
        self._subjects_dir: Path = (
            subjects_dir if subjects_dir is not None else default_subjects_dir()
        )
        self._result: CalibrationResult | None = None

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.id_edit = QLineEdit(default_subject_id)
        self.id_edit.setObjectName("subject_id")
        self.id_edit.textChanged.connect(self._refresh_validity)
        form.addRow("Subject ID:", self.id_edit)

        self.height_spin = QDoubleSpinBox()
        self.height_spin.setObjectName("height_m")
        self.height_spin.setRange(HEIGHT_MIN_M, HEIGHT_MAX_M)
        self.height_spin.setSingleStep(0.01)
        self.height_spin.setDecimals(3)
        self.height_spin.setSuffix(" m")
        self.height_spin.setValue(1.78)
        form.addRow("Height:", self.height_spin)

        self.mass_spin = QDoubleSpinBox()
        self.mass_spin.setObjectName("mass_kg")
        self.mass_spin.setRange(MASS_MIN_KG, MASS_MAX_KG)
        self.mass_spin.setSingleStep(0.5)
        self.mass_spin.setDecimals(2)
        self.mass_spin.setSuffix(" kg")
        self.mass_spin.setValue(75.0)
        form.addRow("Mass:", self.mass_spin)

        self.age_spin = QSpinBox()
        self.age_spin.setObjectName("age_years")
        self.age_spin.setRange(AGE_MIN, AGE_MAX)
        self.age_spin.setSuffix(" yr")
        self.age_spin.setValue(30)
        form.addRow("Age:", self.age_spin)

        self.sex_combo = QComboBox()
        self.sex_combo.setObjectName("sex")
        for value in SEX_VALUES:
            self.sex_combo.addItem(SEX_LABELS[value], value)
        form.addRow("Sex:", self.sex_combo)

        self.estimator_combo = QComboBox()
        self.estimator_combo.setObjectName("estimator")
        self.estimator_combo.addItems(tuple(ESTIMATORS))
        form.addRow("Estimator:", self.estimator_combo)

        layout.addLayout(form)

        self.status_label = QLabel("")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.status_label)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_btn = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn is not None:
            ok_btn.setText("Calibrate && Save")
            ok_btn.setObjectName("calibrate_button")
        self.buttons.accepted.connect(self._on_accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._refresh_validity()

    # ------------------------------------------------------------------ #
    # Public API.                                                        #
    # ------------------------------------------------------------------ #
    @property
    def result_record(self) -> CalibrationResult | None:
        """The :class:`CalibrationResult` produced on accept, or ``None``."""
        return self._result

    def current_form_values(self) -> dict[str, object]:
        """Return the current form state as a plain dict (testing aid)."""
        return {
            "subject_id": self.id_edit.text().strip(),
            "height_m": float(self.height_spin.value()),
            "mass_kg": float(self.mass_spin.value()),
            "age_years": int(self.age_spin.value()),
            "sex": str(self.sex_combo.currentData() or self.sex_combo.currentText()),
            "estimator": self.estimator_combo.currentText(),
        }

    # ------------------------------------------------------------------ #
    # Internals.                                                         #
    # ------------------------------------------------------------------ #
    def _refresh_validity(self) -> None:
        """Enable/disable OK based on a non-empty subject id."""
        ok_btn = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn is None:
            return
        valid = bool(self.id_edit.text().strip())
        ok_btn.setEnabled(valid)
        if not valid:
            self.status_label.setText("Subject ID is required.")
        else:
            self.status_label.setText("")

    def _on_accept(self) -> None:
        values = self.current_form_values()
        try:
            record = build_subject_record(
                subject_id=str(values["subject_id"]),
                height_m=float(values["height_m"]),  # type: ignore[arg-type]
                mass_kg=float(values["mass_kg"]),  # type: ignore[arg-type]
                age_years=int(values["age_years"]),  # type: ignore[arg-type]
                sex=str(values["sex"]),
                estimator=str(values["estimator"]),
            )
        except (ValueError, TypeError) as exc:
            logger.warning("Calibration validation failed: %s", exc)
            self.status_label.setText(f"Invalid input: {exc}")
            return

        try:
            saved_path = self._persist(record)
        except OSError as exc:
            logger.exception("Failed to persist subject record")
            self.status_label.setText(f"Could not save subject: {exc}")
            return

        self._result = CalibrationResult(record=record, saved_path=saved_path)
        self.calibrated.emit(record)
        self.accept()

    def _persist(self, record: SubjectAnthropometrics) -> Path:
        """Write *record* to ``<subjects_dir>/<subject_id>.json``."""
        target_dir = Path(self._subjects_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"{record.subject_id}.json"
        save_subject(record, out_path)
        return out_path


__all__ = [
    "CalibrationDialog",
    "CalibrationResult",
    "ESTIMATORS",
    "SEX_VALUES",
    "build_subject_record",
]
