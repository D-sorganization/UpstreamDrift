"""PyQt6 shell for the Character Builder (CMB-3, #11654).

All behaviour lives in :class:`~.core.CharacterBuilderModel`; this module only
binds widgets to it.
"""

from __future__ import annotations

from typing import Any

from PyQt6 import QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.club_models import CLUBS

from .core import PARAMETER_RANGES, CharacterBuilderModel

logger = get_logger(__name__)

__all__ = ["CharacterBuilderWidget"]

_LABELS = {
    "stature_m": "Stature (m)",
    "mass_kg": "Mass (kg)",
    "trunk_scale": "Trunk Length Scale",
    "arm_scale": "Arm Length Scale",
    "shoulder_scale": "Shoulder Breadth Scale",
    "grip_roll_deg": "Grip Roll (deg)",
}
_EXPORTS = (
    ("spec", "Export Spec JSON"),
    ("urdf", "Export URDF"),
    ("mjcf", "Export MJCF"),
    ("osim", "Export OpenSim"),
)


class CharacterBuilderWidget(QtWidgets.QWidget):
    """Preset picker, parameter form, live summary and export buttons."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.model = CharacterBuilderModel()
        self._syncing = False
        self._build_ui()
        self._load_preset_into_form(self.preset_combo.currentData())

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("Custom", None)
        for preset_id in self.model.preset_ids():
            self.preset_combo.addItem(preset_id.replace("_", " ").title(), preset_id)
        form.addRow("Preset", self.preset_combo)
        self.spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for name, (low, high) in PARAMETER_RANGES.items():
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(3 if name != "mass_kg" else 1)
            spin.setSingleStep(0.01 if name != "mass_kg" else 1.0)
            spin.setObjectName(f"spin_{name}")
            spin.valueChanged.connect(lambda v, n=name: self._on_spin(n, v))
            self.spins[name] = spin
            form.addRow(_LABELS[name], spin)
        self.club_combo = QtWidgets.QComboBox()
        self.club_combo.addItems(sorted(CLUBS))
        self.club_combo.currentTextChanged.connect(
            lambda text: self._on_edit("club", text)
        )
        form.addRow("Club", self.club_combo)
        layout.addLayout(form)
        self.summary = QtWidgets.QPlainTextEdit()
        self.summary.setReadOnly(True)
        layout.addWidget(self.summary, 1)
        self.status = QtWidgets.QLabel("")
        layout.addWidget(self.status)
        buttons = QtWidgets.QHBoxLayout()
        self.export_buttons: dict[str, QtWidgets.QPushButton] = {}
        for fmt, text in _EXPORTS:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda _checked=False, f=fmt: self.export(f))
            self.export_buttons[fmt] = button
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.preset_combo.currentIndexChanged.connect(
            lambda _i: self._load_preset_into_form(self.preset_combo.currentData())
        )

    def _load_preset_into_form(self, preset_id: str | None) -> None:
        if preset_id is not None:
            self.model.apply_preset(preset_id)
        self._syncing = True
        try:
            values = self.model.parameters.to_dict()
            for name, spin in self.spins.items():
                spin.setValue(values[name])
            self.club_combo.setCurrentText(values["club"])
        finally:
            self._syncing = False
        self._refresh()

    def _on_spin(self, name: str, value: float) -> None:
        self._on_edit(name, value)

    def _on_edit(self, name: str, value: float | str) -> None:
        if self._syncing:
            return
        try:
            self.model.set_parameter(name, value)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self._syncing = True
        self.preset_combo.setCurrentIndex(0)
        self._syncing = False
        self._refresh()

    def _refresh(self) -> None:
        try:
            self.summary.setPlainText(self.model.summary_text())
            self.status.setText("")
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("Character compile failed: %s", exc)
            self.summary.setPlainText("")
            self.status.setText(f"Cannot compile: {exc}")

    def export(self, fmt: str, directory: str | None = None) -> str | None:
        """Export in ``fmt``; asks for a folder unless ``directory`` is given."""
        if directory is None:
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Choose Export Folder"
            )
        if not directory:
            return None
        try:
            path = self.model.export(fmt, directory)
        except (ValueError, FileNotFoundError, OSError) as exc:
            logger.warning("Character export failed: %s", exc)
            self.status.setText(f"Export failed: {exc}")
            return None
        self.status.setText(f"Wrote {path}")
        return str(path)

    def cleanup(self) -> None:
        """Nothing to release; kept idempotent for the embed contract."""
