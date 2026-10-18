"""Controls of the Variability & Consistency tab (NSX-100, #12729).

Every list is the backend catalogue's; the controls only collect choices and
hand them to :func:`~.variability_request.sensitivity_options`.  Nothing is
computed.  An engine that cannot supply derivatives is listed with its
``UNAVAILABLE`` reason (choosing it makes the stage report that reason).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from src.tools.grip_wrench_plots.variability_request import (
    DisturbanceChoice,
    VariabilityChoices,
    parse_channels,
    sensitivity_options,
)
from src.tools.grip_wrench_plots.variability_view import UNAVAILABLE

__all__ = ["VariabilityControls"]

DEFAULT_ENGINE = "pinocchio"


def _combo(entries: list[dict[str, Any]]) -> QComboBox:
    combo = QComboBox()
    for entry in entries:
        combo.addItem(str(entry["label"]), entry["id"])
    return combo


class VariabilityControls(QWidget):
    """Disturbances, mechanisms, strategies, outcome priority and engine."""

    def __init__(
        self, catalogue: Mapping[str, Any], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._catalogue = catalogue
        root = QVBoxLayout(self)
        self.enable_check = QCheckBox("Run variability analysis")
        root.addWidget(self.enable_check)
        form = QFormLayout()
        self.name_edit = QLineEdit("disturbance")
        self.stage_combo = _combo(list(catalogue["stages"]))
        self.spectrum_combo = _combo(list(catalogue["spectra"]))
        self.persistent_check = QCheckBox("Persistent per-trial state")
        self.channels_edit = QLineEdit()
        self.channels_edit.setPlaceholderText("channel=variance, channel=variance")
        self.units_edit = QLineEdit("N m")
        self.bandwidth_edit = QLineEdit()
        for label, widget in (
            ("Disturbance", self.name_edit),
            ("Stage", self.stage_combo),
            ("Spectrum", self.spectrum_combo),
            ("", self.persistent_check),
            ("Channels and amplitudes (variance)", self.channels_edit),
            ("Units", self.units_edit),
            ("Bandwidth (Hz)", self.bandwidth_edit),
        ):
            form.addRow(QLabel(label), widget)
        root.addLayout(form)
        self.mechanism_checks = self._checks(
            root, "Mechanisms", catalogue["mechanisms"], False
        )
        self.strategy_checks = self._checks(
            root, "Strategies", catalogue["strategies"], True
        )
        row = QHBoxLayout()
        self.outcome_combo = _combo(list(catalogue["outcomes"]))
        first = catalogue["default_outcome_priority"][0]
        self.outcome_combo.setCurrentIndex(self.outcome_combo.findData(first))
        self.engine_combo = QComboBox()
        for engine in catalogue["engines"]:
            text = engine["id"]
            if not engine["available"]:
                text += f" ({UNAVAILABLE}{engine['reason']})"
            self.engine_combo.addItem(text, engine["id"])
            self.engine_combo.setItemData(
                self.engine_combo.count() - 1,
                engine["reason"],
                Qt.ItemDataRole.ToolTipRole,
            )
        start = self.engine_combo.findData(DEFAULT_ENGINE)
        self.engine_combo.setCurrentIndex(max(start, 0))
        for label, widget in (
            ("Primary outcome", self.outcome_combo),
            ("Engine", self.engine_combo),
        ):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        root.addLayout(row)

    def _checks(
        self,
        root: QVBoxLayout,
        title: str,
        entries: list[dict[str, Any]],
        checked: bool,
    ) -> dict[str, QCheckBox]:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        checks: dict[str, QCheckBox] = {}
        for entry in entries:
            text = f"{entry['id']}: {entry.get('name') or entry['label']}"
            note = entry.get("note") or entry.get("reason") or ""
            status = entry.get("availability", "implemented")
            if status != "implemented":
                note = f"{status}: {note}"
            if note:
                text += f" ({note})"
            check = QCheckBox(text)
            check.setChecked(checked)
            layout.addWidget(check)
            checks[entry["id"]] = check
        root.addWidget(box)
        return checks

    def engine(self) -> str:
        """Catalogue id of the chosen engine."""
        return str(self.engine_combo.currentData())

    def choices(self) -> VariabilityChoices:
        """The choices held (``ValueError`` for text that does not parse)."""
        text = self.bandwidth_edit.text().strip()
        try:
            bandwidth = float(text) if text else None
        except ValueError:
            raise ValueError(f"bandwidth is not a number: {text!r}") from None
        chosen = tuple(k for k, c in self.strategy_checks.items() if c.isChecked())
        every = len(chosen) == len(self.strategy_checks)
        disturbance = DisturbanceChoice(
            name=self.name_edit.text(),
            stage=str(self.stage_combo.currentData()),
            spectrum=str(self.spectrum_combo.currentData()),
            persistent=self.persistent_check.isChecked(),
            channels=parse_channels(self.channels_edit.text()),
            units=self.units_edit.text(),
            bandwidth_hz=bandwidth,
        )
        return VariabilityChoices(
            disturbances=(disturbance,),
            mechanisms=tuple(
                k for k, c in self.mechanism_checks.items() if c.isChecked()
            ),
            strategies=None if every else chosen,
            outcome_priority=(str(self.outcome_combo.currentData()),),
        )

    def request_options(self) -> dict[str, Any] | None:
        """The ``SensitivityOptions`` JSON, or ``None`` while the analysis is off.

        Raises:
            ValueError: A choice does not parse or is not in the catalogue.
        """
        if not self.enable_check.isChecked():
            return None
        return sensitivity_options(self.choices(), self._catalogue)
