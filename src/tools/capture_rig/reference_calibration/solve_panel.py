"""Explicit world-anchor review and canonical solver evidence in the player UI."""

from __future__ import annotations

import html
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class SolvePanel(QWidget):
    solve_requested = pyqtSignal(dict)
    accept_requested = pyqtSignal(dict)
    restore_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._result_id: str | None = None
        self._result_sha256: str | None = None
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Use at least two camera views and two stationary reference placements. "
            "Choose a paper placement as the world anchor: its marked long edge must point toward the target, "
            "its face must be flat and facing upward. Enter the marked corner’s measured offset from the ball. "
            "Keep all cameras and optical settings fixed. Reserve additional views for independent validation."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.anchor = QComboBox()
        self.offsets = []
        form = QFormLayout()
        form.addRow("World Anchor Placement", self.anchor)
        for label in ("Toward Target (+X)", "Up (+Y)", "Golfer’s Right (+Z)"):
            spin = QDoubleSpinBox()
            spin.setRange(-100, 100)
            spin.setDecimals(4)
            spin.setSuffix(" m")
            self.offsets.append(spin)
            form.addRow(label, spin)
        layout.addLayout(form)
        self.settings = QCheckBox(
            "I Rechecked the Recorded Camera Positions, Lens, Zoom and Focus"
        )
        self.anchor_confirmed = QCheckBox(
            "I Verified the Anchor Offset, Flat Face and Target-Pointing Arrow"
        )
        layout.addWidget(self.settings)
        layout.addWidget(self.anchor_confirmed)
        for spin in self.offsets:
            spin.valueChanged.connect(lambda: self.anchor_confirmed.setChecked(False))
        self.anchor.currentTextChanged.connect(
            lambda: self.anchor_confirmed.setChecked(False)
        )
        self.solve = QPushButton("Estimate Camera Positions")
        self.solve.clicked.connect(self._solve)
        layout.addWidget(self.solve)
        restore = QPushButton("Open a Saved Camera Estimate…")
        restore.clicked.connect(self.restore_requested)
        layout.addWidget(restore)
        self.evidence = QTextBrowser()
        self.evidence.setOpenExternalLinks(False)
        layout.addWidget(self.evidence, 1)
        self.reviewed = QCheckBox(
            "I Reviewed the Fit, Validation Errors and Stated Limitations"
        )
        layout.addWidget(self.reviewed)
        self.use = QPushButton("Use Reviewed Camera Layout")
        self.use.clicked.connect(self._accept)
        layout.addWidget(self.use)
        self.reviewed.toggled.connect(self._readiness)
        self.settings.toggled.connect(self._readiness)
        self.anchor_confirmed.toggled.connect(self._readiness)
        self._readiness()

    def _readiness(self) -> None:
        self.solve.setEnabled(
            bool(self.anchor.currentText())
            and self.settings.isChecked()
            and self.anchor_confirmed.isChecked()
        )
        self.use.setEnabled(
            self.solve.isEnabled()
            and self.reviewed.isChecked()
            and self._result_id is not None
        )

    def show_session(self, session: dict[str, Any]) -> None:
        self._result_id = None
        self._result_sha256 = None
        self.reviewed.setChecked(False)
        previous = self.anchor.currentText()
        self.anchor.clear()
        self.anchor.addItems(
            list(
                dict.fromkeys(
                    sample["observation"]["placement_id"]
                    for sample in session["samples"]
                    if sample["enabled"]
                )
            )
        )
        if previous:
            self.anchor.setCurrentText(previous)
        self.settings.setChecked(False)
        self.anchor_confirmed.setChecked(False)
        self.evidence.setPlainText(
            "No result for this revision. Review the recorded optics and reference placements before estimating camera positions."
        )
        self._readiness()

    def _solve(self) -> None:
        self.solve_requested.emit(self.parameters())

    def parameters(self) -> dict[str, Any]:
        return {
            "settings_confirmed": self.settings.isChecked(),
            "anchor_confirmed": self.anchor_confirmed.isChecked(),
            "anchor_placement_id": self.anchor.currentText(),
            "anchor_translation_m": [spin.value() for spin in self.offsets],
        }

    def _accept(self) -> None:
        self.accept_requested.emit(
            {
                **self.parameters(),
                "result_id": self._result_id,
                "result_sha256": self._result_sha256,
                "reviewed": self.reviewed.isChecked(),
            }
        )

    def show_result(self, result: dict[str, Any], digest: str) -> None:
        self._result_id = result["layout_id"]
        self._result_sha256 = digest
        self.anchor.setCurrentText(result["anchor_placement_id"])
        for spin, value in zip(
            self.offsets, result["anchor"]["translation_m"], strict=True
        ):
            spin.setValue(value)
        self.reviewed.setChecked(False)
        rows = []
        for item in result["residuals"]:
            values = [
                item["placement_id"],
                item["camera_key"],
                "Validation" if item["held_out"] else "Fit",
                f"{item['mean_error_px']:.3f} px",
                f"{item['max_error_px']:.3f} px",
            ]
            rows.append(
                "<tr>"
                + "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
                + "</tr>"
            )
        limitations = "".join(
            f"<li>{html.escape(item)}</li>" for item in result["limitations"]
        )
        self.evidence.setHtml(
            "<h3>Camera Position Estimate</h3><p>Saved numerical result. Review the independent validation errors and physical setup before use.</p>"
            "<table cellspacing='8'><tr><th>Placement</th><th>Camera</th><th>Use</th><th>Mean Error</th><th>Max Error</th></tr>"
            + "".join(rows)
            + "</table><ul>"
            + limitations
            + "</ul>"
        )
        self._readiness()
