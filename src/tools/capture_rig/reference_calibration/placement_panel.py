"""Placement selection and readable evidence, without importing provider records."""

from __future__ import annotations

from typing import Any
from .target_dialog import MeasuredTargetDialog

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class PlacementPanel(QWidget):
    """Each physical placement ID is shared across its observing camera views."""

    frame_requested = pyqtSignal(dict)
    revision_requested = pyqtSignal(dict)
    target_requested = pyqtSignal(dict)

    def __init__(self) -> None:
        super().__init__()
        self._session: dict[str, Any] = {}
        self.view = QComboBox()
        self.target = QComboBox()
        self.placement = QComboBox()
        self.placement.setEditable(True)
        self.placement.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.placement.setEditText("Near Ball")
        self.frame_number = QSpinBox()
        self.frame_number.setRange(0, 2_147_483_647)
        self.frame_number.setToolTip("Original recording frame number, starting at 0")
        self.notes = QLineEdit()
        self.notes.setMaxLength(4000)
        self.held_out = QCheckBox("Reserve This View for Validation")
        self.held_out.setToolTip(
            "Other fitted views must locate this same placement. Reserved views do not seed the solve."
        )
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Placement", "Camera", "Points", "Use", "Notes"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        hint = QLabel(
            "Keep the reference still while every camera records it. Reuse its placement name across views. "
            "After moving the reference, enter a new name. Keep the cameras fixed. "
            "Paper must lie flat, with the same marked corner and long-edge arrow visible. "
            "Ruler endpoints can provide scale evidence; they cannot locate cameras on their own."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        for label, widget in (
            ("Physical Placement", self.placement),
            ("Reference", self.target),
            ("Camera View", self.view),
            ("Original Frame", self.frame_number),
            ("Observation Notes", self.notes),
        ):
            form.addRow(label, widget)
        form.addRow(self.held_out)
        layout.addLayout(form)
        measured = QPushButton("Use Measured Reference Dimensions…")
        measured.clicked.connect(self._measured)
        layout.addWidget(measured)
        self.add_placement = QPushButton("Add Another Placement")
        self.add_placement.clicked.connect(self._add_placement)
        self.add_placement.setToolTip(
            "Move only the reference, then select the corresponding original frame in each camera."
        )
        layout.addWidget(self.add_placement)
        mark = QPushButton("Choose Frame and Mark Points…")
        mark.clicked.connect(self._request_frame)
        layout.addWidget(mark)
        layout.addWidget(self.table, 1)
        row = QHBoxLayout()
        toggle = QPushButton("Include / Exclude Selected Observation")
        toggle.setEnabled(False)
        self.table.itemSelectionChanged.connect(
            lambda: toggle.setEnabled(self.table.currentRow() >= 0)
        )
        toggle.clicked.connect(self._toggle)
        row.addWidget(toggle)
        layout.addLayout(row)
        self.table.cellDoubleClicked.connect(self._select_observation)

    def _add_placement(self) -> None:
        existing = {
            item["observation"]["placement_id"] for item in self._session["samples"]
        }
        number = 1
        while f"Placement {number}" in existing:
            number += 1
        self.placement.setEditText(f"Placement {number}")
        self.notes.clear()
        self.held_out.setChecked(False)
        self.placement.setFocus()

    def _select_observation(self, row: int, _column: int) -> None:
        sample = self._session["samples"][row]
        observation = sample["observation"]
        self.placement.setEditText(observation["placement_id"])
        self.view.setCurrentText(observation["camera_key"])
        self.target.setCurrentIndex(self.target.findData(observation["reference_id"]))
        self.frame_number.setValue(observation["frame_sequence"])
        self.notes.setText(sample["notes"])
        self.held_out.setChecked(observation["held_out"])

    def _request_frame(self) -> None:
        self.frame_requested.emit(
            {"view": self.view.currentText(), "frame_index": self.frame_number.value()}
        )

    def _measured(self) -> None:
        target = next(
            item
            for item in self._session["targets"]
            if item["reference_id"] == self.target.currentData()
        )
        dialog = MeasuredTargetDialog(target, self)
        if dialog.exec():
            self.target_requested.emit(dialog.parameters())

    def marking_parameters(self) -> dict[str, Any]:
        return {
            "placement_id": self.placement.currentText().strip(),
            "reference_id": self.target.currentData(),
            "held_out": self.held_out.isChecked(),
            "notes": self.notes.text(),
        }

    def point_ids(self) -> tuple[str, ...]:
        target = next(
            item
            for item in self._session["targets"]
            if item["reference_id"] == self.target.currentData()
        )
        return tuple(target["point_ids"])

    def show_session(self, session: dict[str, Any]) -> None:
        self._session = session
        previous_view = self.view.currentText()
        previous_target = self.target.currentData()
        previous_placement = self.placement.currentText()
        self.view.clear()
        self.view.addItems([item["view"] for item in session["cameras"]])
        if previous_view:
            self.view.setCurrentText(previous_view)
        self.target.clear()
        for target in session["targets"]:
            key = target["reference_id"]
            self.target.addItem(key.replace("-", " ").title(), key)
        index = self.target.findData(previous_target)
        if index >= 0:
            self.target.setCurrentIndex(index)
        samples = session["samples"]
        placements = list(
            dict.fromkeys(item["observation"]["placement_id"] for item in samples)
        )
        self.placement.clear()
        self.placement.addItems(placements)
        self.placement.setEditText(previous_placement)
        self.table.setRowCount(len(samples))
        for row, sample in enumerate(samples):
            observation = sample["observation"]
            state = (
                "Excluded"
                if not sample["enabled"]
                else "Validation"
                if observation["held_out"]
                else "Fit"
            )
            values = [
                observation["placement_id"],
                observation["camera_key"],
                str(len(observation["point_ids"])),
                state,
                sample["notes"],
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, row)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()

    def _toggle(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        samples = [dict(item) for item in self._session["samples"]]
        samples[row]["enabled"] = not samples[row]["enabled"]
        self.revision_requested.emit({"samples": samples})
