"""Club-only source panel widgets for Motion Matching (CO-09)."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.tools.motion_matching import club_only_ui as cui


class ClubOnlySourcePanel(QGroupBox):
    """Excel path, trial list, coverage/conflicts, and match action buttons."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Club-Only Excel Source", parent)
        self.workbook_path = QLineEdit()
        self.workbook_path.setPlaceholderText("Select Club_Data.xlsx…")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_workbook)
        path_row = QHBoxLayout()
        path_row.addWidget(self.workbook_path)
        path_row.addWidget(browse)

        self.trial = QComboBox()
        self.model = QComboBox()
        self.model.addItems(
            [
                "driven_double_pendulum",
                "driven_triple_pendulum",
                "constrained_upper_body_golfer",
                "full_body_mujoco",
                "full_body_pinocchio",
            ]
        )

        self.coverage = QLabel("No workbook loaded.")
        self.coverage.setWordWrap(True)
        self.conflicts = QLabel("")
        self.conflicts.setWordWrap(True)
        self.disclaimer = QLabel(cui.BODY_MOTION_DISCLAIMER)
        self.disclaimer.setWordWrap(True)
        self.disclaimer.setObjectName("club_only_disclaimer")

        self.preview_btn = QPushButton("Preview (fast)")
        self.verified_btn = QPushButton("Verified fit")
        self.compare_btn = QPushButton("Compare candidates")
        actions = QHBoxLayout()
        actions.addWidget(self.preview_btn)
        actions.addWidget(self.verified_btn)
        actions.addWidget(self.compare_btn)

        form = QFormLayout()
        form.addRow("Workbook", path_row)
        form.addRow("Trial", self.trial)
        form.addRow("Model", self.model)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.disclaimer)
        layout.addWidget(self.coverage)
        layout.addWidget(self.conflicts)
        layout.addLayout(actions)

    def _browse_workbook(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Select Club-Only Excel workbook",
            filter="Excel files (*.xlsx)",
        )
        if chosen:
            self.workbook_path.setText(chosen)
            self.import_workbook(Path(chosen))

    def import_workbook(self, path: Path) -> cui.ClubOnlyImportResult:
        result = cui.import_club_only_workbook(path)
        self.set_trials(list(result.trials))
        coverage_bits = list(result.coverage_notes)
        if result.errors:
            coverage_bits.extend(result.errors)
        self.coverage.setText(" | ".join(coverage_bits) if coverage_bits else "OK")
        self.conflicts.setText(
            "Conflicts: " + "; ".join(result.alias_conflicts)
            if result.alias_conflicts
            else "No alias conflicts."
        )
        return result

    def set_trials(self, trials: list[str]) -> None:
        self.trial.clear()
        self.trial.addItems(trials)
