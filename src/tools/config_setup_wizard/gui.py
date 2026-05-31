"""Minimal PyQt surface for the deterministic setup wizard."""

from __future__ import annotations

import json
from typing import Any

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.config.setup_wizard import SetupWizardViewModel

__all__ = ["ConfigSetupWizardWidget"]


class ConfigSetupWizardWidget(QWidget):  # pragma: no cover - GUI smoke follows later
    """Small embeddable widget backed by :class:`SetupWizardViewModel`."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._model = SetupWizardViewModel()
        self._input = QPlainTextEdit(self)
        self._input.setPlaceholderText("Paste canonical-core setup JSON")
        self._output = QTextEdit(self)
        self._output.setReadOnly(True)
        self._status = QLabel(self)

        validate_button = QPushButton("Validate", self)
        next_button = QPushButton("Next", self)
        back_button = QPushButton("Back", self)

        buttons = QHBoxLayout()
        buttons.addWidget(back_button)
        buttons.addWidget(validate_button)
        buttons.addWidget(next_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self._status)
        layout.addWidget(self._input)
        layout.addLayout(buttons)
        layout.addWidget(self._output)

        validate_button.clicked.connect(self._validate_current)
        next_button.clicked.connect(self._advance)
        back_button.clicked.connect(self._retreat)
        self._render(self._model.snapshot())

    def cleanup(self) -> None:
        """Release the wrapped Qt widget."""

        self.deleteLater()

    def _candidate_config(self) -> dict[str, Any]:
        text = self._input.toPlainText().strip()
        if not text:
            return {}
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("setup JSON must decode to an object")
        return payload

    def _validate_current(self) -> None:
        try:
            snapshot = self._model.validate(self._candidate_config())
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._status.setText(f"Invalid input: {exc}")
            return
        self._render(snapshot)

    def _advance(self) -> None:
        try:
            snapshot = self._model.advance(self._candidate_config())
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._status.setText(f"Invalid input: {exc}")
            return
        self._render(snapshot)

    def _retreat(self) -> None:
        self._render(self._model.retreat())

    def _render(self, snapshot: Any) -> None:
        current = snapshot.current_step.replace("_", " ")
        status = "valid" if snapshot.report.is_valid else "needs fixes"
        self._status.setText(f"Step: {current} | {status}")
        lines = []
        for step in snapshot.steps:
            lines.append(f"{step.title}: {step.status} ({step.issue_count})")
        if snapshot.report.issues:
            lines.append("")
            for issue in snapshot.report.issues:
                lines.append(f"{issue.field_path}: {issue.message}")
                lines.append(f"Fix: {issue.suggested_fix}")
        self._output.setPlainText("\n".join(lines))
