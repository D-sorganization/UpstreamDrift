# Copyright (c) 2026 D-Sorganization. All rights reserved.
# mypy: disable-error-code="misc,has-type"
# Mixin pattern: methods annotate self as MainWindow to access its attributes,
# but mypy cannot verify this pattern without the concrete class in scope.
"""Comparison trial helpers extracted from MainWindow.

Provides add/compare/clear trial comparison actions as a mixin class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QMessageBox

from .comparison_dialog import ComparisonDialog

if TYPE_CHECKING:
    from .main_window import MainWindow


class ComparisonMixin:
    """Mixin providing trial comparison actions for MainWindow."""

    def _add_comparison(self: MainWindow) -> None:
        idx = self.tabs.currentIndex()
        r, _fi, _body, _dyn = self._snapshot_idx_state(idx)
        if r is None:
            return
        display_name, _etype = self.EXERCISE_CONFIGS[idx]
        body_params, bar = self.sidebar.get_comparison_trial_data()
        n = len(self._comparison_store.get_trials()) + 1
        trial_name = f"{display_name} #{n} ({bar:.0f}kg)"
        self._comparison_store.add_trial(trial_name, r, body_params, bar)

        # Enable clearing immediately after first trial is added
        self.sidebar.set_clear_comparison_available(True)
        # Enable comparison dialog only when multiple trials exist
        if len(self._comparison_store.get_trials()) >= 2:
            self.sidebar.set_comparison_available(True)

        self.status_label.setText(f"Added '{trial_name}' to comparison list.")

    def _compare_trials(self: MainWindow) -> None:
        trials = self._comparison_store.get_trials()
        if not trials:
            QMessageBox.information(self, "No Trials", "Add trials to compare first.")
            return
        dlg = ComparisonDialog(trials, self)
        dlg.exec()

    def _clear_comparison(self: MainWindow) -> None:
        self._comparison_store.clear()
        self.sidebar.set_comparison_available(False)
        self.sidebar.set_clear_comparison_available(False)
        self.status_label.setText("Comparison list cleared.")
