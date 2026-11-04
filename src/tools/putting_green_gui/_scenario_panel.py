"""Saved scenarios and JSON setups for the Putting Green Simulator (#14471, epic #14452).

Saves the dialled green and stroke as named scenarios through
:func:`service_setup.scenario_study` (on a worker: each runs a dispersion
batch), shows :func:`service_setup.compare_scenarios` as a table, and writes
or reads the shared setup document, the studio export's schema included.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from src.shared.python.putting_analytics import service_setup
from src.shared.python.ui.units import (
    UnitSystem,
    distance_suffix,
    from_display_distance,
)
from src.tools.putting_green_gui._batch_panel import _Job

_M_PER_FT = from_display_distance(1.0, UnitSystem.IMPERIAL, unit="ft")
_COLUMNS = ("Scenario", "Make", "Δ Make (pts)", "Leave", "Δ Leave")
_RUN_KEYS = ("n", "seed")


class ScenarioPanel(QGroupBox):
    """Scenario name, save and file buttons, the comparison table and a status.

    ``green`` gives the green and stroke arguments (the batch panel's),
    ``apply`` puts an imported setup back on the simulator's controls, and
    ``units`` gives the display units.
    """

    def __init__(
        self,
        green: Callable[[], dict[str, Any]],
        apply: Callable[[dict[str, float]], None],
        units: Callable[[], UnitSystem],
    ) -> None:
        super().__init__("Scenarios")
        self._green = green
        self._apply = apply
        self._units = units
        self._busy = False
        self._scenarios: list[dict[str, Any]] = []
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Scenario name")
        self._save_btn = QPushButton("Save Scenario")
        self._save_btn.clicked.connect(lambda: self.start_save(self._next_name()))
        row.addWidget(self._name_edit)
        row.addWidget(self._save_btn)
        layout.addLayout(row)
        files = QHBoxLayout()
        export_btn = QPushButton("Export JSON...")
        export_btn.clicked.connect(self._choose_export)
        import_btn = QPushButton("Import JSON...")
        import_btn.clicked.connect(self._choose_import)
        files.addWidget(export_btn)
        files.addWidget(import_btn)
        layout.addLayout(files)
        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.setMinimumHeight(120)
        layout.addWidget(self._table)
        self._status = QLabel("Save the dialled green to compare it later.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    def scenarios(self) -> list[dict[str, Any]]:
        return list(self._scenarios)

    def is_busy(self) -> bool:
        return self._busy

    def scenario_args(self, name: str) -> dict[str, Any]:
        """Service arguments for saving the dialled green as ``name``."""
        green = self._green()
        run = {key: green[key] for key in _RUN_KEYS if key in green}
        return {"name": name, "setup": green, **run}

    def run_save(self, name: str) -> dict[str, Any]:
        """Save the dialled green as ``name`` now, on this thread."""
        scenario = service_setup.scenario_study(self.scenario_args(name))
        self.add_scenario(scenario)
        return scenario

    def start_save(self, name: str) -> None:
        """Save the dialled green as ``name`` on a worker."""
        pool = QThreadPool.globalInstance()  # Qt makes it with the application
        if self._busy or pool is None:
            return
        self._set_busy(True)
        self._status.setText(f"Rolling putts for '{name}'...")
        job = _Job(service_setup.scenario_study, self.scenario_args(name))
        job.signals.done.connect(self.add_scenario)
        job.signals.failed.connect(self._report)
        self._job = job  # keep the signals alive until delivery
        pool.start(job)

    def add_scenario(self, scenario: dict[str, Any]) -> None:
        self._set_busy(False)
        self._scenarios.append(scenario)
        self._status.setText(f"Saved '{scenario['name']}'.")
        self.redraw()

    def export_to(self, path: Path) -> None:
        """Write the dialled setup and the saved scenarios to ``path``."""
        doc = service_setup.export_setup(
            {"setup": self._green(), "scenarios": self._scenarios}
        )
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        self._status.setText(f"Exported {path.name}.")

    def import_from(self, path: Path) -> None:
        """Read a setup document (or studio export) and apply its setup."""
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            loaded = service_setup.import_setup({"document": doc})
        except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
            self._report(str(exc))
            return
        self._apply(loaded["setup"])
        self._scenarios = list(loaded["scenarios"])
        studio = loaded["studio_scenarios"]
        note = f" ({studio} studio rows are text only)" if studio else ""
        self._status.setText(f"Imported {path.name}{note}.")
        self.redraw()

    def redraw(self) -> None:
        """Refill the comparison table in the current units."""
        rows = (
            service_setup.compare_scenarios({"scenarios": self._scenarios})["rows"]
            if self._scenarios
            else []
        )
        self._table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            cells = (
                row["name"],
                f"{100 * row['make_rate']:.0f}%",
                f"{100 * row['make_rate_delta']:+.0f}",
                self._length(row["mean_leave_ft"], ""),
                self._length(row["mean_leave_delta_ft"], "+"),
            )
            for j, text in enumerate(cells):
                self._table.setItem(i, j, QTableWidgetItem(text))

    def table_text(self) -> list[list[str]]:
        return [
            [self._cell(i, j) for j in range(len(_COLUMNS))]
            for i in range(self._table.rowCount())
        ]

    def status_text(self) -> str:
        return self._status.text()

    def _cell(self, row: int, column: int) -> str:
        item = self._table.item(row, column)
        return "" if item is None else item.text()

    def _length(self, value_ft: float | None, sign: str) -> str:
        if value_ft is None:
            return "-"
        system = self._units()
        scale = _M_PER_FT if system == UnitSystem.METRIC else 1.0
        unit = distance_suffix(system, unit="ft").strip()
        return f"{value_ft * scale:{sign}.2f} {unit}"

    def _next_name(self) -> str:
        return self._name_edit.text().strip() or f"Scenario {len(self._scenarios) + 1}"

    def _report(self, message: str) -> None:
        self._set_busy(False)
        self._status.setText(f"Rejected: {message}")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._save_btn.setEnabled(not busy)

    def _choose_export(self) -> None:
        name, _ = QFileDialog.getSaveFileName(self, "Export Setup", "", "JSON (*.json)")
        if name:
            self.export_to(Path(name))

    def _choose_import(self) -> None:
        name, _ = QFileDialog.getOpenFileName(self, "Import Setup", "", "JSON (*.json)")
        if name:
            self.import_from(Path(name))


__all__ = ["ScenarioPanel"]
