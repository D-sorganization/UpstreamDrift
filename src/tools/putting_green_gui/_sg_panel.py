"""Strokes-gained panel for the Putting Green Simulator (#14467, epic #14452).

Solves the player's own expected putts E(d) on the dialled green with
:func:`service_sg.sg_study` (on a worker: the Bellman solve takes seconds)
and scores the putt against it. The curve is cached per green and player, so
later putts from the same spot score at once with :func:`service_sg.score_putt`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
)

from src.shared.python.putting_analytics import service_sg
from src.shared.python.putting_analytics.aggregate import DEFAULT_GRID_FT
from src.shared.python.ui.units import (
    UnitSystem,
    distance_suffix,
    from_display_distance,
)
from src.tools.putting_green_gui._batch_panel import _Job

_M_PER_FT = from_display_distance(1.0, UnitSystem.IMPERIAL, unit="ft")
_PUTT_KEYS = ("holed", "leave_ft")


class SGPanel(QGroupBox):
    """The putt's result, a score button, the E(d) chart and the SG readout.

    ``green`` gives the green and stroke arguments (the batch panel's), and
    ``units`` the display units.
    """

    def __init__(
        self,
        green: Callable[[], dict[str, Any]],
        units: Callable[[], UnitSystem],
        *,
        n: int = service_sg.DEFAULT_SG_N,
        paces_ft: Sequence[float] = service_sg.DEFAULT_SG_PACES_FT,
        distances_ft: Sequence[float] = DEFAULT_GRID_FT,
    ) -> None:
        super().__init__("Strokes Gained")
        self._green = green
        self._units = units
        self._shown_units = units()
        self._study = {
            "n": n,
            "paces_ft": list(paces_ft),
            "distances_ft": list(distances_ft),
        }
        self._busy = False
        self._cache: dict[str, dict[str, Any]] = {}  # curve report per green
        self._report: dict[str, Any] | None = None
        form = QFormLayout(self)
        self._holed_box = QCheckBox("Holed")
        form.addRow("Result:", self._holed_box)
        self._leave_spin = QDoubleSpinBox()
        self._leave_spin.setDecimals(2)
        self._label_leave()
        form.addRow("Leave:", self._leave_spin)
        self._score_btn = QPushButton("Score Putt")
        self._score_btn.clicked.connect(self.start_score)
        form.addRow(self._score_btn)
        self._summary = QLabel("Score a putt against your own expected putts.")
        self._summary.setWordWrap(True)
        form.addRow(self._summary)
        self._figure = Figure(figsize=(5.0, 2.4), tight_layout=True)
        self._ax = self._figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasQTAgg(self._figure)
        self.canvas.setMinimumSize(160, 200)

    def set_putt(self, *, holed: bool, leave_ft: float) -> None:
        """Enter the putt's result (leave in feet)."""
        self._holed_box.setChecked(holed)
        self.set_leave_display(leave_ft * self._length_scale()[0])

    def set_leave_display(self, leave: float) -> None:
        """Enter the leave in the display units."""
        self._leave_spin.setValue(leave)

    def study_args(self) -> dict[str, Any]:
        """Service arguments: the green, the player, the study size and the putt."""
        holed = self._holed_box.isChecked()
        leave_ft = 0.0 if holed else self._leave_spin.value() / self._length_scale()[0]
        return {**self._green(), **self._study, "holed": holed, "leave_ft": leave_ft}

    def is_busy(self) -> bool:
        return self._busy

    def has_cached_curve(self) -> bool:
        return _curve_key(self.study_args()) in self._cache

    def run_score(self) -> dict[str, Any]:
        """Solve the curve and score the putt now, on this thread."""
        args = self.study_args()
        report = service_sg.sg_study(args)
        self._store(_curve_key(args), report)
        return report

    def start_score(self) -> None:
        """Score on the cached curve, or solve it on a worker first."""
        args = self.study_args()
        cached = self._cache.get(_curve_key(args))
        if cached is not None:
            sg = service_sg.score_putt(cached, args["holed"], args["leave_ft"])
            self.show_report({**cached, "sg_player": sg})
            return
        pool = QThreadPool.globalInstance()  # Qt makes it with the application
        if self._busy or pool is None:
            return
        self._set_busy(True)
        self._summary.setText("Solving your expected putts on this green...")
        job = _Job(service_sg.sg_study, args)
        key = _curve_key(args)
        job.signals.done.connect(lambda report: self._store(key, report))
        job.signals.failed.connect(self._on_failed)
        self._job = job  # keep the signals alive until delivery
        pool.start(job)

    def show_report(self, report: dict[str, Any]) -> None:
        """Plot E(d) and summarise the putt's strokes gained."""
        self._set_busy(False)
        self._report = report
        scale, unit = self._length_scale()
        curve = report["curve"]
        ax = self._ax
        ax.clear()
        xs = [d * scale for d in curve["distances_ft"]]
        ax.plot(xs, curve["expected_putts"], "o-", color="#2b7bd6")
        start = report["distance_ft"] * scale
        ax.plot([start], [report["expected_putts"]], "o", color="#d2553f")
        ax.set_xlabel(f"Distance ({unit})")
        ax.set_ylabel("Expected Putts")
        ax.set_title("Expected Putts", fontsize=10)
        self.canvas.draw_idle()
        sg = report["sg_player"]
        sg_text = "" if sg is None else f" This putt gained {sg:+.2f} strokes."
        self._summary.setText(
            f"From {start:.1f} {unit} you average "
            f"{report['expected_putts']:.2f} putts.{sg_text}"
        )

    def redraw(self) -> None:
        """Re-express the leave and the chart in the current units."""
        leave_ft = self._leave_spin.value() / self._length_scale()[0]
        self._shown_units = self._units()
        self._label_leave()
        self.set_leave_display(leave_ft * self._length_scale()[0])
        if self._report is not None and not self._busy:
            self.show_report(self._report)

    def summary_text(self) -> str:
        return self._summary.text()

    def curve_axis_label(self) -> str:
        return str(self._ax.get_xlabel())

    def curve_title_extent(self) -> Bbox:
        """The drawn E(d) title's box in display pixels."""
        title = self._ax.title
        return title.get_window_extent()

    def curve_points(self) -> list[tuple[float, float]]:
        curve = (self._report or {"curve": {}})["curve"]
        return list(
            zip(
                curve.get("distances_ft", []),
                curve.get("expected_putts", []),
                strict=True,
            )
        )

    def _store(self, key: str, report: dict[str, Any]) -> None:
        """Cache a solved curve under the green and player that shaped it."""
        self._cache[key] = report
        self.show_report(report)

    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        self._summary.setText(f"Study rejected: {message}")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._score_btn.setEnabled(not busy)

    def _label_leave(self) -> None:
        scale, unit = self._length_scale()
        self._leave_spin.setRange(0.0, 45.0 * scale)
        self._leave_spin.setSuffix(f" {unit}")

    def _length_scale(self) -> tuple[float, str]:
        """Feet -> display length factor and unit, from the shared units module."""
        system = self._shown_units
        scale = _M_PER_FT if system == UnitSystem.METRIC else 1.0
        return scale, distance_suffix(system, unit="ft").strip()


def _curve_key(args: dict[str, Any]) -> str:
    """Everything that shapes the curve and the start, without the putt."""
    shape = {k: v for k, v in args.items() if k not in _PUTT_KEYS}
    return json.dumps(shape, sort_keys=True, default=str)


__all__ = ["SGPanel"]
