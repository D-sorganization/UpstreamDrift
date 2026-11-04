"""Batch dispersion panel for the Putting Green Simulator (#14459, epic #14452).

Runs the shared dispersion service
(:mod:`src.shared.python.putting_analytics.service_dispersion`) on the green
the user has dialled in: a batch of 40/100/1000 putts with stroke noise
(face-angle SD and speed CV) plotted as rest positions around the hole, and
the 3-45 ft make-% spread. No physics lives here; the panel turns the
widget's :class:`PuttConfig` into service arguments and draws the answer in
the user's chosen units.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QEvent, QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
)

from src.shared.python.putting_analytics import service_dispersion
from src.shared.python.ui.units import UnitSystem
from src.tools.putting_green_gui._scene_builder import PuttConfig

BATCH_SIZES: tuple[int, ...] = (40, 100, 1000)
# Putts per distance. A spread's cost is dominated by planning one line fan
# per distance (seconds each), not by n, so the buttons run on a worker.
SPREAD_N = 60
_STACK_BELOW_PX = 520  # narrower plot canvases stack the two charts
_FT_M = 0.3048
# The scene's cross-slope raises the green to the left of the line
# (``_scene_builder._cross_slope_heightmap``), so it falls to the right:
# -90 deg, counter-clockwise from the line.
_FALL_RIGHT_DEG = -90.0


def batch_args(
    config: PuttConfig, *, face_sd_deg: float, speed_cv: float, n: int, seed: int
) -> dict[str, Any]:
    """Dispersion-service arguments for the dialled putt and stroke noise."""
    grade_pct = 100.0 * math.tan(math.radians(config.slope_deg))
    return {
        "distance_ft": config.cup_distance_m / _FT_M,
        "grade_pct": grade_pct,
        "aspect_deg": _FALL_RIGHT_DEG if grade_pct else 0.0,
        "stimp": config.stimp,
        "face_sd_deg": face_sd_deg,
        "speed_cv": speed_cv,
        "n": n,
        "seed": seed,
    }


class _Signals(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)


class _Job(QRunnable):
    """Runs one service study off the GUI thread (batches take seconds)."""

    def __init__(
        self, study: Callable[[dict[str, Any]], dict[str, Any]], args: dict[str, Any]
    ) -> None:
        super().__init__()
        self.signals = _Signals()
        self._study = study
        self._args = args

    def run(self) -> None:
        try:
            report = self._study(self._args)
        except ValueError as exc:
            self.signals.failed.emit(str(exc))
            return
        self.signals.done.emit(report)


class BatchPanel(QGroupBox):
    """Stroke-noise inputs, batch and spread buttons, and the result plot.

    ``config`` returns the widget's current putt; ``units`` its display units.
    The buttons run the service on a worker thread (:meth:`start_batch`,
    :meth:`start_spread`); :meth:`run_batch` and :meth:`run_spread` are the
    synchronous equivalents. The plot canvas is :attr:`canvas`.
    """

    def __init__(
        self,
        config: Callable[[], PuttConfig],
        units: Callable[[], UnitSystem],
        *,
        seed: int = 0,
        spread_n: int = SPREAD_N,
        spread_grid_ft: tuple[float, ...] = service_dispersion.DEFAULT_SPREAD_GRID_FT,
    ) -> None:
        super().__init__("Batch Dispersion")
        self._config = config
        self._units = units
        self._seed = seed
        self._spread_n = spread_n
        self._spread_grid_ft = spread_grid_ft
        self._busy = False
        form = QFormLayout(self)
        self._face_spin = _spin(0.0, 10.0, 1.5, "°")
        form.addRow("Face SD:", self._face_spin)
        self._speed_spin = _spin(0.0, 50.0, 8.0, " %")
        form.addRow("Speed CV:", self._speed_spin)
        self._size_combo = QComboBox()
        for size in BATCH_SIZES:
            self._size_combo.addItem(f"{size} putts", size)
        self._size_combo.setCurrentIndex(1)
        form.addRow("Batch:", self._size_combo)
        buttons = QHBoxLayout()
        self._batch_btn = QPushButton("Run Batch")
        self._batch_btn.clicked.connect(self.start_batch)
        self._spread_btn = QPushButton("0-45 ft Spread")
        self._spread_btn.clicked.connect(self.start_spread)
        buttons.addWidget(self._batch_btn)
        buttons.addWidget(self._spread_btn)
        form.addRow(buttons)
        self._summary = QLabel("Run a batch to see the make rate and leave.")
        self._summary.setWordWrap(True)
        form.addRow(self._summary)
        self._figure = Figure(figsize=(5.0, 2.6), tight_layout=True)
        self._scatter_ax = self._figure.add_subplot(1, 2, 1)
        self._spread_ax = self._figure.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasQTAgg(self._figure)
        # Let the splitter shrink the plot instead of widening the window.
        self.canvas.setMinimumSize(160, 220)
        self.canvas.installEventFilter(self)
        self._stacked = False
        self._spread: list[tuple[float, float]] = []

    @property
    def size_combo(self) -> QComboBox:
        return self._size_combo

    def current_args(self) -> dict[str, Any]:
        return batch_args(
            self._config(),
            face_sd_deg=self._face_spin.value(),
            speed_cv=self._speed_spin.value() / 100.0,
            n=int(self._size_combo.currentData()),
            seed=self._seed,
        )

    def spread_args(self) -> dict[str, Any]:
        args = self.current_args()
        args.pop("distance_ft")
        args["n"] = self._spread_n
        args["distances_ft"] = list(self._spread_grid_ft)
        return args

    def is_busy(self) -> bool:
        return self._busy

    def start_batch(self) -> None:
        self._start(
            service_dispersion.dispersion_study, self.current_args(), self.show_batch
        )

    def start_spread(self) -> None:
        self._start(
            service_dispersion.make_spread_study, self.spread_args(), self.show_spread
        )

    def run_batch(self) -> dict[str, Any]:
        """Roll the batch through the service now and plot it."""
        report = service_dispersion.dispersion_study(self.current_args())
        self.show_batch(report)
        return report

    def run_spread(self) -> dict[str, Any]:
        """Make rate by distance (3-45 ft) on the dialled green, now."""
        report = service_dispersion.make_spread_study(self.spread_args())
        self.show_spread(report)
        return report

    def show_batch(self, report: dict[str, Any]) -> None:
        """Plot a dispersion report's rest positions and summarise it."""
        self._set_busy(False)
        scale, unit = self._length_scale()
        ax = self._scatter_ax
        ax.clear()
        holed = report["holed"]
        xs = [x * scale for x in report["rest_x_ft"]]
        ys = [y * scale for y in report["rest_y_ft"]]
        ax.scatter(xs, ys, s=6, c=["#2e9d4a" if h else "#d2553f" for h in holed])
        ax.plot([0.0], [0.0], "ko", markersize=4)
        ax.set_xlabel(f"Past Hole ({unit})")
        ax.set_ylabel(f"Left of Line ({unit})")
        ax.set_title("Rest Positions", fontsize=10)
        self.canvas.draw_idle()
        leave = report["mean_leave_ft"]
        leave_text = "-" if leave is None else f"{leave * scale:.2f} {unit}"
        self._summary.setText(
            f"Make rate {100 * report['make_rate']:.0f}% of {report['n']} putts; "
            f"mean leave on misses {leave_text}."
        )

    def show_spread(self, report: dict[str, Any]) -> None:
        """Plot a make-spread report's make % against distance."""
        self._set_busy(False)
        self._spread = list(
            zip(report["distances_ft"], report["make_rate"], strict=True)
        )
        scale, unit = self._length_scale()
        ax = self._spread_ax
        ax.clear()
        ax.plot(
            [d * scale for d in report["distances_ft"]],
            [100 * m for m in report["make_rate"]],
            "o-",
            color="#2b7bd6",
        )
        ax.set_xlabel(f"Distance ({unit})")
        ax.set_ylabel("Make %")
        ax.set_ylim(0, 100)
        ax.set_title("Make % by Distance", fontsize=10)
        self.canvas.draw_idle()

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        if obj is self.canvas and event is not None:
            if event.type() == QEvent.Type.Resize:
                self.arrange_plots(self.canvas.width())
        return False

    def arrange_plots(self, width_px: int) -> None:
        """Charts side by side on wide canvases, stacked on narrow ones."""
        stacked = width_px < _STACK_BELOW_PX
        if stacked == self._stacked:
            return
        self._stacked = stacked
        grid = (
            self._figure.add_gridspec(2, 1)
            if stacked
            else self._figure.add_gridspec(1, 2)
        )
        self._scatter_ax.set_subplotspec(grid[0])
        self._spread_ax.set_subplotspec(grid[1])
        self.canvas.setMinimumHeight(440 if stacked else 220)
        self.canvas.draw_idle()

    def plots_stacked(self) -> bool:
        return self._stacked

    def summary_text(self) -> str:
        return self._summary.text()

    def scatter_axis_label(self) -> str:
        return str(self._scatter_ax.get_xlabel())

    def scatter_x(self) -> list[float]:
        offsets = self._scatter_ax.collections[0].get_offsets()
        return [float(x) for x in offsets[:, 0]]

    def spread_points(self) -> list[tuple[float, float]]:
        return list(self._spread)

    def _start(
        self,
        study: Callable[[dict[str, Any]], dict[str, Any]],
        args: dict[str, Any],
        show: Callable[[dict[str, Any]], None],
    ) -> None:
        if self._busy:
            return
        self._set_busy(True)
        self._summary.setText("Rolling putts on the dialled green...")
        job = _Job(study, args)
        job.signals.done.connect(show)
        job.signals.failed.connect(self._on_failed)
        self._job = job  # keep the signals alive until delivery
        QThreadPool.globalInstance().start(job)

    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        self._summary.setText(f"Study rejected: {message}")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._batch_btn.setEnabled(not busy)
        self._spread_btn.setEnabled(not busy)

    def _length_scale(self) -> tuple[float, str]:
        if self._units() == UnitSystem.METRIC:
            return _FT_M, "m"
        return 1.0, "ft"


def _spin(lo: float, hi: float, value: float, suffix: str) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setDecimals(1)
    spin.setValue(value)
    spin.setSuffix(suffix)
    return spin


__all__ = ["BATCH_SIZES", "SPREAD_N", "BatchPanel", "batch_args"]
