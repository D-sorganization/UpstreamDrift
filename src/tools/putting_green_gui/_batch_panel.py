"""Batch dispersion panel for the Putting Green Simulator (#14459, epic #14452).

Maps :class:`PuttConfig` to :mod:`service_dispersion` arguments (a noisy
batch and the 3-45 ft make spread) and plots the answer in the user's units.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from collections.abc import Callable
from typing import Any

import numpy as np
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
from src.shared.python.putting_analytics.service_variance import profile_from_args
from src.shared.python.ui.units import (
    UnitSystem,
    distance_suffix,
    from_display_distance,
)
from src.tools.putting_green_gui._scene_builder import PuttConfig

BATCH_SIZES: tuple[int, ...] = (40, 100, 1000)
# Putts per distance; one line fan per distance costs seconds, so use a worker.
SPREAD_N = 60
_STACK_BELOW_PX = 520  # narrower plot canvases stack the two charts
# The scene's cross-slope (``_cross_slope_heightmap``) raises the green to the
# left of the line, so it falls to the right: -90 deg, CCW from the line.
_FALL_RIGHT_DEG = -90.0
_M_PER_FT = from_display_distance(1.0, UnitSystem.IMPERIAL, unit="ft")
_MM_PER_IN = 1000.0 * _M_PER_FT / 12.0
# Noise channel -> (label, core value per shown value, upper bound, suffix).
# Strike is shown in mm or in, so its scale and suffix come from the units.
_CHANNELS: dict[str, tuple[str, float, float, str]] = {
    "face_sd_deg": ("Face SD:", 1.0, 10.0, "°"),
    "path_sd_deg": ("Path SD:", 1.0, 15.0, "°"),
    "speed_cv": ("Speed CV:", 0.01, 50.0, " %"),
    "strike_toe_sd_mm": ("Strike SD:", 1.0, 25.0, ""),
    "grade_rel_sd": ("Read Slope SD:", 0.01, 100.0, " %"),
    "aspect_sd_deg": ("Read Fall Line SD:", 1.0, 45.0, "°"),
    "stimp_rel_sd": ("Read Speed SD:", 0.01, 50.0, " %"),
}


def batch_args(
    config: PuttConfig, *, n: int, seed: int, **noise: float
) -> dict[str, Any]:
    """Dispersion-service arguments for the dialled putt and noise channels."""
    grade_pct = 100.0 * math.tan(math.radians(config.slope_deg))
    return {
        "distance_ft": config.cup_distance_m / _M_PER_FT,
        "grade_pct": grade_pct,
        "aspect_deg": _FALL_RIGHT_DEG if grade_pct else 0.0,
        "stimp": config.stimp,
        **noise,
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

    ``config`` gives the putt, ``units`` the units; ``start_*`` run on a worker.
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
        self._shown_units = units()
        self._noise: dict[str, QDoubleSpinBox] = {}
        profile = profile_from_args({})  # the core's defaults
        core = {**asdict(profile.stroke), **asdict(profile.read)}
        for key, (label, per_shown, top, suffix) in _CHANNELS.items():
            self._noise[key] = _spin(0.0, top, core[key] / per_shown, suffix)
            form.addRow(label, self._noise[key])
        strike_mm = core["strike_toe_sd_mm"]
        self._express_strike(strike_mm / self._per_shown("strike_toe_sd_mm"))
        self._size_combo = QComboBox()
        for size in BATCH_SIZES:
            self._size_combo.addItem(f"{size} putts", size)
        self._size_combo.setCurrentIndex(1)
        form.addRow("Batch:", self._size_combo)
        buttons = QHBoxLayout()
        self._batch_btn = QPushButton("Run Batch")
        self._batch_btn.clicked.connect(self.start_batch)
        self._spread_btn = QPushButton()
        self._spread_btn.clicked.connect(self.start_spread)
        buttons.addWidget(self._batch_btn)
        buttons.addWidget(self._spread_btn)
        form.addRow(buttons)
        self._label_spread()
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
        self._shown: dict[str, dict[str, Any]] = {}  # last report per chart

    @property
    def size_combo(self) -> QComboBox:
        return self._size_combo

    def noise_spin(self, key: str) -> QDoubleSpinBox:
        """The input for one noise channel, a ``profile_from_args`` key."""
        if key not in self._noise:
            raise ValueError(f"noise channel must be one of {tuple(self._noise)}")
        return self._noise[key]

    def current_args(self) -> dict[str, Any]:
        noise = {
            key: spin.value() * self._per_shown(key)
            for key, spin in self._noise.items()
        }
        return batch_args(
            self._config(),
            n=int(self._size_combo.currentData()),
            seed=self._seed,
            **noise,
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
        self._shown["batch"] = report
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
        self._shown["spread"] = report
        scale, unit = self._length_scale()
        ax = self._spread_ax
        ax.clear()
        xs = [d * scale for d in report["distances_ft"]]
        ax.plot(xs, [100 * m for m in report["make_rate"]], "o-", color="#2b7bd6")
        ax.set_xlabel(f"Distance ({unit})")
        ax.set_ylabel("Make %")
        ax.set_ylim(0, 100)
        ax.set_title("Make % by Distance", fontsize=10)
        self.canvas.draw_idle()

    def redraw(self) -> None:
        """Replot the last results in the current units; a running job redraws."""
        mm = self._noise["strike_toe_sd_mm"].value() * self._per_shown(
            "strike_toe_sd_mm"
        )
        self._shown_units = self._units()
        self._express_strike(mm / self._per_shown("strike_toe_sd_mm"))
        self._label_spread()
        if not self.is_busy():
            for kind, report in list(self._shown.items()):
                (self.show_batch if kind == "batch" else self.show_spread)(report)

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        canvas = getattr(self, "canvas", None)  # gone during teardown
        if obj is canvas and canvas is not None and event is not None:
            if event.type() == QEvent.Type.Resize:
                self.arrange_plots(self.canvas.width())
        return False

    def arrange_plots(self, width_px: int) -> None:
        """Charts side by side on wide canvases, stacked on narrow ones."""
        stacked = width_px < _STACK_BELOW_PX
        if stacked == self._stacked:
            return
        self._stacked = stacked
        grid = self._figure.add_gridspec(*((2, 1) if stacked else (1, 2)))
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
        offsets = np.asarray(self._scatter_ax.collections[0].get_offsets(), float)
        return [float(x) for x in offsets[:, 0]]

    def spread_points(self) -> list[tuple[float, float]]:
        report = self._shown.get("spread", {"distances_ft": [], "make_rate": []})
        return list(zip(report["distances_ft"], report["make_rate"], strict=True))

    def _start(
        self,
        study: Callable[[dict[str, Any]], dict[str, Any]],
        args: dict[str, Any],
        show: Callable[[dict[str, Any]], None],
    ) -> None:
        pool = QThreadPool.globalInstance()  # Qt makes it with the application
        if self._busy or pool is None:
            return
        self._set_busy(True)
        self._summary.setText("Rolling putts on the dialled green...")
        job = _Job(study, args)
        job.signals.done.connect(show)
        job.signals.failed.connect(self._on_failed)
        self._job = job  # keep the signals alive until delivery
        pool.start(job)

    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        self._summary.setText(f"Study rejected: {message}")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._batch_btn.setEnabled(not busy)
        self._spread_btn.setEnabled(not busy)

    def spread_label(self) -> str:
        """The spread button's text, e.g. ``0-45 ft Spread``."""
        return self._spread_btn.text()

    def _label_spread(self) -> None:
        scale, unit = self._length_scale()
        top = max(self._spread_grid_ft) * scale
        self._spread_btn.setText(f"0-{top:.0f} {unit} Spread")

    def _per_shown(self, key: str) -> float:
        """Core units per shown unit for a channel, in the units it shows."""
        if key == "strike_toe_sd_mm" and self._shown_units != UnitSystem.METRIC:
            return _MM_PER_IN
        return _CHANNELS[key][1]

    def _express_strike(self, shown: float) -> None:
        """Show the strike spread in mm (metric) or in (imperial)."""
        spin = self._noise["strike_toe_sd_mm"]
        metric = self._shown_units == UnitSystem.METRIC
        spin.setDecimals(1 if metric else 2)
        spin.setSingleStep(0.5 if metric else 0.05)
        spin.setRange(
            0.0, _CHANNELS["strike_toe_sd_mm"][2] / self._per_shown("strike_toe_sd_mm")
        )
        spin.setSuffix(" mm" if metric else " in")
        spin.setValue(shown)

    def _length_scale(self) -> tuple[float, str]:
        """Feet -> display length factor and unit, from the shared units module."""
        system = self._units()
        scale = _M_PER_FT if system == UnitSystem.METRIC else 1.0
        return scale, distance_suffix(system, unit="ft").strip()


def _spin(lo: float, hi: float, value: float, suffix: str) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setDecimals(1)
    spin.setValue(value)
    spin.setSuffix(suffix)
    return spin


__all__ = ["BATCH_SIZES", "SPREAD_N", "BatchPanel", "batch_args"]
