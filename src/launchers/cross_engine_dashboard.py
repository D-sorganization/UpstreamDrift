# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Cross-Engine Perturbation Comparison Dashboard.

Addresses GH2020: provides a PyQt6 interactive dashboard (and optional CLI)
for configuring and running ``CrossEnginePerturbationRunner`` across multiple
physics engines, then visualising the results as Matplotlib bar charts.

Features
--------
- Engine selection checkboxes (mujoco, drake, pinocchio, pendulum stub)
- Configurable ``CrossEngineSimConfig`` parameters via spin-boxes
- Robustness Score bar chart (1 - CV, per engine, per metric)
- CV bar chart (per metric across engines)
- CLI mode via ``--no-gui`` for headless / CI use

Design by Contract
------------------
- At least one engine must be checked before "Run Comparison" is clicked.
- All heavy imports (PyQt6, Matplotlib, engine modules) are deferred to
  runtime to ensure strict lazy-loading at import time.
- No print() calls; all output uses logging.

DRY
---
- Reuses ``CrossEnginePerturbationRunner``, ``CrossEngineSimConfig``, and
  ``SteppableEngine`` from cross_engine_perturbation.py.
- Style constants match existing dark-theme widgets in the codebase.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.pendulum_simulator.cross_engine_perturbation import (
    CrossEnginePerturbationRunner,
    CrossEngineSimConfig,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_STYLE = """
QMainWindow, QWidget#central {
    background: #12121e;
}
QGroupBox {
    color: #9090c8; font-size: 11px; font-weight: bold;
    border: 1px solid #303050; border-radius: 4px;
    margin-top: 8px; padding-top: 14px;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; }
QLabel { color: #8080b0; font-size: 11px; }
QPushButton {
    background: #262650; color: #b0b0e8; border: 1px solid #404070;
    border-radius: 3px; padding: 4px 12px; font-size: 11px;
}
QPushButton:hover { background: #303068; }
QPushButton:disabled { color: #505060; }
QSpinBox, QDoubleSpinBox {
    background: #1a1a2a; color: #b0b0e8; border: 1px solid #303050;
    border-radius: 2px; font-size: 11px; padding: 2px;
}
QCheckBox { color: #8080b0; font-size: 11px; }
QCheckBox::indicator:checked { background: #5555b0; }
"""

# ---------------------------------------------------------------------------
# Engine stub (graceful degradation when physics package is not installed)
# ---------------------------------------------------------------------------


class _StubEngine:
    """Minimal SteppableEngine stub for engines not available in the environment.

    Simulates a trivial overdamped first-order system for demonstration purposes.
    All state values remain near zero with random perturbations applied.

    Design by Contract
    ------------------
    Pre:  name must be a non-empty string
    Post: get_state() returns two 1-D arrays of equal length
    """

    def __init__(self, name: str, n_dof: int = 2) -> None:
        if not name:
            raise ValueError("Engine stub name must be non-empty")
        self._name = name
        self._n_dof = n_dof
        self._q = np.zeros(n_dof)
        self._v = np.zeros(n_dof)

    def reset(self) -> None:
        """Reset state to zero."""
        self._q = np.zeros(self._n_dof)
        self._v = np.zeros(self._n_dof)

    def set_control(self, u: np.ndarray) -> None:
        """Apply control as an impulse to velocity."""
        u_arr = np.asarray(u, dtype=float)
        n = min(len(u_arr), self._n_dof)
        self._v[:n] += u_arr[:n] * 0.01  # small gain

    def step(self, dt: float | None = None) -> None:
        """Integrate with Euler + damping."""
        effective_dt = dt if dt is not None else 0.01
        damping = 0.95
        self._q = self._q + self._v * effective_dt
        self._v = self._v * damping

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (positions, velocities)."""
        return self._q.copy(), self._v.copy()


# ---------------------------------------------------------------------------
# Engine registry helpers
# ---------------------------------------------------------------------------

_ENGINE_NAMES = ("mujoco", "drake", "pinocchio", "pendulum_stub")


@dataclass(frozen=True, slots=True)
class _QtBackends:
    """Deferred Qt and Matplotlib imports for the dashboard window."""

    qapplication: type
    qcheckbox: type
    qdoublespinbox: type
    qgroupbox: type
    qhboxlayout: type
    qlabel: type
    qmainwindow: type
    qpushbutton: type
    qspinbox: type
    qvboxlayout: type
    qwidget: type
    figure_canvas: type | None
    figure: type | None
    has_mpl: bool


def _load_qt_backends() -> _QtBackends:
    """Import the Qt widgets and optional Matplotlib backend on demand."""
    from PyQt6.QtWidgets import (  # noqa: PLC0415
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: PLC0415
        from matplotlib.figure import Figure  # noqa: PLC0415

        has_mpl = True
    except ImportError:
        FigureCanvasQTAgg = None
        Figure = None
        has_mpl = False

    return _QtBackends(
        qapplication=QApplication,
        qcheckbox=QCheckBox,
        qdoublespinbox=QDoubleSpinBox,
        qgroupbox=QGroupBox,
        qhboxlayout=QHBoxLayout,
        qlabel=QLabel,
        qmainwindow=QMainWindow,
        qpushbutton=QPushButton,
        qspinbox=QSpinBox,
        qvboxlayout=QVBoxLayout,
        qwidget=QWidget,
        figure_canvas=FigureCanvasQTAgg,
        figure=Figure,
        has_mpl=has_mpl,
    )


def _initialize_qt_window(window: object, backends: _QtBackends) -> None:
    """Apply the common window shell and add the fixed-width control panel."""
    window.setWindowTitle("Cross-Engine Perturbation Comparison Dashboard")
    window.setMinimumSize(900, 620)
    window.setStyleSheet(_STYLE)

    central = backends.qwidget()
    central.setObjectName("central")
    window.setCentralWidget(central)

    root = backends.qhboxlayout(central)
    root.setContentsMargins(8, 8, 8, 8)
    root.setSpacing(8)
    root.addWidget(_build_qt_config_panel(window, backends), stretch=0)
    if backends.has_mpl:
        root.addWidget(_build_qt_chart_panel(window, backends), stretch=1)


def _build_qt_config_panel(window: object, backends: _QtBackends) -> object:
    """Build the left-side configuration column."""
    panel = backends.qwidget()
    panel.setFixedWidth(260)
    layout = backends.qvboxlayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)

    layout.addWidget(_build_qt_engine_group(window, backends))
    layout.addWidget(_build_qt_sim_config_group(window, backends))
    layout.addWidget(_build_qt_run_group(window, backends))
    layout.addStretch()
    return panel


def _build_qt_engine_group(window: object, backends: _QtBackends) -> object:
    """Build the engine-selection group and cache its checkboxes."""
    grp = backends.qgroupbox("Engines")
    lay = backends.qvboxlayout(grp)
    lay.setSpacing(4)
    window._engine_checks = {}
    for name in _ENGINE_NAMES:
        cb = backends.qcheckbox(name)
        cb.setChecked(name == "pendulum_stub")
        window._engine_checks[name] = cb
        lay.addWidget(cb)
    return grp


def _build_qt_sim_config_group(window: object, backends: _QtBackends) -> object:
    """Build the simulation-parameter controls."""
    grp = backends.qgroupbox("Simulation Config")
    lay = backends.qvboxlayout(grp)
    lay.setSpacing(4)

    row = backends.qhboxlayout()
    row.addWidget(backends.qlabel("Trials:"))
    window._trials_spin = backends.qspinbox()
    window._trials_spin.setRange(1, 500)
    window._trials_spin.setValue(10)
    row.addWidget(window._trials_spin)
    lay.addLayout(row)

    row2 = backends.qhboxlayout()
    row2.addWidget(backends.qlabel("Amplitude:"))
    window._amp_spin = backends.qdoublespinbox()
    window._amp_spin.setRange(0.0, 5.0)
    window._amp_spin.setSingleStep(0.01)
    window._amp_spin.setValue(0.1)
    window._amp_spin.setDecimals(3)
    row2.addWidget(window._amp_spin)
    lay.addLayout(row2)

    row3 = backends.qhboxlayout()
    row3.addWidget(backends.qlabel("t_end (s):"))
    window._tend_spin = backends.qdoublespinbox()
    window._tend_spin.setRange(0.1, 10.0)
    window._tend_spin.setSingleStep(0.1)
    window._tend_spin.setValue(1.5)
    window._tend_spin.setDecimals(2)
    row3.addWidget(window._tend_spin)
    lay.addLayout(row3)

    row4 = backends.qhboxlayout()
    row4.addWidget(backends.qlabel("dt (s):"))
    window._dt_spin = backends.qdoublespinbox()
    window._dt_spin.setRange(0.001, 0.1)
    window._dt_spin.setSingleStep(0.001)
    window._dt_spin.setValue(0.01)
    window._dt_spin.setDecimals(3)
    row4.addWidget(window._dt_spin)
    lay.addLayout(row4)

    return grp


def _build_qt_run_group(window: object, backends: _QtBackends) -> object:
    """Build the run controls and status label."""
    grp = backends.qgroupbox("Run")
    lay = backends.qvboxlayout(grp)
    lay.setSpacing(4)
    window._run_btn = backends.qpushbutton("Run Comparison")
    window._run_btn.clicked.connect(window._on_run)
    lay.addWidget(window._run_btn)
    window._status_label = backends.qlabel("Ready")
    lay.addWidget(window._status_label)
    return grp


def _build_mpl_canvas(
    backends: _QtBackends,
    figsize: tuple[float, float],
) -> tuple[object, object]:
    """Create a themed Matplotlib canvas and axes pair."""
    assert backends.figure_canvas is not None
    assert backends.figure is not None
    figure = backends.figure(figsize=figsize, facecolor="#12121e")
    canvas = backends.figure_canvas(figure)
    axes = figure.add_subplot(111)
    axes.set_facecolor("#1a1a2e")
    _style_axes(axes)
    return canvas, axes


def _build_qt_chart_panel(window: object, backends: _QtBackends) -> object:
    """Build the right-side chart column."""
    panel = backends.qwidget()
    layout = backends.qvboxlayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)

    rs_grp = backends.qgroupbox("Robustness Score (1 − CV, per engine)")
    rs_lay = backends.qvboxlayout(rs_grp)
    window._canvas_rs, window._ax_rs = _build_mpl_canvas(backends, (5, 2.5))
    rs_lay.addWidget(window._canvas_rs)
    layout.addWidget(rs_grp)

    cv_grp = backends.qgroupbox("Coefficient of Variation per Metric")
    cv_lay = backends.qvboxlayout(cv_grp)
    window._canvas_cv, window._ax_cv = _build_mpl_canvas(backends, (5, 2.5))
    cv_lay.addWidget(window._canvas_cv)
    layout.addWidget(cv_grp)

    return panel


def _style_axes(ax: object) -> None:
    """Apply the dashboard's dark theme to a Matplotlib axes."""
    ax.tick_params(colors="#8080b0", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#303050")
    ax.yaxis.label.set_color("#8080b0")
    ax.xaxis.label.set_color("#8080b0")


def _annotate_bars(
    ax: object,
    bars: object,
    values: list[float],
    fmt: str = ".2f",
    offset: float = 0.02,
) -> None:
    """Add value labels above each bar."""
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{val:{fmt}}",
            ha="center",
            va="bottom",
            color="#d0d0f0",
            fontsize=8,
        )


def _draw_robustness_chart(
    window: object,
    engine_names: list[str],
    robustness: float,
) -> None:
    """Draw per-engine robustness score bar chart."""
    robustness_per_engine = [robustness] * len(engine_names)
    ax = window._ax_rs
    ax.clear()
    ax.set_facecolor("#1a1a2e")
    x = np.arange(len(engine_names))
    bars = ax.bar(
        x,
        robustness_per_engine,
        color="#5555b0",
        edgecolor="#303070",
        width=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(engine_names, fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Robustness Score", fontsize=9)
    ax.axhline(0.5, color="#ff6060", linewidth=0.8, linestyle="--")
    _style_axes(ax)
    _annotate_bars(ax, bars, robustness_per_engine, fmt=".2f")
    window._canvas_rs.draw()


def _draw_cv_chart(
    window: object,
    cv_values: list[float],
) -> None:
    """Draw coefficient of variation bar chart per metric."""
    ax = window._ax_cv
    ax.clear()
    ax.set_facecolor("#1a1a2e")
    x = np.arange(len(window._METRIC_LABELS))
    bars = ax.bar(
        x,
        cv_values,
        color="#8040c0",
        edgecolor="#502080",
        width=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(window._METRIC_LABELS, fontsize=9)
    ax.set_ylabel("CV", fontsize=9)
    ax.axhline(1.0, color="#ff6060", linewidth=0.8, linestyle="--")
    _style_axes(ax)
    _annotate_bars(ax, bars, cv_values, fmt=".3f", offset=0.01)
    window._canvas_cv.draw()


def _update_dashboard_charts(
    window: object,
    backends: _QtBackends,
    engine_names: list[str],
    cv_summary: dict[str, float],
) -> None:
    """Refresh the dashboard charts from the latest CV summary."""
    if not backends.has_mpl or not engine_names:
        return

    cv_values = [cv_summary.get(k, 0.0) for k in window._METRIC_KEYS]
    mean_cv = float(np.mean(cv_values)) if cv_values else 0.0
    robustness = max(0.0, min(1.0, 1.0 - mean_cv))

    _draw_robustness_chart(window, engine_names, robustness)
    _draw_cv_chart(window, cv_values)


def _handle_run(window: object, backends: _QtBackends) -> None:
    """Collect form values, run the comparison, and refresh the UI."""
    selected = [name for name, cb in window._engine_checks.items() if cb.isChecked()]
    if not selected:
        window._status_label.setText("Select at least one engine")
        logger.warning("No engines selected for comparison")
        return

    try:
        config = CrossEngineSimConfig(
            t_end=window._tend_spin.value(),
            dt=window._dt_spin.value(),
            noise_amplitude=window._amp_spin.value(),
            n_trials=window._trials_spin.value(),
        )
    except ValueError as exc:
        window._status_label.setText(f"Config error: {exc}")
        logger.error("Invalid CrossEngineSimConfig: %s", exc)
        return

    window._run_btn.setEnabled(False)
    window._status_label.setText("Running…")
    backends.qapplication.processEvents()

    try:
        cv_summary = _run_headless(selected, config)
        _update_dashboard_charts(window, backends, selected, cv_summary)
        window._status_label.setText("Done")
    except Exception as exc:  # noqa: BLE001
        window._status_label.setText(f"Error: {exc}")
        logger.error("Comparison failed: %s", exc, exc_info=True)
    finally:
        window._run_btn.setEnabled(True)


def _try_build_real_engine(name: str) -> object | None:
    """Attempt to instantiate a real physics engine by name.

    Returns the engine instance on success, or None if the package is
    unavailable.  All import errors are caught and logged as warnings.

    Parameters
    ----------
    name : str
        One of 'mujoco', 'drake', 'pinocchio'.

    Returns
    -------
    SteppableEngine instance or None
    """
    try:
        if name == "mujoco":
            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (  # noqa: PLC0415
                MuJoCoPhysicsEngine,
            )

            return MuJoCoPhysicsEngine()
        if name == "drake":
            from src.engines.physics_engines.drake.python.drake_physics_engine import (  # noqa: PLC0415
                DrakePhysicsEngine,
            )

            return DrakePhysicsEngine()
        if name == "pinocchio":
            from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (  # noqa: PLC0415
                PinocchioPhysicsEngine,
            )

            return PinocchioPhysicsEngine()
    except Exception:  # noqa: BLE001  # noqa: BLE001
        logger.warning("Engine '%s' unavailable — will use stub", name, exc_info=False)
    return None


def _build_engine(name: str) -> object:
    """Return a real engine instance or a stub if the real one is unavailable.

    Parameters
    ----------
    name : str
        Engine name.

    Returns
    -------
    SteppableEngine instance (real or stub)
    """
    if name == "pendulum_stub":
        return _StubEngine("pendulum_stub")
    real = _try_build_real_engine(name)
    if real is not None:
        return real
    return _StubEngine(name)


# ---------------------------------------------------------------------------
# Headless / CLI runner
# ---------------------------------------------------------------------------


def _run_headless(
    engine_names: list[str],
    config: CrossEngineSimConfig,
) -> dict[str, float]:
    """Run a cross-engine comparison without a GUI and log the results.

    Parameters
    ----------
    engine_names : list of str
        Names of engines to include.
    config : CrossEngineSimConfig
        Simulation configuration.

    Returns
    -------
    dict — CV summary (keys: cv_total_energy_final, cv_end_effector_speed_final,
    cv_peak_end_effector_speed)

    Design by Contract
    ------------------
    Pre:  len(engine_names) > 0
    Pre:  config is a valid CrossEngineSimConfig
    Post: returns a dict with the three CV keys
    """
    if not engine_names:
        raise ValueError("At least one engine name must be provided")

    runner = CrossEnginePerturbationRunner(config)
    for name in engine_names:
        engine = _build_engine(name)
        runner.register_engine(name, engine)

    n_steps = round(config.t_end / config.dt)
    base_profile = np.zeros(n_steps)
    results = runner.run_comparison(base_profile)
    cv_summary = runner.compute_cv_summary(results)

    logger.info("=== Cross-Engine Perturbation Comparison Results ===")
    for eng_name, result in results.items():
        logger.info(
            "  %s: mean_energy=%.4f, mean_speed=%.4f, mean_peak=%.4f",
            eng_name,
            result.mean_total_energy_final,
            result.mean_end_effector_speed_final,
            result.mean_peak_end_effector_speed,
        )
    logger.info("CV Summary:")
    for key, cv in cv_summary.items():
        logger.info("  %s: %.4f", key, cv)

    return cv_summary


# ---------------------------------------------------------------------------
# GUI — main window
# ---------------------------------------------------------------------------


class CrossEngineDashboardWindow:
    """Main window for the Cross-Engine Perturbation Comparison Dashboard.

    This class is only instantiated when PyQt6 is available.  All Qt imports
    are deferred to this class's module-level import block (see below).

    Parameters
    ----------
    parent : QWidget, optional

    Design by Contract
    ------------------
    Pre:  PyQt6 must be importable
    Post: window is shown and interactive after __init__ returns
    """

    def __new__(cls, *args: object, **kwargs: object) -> CrossEngineDashboardWindow:
        # Defer actual class body to _CrossEngineDashboardWindowImpl which is
        # created after verifying PyQt6 is available.
        raise NotImplementedError(
            "Do not instantiate CrossEngineDashboardWindow directly. "
            "Use create_window() instead."
        )


def _build_qt_window() -> object:
    """Build and return the QMainWindow instance (deferred Qt import).

    Returns
    -------
    QMainWindow subclass instance.

    Raises
    ------
    ImportError if PyQt6 is not available. Matplotlib-backed charts are
    omitted when Matplotlib cannot be imported.
    """
    backends = _load_qt_backends()

    class _Window(backends.qmainwindow):
        """Cross-Engine Perturbation Comparison Dashboard main window."""

        def __init__(self, parent: object | None = None) -> None:
            super().__init__(parent)
            _initialize_qt_window(self, backends)

        # ------------------------------------------------------------------
        # Config panel
        # ------------------------------------------------------------------

        def _build_config_panel(self) -> object:
            return _build_qt_config_panel(self, backends)

        def _build_engine_group(self) -> object:
            return _build_qt_engine_group(self, backends)

        def _build_sim_config_group(self) -> object:
            return _build_qt_sim_config_group(self, backends)

        def _build_run_group(self) -> object:
            return _build_qt_run_group(self, backends)

        # ------------------------------------------------------------------
        # Chart panel
        # ------------------------------------------------------------------

        def _build_chart_panel(self) -> object:
            return _build_qt_chart_panel(self, backends)

        def _style_ax(self, ax: object) -> None:
            _style_axes(ax)

        # ------------------------------------------------------------------
        # Slots
        # ------------------------------------------------------------------

        def _on_run(self) -> None:
            _handle_run(self, backends)

        # ------------------------------------------------------------------
        # Chart update
        # ------------------------------------------------------------------

        _METRIC_KEYS = [
            "cv_total_energy_final",
            "cv_end_effector_speed_final",
            "cv_peak_end_effector_speed",
        ]
        _METRIC_LABELS = ["Energy", "Speed", "Peak Speed"]

        def _update_charts(
            self,
            engine_names: list[str],
            cv_summary: dict[str, float],
        ) -> None:
            _update_dashboard_charts(self, backends, engine_names, cv_summary)

        def _draw_robustness_chart(
            self, engine_names: list[str], robustness: float
        ) -> None:
            _draw_robustness_chart(self, engine_names, robustness)

        def _draw_cv_chart(self, cv_values: list[float]) -> None:
            _draw_cv_chart(self, cv_values)

        @staticmethod
        def _annotate_bars(
            ax: object,
            bars: object,
            values: list[float],
            fmt: str = ".2f",
            offset: float = 0.02,
        ) -> None:
            _annotate_bars(ax, bars, values, fmt=fmt, offset=offset)

    return _Window()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the dashboard CLI."""
    parser = argparse.ArgumentParser(
        description="Cross-Engine Perturbation Comparison Dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        default=False,
        help="Run headless comparison and print results (no GUI)",
    )
    parser.add_argument(
        "--engines",
        type=str,
        default="pendulum_stub",
        help="Comma-separated engine names to compare (e.g. mujoco,pinocchio,pendulum_stub)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of Monte Carlo trials per engine",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.1,
        help="Noise amplitude (standard deviation, N·m)",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=1.5,
        help="Simulation duration in seconds",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Integration timestep in seconds",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the Cross-Engine Perturbation Comparison Dashboard.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments (defaults to sys.argv[1:]).

    Design by Contract
    ------------------
    Pre:  Python 3.10+
    Post: Runs comparison (headless) or shows GUI window; returns only after
          window is closed (GUI mode) or comparison finishes (headless).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    engine_names = [n.strip() for n in args.engines.split(",") if n.strip()]
    if not engine_names:
        logger.error("No engine names provided via --engines")
        sys.exit(1)

    config = CrossEngineSimConfig(
        t_end=args.t_end,
        dt=args.dt,
        noise_amplitude=args.amplitude,
        n_trials=args.n_trials,
    )

    if args.no_gui:
        _run_headless(engine_names, config)
        return

    # GUI mode
    try:
        from PyQt6.QtWidgets import QApplication  # noqa: PLC0415
    except ImportError:
        logger.warning("PyQt6 not available — falling back to headless mode")
        _run_headless(engine_names, config)
        return

    app = QApplication.instance() or QApplication(sys.argv)
    window = _build_qt_window()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
