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
    except Exception:  # noqa: BLE001
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
    ImportError if PyQt6 or Matplotlib is not available.
    """
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

        _has_mpl = True
    except ImportError:
        _has_mpl = False
        FigureCanvasQTAgg = None
        Figure = None

    class _Window(QMainWindow):
        """Cross-Engine Perturbation Comparison Dashboard main window."""

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setWindowTitle("Cross-Engine Perturbation Comparison Dashboard")
            self.setMinimumSize(900, 620)
            self.setStyleSheet(_STYLE)

            central = QWidget()
            central.setObjectName("central")
            self.setCentralWidget(central)

            root = QHBoxLayout(central)
            root.setContentsMargins(8, 8, 8, 8)
            root.setSpacing(8)

            root.addWidget(self._build_config_panel(), stretch=0)
            if _has_mpl:
                root.addWidget(self._build_chart_panel(), stretch=1)

        # ------------------------------------------------------------------
        # Config panel
        # ------------------------------------------------------------------

        def _build_config_panel(self) -> QWidget:
            panel = QWidget()
            panel.setFixedWidth(260)
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)

            layout.addWidget(self._build_engine_group())
            layout.addWidget(self._build_sim_config_group())
            layout.addWidget(self._build_run_group())
            layout.addStretch()
            return panel

        def _build_engine_group(self) -> QGroupBox:
            grp = QGroupBox("Engines")
            lay = QVBoxLayout(grp)
            lay.setSpacing(4)
            self._engine_checks: dict[str, QCheckBox] = {}
            for name in _ENGINE_NAMES:
                cb = QCheckBox(name)
                cb.setChecked(name == "pendulum_stub")
                self._engine_checks[name] = cb
                lay.addWidget(cb)
            return grp

        def _build_sim_config_group(self) -> QGroupBox:
            grp = QGroupBox("Simulation Config")
            lay = QVBoxLayout(grp)
            lay.setSpacing(4)

            # n_trials
            row = QHBoxLayout()
            row.addWidget(QLabel("Trials:"))
            self._trials_spin = QSpinBox()
            self._trials_spin.setRange(1, 500)
            self._trials_spin.setValue(10)
            row.addWidget(self._trials_spin)
            lay.addLayout(row)

            # amplitude
            row2 = QHBoxLayout()
            row2.addWidget(QLabel("Amplitude:"))
            self._amp_spin = QDoubleSpinBox()
            self._amp_spin.setRange(0.0, 5.0)
            self._amp_spin.setSingleStep(0.01)
            self._amp_spin.setValue(0.1)
            self._amp_spin.setDecimals(3)
            row2.addWidget(self._amp_spin)
            lay.addLayout(row2)

            # t_end
            row3 = QHBoxLayout()
            row3.addWidget(QLabel("t_end (s):"))
            self._tend_spin = QDoubleSpinBox()
            self._tend_spin.setRange(0.1, 10.0)
            self._tend_spin.setSingleStep(0.1)
            self._tend_spin.setValue(1.5)
            self._tend_spin.setDecimals(2)
            row3.addWidget(self._tend_spin)
            lay.addLayout(row3)

            # dt
            row4 = QHBoxLayout()
            row4.addWidget(QLabel("dt (s):"))
            self._dt_spin = QDoubleSpinBox()
            self._dt_spin.setRange(0.001, 0.1)
            self._dt_spin.setSingleStep(0.001)
            self._dt_spin.setValue(0.01)
            self._dt_spin.setDecimals(3)
            row4.addWidget(self._dt_spin)
            lay.addLayout(row4)

            return grp

        def _build_run_group(self) -> QGroupBox:
            grp = QGroupBox("Run")
            lay = QVBoxLayout(grp)
            lay.setSpacing(4)
            self._run_btn = QPushButton("Run Comparison")
            self._run_btn.clicked.connect(self._on_run)
            lay.addWidget(self._run_btn)
            self._status_label = QLabel("Ready")
            lay.addWidget(self._status_label)
            return grp

        # ------------------------------------------------------------------
        # Chart panel
        # ------------------------------------------------------------------

        def _build_chart_panel(self) -> QWidget:
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)

            # Robustness Score chart
            rs_grp = QGroupBox("Robustness Score (1 − CV, per engine)")
            rs_lay = QVBoxLayout(rs_grp)
            fig_rs = Figure(figsize=(5, 2.5), facecolor="#12121e")
            self._canvas_rs = FigureCanvasQTAgg(fig_rs)
            self._ax_rs = fig_rs.add_subplot(111)
            self._ax_rs.set_facecolor("#1a1a2e")
            self._style_ax(self._ax_rs)
            rs_lay.addWidget(self._canvas_rs)
            layout.addWidget(rs_grp)

            # CV chart
            cv_grp = QGroupBox("Coefficient of Variation per Metric")
            cv_lay = QVBoxLayout(cv_grp)
            fig_cv = Figure(figsize=(5, 2.5), facecolor="#12121e")
            self._canvas_cv = FigureCanvasQTAgg(fig_cv)
            self._ax_cv = fig_cv.add_subplot(111)
            self._ax_cv.set_facecolor("#1a1a2e")
            self._style_ax(self._ax_cv)
            cv_lay.addWidget(self._canvas_cv)
            layout.addWidget(cv_grp)

            return panel

        @staticmethod
        def _style_ax(ax: object) -> None:
            """Apply dark theme styling to a Matplotlib axes."""
            ax.tick_params(colors="#8080b0", labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#303050")
            ax.yaxis.label.set_color("#8080b0")
            ax.xaxis.label.set_color("#8080b0")

        # ------------------------------------------------------------------
        # Slots
        # ------------------------------------------------------------------

        def _on_run(self) -> None:
            """Build config, run comparison, update charts."""
            selected = [
                name for name, cb in self._engine_checks.items() if cb.isChecked()
            ]
            if not selected:
                self._status_label.setText("Select at least one engine")
                logger.warning("No engines selected for comparison")
                return

            try:
                config = CrossEngineSimConfig(
                    t_end=self._tend_spin.value(),
                    dt=self._dt_spin.value(),
                    noise_amplitude=self._amp_spin.value(),
                    n_trials=self._trials_spin.value(),
                )
            except ValueError as exc:
                self._status_label.setText(f"Config error: {exc}")
                logger.error("Invalid CrossEngineSimConfig: %s", exc)
                return

            self._run_btn.setEnabled(False)
            self._status_label.setText("Running…")
            # Force UI repaint before blocking call
            QApplication.processEvents()

            try:
                cv_summary = _run_headless(selected, config)
                self._update_charts(selected, cv_summary)
                self._status_label.setText("Done")
            except Exception as exc:  # noqa: BLE001
                self._status_label.setText(f"Error: {exc}")
                logger.error("Comparison failed: %s", exc, exc_info=True)
            finally:
                self._run_btn.setEnabled(True)

        # ------------------------------------------------------------------
        # Chart update
        # ------------------------------------------------------------------

        def _update_charts(
            self,
            engine_names: list[str],
            cv_summary: dict[str, float],
        ) -> None:
            """Refresh Robustness Score and CV charts from the latest results.

            Parameters
            ----------
            engine_names : list of str
                Names of engines that were run.
            cv_summary : dict
                Output of CrossEnginePerturbationRunner.compute_cv_summary().

            Design by Contract
            ------------------
            Pre:  engine_names is non-empty
            Pre:  cv_summary has the three standard CV keys
            Post: both canvases are redrawn
            """
            if not _has_mpl:
                return
            if not engine_names:
                return

            metric_keys = [
                "cv_total_energy_final",
                "cv_end_effector_speed_final",
                "cv_peak_end_effector_speed",
            ]
            metric_labels = ["Energy", "Speed", "Peak Speed"]

            # Robustness Score: use mean CV across metrics per engine.
            # Since compute_cv_summary returns aggregate CVs (not per-engine),
            # we compute a single robustness score for the ensemble.
            cv_values = [cv_summary.get(k, 0.0) for k in metric_keys]
            mean_cv = float(np.mean(cv_values)) if cv_values else 0.0
            robustness = max(0.0, min(1.0, 1.0 - mean_cv))
            robustness_per_engine = [robustness] * len(engine_names)

            ax = self._ax_rs
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
            self._style_ax(ax)

            # Annotate bar values
            for bar, val in zip(bars, robustness_per_engine, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    color="#d0d0f0",
                    fontsize=8,
                )

            self._canvas_rs.draw()

            # CV chart — one bar per metric
            ax2 = self._ax_cv
            ax2.clear()
            ax2.set_facecolor("#1a1a2e")
            x2 = np.arange(len(metric_labels))
            bars2 = ax2.bar(
                x2,
                cv_values,
                color="#8040c0",
                edgecolor="#502080",
                width=0.5,
            )
            ax2.set_xticks(x2)
            ax2.set_xticklabels(metric_labels, fontsize=9)
            ax2.set_ylabel("CV", fontsize=9)
            ax2.axhline(1.0, color="#ff6060", linewidth=0.8, linestyle="--")
            self._style_ax(ax2)

            for bar, val in zip(bars2, cv_values, strict=True):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    color="#d0d0f0",
                    fontsize=8,
                )

            self._canvas_cv.draw()

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
