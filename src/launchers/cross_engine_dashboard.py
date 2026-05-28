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
import contextlib
import logging
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.pendulum_simulator.cross_engine_perturbation import (
    CrossEnginePerturbationRunner,
    CrossEngineSimConfig,
    CrossEngineRunResult,
)
from src.shared.python.plot_style import (
    MarkerShape,
    MarkerStyle,
    MatplotlibMarkerRenderer,
    PaletteColor,
    PlotStyleSet,
    PlotStyleSpec,
    PresetLibrary,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

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
        self._q = self._q + self._v * effective_dt  # type: ignore[assignment]
        self._v = self._v * damping  # type: ignore[assignment]

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (positions, velocities)."""
        return self._q.copy(), self._v.copy()


# ---------------------------------------------------------------------------
# Engine registry helpers
# ---------------------------------------------------------------------------

_ENGINE_NAMES = ("mujoco", "drake", "pinocchio", "pendulum_stub")

# ---------------------------------------------------------------------------
# Plot-style integration (issue #4810)
# ---------------------------------------------------------------------------
#
# Curated palette indices for trajectory overlays.  ``tab10`` is the
# colour-blind-friendly default; engines listed here get a deterministic
# entry, anything outside the table falls back to a hash-stable index so
# the overlay still renders.
#
# Five core engines are mapped explicitly per the issue acceptance
# criteria; ``pendulum_stub`` is included so the default GUI selection
# (which checks ``pendulum_stub``) also gets a deterministic colour.
_TRAJECTORY_PALETTE_NAME: str = "tab10"

_ENGINE_PALETTE_INDICES: dict[str, int] = {
    "drake": 0,
    "mujoco": 1,
    "pinocchio": 2,
    "opensim": 3,
    "simscape": 4,
    "pendulum_stub": 7,
}

# Distinct shapes per engine so colour-blind users can still discriminate
# overlapping traces.  Sphere is the default fallback.
_ENGINE_SHAPES: dict[str, MarkerShape] = {
    "drake": MarkerShape.SPHERE,
    "mujoco": MarkerShape.CUBE,
    "pinocchio": MarkerShape.DIAMOND,
    "opensim": MarkerShape.STAR,
    "simscape": MarkerShape.PLUS,
    "pendulum_stub": MarkerShape.POINT,
}


def _engine_palette_index(name: str) -> int:
    """Return the ``tab10`` palette index for ``name`` (deterministic)."""
    if name in _ENGINE_PALETTE_INDICES:
        return _ENGINE_PALETTE_INDICES[name]
    # Stable fallback — wraps modulo 10 for any unknown engine.
    return abs(hash(name)) % 10


def _default_marker_style_template() -> MarkerStyle | None:
    """Return the first style from the ``"default"`` preset, if available.

    The packaged preset is loaded lazily and missing-preset / load-error
    conditions degrade gracefully to ``None`` so the dashboard never
    breaks when run in a stripped environment.
    """
    try:
        library = PresetLibrary.default()
    except Exception:  # pragma: no cover - missing presets package  # noqa: BLE001
        logger.debug("PresetLibrary.default() unavailable; using fallback style")
        return None
    if "default" not in library:
        return None
    preset = library["default"]
    if not preset.entries:
        return None
    return preset.entries[0].style


def _build_engine_marker_style(
    name: str,
    *,
    shape_per_engine: bool = True,
    template: MarkerStyle | None = None,
) -> MarkerStyle:
    """Construct a per-engine :class:`MarkerStyle` for trajectory overlays.

    Parameters
    ----------
    name:
        Engine identifier.
    shape_per_engine:
        When ``True`` each engine uses its distinct :class:`MarkerShape`
        (colour-blind aid). When ``False`` every engine uses
        :data:`MarkerShape.SPHERE`.
    template:
        Optional template style (typically loaded from
        ``PresetLibrary.default()["default"]``); its size / edge / opacity
        attributes are reused so the overlay matches the global theme.
        ``None`` falls back to hard-coded sane defaults.

    Returns
    -------
    MarkerStyle
        A new frozen :class:`MarkerStyle` with a :class:`PaletteColor`
        fill keyed off ``tab10``.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"name must be a non-empty string; got {name!r}")
    fill = PaletteColor(
        palette_name=_TRAJECTORY_PALETTE_NAME,
        palette_index=_engine_palette_index(name),
    )
    shape = (
        _ENGINE_SHAPES.get(name, MarkerShape.SPHERE)
        if shape_per_engine
        else (MarkerShape.SPHERE)
    )
    if template is None:
        return MarkerStyle(
            shape=shape,
            size_px=6.0,
            edge_color="#101020",
            edge_width=0.5,
            fill_color=fill,
            opacity=1.0,
        )
    return MarkerStyle(
        shape=shape,
        size_px=float(template.size_px),
        edge_color=str(template.edge_color),
        edge_width=float(template.edge_width),
        fill_color=fill,
        opacity=float(template.opacity),
    )


def _render_trajectory_overlay(
    ax: Axes,
    trajectories: dict[str, np.ndarray],
    renderer: MatplotlibMarkerRenderer,
    *,
    shape_per_engine: bool = True,
    template: MarkerStyle | None = None,
) -> dict[str, str]:
    """Plot one trajectory per engine on ``ax`` using ``renderer``.

    A single :class:`MatplotlibMarkerRenderer` is reused across every
    engine series (DRY — see issue #4810) by binding it to ``ax`` and
    calling :meth:`MatplotlibMarkerRenderer.add_markers` per engine.

    Parameters
    ----------
    ax:
        Matplotlib 2D ``Axes`` already attached to a figure.
    trajectories:
        Mapping ``engine_name -> (T, D) ndarray`` of trajectories where
        ``D`` is at least 2.  The first two columns are plotted.
    renderer:
        :class:`MatplotlibMarkerRenderer` instance bound to ``ax``.  The
        same instance must be reused — ``RuntimeError`` if its default
        axes don't match.
    shape_per_engine:
        Forwarded to :func:`_build_engine_marker_style`.
    template:
        Forwarded to :func:`_build_engine_marker_style`.

    Returns
    -------
    dict[str, str]
        Mapping engine_name -> renderer handle (useful for later
        :meth:`MatplotlibMarkerRenderer.remove`).

    Design by Contract
    ------------------
    Pre:  trajectories is non-empty and every value has shape (T, D >= 2)
    Pre:  renderer's default axes is ``ax`` (DRY enforcement)
    Post: returns one handle per engine
    """
    if not trajectories:
        raise ValueError("trajectories must be non-empty")
    if getattr(renderer, "_default_ax", None) is not ax:
        raise RuntimeError(
            "MatplotlibMarkerRenderer must be bound to the supplied ax "
            "(DRY: a single renderer instance is required across overlays)"
        )
    handles: dict[str, str] = {}
    for engine_name, traj in trajectories.items():
        arr = np.asarray(traj, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(
                f"trajectory for {engine_name!r} must have shape (T, D>=2); "
                f"got {arr.shape}"
            )
        # Plot first two DOF as 2D scatter overlay.
        positions = arr[:, :2]
        style = _build_engine_marker_style(
            engine_name,
            shape_per_engine=shape_per_engine,
            template=template,
        )
        handle = renderer.add_markers(positions, style, label=engine_name)
        handles[engine_name] = handle
    return handles


def build_dashboard_style_set(
    engine_names: list[str] | tuple[str, ...] = _ENGINE_NAMES,
    *,
    shape_per_engine: bool = True,
) -> PlotStyleSet:
    """Return a :class:`PlotStyleSet` describing the dashboard overlay styles.

    Useful for persisting the styled session (issue #4810 — round-trip
    test). Each engine becomes a :class:`PlotStyleSpec` with target
    ``"trace:<engine>"`` so callers can save / load the configuration via
    :meth:`PlotStyleSet.save` / :meth:`PlotStyleSet.load`.
    """
    template = _default_marker_style_template()
    entries = tuple(
        PlotStyleSpec(
            name=name,
            target=f"trace:{name}",
            style=_build_engine_marker_style(
                name,
                shape_per_engine=shape_per_engine,
                template=template,
            ),
        )
        for name in engine_names
    )
    return PlotStyleSet(entries=entries)


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

            return MuJoCoPhysicsEngine()  # type: ignore[abstract]
        if name == "drake":
            from src.engines.physics_engines.drake.python.drake_physics_engine import (  # noqa: PLC0415
                DrakePhysicsEngine,
            )

            return DrakePhysicsEngine()  # type: ignore[abstract]
        if name == "pinocchio":
            from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (  # noqa: PLC0415
                PinocchioPhysicsEngine,
            )

            return PinocchioPhysicsEngine()

    except (ImportError, ValueError, RuntimeError):  # noqa: BLE001
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


def _run_with_results(
    engine_names: list[str],
    config: CrossEngineSimConfig,
) -> tuple[dict[str, CrossEngineRunResult], dict[str, float]]:
    """Execute the comparison and return both per-engine results and CV summary.

    DRY helper used by both the GUI worker (which needs trajectories for
    overlay rendering) and :func:`_run_headless` (which only logs the
    summary).
    """
    if not engine_names:
        raise ValueError("At least one engine name must be provided")
    runner = CrossEnginePerturbationRunner(config)
    for name in engine_names:
        runner.register_engine(name, _build_engine(name))  # type: ignore[arg-type]
    n_steps = round(config.t_end / config.dt)
    base_profile = np.zeros(n_steps)
    results = runner.run_comparison(base_profile)
    cv_summary = runner.compute_cv_summary(results)
    return results, cv_summary


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
    results, cv_summary = _run_with_results(engine_names, config)

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


def _create_dashboard_window_class() -> type:  # noqa: C901
    """Construct and return the _Window class with deferred Qt/mpl imports.

    Returns
    -------
    type
        A QMainWindow subclass ready to be instantiated.

    Raises
    ------
    ImportError if PyQt6 or Matplotlib is not available.
    """
    from PyQt6.QtCore import (  # noqa: PLC0415
        QObject,
        QRunnable,
        QThreadPool,
        pyqtSignal,
    )
    from PyQt6.QtWidgets import (  # noqa: PLC0415
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

    FigureCanvasQTAgg: Any = None
    Figure: Any = None
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: PLC0415
        from matplotlib.figure import Figure  # noqa: PLC0415

        _has_mpl = True
    except ImportError:
        _has_mpl = False

    class ComparisonWorkerSignals(QObject):
        """Signals for the ComparisonWorker."""

        # Payload: (engine_names, cv_summary, trajectories_per_engine)
        finished = pyqtSignal(list, dict, dict)
        error = pyqtSignal(str)

    class ComparisonWorker(QRunnable):
        """Worker thread for cross-engine comparison (issue #2715).

        Runs the CPU-heavy Monte Carlo simulation comparison in a background thread
        to prevent UI blocking.
        """

        def __init__(
            self, engine_names: list[str], config: CrossEngineSimConfig
        ) -> None:
            super().__init__()
            self.signals = ComparisonWorkerSignals()
            self.engine_names = engine_names
            self.config = config

        def run(self) -> None:
            """Execute comparison and emit results."""
            try:
                results, cv_summary = _run_with_results(self.engine_names, self.config)
                # Extract trial-0 position trajectory per engine for the overlay
                trajectories: dict[str, np.ndarray] = {}
                for name, run_result in results.items():
                    if run_result.metrics_per_trial:
                        traj = np.asarray(
                            run_result.metrics_per_trial[0].trajectory_q,
                            dtype=float,
                        )
                        if traj.ndim == 2 and traj.shape[1] >= 2:
                            trajectories[name] = traj
                self.signals.finished.emit(self.engine_names, cv_summary, trajectories)
            # Worker thread must survive any error to emit a signal back to the
            # GUI thread rather than crashing silently.
            except Exception as e:  # noqa: BLE001
                self.signals.error.emit(str(e))

    class _Window(QMainWindow):
        """Cross-Engine Perturbation Comparison Dashboard main window."""

        def __init__(
            self,
            parent: QWidget | None = None,
            *,
            shape_per_engine: bool = True,
        ) -> None:
            super().__init__(parent)
            self.setWindowTitle("Cross-Engine Perturbation Comparison Dashboard")
            self.setMinimumSize(900, 620)

            try:
                from src.shared.python.theme import apply_theme_to_window

                if callable(apply_theme_to_window):
                    apply_theme_to_window(self)
            except ImportError:
                pass

            self._shape_per_engine = bool(shape_per_engine)
            # Single MatplotlibMarkerRenderer reused across overlays — DRY
            # enforced by _render_trajectory_overlay (#4810).
            self._traj_renderer: MatplotlibMarkerRenderer | None = None
            self._traj_handles: dict[str, str] = {}
            self._style_template: MarkerStyle | None = _default_marker_style_template()

            central = QWidget()
            central.setObjectName("central")
            self.setCentralWidget(central)

            root = QHBoxLayout(central)
            root.setContentsMargins(8, 8, 8, 8)
            root.setSpacing(8)

            root.addWidget(self._build_config_panel(), stretch=0)
            if _has_mpl:
                root.addWidget(self._build_chart_panel(), stretch=1)

            # Thread pool for background tasks
            self._thread_pool = QThreadPool.globalInstance()

        # ------------------------------------------------------------------
        # Config panel
        # ------------------------------------------------------------------

        def _build_config_panel(self) -> QWidget:
            panel = QWidget()
            panel.setMinimumWidth(260)
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

        def _get_theme_colors(self) -> Any:
            """Retrieve the current theme colors mapped as a SimpleNamespace."""
            try:
                from types import SimpleNamespace
                from src.shared.python.theme import DARK_THEME, get_theme_manager

                tm = get_theme_manager()
                raw_c = tm.get_current_colors() if tm else None
                if raw_c:
                    return SimpleNamespace(
                        bg=raw_c.get("bg", "#1a1d23"),
                        bg_elevated=raw_c.get(
                            "table_header", raw_c.get("group_bg", "#2a2d35")
                        ),
                        text_secondary=raw_c.get(
                            "text_secondary", raw_c.get("label", "#8b949e")
                        ),
                        border_default=raw_c.get("border", "#3a3d45"),
                    )
                return DARK_THEME
            except ImportError:

                class FallbackColors:
                    bg = "#12121e"
                    bg_elevated = "#1a1a2e"
                    text_secondary = "#8080b0"
                    border_default = "#303050"

                return FallbackColors()

        # ------------------------------------------------------------------
        # Chart panel
        # ------------------------------------------------------------------

        def _build_chart_panel(self) -> QWidget:
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)

            c = self._get_theme_colors()

            # Robustness Score chart
            rs_grp = QGroupBox("Robustness Score (1 − CV, per engine)")
            rs_lay = QVBoxLayout(rs_grp)
            fig_rs = Figure(figsize=(5, 2.5), facecolor=c.bg)
            self._canvas_rs = FigureCanvasQTAgg(fig_rs)
            self._ax_rs = fig_rs.add_subplot(111)
            self._ax_rs.set_facecolor(c.bg_elevated)
            self._style_ax(self._ax_rs, c)
            rs_lay.addWidget(self._canvas_rs)
            layout.addWidget(rs_grp)

            # CV chart
            cv_grp = QGroupBox("Coefficient of Variation per Metric")
            cv_lay = QVBoxLayout(cv_grp)
            fig_cv = Figure(figsize=(5, 2.5), facecolor=c.bg)
            self._canvas_cv = FigureCanvasQTAgg(fig_cv)
            self._ax_cv = fig_cv.add_subplot(111)
            self._ax_cv.set_facecolor(c.bg_elevated)
            self._style_ax(self._ax_cv, c)
            cv_lay.addWidget(self._canvas_cv)
            layout.addWidget(cv_grp)

            # Trajectory overlay (issue #4810)
            tr_grp = QGroupBox("Trajectory Overlay (per-engine, plot_style)")
            tr_lay = QVBoxLayout(tr_grp)
            fig_tr = Figure(figsize=(5, 2.5), facecolor=c.bg)
            self._canvas_tr = FigureCanvasQTAgg(fig_tr)
            self._ax_tr = fig_tr.add_subplot(111)
            self._ax_tr.set_facecolor(c.bg_elevated)
            self._style_ax(self._ax_tr, c)
            tr_lay.addWidget(self._canvas_tr)
            layout.addWidget(tr_grp)

            # Bind a single renderer to the overlay axes — reused across
            # every comparison run.  DRY (#4810).
            self._traj_renderer = MatplotlibMarkerRenderer(self._ax_tr)

            return panel

        @staticmethod
        def _style_ax(ax: Any, colors: Any) -> None:
            """Apply theme styling to a Matplotlib axes."""
            ax.tick_params(colors=colors.text_secondary, labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor(colors.border_default)
            ax.yaxis.label.set_color(colors.text_secondary)
            ax.xaxis.label.set_color(colors.text_secondary)

        # ------------------------------------------------------------------
        # Slots
        # ------------------------------------------------------------------

        def _on_run(self) -> None:
            """Build config, run comparison in background thread."""
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

            worker = ComparisonWorker(selected, config)
            worker.signals.finished.connect(self._on_comparison_finished)
            worker.signals.error.connect(self._on_comparison_error)
            self._thread_pool.start(worker)

        def _on_comparison_finished(
            self,
            engine_names: list[str],
            cv_summary: dict[str, float],
            trajectories: dict[str, np.ndarray] | None = None,
        ) -> None:
            """Handle successful comparison completion."""
            self._update_charts(engine_names, cv_summary)
            self._update_trajectory_overlay(trajectories or {})
            self._status_label.setText("Done")
            self._run_btn.setEnabled(True)

        def _on_comparison_error(self, error_msg: str) -> None:
            """Handle comparison failure."""
            self._status_label.setText(f"Error: {error_msg}")
            logger.error("Comparison failed: %s", error_msg)
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

            c = self._get_theme_colors()

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
            self._style_ax(ax, c)

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
            self._style_ax(ax2, c)

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

        def _update_trajectory_overlay(
            self,
            trajectories: dict[str, np.ndarray],
        ) -> None:
            """Render per-engine trajectory overlays via plot_style (#4810).

            One :class:`PaletteColor` per engine, recognisable shape per
            engine (when ``shape_per_engine`` is enabled), all routed
            through the single :class:`MatplotlibMarkerRenderer`.
            """
            if not _has_mpl or self._traj_renderer is None:
                return
            c = self._get_theme_colors()
            ax = self._ax_tr
            # Remove any prior handles to avoid stacking artists across runs.
            for handle in list(self._traj_handles.values()):
                with contextlib.suppress(KeyError):  # pragma: no cover - defensive
                    self._traj_renderer.remove(handle)
            self._traj_handles.clear()
            ax.clear()
            ax.set_facecolor("#1a1a2e")
            self._style_ax(ax, c)
            if not trajectories:
                self._canvas_tr.draw()
                return
            self._traj_handles = _render_trajectory_overlay(
                ax,
                trajectories,
                self._traj_renderer,
                shape_per_engine=self._shape_per_engine,
                template=self._style_template,
            )
            ax.set_xlabel("q[0]", fontsize=9)
            ax.set_ylabel("q[1]", fontsize=9)
            ax.legend(
                list(self._traj_handles.keys()),
                fontsize=8,
                loc="best",
                facecolor="#1a1a2e",
                edgecolor="#303050",
                labelcolor="#d0d0f0",
            )
            self._canvas_tr.draw()

    return _Window


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def get_dockable_ui() -> object:
    """Return the dashboard window instance for docking in the unified launcher."""
    return _build_qt_window()


def _build_qt_window(*, shape_per_engine: bool = True) -> Any:
    """Build and return the QMainWindow instance (deferred Qt import).

    Parameters
    ----------
    shape_per_engine:
        Forwarded to the window constructor — when ``True`` each engine
        trajectory uses a distinct :class:`MarkerShape` (issue #4810).
    """
    cls = _create_dashboard_window_class()
    return cls(shape_per_engine=shape_per_engine)


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
    parser.add_argument(
        "--shape-per-engine",
        dest="shape_per_engine",
        action="store_true",
        default=True,
        help="Use a distinct MarkerShape per engine in trajectory overlays "
        "(colour-blind aid; issue #4810)",
    )
    parser.add_argument(
        "--no-shape-per-engine",
        dest="shape_per_engine",
        action="store_false",
        help="Force every engine to share MarkerShape.SPHERE",
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
    window = _build_qt_window(shape_per_engine=args.shape_per_engine)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
