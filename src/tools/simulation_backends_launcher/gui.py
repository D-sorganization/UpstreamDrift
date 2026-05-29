"""Simulation Backends main window.

Composes the backend factory and validation helpers from
:mod:`src.shared.python.simulation_backends` into a single, labelled PyQt6
tool. The window is intentionally thin: every action a test cares about is a
synchronous, dialog-free public method on :class:`MainWidget`
(:meth:`~MainWidget.run_rollout`, :meth:`~MainWidget.run_sweep`,
:meth:`~MainWidget.run_cross_validation`, :meth:`~MainWidget.export_trace_to`).
The button handlers merely wrap those methods (the export handler adds a
``QFileDialog``); they never block on a modal dialog inside the core logic.

Rollouts of the 2-DoF golf model are sub-millisecond, so the GUI runs them
synchronously rather than on a worker thread.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, cast

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtGui, QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.simulation_backends import (
    BackendError,
    GolfModelParams,
    available_backends,
    has_mujoco,
    has_warp,
    make_backend,
    validation,
)
from src.shared.python.simulation_backends.protocol import DynamicsProvider, SimState
from src.shared.python.simulation_backends.trace_io import write_trace

if TYPE_CHECKING:
    from src.shared.python.simulation_backends.protocol import Trace

logger = get_logger(__name__)

__all__ = ["MainWidget", "SimulationBackendsWindow", "main"]

#: Number of clubhead-mass samples evaluated by the parameter sweep. Kept small
#: so the whole sweep (24 short ODE rollouts) finishes well under two seconds.
_SWEEP_SAMPLES = 24

#: Raised initial pose used for rollouts / sweeps so gravity drives a visible
#: swing. A passive rollout from the hanging zero pose would not move.
_INITIAL_Q = (1.2, -0.6)

#: Errors the synchronous actions catch and surface in the UI rather than
#: letting them crash the event loop.
_ACTION_ERRORS = (BackendError, ValueError)

#: Backends that expose a CPU dynamics provider usable for the sweep and
#: cross-validation (``mjwarp`` is GPU-only and unavailable on this host).
_CPU_BACKENDS = ("ode", "mujoco")


def _backend_display_label(name: str) -> str:
    """Return a user-facing label annotating unavailable backends.

    Args:
        name: Raw backend name from
            :func:`~src.shared.python.simulation_backends.available_backends`.

    Returns:
        The name, suffixed with a parenthetical note when the backend cannot
        actually run on this host (no GPU for ``mjwarp``; MuJoCo not installed).
    """
    if name == "mjwarp" and not has_warp():
        return "mjwarp (GPU not available)"
    if name == "mujoco" and not has_mujoco():
        return "mujoco (not installed)"
    return name


class MainWidget(QtWidgets.QWidget):
    """Central widget hosting the Simulation Backends UI.

    Holds all controls and the matplotlib canvas. It can be embedded inside a
    launcher tab/dock or wrapped by :class:`SimulationBackendsWindow` for
    standalone use.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._defaults = GolfModelParams.default()
        self._last_trace: Trace | None = None

        self._build_widgets()
        self._build_layout()
        self._wire_signals()
        self._apply_theme_best_effort()

        self._refresh_capabilities_label()
        self.status_label.setText("Ready. Pick a backend and run a rollout.")

    # ---- construction --------------------------------------------------

    def _build_widgets(self) -> None:
        self._build_backend_group()
        self._build_param_group()
        self._build_run_group()
        self._build_output_widgets()

    def _build_backend_group(self) -> None:
        self.backend_group = QtWidgets.QGroupBox("Physics backend")
        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.setToolTip(
            "Backend that integrates the golf double-pendulum model. "
            "Unavailable backends are annotated; ODE is always available."
        )
        for name in available_backends():
            self.backend_combo.addItem(_backend_display_label(name), userData=name)
        # Prefer ODE (always available) as the initial selection.
        ode_index = self.backend_combo.findData("ode")
        if ode_index >= 0:
            self.backend_combo.setCurrentIndex(ode_index)

        self.capabilities_label = QtWidgets.QLabel()
        self.capabilities_label.setObjectName("CapabilitiesLabel")
        self.capabilities_label.setWordWrap(True)
        self.capabilities_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.capabilities_label.setToolTip(
            "Static capabilities reported by the selected backend."
        )

    def _build_param_group(self) -> None:
        self.param_group = QtWidgets.QGroupBox("Model parameters")
        upper = self._defaults.upper
        lower = self._defaults.lower

        self.upper_mass_spin = self._make_double_spin(
            value=upper.mass_kg,
            minimum=0.5,
            maximum=30.0,
            step=0.1,
            decimals=3,
            suffix=" kg",
            tooltip="Combined-arms (upper segment) mass.",
        )
        self.lower_clubhead_mass_spin = self._make_double_spin(
            value=lower.clubhead_mass_kg,
            minimum=0.05,
            maximum=2.0,
            step=0.01,
            decimals=3,
            suffix=" kg",
            tooltip="Clubhead point mass at the distal end of the shaft.",
        )
        self.damping_wrist_spin = self._make_double_spin(
            value=self._defaults.damping_wrist,
            minimum=0.0,
            maximum=5.0,
            step=0.01,
            decimals=3,
            suffix=" N*m*s/rad",
            tooltip="Viscous damping at the wrist joint.",
        )
        self.plane_incl_spin = self._make_double_spin(
            value=self._defaults.plane_inclination_deg,
            minimum=-90.0,
            maximum=90.0,
            step=1.0,
            decimals=2,
            suffix=" deg",
            tooltip="Swing-plane tilt from vertical. Scales effective gravity.",
        )

        self.gravity_check = QtWidgets.QCheckBox("Gravity enabled")
        self.gravity_check.setChecked(self._defaults.gravity_enabled)
        self.gravity_check.setToolTip(
            "When unchecked, gravity is disabled (a conservative, free swing)."
        )

    def _build_run_group(self) -> None:
        self.run_group = QtWidgets.QGroupBox("Run controls")
        self.horizon_spin = QtWidgets.QSpinBox()
        self.horizon_spin.setRange(10, 5000)
        self.horizon_spin.setValue(300)
        self.horizon_spin.setSuffix(" steps")
        self.horizon_spin.setToolTip("Number of integration steps to roll out.")

        self.dt_spin = self._make_double_spin(
            value=0.005,
            minimum=0.0001,
            maximum=0.1,
            step=0.001,
            decimals=4,
            suffix=" s",
            tooltip="Integration step size.",
        )

        self.run_button = QtWidgets.QPushButton("Run Rollout")
        self.run_button.setToolTip(
            "Integrate a passive rollout with the selected backend and plot "
            "the joint-angle trajectories."
        )
        self.sweep_button = QtWidgets.QPushButton("Run Parameter Sweep")
        self.sweep_button.setToolTip(
            "Sweep the clubhead mass and plot a clubhead-speed proxy."
        )
        self.crossval_button = QtWidgets.QPushButton("Cross-validate vs ODE")
        self.crossval_button.setToolTip(
            "Compare MuJoCo's mass matrix and trajectory against the analytical "
            "ODE reference."
        )
        self.export_button = QtWidgets.QPushButton("Export HDF5...")
        self.export_button.setToolTip(
            "Write the last rollout to a versioned HDF5 trace file."
        )

    def _build_output_widgets(self) -> None:
        self.figure = Figure(figsize=(5.0, 3.5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self._axes = self.figure.add_subplot(111)
        self._init_axes()

        self.report_text = QtWidgets.QPlainTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setPlaceholderText(
            "Rollout, sweep, and cross-validation reports appear here."
        )
        self.report_text.setToolTip("Numeric results and validation reports.")

        self.status_label = QtWidgets.QLabel()
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setWordWrap(True)

    def _make_double_spin(
        self,
        *,
        value: float,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int,
        suffix: str,
        tooltip: str,
    ) -> QtWidgets.QDoubleSpinBox:
        """Build a configured ``QDoubleSpinBox`` (DRY helper)."""
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setToolTip(tooltip)
        return spin

    def _build_layout(self) -> None:
        backend_form = QtWidgets.QFormLayout(self.backend_group)
        backend_form.addRow("Backend:", self.backend_combo)
        backend_form.addRow("Capabilities:", self.capabilities_label)

        param_form = QtWidgets.QFormLayout(self.param_group)
        param_form.addRow("Upper segment mass:", self.upper_mass_spin)
        param_form.addRow("Clubhead mass:", self.lower_clubhead_mass_spin)
        param_form.addRow("Wrist damping:", self.damping_wrist_spin)
        param_form.addRow("Plane inclination:", self.plane_incl_spin)
        param_form.addRow("", self.gravity_check)

        run_layout = QtWidgets.QVBoxLayout(self.run_group)
        run_form = QtWidgets.QFormLayout()
        run_form.addRow("Horizon:", self.horizon_spin)
        run_form.addRow("Time step:", self.dt_spin)
        run_layout.addLayout(run_form)
        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.sweep_button)
        run_layout.addWidget(self.crossval_button)
        run_layout.addWidget(self.export_button)
        run_layout.addStretch(1)

        controls_panel = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(self.backend_group)
        controls_layout.addWidget(self.param_group)
        controls_layout.addWidget(self.run_group)
        controls_layout.addStretch(1)

        output_panel = QtWidgets.QWidget()
        output_layout = QtWidgets.QVBoxLayout(output_panel)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.canvas, stretch=3)
        report_label = QtWidgets.QLabel("Report")
        output_layout.addWidget(report_label)
        output_layout.addWidget(self.report_text, stretch=2)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(controls_panel)
        splitter.addWidget(output_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 560])

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.addWidget(splitter, stretch=1)
        outer.addWidget(self.status_label)

    def _wire_signals(self) -> None:
        self.backend_combo.currentIndexChanged.connect(self._refresh_capabilities_label)
        self.run_button.clicked.connect(self.run_rollout)
        self.sweep_button.clicked.connect(self.run_sweep)
        self.crossval_button.clicked.connect(self.run_cross_validation)
        self.export_button.clicked.connect(self._on_export_clicked)

    def _apply_theme_best_effort(self) -> None:
        """Apply the app theme if available; never fatal when absent."""
        try:
            from sidekick.theme import apply_theme_to_window
        except ImportError:
            return
        try:
            apply_theme_to_window(self)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.debug("Theme application skipped: %s", exc)

    def _init_axes(self) -> None:
        """Reset the plot axes to the empty trajectory state."""
        self._axes.clear()
        self._axes.set_xlabel("time [s]")
        self._axes.set_ylabel("joint angle [rad]")
        self._axes.set_title("Joint trajectories")
        self._axes.grid(True, alpha=0.3)

    def _refresh_capabilities_label(self) -> None:
        """Update the capabilities label from the selected backend.

        Builds the backend with the *default* model just to read its static
        capability flags. Construction failures (e.g. a GPU backend on a host
        without CUDA) are reported inline rather than raised.
        """
        name = self.current_backend_name()
        try:
            backend = make_backend(name, self._defaults)
        except _ACTION_ERRORS as exc:
            self.capabilities_label.setText(
                f"{name}: unavailable on this host ({exc})."
            )
            return
        caps = backend.capabilities
        self.capabilities_label.setText(
            f"name={caps.name}, device={caps.device}, "
            f"batched={caps.supports_batched}, "
            f"differentiable={caps.is_differentiable}, "
            f"dynamics={caps.provides_dynamics}"
        )

    # ---- testable core methods -----------------------------------------

    def current_backend_name(self) -> str:
        """Return the raw name of the currently selected backend.

        Returns:
            The backend id stored as combo ``itemData`` (e.g. ``"ode"``),
            falling back to the first available backend if no data is set.
        """
        data = self.backend_combo.currentData()
        if isinstance(data, str) and data:
            return data
        return available_backends()[0]

    def current_params(self) -> GolfModelParams:
        """Derive a :class:`GolfModelParams` from the current spin controls.

        Returns:
            An immutable model derived from the frozen defaults with the
            upper-segment mass, clubhead mass, wrist damping, plane inclination,
            and gravity flag overridden from the UI.
        """
        base = self._defaults
        upper = base.upper.model_copy(
            update={"mass_kg": float(self.upper_mass_spin.value())}
        )
        lower = base.lower.model_copy(
            update={"clubhead_mass_kg": float(self.lower_clubhead_mass_spin.value())}
        )
        return base.model_copy(
            update={
                "upper": upper,
                "lower": lower,
                "damping_wrist": float(self.damping_wrist_spin.value()),
                "plane_inclination_deg": float(self.plane_incl_spin.value()),
                "gravity_enabled": bool(self.gravity_check.isChecked()),
            }
        )

    def run_rollout(self) -> None:
        """Run a passive rollout and plot the joint-angle trajectories.

        Builds the selected backend from :meth:`current_params`, integrates a
        passive (zero-torque) rollout from a raised initial pose, stores the
        result on ``self._last_trace``, and updates the plot and status label.
        Backend/validation errors are caught and reported, never raised.
        """
        name = self.current_backend_name()
        horizon = int(self.horizon_spin.value())
        dt = float(self.dt_spin.value())
        try:
            backend = make_backend(name, self.current_params())
            backend.reset(self._initial_state())
            trace = backend.rollout(None, horizon, dt)
        except _ACTION_ERRORS as exc:
            self._report_failure("Rollout failed", exc)
            return

        self._last_trace = trace
        self._plot_trajectory(trace)
        self.status_label.setText(
            f"Rollout complete: backend={trace.backend}, "
            f"{trace.num_steps} samples, dt={dt:g} s."
        )

    def run_sweep(self) -> None:
        """Sweep the clubhead mass and plot a clubhead-speed proxy.

        Evaluates :data:`_SWEEP_SAMPLES` clubhead masses centred on the current
        value, rolls each out passively on a CPU backend, and plots
        ``|v_final|`` (a clubhead-speed proxy [rad/s]) versus clubhead mass. A
        short textual summary is written to the report pane.
        """
        backend_name = self._cpu_backend_name()
        horizon = int(self.horizon_spin.value())
        dt = float(self.dt_spin.value())
        center = float(self.lower_clubhead_mass_spin.value())
        masses = self._sweep_masses(center)
        try:
            speeds = self._sweep_speeds(backend_name, masses, horizon, dt)
        except _ACTION_ERRORS as exc:
            self._report_failure("Parameter sweep failed", exc)
            return

        self._plot_sweep(masses, speeds)
        best_idx = int(np.argmax(speeds))
        self.report_text.setPlainText(
            "Clubhead-mass sweep\n"
            "===================\n"
            f"backend: {backend_name}\n"
            f"samples: {len(masses)} "
            f"({masses[0]:.3f}..{masses[-1]:.3f} kg)\n"
            f"horizon: {horizon} steps, dt={dt:g} s\n\n"
            f"fastest swing: {speeds[best_idx]:.4f} rad/s "
            f"at {masses[best_idx]:.3f} kg\n"
            f"slowest swing: {min(speeds):.4f} rad/s\n\n"
            "Speed proxy = norm(final joint velocity) of a passive,\n"
            "gravity-driven swing from a raised initial pose."
        )
        self.status_label.setText(
            f"Sweep complete: {len(masses)} clubhead masses on {backend_name}."
        )

    def run_cross_validation(self) -> None:
        """Cross-validate MuJoCo against the analytical ODE backend.

        Builds the ODE and MuJoCo backends from :meth:`current_params`, runs the
        mass-matrix and trajectory cross-checks, and renders both
        :class:`~src.shared.python.simulation_backends.validation.ValidationReport`
        objects into the report pane. When MuJoCo is unavailable an explanatory
        note is written instead. Never raises.
        """
        if not has_mujoco():
            self.report_text.setPlainText(
                "Cross-validation unavailable\n"
                "============================\n"
                "MuJoCo is not installed, so there is no second backend to "
                "compare the analytical ODE reference against.\n\n"
                "Install the MuJoCo extra to enable cross-validation:\n"
                "  pip install upstream-drift[mujoco]"
            )
            self.status_label.setText("Cross-validation skipped: MuJoCo not installed.")
            return

        params = self.current_params()
        q_samples = [
            np.array([0.1, -0.2]),
            np.array([0.6, 0.3]),
            np.array([-0.4, 0.5]),
        ]
        try:
            ode = make_backend("ode", params)
            mj = make_backend("mujoco", params)
            # ode/mujoco backends both implement DynamicsProvider at runtime;
            # narrow the static SimulationBackend type for the mass-matrix check.
            mass_report = validation.cross_validate_mass_matrix(
                cast(DynamicsProvider, ode), cast(DynamicsProvider, mj), q_samples
            )
            traj_report = validation.cross_validate_trajectory(
                ode, mj, None, min(200, int(self.horizon_spin.value())), 0.005
            )
        except _ACTION_ERRORS as exc:
            self._report_failure("Cross-validation failed", exc)
            return

        self.report_text.setPlainText(
            "ODE vs MuJoCo cross-validation\n"
            "==============================\n"
            + _format_report(mass_report)
            + "\n\n"
            + _format_report(traj_report)
        )
        both_passed = mass_report.passed and traj_report.passed
        verdict = "PASS" if both_passed else "FAIL"
        self.status_label.setText(f"Cross-validation complete: {verdict}.")

    def export_trace_to(self, path: str) -> bool:
        """Write the last rollout to an HDF5 trace file.

        Args:
            path: Destination filesystem path for the HDF5 trace.

        Returns:
            ``True`` once the trace is written.

        Raises:
            ValueError: If no rollout has been run yet (nothing to export) or
                ``path`` is empty.
        """
        if self._last_trace is None:
            raise ValueError(
                "No rollout to export yet; run a rollout before exporting."
            )
        if not str(path).strip():
            raise ValueError("export path must be a non-empty filesystem path")
        write_trace(self._last_trace, path)
        self.status_label.setText(f"Exported trace to {path}.")
        return True

    # ---- internal helpers ----------------------------------------------

    def _initial_state(self) -> SimState:
        """Return the raised initial state used for rollouts/sweeps."""
        return SimState(q=np.array(_INITIAL_Q, dtype=float), v=np.zeros(2))

    def _cpu_backend_name(self) -> str:
        """Return a CPU backend usable for sweep/dynamics work.

        Prefers the current selection when it is a CPU backend; otherwise falls
        back to ODE (always available).
        """
        name = self.current_backend_name()
        if name in _CPU_BACKENDS and (name != "mujoco" or has_mujoco()):
            return name
        return "ode"

    def _sweep_masses(self, center: float) -> np.ndarray:
        """Return the clubhead-mass sample grid centred on ``center``."""
        low = max(0.05, center * 0.5)
        high = max(low + 0.05, center * 1.5)
        return np.linspace(low, high, _SWEEP_SAMPLES)

    def _sweep_speeds(
        self, backend_name: str, masses: np.ndarray, horizon: int, dt: float
    ) -> list[float]:
        """Roll out each clubhead mass passively and return speed proxies."""
        base = self.current_params()
        initial = self._initial_state()
        speeds: list[float] = []
        for mass in masses:
            lower = base.lower.model_copy(update={"clubhead_mass_kg": float(mass)})
            params = base.model_copy(update={"lower": lower})
            backend = make_backend(backend_name, params)
            backend.reset(initial.copy())
            trace = backend.rollout(None, horizon, dt)
            speeds.append(float(np.linalg.norm(trace.v[-1])))
        return speeds

    def _plot_trajectory(self, trace: Trace) -> None:
        """Plot ``theta1``/``theta2`` versus time for ``trace``."""
        self._init_axes()
        self._axes.plot(trace.t, trace.q[:, 0], label="theta1 (shoulder)")
        self._axes.plot(trace.t, trace.q[:, 1], label="theta2 (wrist)")
        self._axes.legend(loc="best", fontsize="small")
        self._axes.set_title(f"Joint trajectories ({trace.backend})")
        self.canvas.draw_idle()

    def _plot_sweep(self, masses: np.ndarray, speeds: list[float]) -> None:
        """Plot the clubhead-speed proxy versus clubhead mass."""
        self._axes.clear()
        self._axes.plot(masses, speeds, marker="o")
        self._axes.set_xlabel("clubhead mass [kg]")
        self._axes.set_ylabel("clubhead-speed proxy [rad/s]")
        self._axes.set_title("Clubhead-mass sweep")
        self._axes.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def _report_failure(self, headline: str, exc: Exception) -> None:
        """Surface an action failure in the status/report panes and log it."""
        logger.exception(headline)
        message = f"{headline}: {exc}"
        self.status_label.setText(message)
        self.report_text.setPlainText(message)

    def _on_export_clicked(self) -> None:
        """Button handler: pick a path via dialog, then export."""
        if self._last_trace is None:
            self.status_label.setText("Nothing to export yet; run a rollout first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export trace to HDF5",
            "trace.h5",
            "HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        if not path:
            return
        try:
            self.export_trace_to(path)
        except (ValueError, OSError) as exc:
            self._report_failure("Export failed", exc)


class SimulationBackendsWindow(QtWidgets.QMainWindow):
    """Standalone window wrapping :class:`MainWidget` with a menu bar."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Simulation Backends")
        self.resize(1100, 720)
        self._main_widget = MainWidget(self)
        self.setCentralWidget(self._main_widget)
        self._build_menu_bar()

    @property
    def main_widget(self) -> MainWidget:
        """Return the embedded :class:`MainWidget`."""
        return self._main_widget

    def _build_menu_bar(self) -> None:
        menubar = self.menuBar()
        assert menubar is not None  # noqa: S101 — Qt invariant

        file_menu = menubar.addMenu("&File")
        help_menu = menubar.addMenu("&Help")
        assert file_menu is not None  # noqa: S101 — Qt invariant
        assert help_menu is not None  # noqa: S101 — Qt invariant

        act_quit = QtGui.QAction("&Quit", self)
        act_quit.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        act_about = QtGui.QAction("&About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _show_about(self) -> None:
        """Show an about dialog explaining what the tool is and how to use it."""
        QtWidgets.QMessageBox.about(
            self,
            "About Simulation Backends",
            "<b>Simulation Backends</b><br><br>"
            "Compare physics backends for the golf double-pendulum model: "
            "the analytical <i>ODE</i> reference, <i>MuJoCo</i> (CPU), and "
            "<i>MuJoCo Warp</i> (GPU, when available).<br><br>"
            "<b>How to use</b>:"
            "<ol>"
            "<li><b>Run Rollout</b> &mdash; integrate a passive swing with the "
            "selected backend and plot the joint angles.</li>"
            "<li><b>Run Parameter Sweep</b> &mdash; vary the clubhead mass and "
            "plot a clubhead-speed proxy.</li>"
            "<li><b>Cross-validate vs ODE</b> &mdash; check that MuJoCo's mass "
            "matrix and trajectory match the analytical reference.</li>"
            "<li><b>Export HDF5</b> &mdash; save the last rollout to a versioned "
            "trace file.</li>"
            "</ol>",
        )


def _format_report(report: validation.ValidationReport) -> str:
    """Render a :class:`ValidationReport` as a compact, readable block."""
    verdict = "PASS" if report.passed else "FAIL"
    return (
        f"[{verdict}] {report.name}\n"
        f"  max_abs_error = {report.max_abs_error:.3e}\n"
        f"  rtol = {report.rtol:.1e}, atol = {report.atol:.1e}\n"
        f"  {report.detail}"
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point used by ``python -m src.tools.simulation_backends_launcher``.

    Args:
        argv: Optional argument vector; defaults to :data:`sys.argv`.

    Returns:
        The Qt application exit code.
    """
    if argv is None:
        argv = sys.argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    window = SimulationBackendsWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
