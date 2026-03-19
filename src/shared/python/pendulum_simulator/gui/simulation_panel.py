"""
Shared simulation panel for double and triple pendulum tabs.
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from PyQt6.QtCore import QByteArray, QObject, QSettings, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap  # noqa: F401 — used by subclasses
from PyQt6.QtSvg import QSvgGenerator
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QWidget,
)

if TYPE_CHECKING:
    from .controls_widget import ControlsWidget
    from .controls_widget_golfer import ControlsWidgetGolfer
    from .controls_widget_triple import ControlsWidgetTriple

logger = logging.getLogger(__name__)

_SCROLL_STYLE = "QScrollArea { border: none; background: transparent; }"


# ---------------------------------------------------------------------------
# Background simulation worker
# ---------------------------------------------------------------------------


class _SimWorker(QObject):
    """Runs the ODE integration on a background thread.

    Emits ``finished`` with the result object on success,
    or ``error`` with an error message string on failure.
    """

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(
        self,
        run_fn: Any,
        run_kwargs: dict,
    ) -> None:
        assert run_kwargs is not None, "run_kwargs must be provided"
        super().__init__()
        self._run_fn = run_fn
        self._run_kwargs = run_kwargs

    def run(self) -> None:
        """Called by QThread.started — executes the ODE integration."""
        try:
            self.progress.emit(0)  # Start at 0%
            result = self._run_fn(**self._run_kwargs)
            self.progress.emit(100)  # End at 100%
            self.finished.emit(result)
        except (RuntimeError, ValueError, AssertionError, OSError) as exc:
            logger.error("Simulation worker error: %s", exc)
            self.error.emit(str(exc))


class _SimViewer(Protocol):
    """Structural typing for pendulum/matrix/torque_history widgets."""

    def set_simulation(self, result: object) -> None: ...
    def set_frame(self, idx: int) -> None: ...
    def clear(self) -> None: ...


class SimulationPanel(QWidget):
    """Reusable panel that hosts controls, pendulum, and matrix widgets."""

    ANIMATION_INTERVAL_MS = 16  # ~60 fps

    #: Emitted when ODE integration starts (background thread launched)
    sim_started = pyqtSignal()
    #: Emitted when simulation finishes successfully or with an error
    sim_finished = pyqtSignal()
    #: Emitted each time the displayed frame changes (idx: int)
    frame_changed = pyqtSignal(int)
    #: Emitted when animation playback reaches end (for toolstrip play reset)
    playback_ended = pyqtSignal()

    def __init__(
        self,
        controls: ControlsWidget | ControlsWidgetTriple | ControlsWidgetGolfer,
        pendulum: _SimViewer,
        matrix: _SimViewer,
        params_builder: Callable[[dict], object],
        torque_builder: Callable[[dict], Any],
        state_builder: Callable[[dict], np.ndarray],
        run_simulation: Callable,
        torque_history: _SimViewer | None = None,
        limits_builder: Callable[[dict], Any] | None = None,
        clamp_builder: Callable[[dict], Any] | None = None,
        optimizer: QWidget | None = None,
        objective_builder: Any | None = None,
        parent: QWidget | None = None,
    ) -> None:
        assert controls is not None, "controls must be provided"
        super().__init__(parent)
        self.controls = controls
        self.pendulum = pendulum
        self.matrix = matrix
        self.torque_history = torque_history
        self.optimizer = optimizer
        self.objective_builder = objective_builder
        self.perturbation_panel: QWidget | None = None
        self._params_builder = params_builder
        self._torque_builder = torque_builder
        self._state_builder = state_builder
        self._run_simulation = run_simulation
        self._limits_builder = limits_builder
        self._clamp_builder = clamp_builder

        self._settings_key: str = "splitter_double"

        self._result: Any | None = None
        self._anim_idx = 0
        self._anim_frac: float = 0.0  # fractional frame accumulator (#1097)
        self._playback_speed = 1.0
        self._sim_dt: float = 0.005  # simulation time step (updated on sim completion)
        self._loop_playback: bool = False  # loop animation when it reaches the end

        self._build_ui()
        self._connect_signals()
        self._setup_timer()

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Wrap controls in a scroll area so it never clips on small heights
        scroll = QScrollArea()
        scroll.setWidget(self.controls)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(380)
        scroll.setStyleSheet(_SCROLL_STYLE)
        splitter.addWidget(scroll)

        splitter.addWidget(cast("QWidget", self.pendulum))
        splitter.addWidget(cast("QWidget", self.matrix))

        # Use proportional sizes: compute from available screen width
        screen = QApplication.primaryScreen()
        sw = screen.availableGeometry().width() if screen else 1400
        ctrl_w = min(320, int(sw * 0.20))
        matrix_w = min(280, int(sw * 0.18))

        if self.torque_history is not None:
            splitter.addWidget(cast("QWidget", self.torque_history))
            torque_w = min(260, int(sw * 0.16))
            pend_w = sw - ctrl_w - matrix_w - torque_w - 20
            splitter.setSizes([ctrl_w, pend_w, matrix_w, torque_w])
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 3)
            splitter.setStretchFactor(2, 1)
            splitter.setStretchFactor(3, 1)
        else:
            pend_w = sw - ctrl_w - matrix_w - 20
            splitter.setSizes([ctrl_w, pend_w, matrix_w])
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 3)
            splitter.setStretchFactor(2, 1)

        # Add optimizer panel if provided (#1108, #1109, #1110)
        if self.optimizer is not None:
            opt_scroll = QScrollArea()
            opt_scroll.setWidget(self.optimizer)
            opt_scroll.setWidgetResizable(True)
            opt_scroll.setMinimumWidth(200)
            opt_scroll.setMaximumWidth(300)
            opt_scroll.setStyleSheet(_SCROLL_STYLE)
            splitter.addWidget(opt_scroll)
            splitter.setStretchFactor(splitter.count() - 1, 0)

        main_layout.addWidget(splitter)
        self._splitter = splitter  # keep reference for save/restore

        # Progress bar (hidden by default, positioned absolutely like busy indicator)
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #404070; border-radius: 3px; "
            "background: #1a1a2e; padding: 2px; }"
            "QProgressBar::chunk { background: #4a7bdb; }"
        )
        self._progress_bar.setFixedSize(200, 24)
        self._progress_bar.hide()

        # Restore saved splitter state
        settings = QSettings("D-sorganization", "PendulumSimulator")
        saved = settings.value(self._settings_key)
        if isinstance(saved, QByteArray):
            self._splitter.restoreState(saved)

    def set_perturbation_panel(self, panel: QWidget) -> None:
        """Attach a perturbation panel to the right side of the splitter.

        Must be called after construction but before the widget is shown.
        """
        assert panel is not None, "perturbation panel must not be None"
        self.perturbation_panel = panel
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(200)
        scroll.setMaximumWidth(320)
        scroll.setStyleSheet(_SCROLL_STYLE)
        self._splitter.addWidget(scroll)
        self._splitter.setStretchFactor(self._splitter.count() - 1, 0)

    def save_layout(self) -> None:
        """Persist the current splitter positions to QSettings."""
        settings = QSettings("D-sorganization", "PendulumSimulator")
        settings.setValue(self._settings_key, self._splitter.saveState())

    def _connect_signals(self) -> None:
        self.controls.run_requested.connect(self._on_run)
        self.controls.reset_requested.connect(self._on_reset)
        self.controls.play_toggled.connect(self._on_play_toggle)
        self.controls.speed_changed.connect(self._on_speed_change)
        self.controls.frame_changed.connect(self._on_frame_change)
        self.controls.export_data_requested.connect(self._on_export_data)
        self.controls.export_video_requested.connect(self._on_export_video)
        self.controls.export_image_requested.connect(self.export_image)

        # Wire new physics/display toggles if the pendulum widget supports them
        if hasattr(self.controls, "gravity_changed") and hasattr(
            self.pendulum,
            "set_gravity_on",
        ):
            self.controls.gravity_changed.connect(self.pendulum.set_gravity_on)
        if hasattr(self.controls, "forces_changed") and hasattr(
            self.pendulum,
            "set_show_forces",
        ):
            self.controls.forces_changed.connect(self.pendulum.set_show_forces)
        if hasattr(self.controls, "force_scale_changed") and hasattr(
            self.pendulum,
            "set_force_scale",
        ):
            self.controls.force_scale_changed.connect(self.pendulum.set_force_scale)

        # Wire real-time rotation controls (#1146)
        if hasattr(self.controls, "tilt_changed") and hasattr(
            self.pendulum, "set_tilt_angle"
        ):
            self.controls.tilt_changed.connect(self.pendulum.set_tilt_angle)
        if hasattr(self.controls, "azimuth_changed") and hasattr(
            self.pendulum, "set_view_azimuth"
        ):
            self.controls.azimuth_changed.connect(self.pendulum.set_view_azimuth)

        # Persist splitter when it changes
        if hasattr(self, "_splitter"):
            self._splitter.splitterMoved.connect(lambda *_: self.save_layout())

        # Wire optimizer (#1151): build objective from current params on demand
        if self.optimizer is not None and self.objective_builder is not None:
            from .optimization_widget import OptimizationWidget

            opt = cast("OptimizationWidget", self.optimizer)
            _obj_builder = self.objective_builder  # capture for closure
            assert callable(_obj_builder), "objective_builder must be callable"

            # Before each optimizer run, rebuild the objective with current params
            orig_on_run = opt._on_run

            def _patched_on_run() -> None:
                """Build objective from current UI params, then run optimizer."""
                try:
                    p = self.controls.get_params()
                    obj_fn = _obj_builder(p)
                    opt.set_objective_function(obj_fn)
                except (ValueError, AssertionError) as e:
                    opt._log.append(f"⚠ Cannot build objective: {e}")
                    return
                orig_on_run()

            opt._btn_run.clicked.disconnect()
            opt._btn_run.clicked.connect(_patched_on_run)

            # Wire apply back to controls (set torque coefficients)
            opt.optimized_coefficients.connect(self._apply_optimized_coefficients)

    def _setup_timer(self) -> None:
        self._timer = QTimer(self)
        self._timer.setInterval(self.ANIMATION_INTERVAL_MS)
        self._timer.timeout.connect(self._advance_frame)

    def _on_run(self) -> None:
        from .diagnostics import get_tracker

        try:
            p = self.controls.get_params()
        except ValueError as e:
            logger.warning("Parameter validation failed: %s", e)
            get_tracker().record_exception(
                "simulation", e, context="Parameter validation"
            )
            QMessageBox.warning(self, "Input Error", str(e))
            return

        try:
            params = self._params_builder(p)
        except (AssertionError, Exception) as e:
            logger.warning("Parameter build failed: %s", e, exc_info=True)
            get_tracker().record_exception("simulation", e, context="Parameter build")
            QMessageBox.warning(self, "Parameter Error", str(e))
            return

        if p["t_end"] <= 0:
            QMessageBox.warning(self, "Input Error", "Duration must be positive")
            return

        try:
            initial_state = self._state_builder(p)
            torque_func = self._torque_builder(p)
        except (ValueError, TypeError, KeyError) as e:
            logger.warning("State/torque build failed: %s", e, exc_info=True)
            get_tracker().record_exception(
                "simulation", e, context="State/torque build"
            )
            QMessageBox.warning(self, "Build Error", str(e))
            return

        self.controls.btn_run.setEnabled(False)
        self.controls.btn_reset.setEnabled(False)
        self._show_busy(True)
        self._show_progress(True)
        self.sim_started.emit()
        logger.info(
            "Simulation started: t_end=%.3f, dt=%.4f",
            p["t_end"],
            float(p.get("dt", 0.005)),
        )

        # Build optional joint limits and torque clamp
        limits = self._limits_builder(p) if self._limits_builder else None
        clamp = self._clamp_builder(p) if self._clamp_builder else None

        # Build kwargs for the runner function
        run_kwargs: dict = {
            "params": params,
            "initial_state": initial_state,
            "t_end": p["t_end"],
            "torque_func": torque_func,
            "dt": float(p.get("dt", 0.005)),
        }
        if limits is not None:
            run_kwargs["limits"] = limits
        if clamp is not None:
            run_kwargs["clamp"] = clamp

        # Spin up background thread
        self._sim_thread = QThread()
        self._sim_worker = _SimWorker(self._run_simulation, run_kwargs)
        self._sim_worker.moveToThread(self._sim_thread)
        self._sim_thread.started.connect(self._sim_worker.run)
        self._sim_worker.progress.connect(self._progress_bar.setValue)
        self._sim_worker.finished.connect(self._on_sim_done)
        self._sim_worker.error.connect(self._on_sim_error)
        self._sim_worker.finished.connect(self._sim_thread.quit)
        self._sim_worker.error.connect(self._sim_thread.quit)
        self._sim_thread.finished.connect(self._sim_thread.deleteLater)
        self._sim_thread.start()

    def _on_sim_done(self, result: object) -> None:
        """Called on the main thread when simulation completes.

        Pre: result has n_steps, t, states attributes (TrajectoryResultMixin).
        """
        assert result is not None, "Simulation result must not be None"
        assert hasattr(result, "n_steps"), "Result must have n_steps attribute"
        assert hasattr(result, "t"), "Result must have t attribute"

        res: Any = result  # pyqtSignal emits object; cast for attribute access

        self._show_busy(False)
        self._show_progress(False)
        self.controls.btn_run.setEnabled(True)
        self.controls.btn_reset.setEnabled(True)
        self.sim_finished.emit()

        self._result = res
        self._anim_idx = 0

        # Compute simulation dt for real-time playback (#1115)
        if res.n_steps > 1:
            self._sim_dt = float(res.t[-1] - res.t[0]) / (res.n_steps - 1)
        else:
            self._sim_dt = 0.005

        logger.info(
            "Simulation finished: %d steps, t=[0, %.3f]s, dt=%.4fms",
            res.n_steps,
            float(res.t[-1]),
            self._sim_dt * 1000,
        )

        self.pendulum.set_simulation(res)
        self.matrix.set_simulation(res)
        if self.torque_history is not None:
            self.torque_history.set_simulation(res)
        self.controls.set_slider_range(res.n_steps - 1)
        self.controls.set_slider_value(0)
        self._display_frame(0)

        # Auto-play
        self.controls.btn_play.setChecked(True)

    def _on_sim_error(self, msg: str) -> None:
        """Called on the main thread when simulation fails."""
        assert msg is not None, "msg must be provided"
        from .diagnostics import get_tracker

        logger.error("Simulation failed: %s", msg)
        get_tracker().record(
            "simulation",
            f"Simulation failed: {msg}",
            severity="error",
            details=msg,
        )
        self._show_busy(False)
        self._show_progress(False)
        self.controls.btn_run.setEnabled(True)
        self.controls.btn_reset.setEnabled(True)
        self.sim_finished.emit()
        QMessageBox.critical(self, "Simulation Error", msg)

    def _show_busy(self, busy: bool) -> None:
        """Show / hide a 'Simulating…' indicator in the top-right."""
        assert busy is not None, "busy must be provided"
        if not hasattr(self, "_busy_label"):
            self._busy_label = QLabel("⏳  Simulating…", self)
            self._busy_label.setStyleSheet(
                "background: #202040; color: #b0b0e8; border: 1px solid #404070;"
                "border-radius: 4px; padding: 4px 10px; font-size: 12px;"
            )
            self._busy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if busy:
            # Position top-centre of this panel
            self._busy_label.setFixedWidth(160)
            self._busy_label.move(
                (self.width() - 160) // 2,
                8,
            )
            self._busy_label.raise_()
            self._busy_label.show()
        else:
            self._busy_label.hide()

    def _show_progress(self, show: bool) -> None:
        """Show / hide the progress bar during simulation."""
        if show:
            # Position below busy indicator
            self._progress_bar.move(
                (self.width() - 200) // 2,
                38,
            )
            self._progress_bar.raise_()
            self._progress_bar.show()
        else:
            self._progress_bar.hide()

    def _on_reset(self) -> None:
        self._timer.stop()
        self._result = None
        self._anim_idx = 0
        self._anim_frac = 0.0
        self.pendulum.clear()
        self.matrix.clear()
        if self.torque_history is not None:
            self.torque_history.clear()
        self.controls.stop_playback()
        self.controls.set_slider_value(0)

    def _on_play_toggle(self, playing: bool) -> None:
        if self._result is None:
            self.controls.stop_playback()
            return
        if playing:
            if self._anim_idx >= self._result.n_steps - 1:
                self._anim_idx = 0
                self._anim_frac = 0.0
                if hasattr(self.pendulum, "_trail"):
                    self.pendulum._trail.clear()
            self._timer.start()
        else:
            self._timer.stop()

    def _on_speed_change(self, speed: float) -> None:
        """Pre: speed > 0"""
        assert speed > 0, f"Playback speed must be positive, got {speed}"
        self._playback_speed = speed

    def _on_frame_change(self, frame: int) -> None:
        assert frame is not None, "frame must be provided"
        if self._result is None:
            return
        self._anim_idx = frame
        if hasattr(self.pendulum, "_trail"):
            self.pendulum._trail.clear()
            trail_len = getattr(self.pendulum, "TRAIL_LENGTH", 200)
            trail_start = max(0, frame - trail_len)
            # Fast path: use vectorized cache if available
            cache = getattr(self.pendulum, "_tip_positions_cache", None)
            if cache is not None:
                for i in range(trail_start, frame + 1):
                    self.pendulum._trail.append(tuple(cache[i]))
            else:
                for i in range(trail_start, frame + 1):
                    pos = self._result.positions_at(i)
                    tip = pos.get("tip")
                    if tip is not None:
                        self.pendulum._trail.append(tip)
        self._display_frame(frame)

    def _advance_frame(self) -> None:
        """Advance the animation by a fractional frame count.

        Uses a fractional accumulator so that high-speed playback (e.g. 5×)
        still produces smooth visual transitions instead of integer jumps.

        Closes #1097.
        """
        if self._result is None:
            self._timer.stop()
            return

        # Inv: _playback_speed > 0, _sim_dt > 0
        assert self._playback_speed > 0, "Playback speed invariant violated"

        # Compute real-time frame advance (#1115)
        # frames_per_tick = wall_clock_tick / sim_dt × speed_multiplier
        dt_wall = self.ANIMATION_INTERVAL_MS / 1000.0
        frames_per_tick = (dt_wall / max(self._sim_dt, 1e-6)) * self._playback_speed
        self._anim_frac += frames_per_tick
        advance = int(self._anim_frac)
        if advance < 1:
            return  # sub-frame accumulation — wait for next tick
        self._anim_frac -= advance

        self._anim_idx += advance

        if self._anim_idx >= self._result.n_steps:
            if self._loop_playback:
                # Loop: restart from beginning
                self._anim_idx = 0
                self._anim_frac = 0.0
            else:
                # Stop at end and reset play button
                self._anim_idx = self._result.n_steps - 1
                self._anim_frac = 0.0
                self._timer.stop()
                self.controls.stop_playback()
                self.playback_ended.emit()

        self._display_frame(self._anim_idx)
        self.controls.set_slider_value(self._anim_idx)

    def _display_frame(self, idx: int) -> None:
        assert self._result is not None
        idx = max(0, min(idx, self._result.n_steps - 1))
        self.pendulum.set_frame(idx)
        self.matrix.set_frame(idx)
        if self.torque_history is not None:
            self.torque_history.set_frame(idx)
        self.frame_changed.emit(idx)

    def scrub_to_frame(self, idx: int) -> None:
        """Jump to a specific frame index (called by toolstrip slider)."""
        assert idx is not None, "idx must be provided"
        if self._result is None:
            return
        idx = max(0, min(idx, self._result.n_steps - 1))
        self._anim_idx = idx
        self.controls.set_slider_value(idx)
        self._display_frame(idx)

    def current_n_steps(self) -> int:
        """Return the number of frames in the current simulation (0 if none)."""
        if self._result is None:
            return 0
        return int(self._result.n_steps)

    def _apply_optimized_coefficients(self, result: object) -> None:
        """Apply optimizer results back to the control widget's torque fields.

        Splits the flat coefficient array into per-joint polynomial strings.
        Supports double (2), triple (3), and golfer (7) torque-param models.

        Pre: result has 'coeffs' key with a numpy array.
        Closes #1151.
        """
        assert result is not None, "result must be provided"
        import logging

        logger = logging.getLogger(__name__)

        if not isinstance(result, dict):
            logger.warning("Optimizer result is not a dict, skipping apply")
            return
        coeffs = result.get("coeffs")
        if coeffs is None:
            logger.warning("No coefficients in optimizer result")
            return

        coeffs = np.asarray(coeffs, dtype=float)

        # Determine model type by which control widget is active
        from .controls_widget import ControlsWidget
        from .controls_widget_golfer import ControlsWidgetGolfer
        from .controls_widget_triple import ControlsWidgetTriple

        def _fmt_coeffs(arr: np.ndarray) -> str:
            """Format coefficient array as comma-separated string."""
            return ", ".join(f"{v:.4f}" for v in arr)

        if isinstance(self.controls, ControlsWidget):
            # Double: split into 2 groups (shoulder, wrist)
            n_half = len(coeffs) // 2
            self.controls.inp_tau_shoulder.set_value(_fmt_coeffs(coeffs[:n_half]))
            self.controls.inp_tau_wrist.set_value(_fmt_coeffs(coeffs[n_half:]))
            logger.info("Applied double pendulum optimizer coefficients")

        elif isinstance(self.controls, ControlsWidgetTriple):
            # Triple: split into 3 groups (shoulder, elbow, wrist)
            n_third = len(coeffs) // 3
            self.controls.inp_tau_shoulder.set_value(_fmt_coeffs(coeffs[:n_third]))
            self.controls.inp_tau_elbow.set_value(
                _fmt_coeffs(coeffs[n_third : 2 * n_third])
            )
            self.controls.inp_tau_wrist.set_value(_fmt_coeffs(coeffs[2 * n_third :]))
            logger.info("Applied triple pendulum optimizer coefficients")

        elif isinstance(self.controls, ControlsWidgetGolfer):
            # Golfer: split into 7 groups
            n_seventh = max(1, len(coeffs) // 7)
            field_names = [
                "inp_tau_hub",
                "inp_tau_rs",
                "inp_tau_re",
                "inp_tau_rh",
                "inp_tau_ls",
                "inp_tau_le",
                "inp_tau_lh",
            ]
            for i, field_name in enumerate(field_names):
                field = getattr(self.controls, field_name, None)
                if field is not None:
                    start = i * n_seventh
                    end = (i + 1) * n_seventh if i < 6 else len(coeffs)
                    field.set_value(_fmt_coeffs(coeffs[start:end]))
            logger.info("Applied golfer optimizer coefficients")

    def export_image(self) -> None:
        """Export the current pendulum visualization as PNG, SVG, or PDF."""
        if self._result is None:
            QMessageBox.information(self, "Export Image", "Run a simulation first.")
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            "",
            "PNG Files (*.png);;SVG Files (*.svg);;PDF Files (*.pdf)",
        )
        if not path:
            return

        try:
            if path.endswith(".png"):
                self._export_as_png(path)
            elif path.endswith(".svg"):
                self._export_as_svg(path)
            elif path.endswith(".pdf"):
                self._export_as_pdf(path)
            else:
                # Default to PNG if extension unclear
                if not path.endswith("."):
                    path += ".png"
                self._export_as_png(path)

            QMessageBox.information(self, "Export Image", f"Saved image to:\n{path}")
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to export image: %s", e)
            QMessageBox.critical(
                self,
                "Export Image",
                f"Failed to export image:\n{e}",
            )

    def _export_as_png(self, path: str) -> None:
        """Export the pendulum widget as a PNG image."""
        pix = cast("QWidget", self.pendulum).grab()
        if not pix.save(path):
            raise OSError(f"Failed to save PNG to {path}")
        logger.info("Exported PNG: %s", path)

    def _export_as_svg(self, path: str) -> None:
        """Export the pendulum widget as an SVG image."""
        assert path is not None, "path must be provided"
        from PyQt6.QtCore import QRect
        from PyQt6.QtGui import QPainter

        widget = cast("QWidget", self.pendulum)
        rect = QRect(0, 0, widget.width(), widget.height())

        generator = QSvgGenerator()
        generator.setFileName(path)
        generator.setSize(rect.size())
        generator.setViewBox(rect)
        generator.setTitle("Pendulum Visualization")
        generator.setDescription("Exported from Pendulum Simulator")

        painter = QPainter()
        painter.begin(generator)
        widget.render(painter)
        painter.end()

        logger.info("Exported SVG: %s", path)

    def _export_as_pdf(self, path: str) -> None:
        """Export the pendulum widget as a PDF (via QPrinter)."""
        assert path is not None, "path must be provided"
        from PyQt6.QtCore import QMarginsF
        from PyQt6.QtGui import QPainter
        from PyQt6.QtPrintSupport import QPrinter

        widget = cast("QWidget", self.pendulum)

        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(path)
        printer.setPageMargins(QMarginsF(0, 0, 0, 0))

        painter = QPainter()
        painter.begin(printer)
        widget.render(painter)
        painter.end()

        logger.info("Exported PDF: %s", path)

    def _on_export_data(self) -> None:
        if self._result is None:
            QMessageBox.information(self, "Export Data", "Run a simulation first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Data",
            "",
            "CSV Files (*.csv)",
        )
        if not path:
            return

        headers = ["t"]
        if self._result.states.shape[1] == 4:
            headers += [
                "tau_drive_1",
                "tau_drive_2",
                "tau_friction_1",
                "tau_friction_2",
                "tau_total_1",
                "tau_total_2",
                "shoulder_fx",
                "shoulder_fy",
                "wrist_fx",
                "wrist_fy",
            ]
        else:
            headers += [
                "tau_drive_1",
                "tau_drive_2",
                "tau_drive_3",
                "tau_friction_1",
                "tau_friction_2",
                "tau_friction_3",
                "tau_total_1",
                "tau_total_2",
                "tau_total_3",
                "shoulder_fx",
                "shoulder_fy",
                "wrist1_fx",
                "wrist1_fy",
                "wrist2_fx",
                "wrist2_fy",
            ]

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for i in range(self._result.n_steps):
                    t = self._result.t[i]
                    tau_drive = self._result.torques_at(i)
                    forces = self._result.joint_forces_at(i)

                    if self._result.states.shape[1] == 4:
                        tau_friction = self._result.friction_torques_at(i)
                        tau_total = self._result.total_torques_at(i)
                        row = [
                            t,
                            tau_drive[0],
                            tau_drive[1],
                            tau_friction[0],
                            tau_friction[1],
                            tau_total[0],
                            tau_total[1],
                            forces["shoulder"][0],
                            forces["shoulder"][1],
                            forces["wrist"][0],
                            forces["wrist"][1],
                        ]
                    else:
                        row = [
                            t,
                            tau_drive[0],
                            tau_drive[1],
                            tau_drive[2],
                            0.0,
                            0.0,
                            0.0,  # friction not yet in triple model
                            tau_drive[0],
                            tau_drive[1],
                            tau_drive[2],
                            forces["shoulder"][0],
                            forces["shoulder"][1],
                            forces["wrist1"][0],
                            forces["wrist1"][1],
                            forces["wrist2"][0],
                            forces["wrist2"][1],
                        ]
                    writer.writerow(row)

        except OSError as e:
            QMessageBox.critical(self, "Export Data", f"Failed to write file: {e}")
            return

        QMessageBox.information(self, "Export Data", f"Saved data to:\n{path}")

    def _on_export_video(self) -> None:
        if self._result is None:
            QMessageBox.information(self, "Export Video", "Run a simulation first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Video",
            "",
            "MP4 Video (*.mp4);;GIF (*.gif)",
        )
        if not path:
            return

        ffmpeg_path = shutil.which("ffmpeg")
        was_playing = self._timer.isActive()
        self._timer.stop()

        tmp_dir = tempfile.mkdtemp(prefix="pendulum_frames_")
        try:
            for i in range(self._result.n_steps):
                self._display_frame(i)
                QApplication.processEvents()
                pix = cast("QWidget", self.pendulum).grab()
                frame_path = os.path.join(tmp_dir, f"frame_{i:05d}.png")
                pix.save(frame_path)

            if ffmpeg_path is None:
                out_dir = os.path.splitext(path)[0] + "_frames"
                os.makedirs(out_dir, exist_ok=True)
                for name in os.listdir(tmp_dir):
                    shutil.move(
                        os.path.join(tmp_dir, name),
                        os.path.join(out_dir, name),
                    )
                QMessageBox.warning(
                    self,
                    "Export Video",
                    "ffmpeg not found. Exported PNG frames instead:\n" + out_dir,
                )
                return

            fps = int(1000 / self.ANIMATION_INTERVAL_MS)
            cmd = [
                ffmpeg_path,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(tmp_dir, "frame_%05d.png"),
                "-pix_fmt",
                "yuv420p",
                path,
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                QMessageBox.critical(
                    self,
                    "Export Video",
                    "ffmpeg failed. Check your ffmpeg installation.",
                )
                return

            QMessageBox.information(self, "Export Video", f"Saved video to:\n{path}")
        finally:
            if was_playing:
                self._timer.start()
            shutil.rmtree(tmp_dir, ignore_errors=True)
