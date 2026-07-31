# Copyright (c) 2026 D-Sorganization. All rights reserved.
# mypy: disable-error-code="misc,has-type"
# Mixin pattern: methods annotate self as MainWindow to access its attributes,
# but mypy cannot verify this pattern without the concrete class in scope.
# Typing ``self`` as ``MainWindow`` lets mypy resolve every attribute and
# sibling-mixin method directly, which is why the per-call ``# type: ignore``
# comments that previously littered this module are no longer required.
"""Mixin for optimization controller logic in the Movement Optimizer GUI."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ..cli import EXERCISE_FACTORIES
from ..constants import trapezoid
from ..errors import (
    MovementOptimizerError,
    OptimizationError,
    PhysicsError,
    ValidationError,
)
from ..models import BodyModel
from ..trajectory import (
    CancelledError,
    OptimizationResult,
    ProgressReport,
    TrajectoryOptimizer,
)

if TYPE_CHECKING:
    from .main_window import MainWindow

logger = logging.getLogger(__name__)


class OptimizationMixin:
    """Contains logic for preparing and running optimization background tasks.

    Threading contract (provided by ``MainWindow``): ``_opt_lock`` is an
    ``RLock`` so the same thread may re-enter critical sections (e.g. a locked
    helper calling another locked helper) without deadlocking. All writes to
    ``exercise_states`` (and the corresponding cross-thread reads) must hold
    this lock to prevent torn updates between the worker thread and the GUI
    main thread.
    """

    def __init__(self) -> None:
        """Initialise the next class in the cooperative Qt MRO."""
        super().__init__()

    def _snapshot_idx_state(
        self: MainWindow, idx: int
    ) -> tuple[OptimizationResult | None, int, BodyModel | None, Any]:
        """Return a consistent snapshot of (result, anim_frame, body, dyn) for ``idx``.

        Acquires ``_opt_lock`` briefly to copy the four shared per-index values
        out so callers can work with the snapshot outside the critical section,
        avoiding torn reads while the worker thread is publishing a new result.
        """
        with self._opt_lock:
            state = self.exercise_states[idx]
            return (
                state.result,
                state.anim_frame,
                state.body,
                state.dynamics,
            )

    def _set_anim_frame(self: MainWindow, idx: int, frame: int) -> None:
        """Atomically write an exercise animation frame under the optimizer lock."""
        with self._opt_lock:
            self.exercise_states[idx].anim_frame = frame

    def _set_exercise_result(
        self: MainWindow, idx: int, result: OptimizationResult, *, frame: int = 0
    ) -> None:
        """Atomically publish an optimization result and reset playback frame."""
        with self._opt_lock:
            state = self.exercise_states[idx]
            state.result = result
            state.anim_frame = frame

    def _resolve_exercise_params(
        self: MainWindow, idx: int
    ) -> tuple[Any, Any, str, float, float, float]:
        body = self.sidebar.get_body_model()
        bar, dur, smoothness = self.sidebar.get_optimization_params()
        _, etype = self.EXERCISE_CONFIGS[idx]

        factory = EXERCISE_FACTORIES[etype]
        config = factory(body, bar)
        if len(config) == 5:
            dyn, qs, qe, qb, q_via = config
        else:
            dyn, qs, qe, qb = config
            q_via = None

        _min_durations = {
            "full_squat": 3.0,
            "bench_press": 3.0,
            "clean": 2.5,
            "jerk": 2.0,
            "snatch": 3.0,
        }
        if etype in _min_durations:
            dur = max(dur, _min_durations[etype])

        with self._opt_lock:
            state = self.exercise_states[idx]
            state.body = body
            state.dynamics = dyn
            self._last_config = (dyn, qs, qe, qb, q_via, etype)
        return body, dyn, etype, bar, dur, smoothness

    def _seg_mults(self: MainWindow) -> dict[str, float]:
        return self.sidebar.get_segment_multipliers()

    def _run_optimizer(
        self: MainWindow,
        body: Any,
        bar: float,
        dur: float,
        smoothness: float,
    ) -> OptimizationResult:
        dyn, qs, qe, qb, q_via, etype = self._last_config
        logger.info(
            "Starting %s optimisation: mass=%.0f, height=%.2f, bar=%.0f",
            etype,
            body.body_mass,
            body.height,
            bar,
        )
        opt = TrajectoryOptimizer(
            body,
            dyn,
            etype,
            bar,
            qs,
            qe,
            qb,
            q_via=q_via,
            duration=dur,
            n_waypoints=12,
            smoothness=smoothness,
            progress_cb=self._make_progress_cb(),
            cancel_event=self._cancel_event,
        )
        return opt.optimize()

    def _opt_worker(self: MainWindow, idx: int, then_chain: list[int] | None) -> None:
        try:
            body, _dyn, etype, bar, dur, smoothness = self._resolve_exercise_params(idx)
            seg_mults = self._seg_mults()

            b_depth = getattr(body, "squat_bar_depth", 0.0)
            b_height = getattr(body, "squat_bar_height", 0.0)
            cached = self._cache.get(
                etype,
                body.body_mass,
                body.height,
                seg_mults,
                bar,
                dur,
                smoothness,
                b_depth,
                b_height,
            )
            if cached is not None:
                logger.info("Cache hit for %s", etype)
                self._set_exercise_result(idx, cached)
                self._sig_done.emit(idx, cached, body, bar, then_chain)
                return

            result = self._run_optimizer(body, bar, dur, smoothness)
            self._set_exercise_result(idx, result)

            self._cache.put(
                etype,
                body.body_mass,
                body.height,
                seg_mults,
                bar,
                dur,
                smoothness,
                result,
                b_depth,
                b_height,
            )
            self._sig_done.emit(idx, result, body, bar, then_chain)
        except CancelledError:
            self._sig_cancelled.emit()
        except NotImplementedError as exc:
            tb = traceback.format_exc()
            logger.error("Optimisation failed (feature not implemented):\n%s", tb)
            err: MovementOptimizerError = OptimizationError(
                f"Feature not yet implemented: {exc}",
                error_code="OPT_NOT_IMPLEMENTED",
                recoverable=False,
                suggestion="This exercise type may not be supported yet. Try a different exercise.",
            )
            self._sig_error.emit(err)
        except np.linalg.LinAlgError as exc:
            tb = traceback.format_exc()
            logger.error("Physics computation failed (linear algebra):\n%s", tb)
            physics_err = PhysicsError(
                f"A numerical error occurred in the physics engine: {exc}",
                error_code="PHYSICS_LINALG_ERROR",
                suggestion=(
                    "Verify that the body model parameters are physically plausible "
                    "and try adjusting the segment multipliers."
                ),
            )
            self._sig_error.emit(physics_err)
        except ValueError as exc:
            tb = traceback.format_exc()
            logger.error("Validation or parameter error during optimisation:\n%s", tb)
            validation_err = ValidationError(
                f"Invalid parameters: {exc}",
                error_code="VALIDATION_ERROR",
                suggestion=(
                    "Check that all body and exercise parameters are within valid ranges."
                ),
            )
            self._sig_error.emit(validation_err)
        except (RuntimeError, OSError) as exc:
            tb = traceback.format_exc()
            logger.error("Optimisation failed:\n%s", tb)
            err = OptimizationError(
                f"Optimization failed: {exc}",
                error_code="OPT_RUNTIME_ERROR",
                suggestion=(
                    "Try increasing the movement duration or reducing the range of motion."
                ),
            )
            self._sig_error.emit(err)

    def _make_progress_cb(self: MainWindow) -> Callable[[ProgressReport], None]:
        def cb(report: ProgressReport) -> None:
            logger.debug(
                "iter=%d cost=%.3f best=%.3f improve=%+.3f%% elapsed=%.1fs",
                report.iteration,
                report.cost,
                report.best_cost,
                report.improvement_pct,
                report.elapsed_s,
            )
            self._sig_progress.emit(report)

        return cb

    def _update_progress(self: MainWindow, report: ProgressReport) -> None:
        self.sidebar.update_progress(report)

    def _on_done(
        self: MainWindow,
        idx: int,
        result: OptimizationResult,
        body: BodyModel,
        bar: float,
        then_chain: list[int] | None,
    ) -> None:
        """Handle successful optimization completion (called from main thread via signal)."""
        try:
            name = self.EXERCISE_CONFIGS[idx][0]
            _, etype = self.EXERCISE_CONFIGS[idx]
            self._update_result_summary(name, result, exercise_type=etype)
            tab = self.exercise_tabs[idx]
            tab.draw_all_plots(result, body, bar, exercise_type=etype)
            with self._opt_lock:
                dyn = self.exercise_states[idx].dynamics
            tab.draw_anim_frame(0, result, dyn, body, etype)
            elapsed = result.elapsed_s
            t_str = (
                f"{elapsed:.1f}s"
                if elapsed < 60
                else f"{int(elapsed // 60)}m {elapsed % 60:.0f}s"
            )
            self.sidebar.set_progress_done(t_str, result.n_evals)
            self._enable_post_run_buttons()
            if result.success:
                self.sidebar.clear_stall_message()
                status_msg = f"{name} optimization complete in {t_str}!"
            else:
                self.sidebar.set_stall_message(
                    "\u26a0 COM went outside the inner 60% BOS zone. "
                    "Try increasing smoothness or adjusting body parameters."
                )
                status_msg = f"{name} done in {t_str} -- WARNING: COM balance violated"
            self._finish_or_chain(then_chain, status_msg)
        except (ValueError, RuntimeError, OSError, AttributeError) as exc:
            with self._opt_lock:
                self._opt_running = False
            tb = traceback.format_exc()
            logger.error("Error in _on_done:\n%s", tb)
            self.sidebar.show_idle()
            self.status_label.setText(f"Render error: {exc}")

    def _enable_post_run_buttons(self: MainWindow) -> None:
        """Enable export/save/compare buttons after a successful optimization run."""
        self.sidebar.enable_post_run_buttons()

    def _finish_or_chain(
        self: MainWindow, then_chain: list[int] | None, status_msg: str
    ) -> None:
        """Either chain to the next exercise or finalize the run."""
        if then_chain:
            next_idx = then_chain[0]
            remaining = then_chain[1:] if len(then_chain) > 1 else None
            self._run_exercise(next_idx, remaining)
        else:
            with self._opt_lock:
                self._opt_running = False
            self.sidebar.show_idle()
            self.status_label.setText(status_msg)

    def _on_cancelled(self: MainWindow) -> None:
        """Handle user-requested cancellation (called from main thread via signal)."""
        with self._opt_lock:
            self._opt_running = False
        self.sidebar.show_idle()
        self.sidebar.set_cancelled()
        self.status_label.setText("Optimization cancelled by user.")

    def _update_result_summary(
        self: MainWindow, name: str, r: OptimizationResult, exercise_type: str = "squat"
    ) -> None:
        """Build and display the results summary in the sidebar."""
        pk = np.max(np.abs(r.torques), axis=0)
        work = trapezoid(np.einsum("ij->i", np.abs(r.power)), r.t)
        if exercise_type == "bench_press":
            joint_lines = (
                f"  Shoulder: {pk[0]:>6.0f} N\u00b7m\n"
                f"  Elbow:    {pk[1]:>6.0f} N\u00b7m\n"
                f"  Wrist:    {pk[2]:>6.0f} N\u00b7m"
            )
        else:
            balance_ok = "BALANCED" if r.success else "OUT OF BOUNDS"
            joint_lines = (
                f"  Ankle: {pk[0]:>6.0f} N\u00b7m\n"
                f"  Knee:  {pk[1]:>6.0f} N\u00b7m\n"
                f"  Hip:   {pk[2]:>6.0f} N\u00b7m\n"
                f"  COM sway: {r.com_horizontal_range_cm:.1f} cm\n"
                f"  Balance: {balance_ok}"
            )
        self.sidebar.set_result_label(
            f"{name} results:\n{joint_lines}\n  Work: {work:>6.0f} J"
        )

    def _on_err(self: MainWindow, err: object) -> None:
        """Handle optimizer errors (called from main thread via signal)."""
        from PyQt6.QtWidgets import QMessageBox

        from ..errors import MovementOptimizerError

        self._opt_running = False
        self.sidebar.show_idle()

        if isinstance(err, MovementOptimizerError):
            title = "Optimization Failed"
            detail = err.message
            suggestion = err.suggestion
            status_text = f"Error [{err.error_code}]: {err.message}"
        else:
            title = "Optimization Failed"
            detail = str(err)
            suggestion = ""
            status_text = f"Error: {err}"

        self.status_label.setText(status_text)

        body = detail
        if suggestion:
            body = f"{detail}\n\nSuggestion: {suggestion}"

        QMessageBox.critical(self, title, body)  # type: ignore[arg-type]

    def _reset(self: MainWindow) -> None:
        """Reset to defaults and clear the solution cache."""
        self._stop_anim()
        self.sidebar.reset_defaults()
        self._cache.clear()
        self.status_label.setText("Defaults restored. Cache cleared.")
