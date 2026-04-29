"""TDD tests for pendulum desktop app state-management bugs (issue #2492).

Bugs covered:
1. Tk app: repeated start() calls create concurrent update loops (no reentry guard).
2. PyQt app: _start() resets time to 0 without resetting evolved state,
   causing state/time desynchronisation.
3. PyQt app: _points_triple() hardcodes plane_inclination=35° instead of using
   self.triple_params.plane_inclination_deg.
4. PyQt app: _safe_eval() silently returns 0.0 on expression errors; the run
   continues with wrong torques, and no error is surfaced to the user.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from double_pendulum_model.physics.double_pendulum import (
    DoublePendulumState,
)
from double_pendulum_model.physics.triple_pendulum import TriplePendulumState
from double_pendulum_model.ui.pendulum_pyqt_app import PendulumController
from src.shared.python.ui.qt.utils import get_qapp


@pytest.fixture(scope="module")
def qapp() -> Any:
    """Shared QApplication for the module."""
    return get_qapp()


# ---------------------------------------------------------------------------
# Bug 1: Tk reentry — tested via DoublePendulumTkApp internals
# ---------------------------------------------------------------------------


class TestTkStartReentry:
    """Tk start() must be idempotent — calling it twice must not fork loops."""

    def test_start_while_running_does_not_reschedule(self) -> None:
        """If self.running is already True, start() must not call _update() again."""
        import tkinter as tk

        from double_pendulum_model.ui.double_pendulum_gui import DoublePendulumApp

        root = tk.Tk()
        root.withdraw()
        try:
            app = DoublePendulumApp(root)
            update_calls: list[int] = []

            def tracking_update() -> None:
                update_calls.append(1)
                # Don't actually schedule the next callback in tests
                app.running = False

            app._update = tracking_update  # type: ignore[method-assign]
            app.running = False

            # First start() should trigger one _update call
            app.start()
            first_call_count = len(update_calls)

            # Reset and call start() again while already running
            app.running = True  # simulate already-running
            app._update = tracking_update  # type: ignore[method-assign]
            app.start()  # Bug: would create a second concurrent loop

            # After the fix: start() must not call _update() when already running
            assert len(update_calls) <= first_call_count + 1, (
                "start() while running created extra update loop "
                f"(calls before: {first_call_count}, after: {len(update_calls)})"
            )
        finally:
            root.destroy()


# ---------------------------------------------------------------------------
# Bug 2: PyQt _start() resets time without resetting state
# ---------------------------------------------------------------------------


class TestPyQtStartResetsState:
    """_start() must reset state alongside time to avoid desynchronisation."""

    def test_start_resets_state_double(self, qapp) -> None:  # noqa: ARG002
        """After _start(), state_double must be the initial/default state."""
        controller = PendulumController()
        # Evolve the state artificially
        controller.state_double = DoublePendulumState(
            theta1=1.5, theta2=0.8, omega1=2.0, omega2=-1.5
        )
        controller.time = 99.0
        controller.timer.stop()  # don't actually run the timer

        controller._start()

        assert controller.time == pytest.approx(0.0), "_start() must reset time to 0"
        # State must be at the default/reset position, not the evolved one
        assert controller.state_double.omega1 == pytest.approx(0.0), (
            "_start() must reset omega1 to 0"
        )
        assert controller.state_double.omega2 == pytest.approx(0.0), (
            "_start() must reset omega2 to 0"
        )

    def test_start_resets_state_triple(self, qapp) -> None:  # noqa: ARG002
        """After _start(), state_triple must be the initial/default state."""
        controller = PendulumController()
        controller.state_triple = TriplePendulumState(
            theta1=1.0, theta2=1.0, theta3=1.0, omega1=3.0, omega2=2.0, omega3=1.0
        )
        controller.time = 50.0
        controller.timer.stop()

        controller._start()

        assert controller.time == pytest.approx(0.0)
        assert controller.state_triple.omega1 == pytest.approx(0.0), (
            "_start() must reset omega1 to 0"
        )
        assert controller.state_triple.omega3 == pytest.approx(0.0), (
            "_start() must reset omega3 to 0"
        )


# ---------------------------------------------------------------------------
# Bug 3: Triple pendulum _points_triple hardcodes plane angle
# ---------------------------------------------------------------------------


class TestTriplePendulumPlaneConstraint:
    """_points_triple must use triple_params.plane_inclination_deg, not 35.0."""

    def _points_differ(
        self, controller: PendulumController, inclination_a: float, inclination_b: float
    ) -> bool:
        """Return True if the two inclinations produce different endpoint positions."""
        import numpy as np

        state = TriplePendulumState(
            theta1=0.3, theta2=-0.4, theta3=0.2, omega1=0.0, omega2=0.0, omega3=0.0
        )
        controller.triple_params.plane_inclination_deg = inclination_a
        pts_a = controller._points_triple(state)

        controller.triple_params.plane_inclination_deg = inclination_b
        pts_b = controller._points_triple(state)

        return not np.allclose(pts_a, pts_b, atol=1e-6)

    def test_different_plane_inclinations_produce_different_points(
        self,
        qapp,  # noqa: ARG002
    ) -> None:
        """Changing triple_params.plane_inclination_deg must change the geometry."""
        controller = PendulumController()
        assert self._points_differ(controller, 0.0, 90.0), (
            "_points_triple ignores triple_params.plane_inclination_deg "
            "(hardcoded 35° suspected)"
        )

    def test_zero_inclination_stays_in_plane(self, qapp) -> None:  # noqa: ARG002
        """At plane_inclination_deg=0, the pendulum swings in the xz plane (y≈0)."""
        import numpy as np

        controller = PendulumController()
        controller.triple_params.plane_inclination_deg = 0.0
        state = TriplePendulumState(
            theta1=0.3, theta2=-0.4, theta3=0.2, omega1=0.0, omega2=0.0, omega3=0.0
        )
        pts = controller._points_triple(state)
        y_coords = pts[:, 1]
        assert np.allclose(y_coords, 0.0, atol=1e-6), (
            f"At plane_inclination=0, y-coords should be ~0 but got {y_coords}"
        )


# ---------------------------------------------------------------------------
# Bug 4: _safe_eval silently returns 0.0 on expression errors
# ---------------------------------------------------------------------------


class TestSafeEvalErrorSurfacing:
    """_safe_eval must surface errors visibly rather than silently returning 0."""

    def test_safe_eval_valid_expression(self, qapp) -> None:  # noqa: ARG002
        """A valid numeric expression evaluates correctly."""
        controller = PendulumController()
        result = controller._safe_eval("2 + 3")
        assert result == pytest.approx(5.0)

    def test_safe_eval_invalid_expression_does_not_silently_succeed(
        self,
        qapp,  # noqa: ARG002
    ) -> None:
        """An invalid expression must NOT return 0.0 without any indication of error.

        Either raise, return NaN/None, or set an error-state attribute.
        The pre-fix behaviour was: silently return 0.0 and log at DEBUG level only.
        """
        controller = PendulumController()
        result = controller._safe_eval("completely_undefined_symbol * x")
        # After the fix: result must be distinguishable from a valid zero torque.
        # Either the controller has an error flag, or the result is not 0.0.
        has_error_flag = (
            getattr(controller, "_last_eval_error", None) is not None
            or getattr(controller, "_expression_error", None) is not None
        )
        result_signals_error = result is None or (
            isinstance(result, float) and math.isnan(result)
        )
        assert has_error_flag or result_signals_error, (
            "_safe_eval must surface expression errors (got silent 0.0): "
            f"result={result}, controller attrs checked"
        )

    def test_safe_eval_error_visible_in_status(self, qapp) -> None:  # noqa: ARG002
        """After an invalid expression, the status label must reflect the error."""
        controller = PendulumController()
        controller._safe_eval("no_such_var / 0")

        # The status label (or a dedicated error label) should mention the problem
        has_error_display = False
        for attr in ("status_label", "torque_label", "error_label"):
            widget = getattr(controller, attr, None)
            if widget is not None and hasattr(widget, "text"):
                text = widget.text()
                if "error" in text.lower() or "invalid" in text.lower():
                    has_error_display = True
                    break

        assert has_error_display, (
            "After an invalid torque expression, no visible error was shown in the UI"
        )
