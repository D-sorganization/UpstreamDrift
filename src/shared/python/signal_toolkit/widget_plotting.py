"""Signal Toolkit Widget Plotting Mixin.

Contains plot update, secondary plot, and logging methods.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np

from .calculus import compute_tangent_line
from .core import Signal

if TYPE_CHECKING:
    from .widget_protocol import WidgetProtocol

logger = logging.getLogger(__name__)


class PlottingMixin:
    """Mixin providing plotting and logging methods for SignalToolkitWidget."""

    def _update_plot(
        self,
        fitted_signal: Signal | None = None,
    ) -> None:
        """Update the main plot."""
        self_w = cast("WidgetProtocol", self)
        self_w.canvas.axes.clear()
        self_w.canvas.setup_dark_theme()

        if self_w.current_signal is None:
            self_w.canvas.draw()
            return

        # Plot current signal
        self_w.canvas.axes.plot(
            self_w.current_signal.time,
            self_w.current_signal.values,
            color="#4da6ff",
            linewidth=1.5,
            label="Signal",
        )

        # Plot fitted signal if provided
        if fitted_signal:
            self_w.canvas.axes.plot(
                fitted_signal.time,
                fitted_signal.values,
                color="#ff6b6b",
                linewidth=2,
                linestyle="--",
                label="Fit",
            )

        # Plot tangent line if enabled
        if self_w.show_tangent_check.isChecked():
            tangent = compute_tangent_line(
                self_w.current_signal,
                self_w.tangent_t_spin.value(),
            )
            self_w.canvas.axes.plot(
                tangent.t_range,
                tangent.line_values,
                color="#ffd93d",
                linewidth=2,
                label=f"Tangent (slope={tangent.slope:.3f})",
            )
            self_w.canvas.axes.scatter(
                [tangent.t_point],
                [tangent.y_point],
                color="#ffd93d",
                s=50,
                zorder=5,
            )

        self_w.canvas.axes.set_xlabel("Time")
        self_w.canvas.axes.set_ylabel("Value")
        self_w.canvas.axes.set_title(self_w.current_signal.name)
        self_w.canvas.axes.legend(loc="upper right")

        self_w.canvas.draw()

    def _update_secondary_plot(
        self,
        signal: Signal,
        title: str,
    ) -> None:
        """Update the secondary plot."""
        if signal is None:
            raise ValueError("signal must be provided")
        self_w = cast("WidgetProtocol", self)
        self_w.canvas2.axes.clear()
        self_w.canvas2.setup_dark_theme()

        self_w.canvas2.axes.plot(
            signal.time,
            signal.values,
            color="#6bcb77",
            linewidth=1.5,
        )

        self_w.canvas2.axes.set_xlabel("Time")
        self_w.canvas2.axes.set_ylabel("Value")
        self_w.canvas2.axes.set_title(title)

        self_w.canvas2.draw()

    def _update_frequency_response_plot(
        self,
        frequencies: np.ndarray,
        magnitude: np.ndarray,
        phase: np.ndarray,
        title: str = "Frequency Response",
    ) -> None:
        """Update the secondary plot with a Bode-style frequency response.

        Renders magnitude (dB) on the secondary canvas.
        """
        if frequencies is None:
            raise ValueError("frequencies must be provided")
        self_w = cast("WidgetProtocol", self)
        self_w.canvas2.axes.clear()
        self_w.canvas2.setup_dark_theme()

        # Magnitude in dB
        mag_db = 20 * np.log10(np.maximum(magnitude, 1e-10))
        self_w.canvas2.axes.plot(
            frequencies, mag_db, color="#4ecdc4", linewidth=1.5, label="Magnitude (dB)"
        )
        self_w.canvas2.axes.set_xlabel("Frequency (Hz)")
        self_w.canvas2.axes.set_ylabel("Magnitude (dB)")
        self_w.canvas2.axes.set_title(title)
        self_w.canvas2.axes.grid(True, alpha=0.3)
        self_w.canvas2.axes.legend(loc="upper right")

        self_w.canvas2.draw()

    def _log(self, message: str) -> None:
        """Log a message to the result text area."""
        if message is None:
            raise ValueError("message must be provided")
        self_w = cast("WidgetProtocol", self)
        self_w.result_text.append(message)

    def set_joints(self, joints: list[str]) -> None:
        """Set the list of available joints."""
        if joints is None:
            raise ValueError("joints must be provided")
        self_w = cast("WidgetProtocol", self)
        self_w.joint_names = joints  # type: ignore[assignment]
        self_w.joint_combo.clear()
        self_w.joint_combo.addItems(joints)
