"""Signal Toolkit Widget Plotting Mixin.

Contains plot update, secondary plot, and logging methods.
"""

from __future__ import annotations

import logging

from .calculus import compute_tangent_line
from .core import Signal

logger = logging.getLogger(__name__)


class PlottingMixin:
    """Mixin providing plotting and logging methods for SignalToolkitWidget."""

    def _update_plot(
        self,
        fitted_signal: Signal | None = None,
    ) -> None:
        """Update the main plot."""
        self.canvas.axes.clear()  # type: ignore[attr-defined]
        self.canvas.setup_dark_theme()  # type: ignore[attr-defined]

        if self.current_signal is None:  # type: ignore
            self.canvas.draw()  # type: ignore[attr-defined]
            return

        # Plot current signal
        self.canvas.axes.plot(  # type: ignore[attr-defined]
            self.current_signal.time,  # type: ignore
            self.current_signal.values,  # type: ignore
            color="#4da6ff",
            linewidth=1.5,
            label="Signal",
        )

        # Plot fitted signal if provided
        if fitted_signal:
            self.canvas.axes.plot(  # type: ignore[attr-defined]
                fitted_signal.time,
                fitted_signal.values,
                color="#ff6b6b",
                linewidth=2,
                linestyle="--",
                label="Fit",
            )

        # Plot tangent line if enabled
        if self.show_tangent_check.isChecked():  # type: ignore[attr-defined]
            tangent = compute_tangent_line(
                self.current_signal,  # type: ignore
                self.tangent_t_spin.value(),  # type: ignore[attr-defined]
            )
            self.canvas.axes.plot(  # type: ignore[attr-defined]
                tangent.t_range,
                tangent.line_values,
                color="#ffd93d",
                linewidth=2,
                label=f"Tangent (slope={tangent.slope:.3f})",
            )
            self.canvas.axes.scatter(  # type: ignore[attr-defined]
                [tangent.t_point],
                [tangent.y_point],
                color="#ffd93d",
                s=50,
                zorder=5,
            )

        self.canvas.axes.set_xlabel("Time")  # type: ignore[attr-defined]
        self.canvas.axes.set_ylabel("Value")  # type: ignore[attr-defined]
        self.canvas.axes.set_title(self.current_signal.name)  # type: ignore[attr-defined]
        self.canvas.axes.legend(loc="upper right")  # type: ignore[attr-defined]

        self.canvas.draw()  # type: ignore[attr-defined]

    def _update_secondary_plot(
        self,
        signal: Signal,
        title: str,
    ) -> None:
        """Update the secondary plot."""
        assert signal is not None, "signal must be provided"
        assert signal is not None, "signal must be provided"
        self.canvas2.axes.clear()  # type: ignore[attr-defined]
        self.canvas2.setup_dark_theme()  # type: ignore[attr-defined]

        self.canvas2.axes.plot(  # type: ignore[attr-defined]
            signal.time,
            signal.values,
            color="#6bcb77",
            linewidth=1.5,
        )

        self.canvas2.axes.set_xlabel("Time")  # type: ignore[attr-defined]
        self.canvas2.axes.set_ylabel("Value")  # type: ignore[attr-defined]
        self.canvas2.axes.set_title(title)  # type: ignore[attr-defined]

        self.canvas2.draw()  # type: ignore[attr-defined]

    def _log(self, message: str) -> None:
        """Log a message to the result text area."""
        self.result_text.append(message)  # type: ignore[attr-defined]

    def set_joints(self, joints: list[str]) -> None:
        """Set the list of available joints."""
        assert joints is not None, "joints must be provided"
        assert joints is not None, "joints must be provided"
        self.joint_names = joints
        self.joint_combo.clear()  # type: ignore[attr-defined]
        self.joint_combo.addItems(joints)  # type: ignore[attr-defined]
