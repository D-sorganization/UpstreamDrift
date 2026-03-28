"""Signal Toolkit Widget Plotting Mixin.

Contains plot update, secondary plot, and logging methods.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .calculus import compute_tangent_line
from .core import Signal
from .widget_protocol import _SignalToolkitHost

logger = logging.getLogger(__name__)

# Use the protocol as the mixin base so mypy understands that ``self``
# carries all the host widget attributes.  At runtime this resolves to
# ``object``, so there is no actual inheritance from the protocol.
if TYPE_CHECKING:
    _Base = _SignalToolkitHost
else:
    _Base = object


class PlottingMixin(_Base):
    """Mixin providing plotting and logging methods for SignalToolkitWidget."""

    def _update_plot(
        self,
        fitted_signal: Signal | None = None,
    ) -> None:
        """Update the main plot."""
        self.canvas.axes.clear()
        self.canvas.setup_dark_theme()

        if self.current_signal is None:
            self.canvas.draw()
            return

        # Plot current signal
        self.canvas.axes.plot(
            self.current_signal.time,
            self.current_signal.values,
            color="#4da6ff",
            linewidth=1.5,
            label="Signal",
        )

        # Plot fitted signal if provided
        if fitted_signal:
            self.canvas.axes.plot(
                fitted_signal.time,
                fitted_signal.values,
                color="#ff6b6b",
                linewidth=2,
                linestyle="--",
                label="Fit",
            )

        # Plot tangent line if enabled
        if self.show_tangent_check.isChecked():
            tangent = compute_tangent_line(
                self.current_signal,
                self.tangent_t_spin.value(),
            )
            self.canvas.axes.plot(
                tangent.t_range,
                tangent.line_values,
                color="#ffd93d",
                linewidth=2,
                label=f"Tangent (slope={tangent.slope:.3f})",
            )
            self.canvas.axes.scatter(
                [tangent.t_point],
                [tangent.y_point],
                color="#ffd93d",
                s=50,
                zorder=5,
            )

        self.canvas.axes.set_xlabel("Time")
        self.canvas.axes.set_ylabel("Value")
        self.canvas.axes.set_title(self.current_signal.name)
        self.canvas.axes.legend(loc="upper right")

        self.canvas.draw()

    def _update_secondary_plot(
        self,
        signal: Signal,
        title: str,
    ) -> None:
        """Update the secondary plot."""
        if not (signal is not None):
            raise ValueError("signal must be provided")
        self.canvas2.axes.clear()
        self.canvas2.setup_dark_theme()

        self.canvas2.axes.plot(
            signal.time,
            signal.values,
            color="#6bcb77",
            linewidth=1.5,
        )

        self.canvas2.axes.set_xlabel("Time")
        self.canvas2.axes.set_ylabel("Value")
        self.canvas2.axes.set_title(title)

        self.canvas2.draw()

    def _log(self, message: str) -> None:
        """Log a message to the result text area."""
        self.result_text.append(message)

    def set_joints(self, joints: list[str]) -> None:
        """Set the list of available joints."""
        if not (joints is not None):
            raise ValueError("joints must be provided")
        self.joint_names = joints
        self.joint_combo.clear()
        self.joint_combo.addItems(joints)
