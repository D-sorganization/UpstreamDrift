"""Counter-torque explorer for the Putting Green Simulator (#14469, epic #14452).

Charts the shared core's two-hand couple model (:mod:`counter_torque`): the
in-plane hand couples over the stroke, and the face twist a gust starts with
the grip preload held and with a slack grip. Couples and forces follow the
user's units; the model is closed form, so it replots on every change.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
)

from src.shared.python.putting_dynamics import counter_torque as ct
from src.shared.python.putting_dynamics.putter_presets import BALANCE_PRESETS
from src.shared.python.ui.units import (
    UnitSystem,
    force_suffix,
    from_display_torque,
    to_display_force,
    to_display_torque,
    torque_suffix,
)


class TorquePanel(QGroupBox):
    """Grip and stroke inputs, the couple and twist charts, and a summary.

    ``units`` gives the display units; :meth:`redraw` follows a change.
    """

    def __init__(self, units: Callable[[], UnitSystem]) -> None:
        super().__init__("Counter-Torque Stabilization")
        self._units = units
        self._shown_units = units()
        defaults = ct.CounterTorqueInputs()
        form = QFormLayout(self)
        self._preload_spin = _spin(2)
        form.addRow("Grip preload:", self._preload_spin)
        self._off_spin = _spin(2)
        form.addRow("Off-plane:", self._off_spin)
        self._gust_spin = _spin(2)
        form.addRow("Gust impulse:", self._gust_spin)
        self._torque_fields = {  # input -> its core field name and range key
            self._preload_spin: ("preload_nm", "PRELOAD_RANGE_NM"),
            self._off_spin: ("off_preload_nm", "PRELOAD_RANGE_NM"),
            self._gust_spin: ("gust_impulse_nm", "GUST_RANGE_NM"),
        }
        self._set_torque_spins(
            {
                self._preload_spin: defaults.preload_nm,
                self._off_spin: defaults.off_preload_nm,
                self._gust_spin: defaults.gust_impulse_nm,
            }
        )
        self._noise_spin = _spin(2, ct.NOISE_RANGE, defaults.noise_scale)
        form.addRow("Hand noise:", self._noise_spin)
        self._lie_spin = _spin(1, ct.LIE_RANGE_DEG, defaults.lie_deg, "°")
        form.addRow("Lie:", self._lie_spin)
        self._putter_combo = QComboBox()
        self._putter_combo.setSizeAdjustPolicy(  # long names must not widen phones
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._putter_combo.setMinimumContentsLength(8)
        for key, spec in BALANCE_PRESETS.items():
            self._putter_combo.addItem(spec.name, key)
        self._putter_combo.setCurrentIndex(list(BALANCE_PRESETS).index(defaults.putter))
        form.addRow("Putter:", self._putter_combo)
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        form.addRow(self._summary)
        self._figure = Figure(figsize=(5.0, 4.4), tight_layout=True)
        self._couple_ax = self._figure.add_subplot(2, 1, 1)
        self._twist_ax = self._figure.add_subplot(2, 1, 2)
        self.canvas = FigureCanvasQTAgg(self._figure)
        self.canvas.setMinimumSize(160, 380)
        form.addRow(self.canvas)  # in the scrolling controls column
        for spin in (
            self._preload_spin,
            self._off_spin,
            self._gust_spin,
            self._noise_spin,
            self._lie_spin,
        ):
            spin.valueChanged.connect(self.replot)
        self._putter_combo.currentIndexChanged.connect(self.replot)
        self.replot()

    @property
    def putter_combo(self) -> QComboBox:
        return self._putter_combo

    def current_inputs(self) -> ct.CounterTorqueInputs:
        """The dialled settings in SI, as the core takes them."""
        system = self._shown_units
        return ct.CounterTorqueInputs(
            preload_nm=from_display_torque(self._preload_spin.value(), system),
            off_preload_nm=from_display_torque(self._off_spin.value(), system),
            noise_scale=self._noise_spin.value(),
            gust_impulse_nm=from_display_torque(self._gust_spin.value(), system),
            lie_deg=self._lie_spin.value(),
            putter=str(self._putter_combo.currentData()),
        )

    def set_preload_nm(self, preload_nm: float) -> None:
        """Dial the in-plane grip preload, given in N·m."""
        self._set_torque_spins({self._preload_spin: preload_nm})

    def redraw(self) -> None:
        """Re-express the inputs and charts in the current units."""
        inputs = self.current_inputs()
        self._shown_units = self._units()
        self._set_torque_spins(
            {
                self._preload_spin: inputs.preload_nm,
                self._off_spin: inputs.off_preload_nm,
                self._gust_spin: inputs.gust_impulse_nm,
            }
        )
        self.replot()

    def replot(self) -> None:
        """Run the core for the dialled settings and redraw both charts."""
        inputs = self.current_inputs()
        series = ct.stroke_series(inputs)
        system = self._shown_units
        unit = torque_suffix(system).strip()
        times = [s.t_s for s in series]
        ax = self._couple_ax
        ax.clear()
        for field, label in (("tau_left_in_nm", "Lead"), ("tau_right_in_nm", "Trail")):
            values = [to_display_torque(getattr(s, field), system) for s in series]
            ax.plot(times, values, label=f"{label} hand")
        ax.axvline(ct.T_IMPACT_S, color="0.6", linestyle=":")
        ax.set_ylabel(f"Hand Couple ({unit})")
        ax.set_title("Hand Couples", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax = self._twist_ax
        ax.clear()
        ax.plot(times, [s.phi_preload_deg for s in series], label="With preload")
        ax.plot(times, [s.phi_slack_deg for s in series], "--", label="Slack grip")
        ax.axvline(ct.T_IMPACT_S, color="0.6", linestyle=":")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Face Twist (°)")
        ax.set_title("Gust Response", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        self.canvas.draw_idle()
        self._summarise(series, inputs, system)

    def title_extents(self) -> list[Bbox]:
        """Each chart title's drawn box, in figure pixels."""
        titles = (self._couple_ax.title, self._twist_ax.title)
        return [title.get_window_extent() for title in titles]

    def couple_lines(self) -> tuple[list[float], list[float]]:
        lead, trail = self._couple_ax.get_lines()[:2]
        return _ys(lead), _ys(trail)

    def twist_lines(self) -> tuple[list[float], list[float]]:
        held, slack = self._twist_ax.get_lines()[:2]
        return _ys(held), _ys(slack)

    def couple_axis_label(self) -> str:
        return str(self._couple_ax.get_ylabel())

    def summary_text(self) -> str:
        return self._summary.text()

    def _summarise(
        self,
        series: tuple[ct.CounterTorqueSample, ...],
        inputs: ct.CounterTorqueInputs,
        system: UnitSystem,
    ) -> None:
        held = max(abs(s.phi_preload_deg) for s in series)
        slack = max(abs(s.phi_slack_deg) for s in series)
        peak_n = max(
            max(abs(s.force_left_in_n), abs(s.force_right_in_n)) for s in series
        )
        impact = ct.evaluate(ct.T_IMPACT_S, inputs)
        force = f"{to_display_force(peak_n, system):.1f}{force_suffix(system)}"
        self._summary.setText(
            f"Peak gust twist {held:.2f}° with preload vs {slack:.2f}° slack. "
            f"At impact: face {impact.face_angle_impact_deg:+.2f}°, "
            f"path {impact.club_path_impact_deg:+.2f}°. "
            f"Peak hand force {force}."
        )

    def torque_input_ranges(self) -> dict[str, tuple[float, float]]:
        """Each couple input's core range in N·m, keyed by its field name."""
        return {
            field: getattr(ct, range_key)
            for field, range_key in self._torque_fields.values()
        }

    def _set_torque_spins(self, values_nm: dict[QDoubleSpinBox, float]) -> None:
        system = self._shown_units
        for spin, value_nm in values_nm.items():
            _, range_key = self._torque_fields[spin]
            lo, hi = (to_display_torque(v, system) for v in getattr(ct, range_key))
            spin.blockSignals(True)
            spin.setSuffix(torque_suffix(system))
            spin.setRange(lo, hi)
            spin.setValue(to_display_torque(value_nm, system))
            spin.blockSignals(False)
        if hasattr(self, "canvas"):
            self.replot()


def _ys(line: Line2D) -> list[float]:
    """A plotted line's y values as floats."""
    return [float(y) for y in np.asarray(line.get_ydata(), dtype=float)]


def _spin(
    decimals: int,
    bounds: tuple[float, float] = (0.0, 1.0),
    value: float = 0.0,
    suffix: str = "",
) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setDecimals(decimals)
    spin.setRange(*bounds)
    spin.setSingleStep(10.0 ** (1 - decimals))
    spin.setValue(value)
    spin.setSuffix(suffix)
    return spin


__all__ = ["TorquePanel"]
