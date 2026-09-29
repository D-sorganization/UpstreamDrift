"""Editor for selecting a :class:`DataChannel` and its (vmin, vmax) range.

The widget is constructed with a list of available
:class:`DataChannel` objects. The user picks one from a combobox and
edits the optional ``vmin`` / ``vmax`` bounds. A "fit to data" button
asks an injected callable to recompute bounds from the underlying data —
the default implementation calls :meth:`DataChannel.auto_range`, but
callers can supply their own (e.g. windowed range, robust quantiles).

Public API
----------
* ``value() -> DataChannel``                  — current selection.
* ``set_value(channel: DataChannel) -> None`` — programmatic selection.
* ``range_value() -> tuple[float, float]``    — current ``(vmin, vmax)``.
* ``set_range(vmin, vmax) -> None``           — programmatic range.
* ``channelChanged(DataChannel)`` Qt signal
* ``rangeChanged(float, float)`` Qt signal
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..channels import DataChannel

__all__ = ["DataChannelEditor"]

logger = logging.getLogger(__name__)

_RANGE_LIMIT = 1.0e30  # spinbox caps; channel data may exceed only with NaN
_DEFAULT_RANGE = (0.0, 1.0)

FitFn = Callable[[DataChannel], tuple[float, float]]


def _default_fit(channel: DataChannel) -> tuple[float, float]:
    lo, hi = channel.auto_range()
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return _DEFAULT_RANGE
    return (float(lo), float(hi))


class DataChannelEditor(QWidget):
    """Channel + ``(vmin, vmax)`` editor with a "fit to data" button.

    Parameters
    ----------
    channels:
        Non-empty sequence of :class:`DataChannel` candidates. The
        first entry is selected by default unless ``initial`` is given.
    initial:
        Optional starting :class:`DataChannel`. Must appear in
        ``channels``.
    initial_range:
        Optional starting ``(vmin, vmax)``; defaults to the result of
        ``fit_fn`` on the initial channel.
    fit_fn:
        Callable used by the "fit to data" button. Defaults to
        :meth:`DataChannel.auto_range`.
    parent:
        Optional Qt parent.
    """

    channelChanged = pyqtSignal(DataChannel)
    rangeChanged = pyqtSignal(float, float)

    def __init__(
        self,
        channels: Sequence[DataChannel],
        initial: DataChannel | None = None,
        initial_range: tuple[float, float] | None = None,
        fit_fn: FitFn | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        if not isinstance(channels, Sequence) or isinstance(channels, str):
            raise TypeError(
                "channels must be a sequence of DataChannel; "
                f"got {type(channels).__name__}"
            )
        if len(channels) == 0:
            raise ValueError("channels must not be empty")
        for idx, c in enumerate(channels):
            if not isinstance(c, DataChannel):
                raise TypeError(
                    f"channels[{idx}] must be DataChannel; got {type(c).__name__}"
                )

        self._channels: list[DataChannel] = list(channels)
        self._fit_fn: FitFn = fit_fn or _default_fit
        if initial is None:
            initial = self._channels[0]
        if initial not in self._channels:
            raise ValueError("initial channel is not in channels list")

        self._channel: DataChannel = initial
        self._suppress = False

        if initial_range is None:
            initial_range = self._fit_fn(initial)
        vmin, vmax = self._validate_range(initial_range)
        self._range: tuple[float, float] = (vmin, vmax)

        # ----- UI ----------------------------------------------------
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Data channel", self)
        outer.addWidget(group)

        form = QFormLayout(group)

        self._channel_combo = QComboBox(group)
        self._channel_combo.setObjectName("data_channel_combo")
        for c in self._channels:
            self._channel_combo.addItem(c.name, c)
        self._channel_combo.setCurrentIndex(self._channels.index(initial))
        form.addRow("Channel", self._channel_combo)

        self._vmin_spin = self._make_spin("data_channel_vmin")
        self._vmax_spin = self._make_spin("data_channel_vmax")
        form.addRow("vmin", self._vmin_spin)
        form.addRow("vmax", self._vmax_spin)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        self._fit_button = QPushButton("Fit to data", group)
        self._fit_button.setObjectName("data_channel_fit_button")
        button_row.addWidget(self._fit_button)
        button_row.addStretch(1)
        form.addRow(button_row)

        self._sync_widgets_from_state()

        self._channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        self._vmin_spin.valueChanged.connect(self._on_range_changed)
        self._vmax_spin.valueChanged.connect(self._on_range_changed)
        self._fit_button.clicked.connect(self._on_fit_clicked)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def value(self) -> DataChannel:
        """Return the currently selected :class:`DataChannel`."""
        return self._channel

    def set_value(self, channel: DataChannel) -> None:
        """Programmatically select ``channel`` (must be in the constructor list)."""
        if not isinstance(channel, DataChannel):
            raise TypeError(
                f"channel must be DataChannel; got {type(channel).__name__}"
            )
        if channel not in self._channels:
            raise ValueError("channel is not in the editor's channel list")
        if channel is self._channel or channel == self._channel:
            return
        self._channel = channel
        with self._suppressed():
            self._channel_combo.setCurrentIndex(self._channels.index(channel))
        self.channelChanged.emit(channel)

    def range_value(self) -> tuple[float, float]:
        """Return the current ``(vmin, vmax)`` tuple."""
        return self._range

    def set_range(self, vmin: float, vmax: float) -> None:
        """Programmatically set ``(vmin, vmax)``.

        Emits :pyattr:`rangeChanged` only when the values change.
        """
        new_range = self._validate_range((vmin, vmax))
        if new_range == self._range:
            return
        self._range = new_range
        with self._suppressed():
            self._vmin_spin.setValue(new_range[0])
            self._vmax_spin.setValue(new_range[1])
        self.rangeChanged.emit(*new_range)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_spin(self, name: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setObjectName(name)
        spin.setDecimals(6)
        spin.setRange(-_RANGE_LIMIT, _RANGE_LIMIT)
        spin.setSingleStep(0.1)
        return spin

    @staticmethod
    def _validate_range(rng: tuple[float, float]) -> tuple[float, float]:
        if (
            not isinstance(rng, tuple)
            or len(rng) != 2
            or not all(isinstance(v, (int, float)) for v in rng)
            or any(isinstance(v, bool) for v in rng)
        ):
            raise TypeError(
                f"range must be a (vmin, vmax) tuple of numbers; got {rng!r}"
            )
        vmin = float(rng[0])
        vmax = float(rng[1])
        if not math.isfinite(vmin) or not math.isfinite(vmax):
            raise ValueError(
                f"vmin and vmax must be finite; got vmin={vmin!r}, vmax={vmax!r}"
            )
        if vmax <= vmin:
            raise ValueError(
                f"vmax must be strictly greater than vmin; got {vmin}, {vmax}"
            )
        return (vmin, vmax)

    @contextmanager
    def _suppressed(self) -> Iterator[None]:
        prev = self._suppress
        self._suppress = True
        for w in (self._channel_combo, self._vmin_spin, self._vmax_spin):
            w.blockSignals(True)
        try:
            yield
        finally:
            for w in (self._channel_combo, self._vmin_spin, self._vmax_spin):
                w.blockSignals(False)
            self._suppress = prev

    def _sync_widgets_from_state(self) -> None:
        with self._suppressed():
            self._vmin_spin.setValue(self._range[0])
            self._vmax_spin.setValue(self._range[1])

    # --- slots ---------------------------------------------------------

    def _on_channel_changed(self, index: int) -> None:
        if self._suppress or index < 0:
            return
        new_channel = self._channels[index]
        if new_channel == self._channel:
            return
        self._channel = new_channel
        self.channelChanged.emit(new_channel)

    def _on_range_changed(self, _value: float) -> None:
        if self._suppress:
            return
        try:
            new_range = self._validate_range(
                (float(self._vmin_spin.value()), float(self._vmax_spin.value()))
            )
        except (TypeError, ValueError):
            # Invalid intermediate state (e.g. vmax temporarily <= vmin
            # while user types). Don't roll back — just defer until the
            # next valid pair.
            return
        if new_range == self._range:
            return
        self._range = new_range
        self.rangeChanged.emit(*new_range)

    def _on_fit_clicked(self) -> None:
        try:
            new_range = self._validate_range(self._fit_fn(self._channel))
        except (TypeError, ValueError) as exc:
            logger.warning(
                "fit_fn returned invalid range for %s: %s", self._channel.name, exc
            )
            return
        self.set_range(*new_range)
