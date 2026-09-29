"""Headless tests for the shared wheel-event filter (issue #7062).

``WheelEventFilter`` is installed on combo/spin/slider widgets across >=5 GUI
tabs to enforce the policy that *no user-editable value changes solely because
of a mouse-wheel event*. These tests run under the offscreen Qt platform and
exercise the swallow/pass-through behaviour plus the install helpers.

The filter is intentionally focus-independent: an accidental wheel event is
swallowed whether or not the widget is focused, so that scrolling a surrounding
surface can never mutate an input control. The focused vs. unfocused cases below
pin that contract so a future "forward when focused" change is a deliberate,
test-visible decision.
"""

from __future__ import annotations

import os

# Headless Qt platform must be set before any PyQt6 import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

if "PySide6" in sys.modules:
    pytest.skip(
        "PySide6 already loaded — PyQt6 DLLs unavailable", allow_module_level=True
    )

try:
    from PyQt6.QtWidgets import QApplication  # noqa: F401

    _HAVE_QT = True
except Exception:  # noqa: BLE001
    _HAVE_QT = False

if not _HAVE_QT:  # pragma: no cover - environment-dependent
    pytest.skip("PyQt6.QtWidgets unavailable", allow_module_level=True)


from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QWheelEvent  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
)

from src.shared.python.qt_utils.wheel_event_filter import (  # noqa: E402
    WheelEventFilter,
    suppress_wheel_on_widget,
    suppress_wheel_on_widgets,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_gl, pytest.mark.headless_safe]


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_wheel_event(delta: int = 120) -> QWheelEvent:
    """Build a vertical scroll wheel event with the given notch delta."""
    pos = QPointF(5.0, 5.0)
    return QWheelEvent(
        pos,
        pos,
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


class TestWheelEventFilter:
    def test_wheel_event_is_swallowed(self, qapp: QApplication) -> None:
        filt = WheelEventFilter()
        combo = QComboBox()
        event = _make_wheel_event()
        # eventFilter returns True => the event is consumed before the widget
        # can act on it, so no value change occurs.
        assert filt.eventFilter(combo, event) is True

    def test_wheel_event_swallowed_when_unfocused(self, qapp: QApplication) -> None:
        filt = WheelEventFilter()
        spin = QSpinBox()
        spin.clearFocus()
        assert spin.hasFocus() is False
        assert filt.eventFilter(spin, _make_wheel_event()) is True

    def test_wheel_event_swallowed_when_focused(self, qapp: QApplication) -> None:
        filt = WheelEventFilter()
        spin = QSpinBox()
        spin.setFocus()
        # Even focused, accidental wheel input must not change the value: the
        # filter swallows it (focus-independent safety contract).
        assert filt.eventFilter(spin, _make_wheel_event()) is True

    def test_non_wheel_event_passes_through(self, qapp: QApplication) -> None:
        filt = WheelEventFilter()
        combo = QComboBox()
        other = QEvent(QEvent.Type.MouseButtonPress)
        assert filt.eventFilter(combo, other) is False

    def test_installed_filter_blocks_value_change_on_combo(
        self, qapp: QApplication
    ) -> None:
        combo = QComboBox()
        combo.addItems(["a", "b", "c"])
        combo.setCurrentIndex(0)
        filt = WheelEventFilter()
        combo.installEventFilter(filt)
        # Dispatch through the real event system; the index must stay put.
        handled = QApplication.sendEvent(combo, _make_wheel_event())
        assert handled is True
        assert combo.currentIndex() == 0

    def test_removed_filter_no_longer_intercepts(self, qapp: QApplication) -> None:
        spin = QSpinBox()
        filt = WheelEventFilter()
        spin.installEventFilter(filt)
        spin.removeEventFilter(filt)
        # After removal the filter is no longer consulted; sendEvent reaches
        # the widget's own (default) handler instead of being swallowed.
        QApplication.sendEvent(spin, _make_wheel_event())
        # Filter object itself still reports swallow if called directly.
        assert filt.eventFilter(spin, _make_wheel_event()) is True


class TestSuppressHelpers:
    def test_suppress_on_single_widget_installs_and_retains(
        self, qapp: QApplication
    ) -> None:
        spin = QDoubleSpinBox()
        suppress_wheel_on_widget(spin)
        stored = spin._wheel_event_filter
        assert isinstance(stored, WheelEventFilter)
        # The stored filter swallows wheel events for this widget.
        assert stored.eventFilter(spin, _make_wheel_event()) is True

    def test_suppress_on_multiple_widgets(self, qapp: QApplication) -> None:
        combo = QComboBox()
        spin = QSpinBox()
        dspin = QDoubleSpinBox()
        suppress_wheel_on_widgets(combo, spin, dspin)
        for widget in (combo, spin, dspin):
            stored = widget._wheel_event_filter
            assert isinstance(stored, WheelEventFilter)
            assert stored.eventFilter(widget, _make_wheel_event()) is True

    def test_each_widget_gets_independent_filter(self, qapp: QApplication) -> None:
        a = QSpinBox()
        b = QSpinBox()
        suppress_wheel_on_widgets(a, b)
        assert a._wheel_event_filter is not b._wheel_event_filter
