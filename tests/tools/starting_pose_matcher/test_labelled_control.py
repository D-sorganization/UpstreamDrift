"""Unit tests for ``LabelledControl.set_value_silent`` (issue #7740).

``set_value_silent`` encapsulates the spin/slider/scale coupling so callers
(such as session restore in ``gui_session_mixin``) no longer reach into the
private ``_scale`` attribute or poke ``spin``/``slider`` directly. The key
contract: the value is applied to *both* the spin box and the slider, and
neither emits ``valueChanged`` (so transform-change handlers don't re-fire).
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.starting_pose_matcher._gui_common import LabelledControl  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


def test_set_value_silent_updates_spin_and_slider(qapp: QApplication) -> None:
    """Both the spin box and slider reflect the new value."""
    ctrl = LabelledControl(
        "scale", units="x", scale=0.01, slider_range=(-100, 100), decimals=2
    )
    ctrl.set_value_silent(0.5)

    assert ctrl.value() == pytest.approx(0.5)
    # slider value is the scaled tick position: round(0.5 / 0.01) == 50
    assert ctrl.slider.value() == 50


def test_set_value_silent_emits_no_signals(qapp: QApplication) -> None:
    """Neither spin nor slider emits ``valueChanged`` during a silent set."""
    ctrl = LabelledControl(
        "tx", units="m", scale=0.01, slider_range=(-100, 100), decimals=2
    )
    fired: list[str] = []
    ctrl.spin.valueChanged.connect(lambda _v: fired.append("spin"))
    ctrl.slider.valueChanged.connect(lambda _v: fired.append("slider"))

    ctrl.set_value_silent(0.25)

    assert fired == []
    assert ctrl.value() == pytest.approx(0.25)
    assert ctrl.slider.value() == 25


def test_set_value_silent_restores_signal_state(qapp: QApplication) -> None:
    """Signals are re-enabled afterwards so normal edits still propagate."""
    ctrl = LabelledControl(
        "ty", units="m", scale=0.01, slider_range=(-100, 100), decimals=2
    )
    ctrl.set_value_silent(0.1)

    fired: list[str] = []
    ctrl.spin.valueChanged.connect(lambda _v: fired.append("spin"))
    # A subsequent normal set must still emit.
    ctrl.set_value(0.2)
    assert "spin" in fired
