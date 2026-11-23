"""Impact-zone slow-motion toggle on the results PlaybackBar (#15524)."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")

from src.tools.model_explorer.playback_bar import PlaybackBar  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])


@pytest.fixture
def bar(qapp):
    widget = PlaybackBar()
    yield widget
    widget.deleteLater()


def _session(frames: int = 10):
    return SimpleNamespace(frame_count=frames, current_frame=0, current_time_s=0.0)


def _record(bar):
    seen: list[bool] = []
    bar.impact_zone_slow_toggled.connect(seen.append)
    return seen


def test_toggle_starts_unchecked_and_disabled(bar):
    assert not bar.impact_slow_button.isChecked()
    assert not bar.impact_slow_button.isEnabled()
    assert bar.impact_slow_button.toolTip()  # explains why it is unavailable


def test_available_toggle_emits_user_choice(bar):
    seen = _record(bar)
    bar.set_session(_session())
    bar.set_impact_zone_available(True)
    assert bar.impact_slow_button.isEnabled()
    bar.impact_slow_button.click()
    bar.impact_slow_button.click()
    assert seen == [True, False]


def test_losing_availability_while_on_unchecks_and_emits_once(bar):
    seen = _record(bar)
    bar.set_session(_session())
    bar.set_impact_zone_available(True)
    bar.impact_slow_button.click()
    bar.set_impact_zone_available(False)
    bar.set_impact_zone_available(False)  # idempotent
    assert not bar.impact_slow_button.isChecked()
    assert not bar.impact_slow_button.isEnabled()
    assert seen == [True, False]


def test_unavailable_while_off_emits_nothing(bar):
    seen = _record(bar)
    bar.set_impact_zone_available(True)
    bar.set_impact_zone_available(False)
    assert seen == []


def test_clearing_session_resets_slow_motion(bar):
    seen = _record(bar)
    bar.set_session(_session())
    bar.set_impact_zone_available(True)
    bar.impact_slow_button.click()
    bar.set_session(None)
    assert not bar.impact_slow_button.isChecked()
    assert not bar.impact_slow_button.isEnabled()
    assert seen == [True, False]


def test_availability_requires_a_playable_session(bar):
    bar.set_session(_session(frames=1))
    bar.set_impact_zone_available(True)
    assert not bar.impact_slow_button.isEnabled()
    bar.set_session(_session())
    assert bar.impact_slow_button.isEnabled()
