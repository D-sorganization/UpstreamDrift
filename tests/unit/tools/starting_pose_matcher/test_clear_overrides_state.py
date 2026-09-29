"""Unit tests for Starting Pose Matcher 'Clear overrides' state restoration and confirmation (#8889)."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest

if "PySide6" in sys.modules:
    pytest.skip(
        "PySide6 already loaded — PyQt6 DLLs unavailable", allow_module_level=True
    )

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox

    _HAVE_QT = True
except Exception:  # noqa: BLE001
    _HAVE_QT = False

if not _HAVE_QT:
    pytest.skip("PyQt6.QtWidgets unavailable", allow_module_level=True)

from src.tools.starting_pose_matcher.core import MocapEvents
from src.tools.starting_pose_matcher.gui_main_widget import MainWidget

pytestmark = pytest.mark.unit


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


def test_clear_overrides_without_xlsx_restores_original_state_when_confirmed(
    qapp, monkeypatch
) -> None:
    """Issue #8889: Clear overrides without xlsx path must restore original events and ask confirmation."""
    widget = MainWidget()
    widget.df = pd.DataFrame(
        {
            "time": [0.0, 0.01, 0.02, 0.03, 0.04],
            "mid_X": [0.0, 0.0, 0.0, 0.0, 0.0],
            "mid_Y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "mid_Z": [0.0, 0.0, 0.0, 0.0, 0.0],
            "club_X": [0.1, 0.2, 0.3, 0.4, 0.5],
            "club_Y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "club_Z": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    widget._xlsx_path = None
    original_events = MocapEvents(
        A_sample=1.0, T_sample=2.0, I_sample=3.0, F_sample=4.0
    )
    widget._events_original = original_events
    widget.events = MocapEvents(A_sample=1.0, T_sample=2.0, I_sample=3.0, F_sample=4.0)
    original_t_sample = widget.events.T_sample
    assert widget._events_original.T_sample == original_t_sample

    # Manually set override for T
    widget.current_frame = 2
    widget.combo_set_event.setCurrentText("T - Top of Backswing")
    widget._set_event_to_current_frame()

    assert widget.event_overrides.get("T") == 3
    assert widget.events.T_sample == 3.0

    # User cancels dialog -> overrides must remain
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.No,
    )
    widget._clear_event_overrides()
    assert widget.event_overrides.get("T") == 3
    assert widget.events.T_sample == 3.0

    # User confirms dialog -> overrides cleared and original state restored
    dialog_called = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: (
            dialog_called.append(args) or QMessageBox.StandardButton.Yes
        ),
    )
    widget._clear_event_overrides()

    assert len(dialog_called) == 1
    assert widget.event_overrides == {}
    assert widget.events.T_sample == original_t_sample
    assert widget.events.T_sample == widget._events_original.T_sample
