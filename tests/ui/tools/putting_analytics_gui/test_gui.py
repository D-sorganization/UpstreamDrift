"""Offscreen smoke tests for the putting analytics window (#14177)."""

from __future__ import annotations

import os
import time
from typing import Any

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

TIMEOUT_S = 90.0


@pytest.fixture
def qapp() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")
    return QApplication.instance() or QApplication([])


def _wait_until_idle(qapp: Any, widget: Any) -> None:
    deadline = time.monotonic() + TIMEOUT_S
    while widget.is_running and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert not widget.is_running


def _fill(widget: Any) -> None:
    widget.baseline_distances.setText("1 3 6 10 20 40")
    widget.baseline_putts.setText("1.0 1.2 1.5 1.7 1.9 2.2")
    widget.rollouts.setValue(30)
    widget.aims.setText("0 2")
    widget.paces.setText("1")
    # One short class keeps the library study (grid x candidates x n) quick.
    widget.classes.setPlainText("3 1 uphill 10 2")


def test_bad_input_is_reported_without_starting_a_study(qapp: Any) -> None:
    from src.tools.putting_analytics_gui.gui import MainWidget

    widget = MainWidget()
    widget.run_study()
    assert not widget.is_running
    assert "baseline distances" in widget.status_label.text()
    assert widget.cleanup()


@pytest.mark.parametrize("index", [0, 1, 2])
def test_each_study_fills_the_table(qapp: Any, index: int) -> None:
    from src.tools.putting_analytics_gui.gui import MainWidget

    widget = MainWidget()
    _fill(widget)
    widget.mode_combo.setCurrentIndex(index)
    widget.run_study()
    _wait_until_idle(qapp, widget)
    assert widget.table.rowCount() > 0, widget.status_label.text()
    assert widget.status_label.text().startswith("Done")
    assert widget.run_button.isEnabled()
    assert widget.cleanup()


def test_service_rejection_is_shown(qapp: Any) -> None:
    from src.tools.putting_analytics_gui.gui import MainWidget

    widget = MainWidget()
    _fill(widget)
    widget.aims.setText("90")
    widget.run_study()
    _wait_until_idle(qapp, widget)
    assert widget.table.rowCount() == 0
    assert "rejected" in widget.status_label.text()
    assert widget.cleanup()


def test_objective_choices_are_the_ones_the_service_accepts(qapp: Any) -> None:
    from src.shared.python.putting_analytics.optimize import OBJECTIVES
    from src.tools.putting_analytics_gui.gui import MainWidget

    widget = MainWidget()
    choices = [widget.objective.itemText(i) for i in range(widget.objective.count())]
    assert tuple(choices) == OBJECTIVES
    assert widget.cleanup()


def test_studies_run_on_the_shared_async_action_bar(qapp: Any) -> None:
    from src.tools.async_action import AsyncActionBar
    from src.tools.putting_analytics_gui.gui import MainWidget

    widget = MainWidget()
    assert isinstance(widget.action_bar, AsyncActionBar)
    _fill(widget)
    widget.run_study()
    assert widget.is_running and not widget.run_button.isEnabled()
    _wait_until_idle(qapp, widget)
    assert widget.run_button.isEnabled()
    assert widget.cleanup()


def test_an_unexpected_error_is_not_called_a_rejection(
    qapp: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.shared.python.putting_analytics import service
    from src.tools.putting_analytics_gui.gui import MainWidget

    def _boom(_args: Any) -> Any:
        raise RuntimeError("integrator diverged")

    monkeypatch.setattr(service, "optimize_putt", _boom)
    widget = MainWidget()
    _fill(widget)
    widget.run_study()
    _wait_until_idle(qapp, widget)
    text = widget.status_label.text()
    assert "failed" in text and "integrator diverged" in text
    assert "rejected" not in text
    assert widget.cleanup()
