"""End-to-end PyQt6 interaction tests for the Grip Wrench Plots tool (#15522).

``tests/unit/tools/grip_wrench_plots`` and ``tests/unit/grip_overlay`` call
widget methods and ``.click()`` directly. These tests drive the real tabbed
window the way a user does, headless: ``QTest`` key presses type the frame list
and step the combo boxes, real left-button clicks press Run, Cancel and "Open
Grip Wrench JSON...", and the run goes through the real ``QThreadPool`` worker
with queued signals. Assertions read what the user sees: button state, status
line, progress bar, current tab and the drawn figure titles.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QCoreApplication, Qt, QThread  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QFileDialog, QWidget  # noqa: E402

from src.shared.python.biomechanics.grip_plot_model import (  # noqa: E402
    GripPlotSeries,
    build_grip_plot_series,
    plot_series_to_json,
)
from src.shared.python.biomechanics.grip_wrench import (  # noqa: E402
    HandWrench,
    analyze_grip,
)
from src.shared.python.loop_nullspace.facade import (  # noqa: E402
    AnalysisCancelledError,
)
from src.tools.grip_wrench_plots.gui import GripWrenchPlotWidget  # noqa: E402
from src.tools.grip_wrench_plots.nsx_job import NSXRequest  # noqa: E402
from src.tools.grip_wrench_plots.window import GripWrenchTabsWidget  # noqa: E402
from tests.unit.tools.grip_wrench_plots.golden_support import (  # noqa: E402
    golden_model,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

TIMEOUT_S = 30.0
NO_GRIP = "unavailable: no grip wrench loaded"


def _pump_until(predicate: Callable[[], bool], timeout_s: float = TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        QThread.msleep(5)
    QCoreApplication.processEvents()
    return predicate()


def _click(widget: QWidget) -> None:
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton)


def _type_into(widget: QWidget, text: str) -> None:
    """Focus ``widget``, select all and type ``text`` key by key."""
    widget.setFocus()
    QTest.keyClick(widget, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(widget, text)


def _grip_series() -> GripPlotSeries:
    analyses = [
        analyze_grip(
            HandWrench("L", (0, 0, 1.0), (10.0 * k, 0, 0), (0, 0, 0)),
            HandWrench("R", (0, 0, 0.8), (-10.0 * k, 0, 0), (0, 0, 0)),
            split_method="efc_force",
        )
        for k in range(4)
    ]
    return build_grip_plot_series([0.0, 0.01, 0.02, 0.03], analyses)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    assert isinstance(app, QApplication)
    return app


@pytest.fixture
def grip(qapp: QApplication) -> Iterator[GripWrenchPlotWidget]:
    widget = GripWrenchPlotWidget()
    widget.show()
    assert QTest.qWaitForWindowExposed(widget)
    yield widget
    widget.cleanup()
    widget.close()


class _Jobs:
    """Records the requests a user's clicks produced and plays one scripted job."""

    def __init__(self, behaviour: Callable[..., Any]) -> None:
        self.requests: list[NSXRequest] = []
        self.started = threading.Event()
        self._behaviour = behaviour

    def __call__(self, request: NSXRequest) -> Callable[..., Any]:
        self.requests.append(request)
        return lambda on_stage, should_cancel: self._behaviour(
            self, on_stage, should_cancel
        )


def _finish(_jobs: _Jobs, on_stage: Callable[..., None], _cancel: Any) -> Any:
    on_stage("facade", 50.0, None)
    return golden_model()


def _fail(_jobs: _Jobs, _on_stage: Any, _cancel: Any) -> Any:
    raise ValueError("boom")


def _block_until_cancelled(
    jobs: _Jobs, _on_stage: Any, should_cancel: Callable[[], bool]
) -> Any:
    jobs.started.set()
    while not should_cancel():
        time.sleep(0.005)
    raise AnalysisCancelledError


def _open_window(qapp: QApplication, jobs: _Jobs) -> GripWrenchTabsWidget:
    widget = GripWrenchTabsWidget(job_factory=jobs, sources=["fake"])
    widget.show()
    assert QTest.qWaitForWindowExposed(widget)
    return widget


@pytest.fixture
def make_window(qapp: QApplication) -> Iterator[Callable[[Callable[..., Any]], Any]]:
    opened: list[GripWrenchTabsWidget] = []

    def factory(behaviour: Callable[..., Any]) -> tuple[GripWrenchTabsWidget, _Jobs]:
        jobs = _Jobs(behaviour)
        widget = _open_window(qapp, jobs)
        opened.append(widget)
        return widget, jobs

    yield factory
    for widget in opened:
        widget.cleanup()
        widget.close()


def _choose_file(monkeypatch: pytest.MonkeyPatch, path: Path | str) -> None:
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(path), ""))
    )


# ------------------------------------------------------------ grip widget
def test_grip_widget_launches_empty_with_both_couple_frames(
    grip: GripWrenchPlotWidget,
) -> None:
    assert grip.isVisible()
    assert grip.status_label.text() == NO_GRIP
    assert [grip.frame_combo.itemText(i) for i in range(grip.frame_combo.count())] == [
        "world",
        "club",
    ]
    assert grip.open_button.text() == "Open Grip Wrench JSON..."
    assert not grip.figure.axes


def test_open_button_loads_json_then_keyboard_flips_couple_frame(
    grip: GripWrenchPlotWidget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "grip.json"
    path.write_text(plot_series_to_json(_grip_series()), encoding="utf-8")
    _choose_file(monkeypatch, path)
    _click(grip.open_button)
    assert grip.status_label.text() == "Split method: efc_force | 4 samples"
    assert len(grip.figure.axes) == 4
    assert grip.figure.axes[2].get_title().endswith("(World)")
    grip.frame_combo.setFocus()
    QTest.keyClick(grip.frame_combo, Qt.Key.Key_Down)
    assert grip.frame_combo.currentText() == "club"
    assert grip.figure.axes[2].get_title().endswith("(Club)")
    QTest.keyClick(grip.frame_combo, Qt.Key.Key_Up)
    assert grip.figure.axes[2].get_title().endswith("(World)")


def test_cancelled_file_dialog_leaves_the_view_untouched(
    grip: GripWrenchPlotWidget, monkeypatch: pytest.MonkeyPatch
) -> None:
    _choose_file(monkeypatch, "")
    _click(grip.open_button)
    assert grip.status_label.text() == NO_GRIP
    assert not grip.figure.axes


def test_unavailable_payload_says_why_instead_of_plotting_zeros(
    grip: GripWrenchPlotWidget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _grip_series().to_dict()
    payload.update(available=False, reason="no net force from this engine")
    path = tmp_path / "unavailable.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _choose_file(monkeypatch, path)
    _click(grip.open_button)
    assert grip.status_label.text() == "unavailable: no net force from this engine"


@pytest.mark.parametrize(
    "content",
    ["not json at all", "[1, 2, 3]", json.dumps({"traces": {}})],
    ids=["garbage", "not-an-object", "missing-time"],
)
def test_bad_json_is_reported_not_raised_and_keeps_the_prior_plot(
    grip: GripWrenchPlotWidget,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: str,
) -> None:
    good = tmp_path / "good.json"
    good.write_text(plot_series_to_json(_grip_series()), encoding="utf-8")
    _choose_file(monkeypatch, good)
    _click(grip.open_button)
    bad = tmp_path / "bad.json"
    bad.write_text(content, encoding="utf-8")
    _choose_file(monkeypatch, bad)
    _click(grip.open_button)
    assert grip.status_label.text().startswith("unavailable: ")
    assert len(grip.figure.axes) == 4


def test_missing_file_is_reported_as_unavailable(
    grip: GripWrenchPlotWidget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _choose_file(monkeypatch, tmp_path / "absent.json")
    _click(grip.open_button)
    assert grip.status_label.text().startswith("unavailable: ")


# ---------------------------------------------------------- tabbed window
def test_window_launches_idle_with_the_grip_tab_first(
    make_window: Callable[..., Any],
) -> None:
    window, _ = make_window(_finish)
    assert window.isVisible()
    assert window.tabs.tabText(0) == "Grip Wrench"
    assert window.tabs.currentIndex() == 0
    assert window.run_button.isEnabled()
    assert not window.cancel_button.isEnabled()
    assert not window.is_running()
    assert window.progress.value() <= 0  # Qt reports -1 until the first run
    assert window.frames_edit.text() == "0, 3, 6, 9"
    assert window.preset_combo.count() >= 2


def test_typed_frames_keyboard_preset_and_run_click_produce_the_request(
    make_window: Callable[..., Any],
) -> None:
    window, jobs = make_window(_finish)
    finished: list[object] = []
    window.analysis_finished.connect(finished.append)
    first_preset = window.selected_preset()
    window.preset_combo.setFocus()
    QTest.keyClick(window.preset_combo, Qt.Key.Key_Down)
    assert window.selected_preset() != first_preset
    _type_into(window.frames_edit, "4, 1 7")
    _click(window.allocation_check)
    _click(window.run_button)
    assert _pump_until(lambda: bool(finished))
    request = jobs.requests[-1]
    assert request.frame_indices == (4, 1, 7)
    assert request.source_id == "fake"
    assert request.preset == window.selected_preset() != first_preset
    assert request.include_allocation is True


def test_run_click_shows_progress_result_tab_and_returns_to_idle(
    make_window: Callable[..., Any],
) -> None:
    window, _ = make_window(_finish)
    _click(window.run_button)
    assert _pump_until(lambda: not window.is_running())
    assert window.progress.value() == 100
    assert window.tabs.currentIndex() == 1
    assert window.status_label.text() == "complete: preset envelope_only"
    assert window.plot_model() is not None
    assert window.run_button.isEnabled()
    assert not window.cancel_button.isEnabled()


def test_invalid_frame_text_blocks_the_run_with_a_message(
    make_window: Callable[..., Any],
) -> None:
    window, jobs = make_window(_finish)
    _type_into(window.frames_edit, "3 3")
    _click(window.run_button)
    assert window.status_label.text().startswith("cannot run: frame indices must be")
    assert not window.is_running()
    assert window.run_button.isEnabled()
    assert not jobs.requests
    _type_into(window.frames_edit, "two")
    _click(window.run_button)
    assert window.status_label.text().startswith("cannot run: frames must be integers")


def test_failing_job_reports_failure_and_allows_another_run(
    make_window: Callable[..., Any],
) -> None:
    window, jobs = make_window(_fail)
    failures: list[str] = []
    window.analysis_failed.connect(failures.append)
    _click(window.run_button)
    assert _pump_until(lambda: bool(failures))
    assert failures == ["ValueError: boom"]
    assert window.status_label.text() == "failed: ValueError: boom"
    assert window.run_button.isEnabled()
    assert window.plot_model() is None
    _click(window.run_button)
    assert _pump_until(lambda: len(failures) == 2)
    assert len(jobs.requests) == 2


def test_cancel_click_stops_a_running_job(make_window: Callable[..., Any]) -> None:
    window, jobs = make_window(_block_until_cancelled)
    cancelled: list[bool] = []
    window.analysis_cancelled.connect(lambda: cancelled.append(True))
    _click(window.run_button)
    assert jobs.started.wait(TIMEOUT_S)
    assert window.is_running()
    assert not window.run_button.isEnabled()
    assert window.cancel_button.isEnabled()
    _click(window.cancel_button)
    assert window.status_label.text() == "cancelling"
    assert _pump_until(lambda: bool(cancelled))
    assert window.status_label.text() == "cancelled"
    assert not window.is_running()
    assert window.run_button.isEnabled()
    assert not window.cancel_button.isEnabled()
    assert window.plot_model() is None


def test_closing_during_a_run_cancels_it_and_cleanup_is_idempotent(
    make_window: Callable[..., Any],
) -> None:
    window, jobs = make_window(_block_until_cancelled)
    _click(window.run_button)
    assert jobs.started.wait(TIMEOUT_S)
    window.cleanup()
    assert not window.is_running()
    assert window.plot_model() is None
    window.cleanup()
