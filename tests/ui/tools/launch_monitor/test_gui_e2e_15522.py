"""Native workbench flows with real import, analysis and worker lifecycle."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from threading import Event
from typing import Any

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402

from src.tools.async_action import WorkerContext  # noqa: E402
from src.tools.launch_monitor_analytics.gui import (  # noqa: E402
    LaunchMonitorAnalyticsWindow,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
# PyQt6's installed stub incorrectly types the QTest namespace as an instance.
_mouse_click: Any = QTest.mouseClick
FIXTURES = Path(__file__).parents[3] / "fixtures" / "launch_monitor"


@pytest.fixture
def window(qtbot: Any) -> Iterator[LaunchMonitorAnalyticsWindow]:
    result = LaunchMonitorAnalyticsWindow()
    qtbot.addWidget(result)
    result.show()
    yield result
    result.main_widget.cleanup()
    result.close()


def _select_tab(window: LaunchMonitorAnalyticsWindow, label: str) -> None:
    tabs = window.main_widget.tabs
    bar = tabs.tabBar()
    index = next(i for i in range(tabs.count()) if tabs.tabText(i) == label)
    _mouse_click(bar, Qt.MouseButton.LeftButton, pos=bar.tabRect(index).center())
    assert tabs.currentIndex() == index


def test_relationship_click_through_and_reopen(window: Any, qtbot: Any) -> None:
    panel = window.main_widget
    panel.import_file(FIXTURES / "trackman.csv")
    panel.import_file(FIXTURES / "garmin.csv")
    assert panel.session_tree.topLevelItemCount() == 2
    assert panel.data_table.rowCount() == 4
    _select_tab(window, "Relationships")
    for index in range(panel.relationship_metrics.count()):
        item = panel.relationship_metrics.item(index)
        if item.text() in {"club_speed", "ball_speed", "carry_distance"}:
            panel.relationship_metrics.scrollToItem(item)
            rect = panel.relationship_metrics.visualItemRect(item)
            _mouse_click(
                panel.relationship_metrics.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.ControlModifier,
                rect.center(),
            )
    _mouse_click(panel.run_relationship_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: panel.relationship_table.rowCount() == 3, timeout=15000)
    assert panel.relationship_table.rowCount() == 3
    assert len(panel.relationship_plot.axes.images) == 1
    assert "caus" in panel.scientific_boundary.text().lower()
    window.close()
    assert not window.isVisible()
    window.show()
    assert panel.data_table.rowCount() == 4


def test_model_click_fits_real_held_out_shots(
    window: Any, qtbot: Any, tmp_path: Path
) -> None:
    # Synthetic software-test data; no experimental accuracy claim.
    source = tmp_path / "shots.csv"
    header = (FIXTURES / "trackman.csv").read_text().splitlines()[0]
    rows = [
        f"{i},2026-01-02,10:00:00,7 Iron,{80 + i / 5},{110 + i / 3},"
        f"-3,2,1,-1,20,16,1,6000,-2,{160 + i / 2},{166 + i / 2},3,90"
        for i in range(40)
    ]
    source.write_text("\n".join([header, *rows]), encoding="utf-8")
    panel = window.main_widget
    panel.import_file(source)
    _select_tab(window, "Models")
    panel.model_target.setCurrentText("carry_distance")
    panel.model_group.setCurrentText("(random split)")
    for i in range(panel.model_features.count()):
        item = panel.model_features.item(i)
        item.setSelected(item.text() == "ball_speed")
    _mouse_click(panel.run_model_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: "Model complete" in panel.status_label.text(), timeout=15000
    )
    report = json.loads(panel.model_report.toPlainText())
    assert report["features"] == ["ball_speed"]
    assert report["train_count"] + report["test_count"] == 40
    assert report["test_count"] > 0
    assert report["metrics"]["r2"] == pytest.approx(1.0)
    assert len(panel.model_plot.axes.collections) == 1


def test_standalone_close_cancels_active_worker(window: Any, qtbot: Any) -> None:
    started, stopped = Event(), Event()

    def cooperative_job(ctx: WorkerContext) -> None:
        # Gate a real QThread deterministically; no wall-clock race with a fit.
        started.set()
        try:
            while not stopped.wait(0.005):
                ctx.raise_if_cancelled()
        finally:
            stopped.set()

    bar = window.main_widget.action_bar
    bar.start("Lifecycle probe", cooperative_job, on_finished=lambda _: None)
    qtbot.waitUntil(started.is_set)
    window.close()
    assert not window.isVisible()
    assert stopped.wait(1), "Standalone close must cancel its active analysis"
    assert not bar.is_running
