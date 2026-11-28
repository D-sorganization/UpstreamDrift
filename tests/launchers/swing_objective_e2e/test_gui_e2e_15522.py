"""Headless launcher-to-optimizer flows; no substituted physics or workers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, Qt
from PyQt6.QtTest import QSignalSpy, QTest
from PyQt6.QtWidgets import QApplication

from src.launchers.adapters.swing_objective_lab_embed import (
    _SwingObjectiveLabEmbedAdapter,
)
from src.launchers.tools_repo_path import resolve_tools_repo

pytestmark = pytest.mark.unit
# PyQt6 stubs model namespace methods as instance methods.
_qtest = cast(Any, QTest)


@pytest.fixture(params=[(1280, 800), (390, 844)], ids=["desktop", "phone"])
def lab(qapp: QApplication, request: Any, monkeypatch: Any) -> Iterator[Any]:
    """Use the launcher's provider resolver and dispose every completed worker."""
    root = Path(__file__).resolve().parents[3]
    try:
        provider = resolve_tools_repo(root, None)
    except RuntimeError:
        provider = None
    if provider is None or not provider.pinned:
        # unit-test-gate does not materialize vendor/ud-tools (see #9361).
        pytest.skip(
            "requires the pinned vendored Tools checkout (vendor/ud-tools); "
            "not materialized in this CI job"
        )
    monkeypatch.syspath_prepend(str(provider.path / "src/pendulum_simulator/src"))
    adapter = _SwingObjectiveLabEmbedAdapter()
    widget = adapter.create_main_widget(None)
    widget.resize(*request.param)
    widget.show()
    _qtest.qWait(1)
    assert widget.isVisible()
    assert not adapter.is_dirty()
    try:
        yield widget
    finally:
        worker = widget._worker
        if worker is not None:
            assert worker.wait(60_000), "comparison worker did not terminate"
        widget.close()
        adapter.cleanup()
        adapter.cleanup()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert sip.isdeleted(widget)


def _select_first(lab: Any, count: int) -> None:
    """Choose objectives through viewport mouse events, including deselection."""
    selector = lab.objective_list
    for index in range(selector.count()):
        item = selector.item(index)
        if item.isSelected() != (index < count):
            selector.scrollToItem(item)
            _qtest.mouseClick(
                selector.viewport(),
                Qt.MouseButton.LeftButton,
                pos=selector.visualItemRect(item).center(),
            )
    assert len(lab.selected_objective_keys()) == count


def _enter(spin: Any, value: str) -> None:
    """Commit a numeric input using native keyboard events."""
    spin.setFocus()
    spin.selectAll()
    _qtest.keyClicks(spin, value)
    _qtest.keyClick(spin, Qt.Key.Key_Return)


def test_reject_single_objective_then_recover(lab: Any) -> None:
    """Invalid selection starts no worker; a corrected selection can run."""
    _select_first(lab, 1)
    _qtest.mouseClick(lab.run_button, Qt.MouseButton.LeftButton)
    assert lab.status_text() == "Select at least two objectives to compare."
    assert lab._worker is None
    assert lab.run_button.isEnabled()
    assert not lab.progress.isVisible()
    assert lab.result_table.rowCount() == 0
    _select_first(lab, 2)
    assert lab.selected_objective_keys() == ("clubhead_speed", "centrifugal")


def test_compare_real_downswings_and_render_results(lab: Any) -> None:
    """Run the real QThread optimizer and check both rendered result tables."""
    _select_first(lab, 2)
    _enter(lab.duration_spin, "0.30")
    _enter(lab.hub_torque_spin, "300")
    _enter(lab.wrist_torque_spin, "25")
    _enter(lab.node_spin, "9")
    assert lab.duration_spin.value() == pytest.approx(0.30)
    assert lab.hub_torque_spin.value() == 300
    assert lab.wrist_torque_spin.value() == 25
    assert lab.node_spin.value() == 9
    _qtest.mouseClick(lab.run_button, Qt.MouseButton.LeftButton)
    worker = lab._worker
    assert worker is not None, "Run comparison must dispatch the real worker"
    completed = QSignalSpy(worker.finished)
    failed = QSignalSpy(worker.failed)
    assert not lab.run_button.isEnabled()
    assert lab.progress.isVisible()
    assert completed.wait(45_000), (
        f"comparison did not finish: {len(failed)} failure signals"
    )
    assert worker.wait(5_000)
    QCoreApplication.processEvents()
    assert len(failed) == 0
    comparison = completed[0][0]
    assert comparison.objective_keys == ("clubhead_speed", "centrifugal")
    assert np.isfinite(comparison.matrix).all()
    for result in comparison.results.values():
        assert result.states.shape[0] == 9
        assert np.isfinite(result.states).all()
        assert np.isfinite(result.signals.clubhead_speed).all()
    assert lab.run_button.isEnabled()
    assert not lab.progress.isVisible()
    assert lab.result_table.rowCount() == 2
    assert lab.matrix_table.rowCount() == lab.matrix_table.columnCount() == 2
    for row, key in enumerate(comparison.objective_keys):
        speed = comparison.results[key].signals.clubhead_speed[-1]
        assert lab.result_table.item(row, 3).text() == f"{speed:.2f}"
        for column in range(2):
            expected = f"{comparison.matrix[row, column]:.1f}%"
            assert lab.matrix_table.item(row, column).text() == expected
    assert "Solving" not in lab.status_text()
    assert "failed" not in lab.status_text().lower()
