"""Native QTest → real analysis child → saved physics results (#15522)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QAbstractSpinBox, QLineEdit  # noqa: E402
from pytestqt.qtbot import QtBot  # noqa: E402

from src.shared.python.physics.rust_kernel import is_rust_available  # noqa: E402
from src.tools.shot_pattern_analysis.core import PATTERN_NAMES  # noqa: E402
from src.tools.shot_pattern_analysis.gui import MainWidget  # noqa: E402
from src.tools.shot_pattern_analysis.presets import get_club_preset  # noqa: E402

pytestmark = pytest.mark.unit

# Match the real flight simulator's dependency boundary; never mock its kernel.
requires_kernel = pytest.mark.skipif(
    not is_rust_available(), reason="Current upstream-physics Rust kernel required"
)


def _type(widget: QLineEdit | QAbstractSpinBox, text: str) -> None:
    widget.setFocus()
    QTest.keyClick(widget, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(widget, text)
    QTest.keyClick(widget, Qt.Key.Key_Tab)


@pytest.fixture
def window(qtbot: QtBot, tmp_path: Path) -> Iterator[MainWidget]:
    widget = MainWidget()
    qtbot.addWidget(widget)
    widget.resize(1280, 900)
    widget.show()
    _type(widget.output_edit, str(tmp_path / "results"))
    _type(widget.shot_count, "2")
    _type(widget.seed, "15522")
    try:
        yield widget
    finally:
        assert widget.cleanup(), "Analysis workers must stop before Qt teardown"
        widget.close()


def _run(window: MainWidget, qtbot: QtBot) -> None:
    QTest.mouseClick(window.run_button, Qt.MouseButton.LeftButton)
    assert not window.run_button.isEnabled()
    qtbot.waitUntil(
        lambda: window.run_button.isEnabled() and not window.action_bar.is_running,
        timeout=45_000,
    )


def _assert_results(window: MainWidget, output: Path) -> None:
    assert window.status_label.text().startswith("Analysis complete."), (
        window.status_label.text()
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["n_shots"] == 2
    assert summary["config"]["seed"] == 15522
    assert window.summary_table.rowCount() == len(PATTERN_NAMES)
    for row, name in enumerate(PATTERN_NAMES):
        assert window.summary_table.item(row, 0).text() == name
        carry = summary["patterns"][name]["mean_carry_m"]
        assert carry > 0
        assert window.summary_table.item(row, 5).text() == f"{carry:.1f} m"
    for filename, label in window.figure_labels.items():
        assert (output / filename).is_file()
        assert not label.pixmap().isNull()
        assert not label.property("sourcePixmap").isNull()


@requires_kernel
def test_launch_select_club_and_run_real_physics(
    window: MainWidget, qtbot: QtBot, tmp_path: Path
) -> None:
    assert window.isVisible()
    window.club_preset.setFocus()
    QTest.keyClick(window.club_preset, Qt.Key.Key_Down)
    preset = get_club_preset("seven_iron")
    assert window.club_preset.currentData() == preset.id
    assert window.loft.value() == preset.loft_deg
    assert window.club_speed.value() == preset.club_speed_mps
    _run(window, qtbot)
    _assert_results(window, tmp_path / "results")
    summary = json.loads((tmp_path / "results/summary.json").read_text())
    assert summary["config"]["loft_deg"] == preset.loft_deg
    assert summary["config"]["club_id"] == preset.id


@requires_kernel
def test_failed_rerun_clears_stale_results_and_can_recover(
    window: MainWidget, qtbot: QtBot, tmp_path: Path
) -> None:
    _run(window, qtbot)
    _assert_results(window, tmp_path / "results")
    invalid_output = tmp_path / "not-a-directory"
    invalid_output.write_text("An existing file is not a results folder.")
    _type(window.output_edit, str(invalid_output))
    _run(window, qtbot)
    assert window.status_label.text().startswith("Analysis failed:")
    assert window.summary_table.rowCount() == 0
    for label in window.figure_labels.values():
        assert label.property("sourcePixmap") is None
        assert label.pixmap().isNull()
    assert "Strokes Gained" not in window.scoring_label.text()
    _type(window.output_edit, str(tmp_path / "recovered"))
    _run(window, qtbot)
    _assert_results(window, tmp_path / "recovered")


def test_empty_output_is_rejected_without_starting_worker(
    window: MainWidget,
) -> None:
    _type(window.output_edit, " ")
    QTest.mouseClick(window.run_button, Qt.MouseButton.LeftButton)
    assert window.status_label.text() == "Choose an output folder first."
    assert window.run_button.isEnabled()
    assert not window.action_bar.is_running
    assert window.summary_table.rowCount() == 0
