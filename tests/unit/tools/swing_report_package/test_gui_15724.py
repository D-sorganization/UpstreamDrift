"""Swing report package window (#15724, SRP-5), offscreen; skipped when PyQt6
cannot load (AGENTS.md E). The pure logic is tested in test_gui_model_15724."""

from __future__ import annotations

import pytest

from src.tools.swing_report_package.catalog import load_catalog

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

CATALOG = load_catalog()


@pytest.fixture
def window():
    try:
        from PyQt6.QtWidgets import QApplication

        from src.tools.swing_report_package.gui import SwingReportWindow
    except (ImportError, OSError) as exc:  # pragma: no cover - environment
        pytest.skip(f"PyQt6 not loadable: {exc}")
    app = QApplication.instance() or QApplication([])
    win = SwingReportWindow()
    yield win
    win.close()
    app.processEvents()


def test_window_ticks_follow_the_package_and_preview_plans(window) -> None:
    assert window.items.count() == len(CATALOG.items)
    assert set(window.choice().checked) == set(CATALOG.default_package)
    window.package.setCurrentText("midhands")
    assert set(window.choice().checked) == set(CATALOG.packages["midhands"])
    window.preview()
    assert "midhands_grid@face_on" in window.log.toPlainText()
    window.run()
    assert window.process is None and "bundle" in window.log.toPlainText()


def test_window_run_streams_the_command_line_output(window, tmp_path) -> None:
    window.bundle.setText(str(tmp_path / "missing.npz"))
    window.out.setText(str(tmp_path / "out"))
    window.run()
    assert window.process is not None and not window.buttons["run"].isEnabled()
    assert window.process.waitForFinished(60_000)
    text = window.log.toPlainText()
    assert "no bundle file" in text and "exit status 2" in text
    assert window.process is None and window.buttons["run"].isEnabled()
