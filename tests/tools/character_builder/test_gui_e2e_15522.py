"""Native headless input through the Character Builder launch/export surface."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
pytestmark = pytest.mark.unit


@pytest.fixture
def builder():
    """Show the production Character Builder and dispose it on teardown."""
    try:
        from PyQt6.QtCore import QCoreApplication, QEvent
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")

    from src.tools.character_builder._embed_adapter import CharacterBuilderAdapter

    app = QApplication.instance() or QApplication([])
    adapter = CharacterBuilderAdapter()
    widget = adapter.create_main_widget()
    widget.resize(1280, 1000)
    widget.show()
    app.processEvents()
    try:
        yield widget
    finally:
        widget.close()
        adapter.cleanup()
        adapter.cleanup()
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def _choose(combo, index: int) -> None:
    """Move a combo to an index with keyboard events."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    assert 0 <= index < combo.count()
    combo.setFocus()
    QTest.keyClick(combo, Qt.Key.Key_Home)
    for _ in range(index):
        QTest.keyClick(combo, Qt.Key.Key_Down)
    assert combo.currentIndex() == index


def test_custom_selection_preserves_physics_and_clears_preset(builder) -> None:
    """The displayed preset and summary agree after a native Custom selection."""
    _choose(builder.preset_combo, builder.preset_combo.findData("junior"))
    parameters = builder.model.parameters_dict()
    spec_hash = builder.model.compile().sha256
    assert "Preset: Junior" in builder.summary.toPlainText()

    _choose(builder.preset_combo, 0)

    assert builder.model.preset is None
    assert builder.summary.toPlainText().startswith("Preset: Custom\n")
    assert builder.model.parameters_dict() == parameters
    assert builder.model.compile().sha256 == spec_hash


@pytest.mark.parametrize("fmt", ["spec", "urdf"])
def test_edit_and_export_from_real_buttons(builder, monkeypatch, tmp_path, fmt) -> None:
    """Native edits reach the compiler, file exporter and grip-binding sidecar."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QFileDialog

    from src.shared.python.humanoid_character_builder import spec_export

    _choose(builder.preset_combo, builder.preset_combo.findData("junior"))
    old_hash = builder.model.compile().sha256
    spin = builder.spins["mass_kg"]
    spin.setFocus()
    QTest.keyClick(spin, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(spin, "55.0")
    QTest.keyClick(spin, Qt.Key.Key_Tab)
    assert builder.model.parameters.mass_kg == 55.0
    assert builder.preset_combo.currentIndex() == 0
    compiled = builder.model.compile()
    assert compiled.sha256 != old_hash
    assert compiled.sha256 in builder.summary.toPlainText()

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: str(tmp_path))
    QTest.mouseClick(builder.export_buttons[fmt], Qt.MouseButton.LeftButton)

    expected, _media, filename = spec_export.export_character(compiled, fmt)
    output = tmp_path / filename
    assert output.read_text(encoding="utf-8") == expected
    assert builder.status.text() == f"Wrote {output}"
    sidecar = tmp_path / f"{output.stem}.grip_binding.json"
    assert (
        json.loads(sidecar.read_text("utf-8")) == builder.model.grip_binding_manifest()
    )


def test_cancel_export_does_not_write_or_report_success(builder, monkeypatch, tmp_path):
    """A cancelled folder picker writes nothing and reports no success."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QFileDialog

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: "")
    QTest.mouseClick(builder.export_buttons["spec"], Qt.MouseButton.LeftButton)
    assert builder.status.text() == ""
    assert list(Path(tmp_path).iterdir()) == []


def test_export_error_can_be_retried(builder, monkeypatch, tmp_path) -> None:
    """Real filesystem rejection is visible, and the next click can recover."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QFileDialog

    invalid = tmp_path / "not-a-directory"
    invalid.write_text("keep this file", encoding="utf-8")
    choices = iter([str(invalid), str(tmp_path)])
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: next(choices))
    button = builder.export_buttons["spec"]
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert builder.status.text().startswith("Export failed:")
    assert invalid.read_text("utf-8") == "keep this file"
    assert list(tmp_path.iterdir()) == [invalid]

    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert builder.status.text().startswith("Wrote ")
    assert len(list(tmp_path.glob("*.json"))) == 2
