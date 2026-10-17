"""Offscreen tests of the plate declarations dialog and its tile flow (#13087).

Nothing here opens a window: the modal ``exec`` and file dialogs are stubbed.
"""

from __future__ import annotations

import os

import pytest

from tests.unit.biomechanics.test_ground_reaction_plate_import_form import _texts
from tests.unit.biomechanics.test_ground_reaction_plate_import_widget import (
    _model_series,
)
from tests.unit.force_plates import _open_fp_writer as w

pytestmark = pytest.mark.unit

_APP = None


def _qt():
    """Return a real QApplication, or skip when PyQt6 is absent or a stand-in."""
    global _APP
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication, QWidget
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")
    if getattr(QWidget, "__module__", "") != "PyQt6.QtWidgets":
        pytest.skip("PyQt6 is a test stand-in in this environment")
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _dialog(texts=None):
    _qt()
    from src.tools.ground_reaction_plots.plate_import_dialog import PlateImportDialog

    return PlateImportDialog(texts=texts)


def test_physical_facts_start_empty_and_accept_is_refused() -> None:
    dialog = _dialog()
    assert dialog.texts()["capacity_n"] == ""
    assert dialog.texts()["handedness"] == "RH"
    dialog.accept()
    assert dialog.result() != dialog.DialogCode.Accepted
    assert dialog.context() is None
    assert "uncertainty_source" in dialog.error_label.text()
    assert not dialog.error_label.isHidden()


def test_complete_form_is_accepted_with_its_context() -> None:
    dialog = _dialog(_texts())
    assert dialog.texts() == _texts()
    dialog.accept()
    assert dialog.result() == dialog.DialogCode.Accepted
    context = dialog.context()
    assert context is not None and context.declared.capacity_n == 3923.0
    assert dialog.error_label.isHidden()


def test_a_refused_value_names_its_field_and_keeps_the_dialog_open() -> None:
    dialog = _dialog(_texts(capacity_n="-1"))
    dialog.accept()
    assert dialog.result() != dialog.DialogCode.Accepted
    assert "capacity_n" in dialog.error_label.text()


def test_unknown_choice_in_prefill_is_refused() -> None:
    with pytest.raises(ValueError, match="alignment_kind"):
        _dialog(_texts(alignment_kind="cross_correlation"))


class _FakeDialog:
    """Stands in for the modal dialog; ``answer`` is the texts or ``None``."""

    answer: dict[str, str] | None = None
    seen_texts: list[object] = []

    def __init__(self, parent=None, texts=None) -> None:
        type(self).seen_texts.append(texts)

    def ask(self):
        from src.tools.ground_reaction_plots.plate_import_form import (
            context_from_form,
        )

        answer = type(self).answer
        return None if answer is None else context_from_form(answer)

    def texts(self) -> dict[str, str]:
        return dict(type(self).answer or {})


def _widget(monkeypatch, trial_path, answer):
    _qt()
    from src.tools.ground_reaction_plots import gui

    _FakeDialog.answer = answer
    _FakeDialog.seen_texts = []
    monkeypatch.setattr(gui, "PlateImportDialog", _FakeDialog)
    opened: list[str] = []

    def fake_open(*_args, **_kwargs):
        opened.append("file")
        return str(trial_path), ""

    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName", fake_open)
    widget = gui.GroundReactionPlotWidget()
    widget.set_series(_model_series())
    return widget, opened


def test_import_asks_for_declarations_then_overlays(tmp_path, monkeypatch) -> None:
    widget, opened = _widget(monkeypatch, w.write_swing(tmp_path), _texts())
    widget.plate_button.click()
    assert opened == ["file"]
    assert not widget.banner_label.isHidden()
    widget.plate_button.click()
    assert _FakeDialog.seen_texts == [None, _texts()], "second ask is prefilled"
    widget.cleanup()


def test_cancelled_declarations_open_no_file(tmp_path, monkeypatch) -> None:
    widget, opened = _widget(monkeypatch, w.write_swing(tmp_path), None)
    widget.plate_button.click()
    assert opened == []
    assert widget.banner_label.isHidden()
    widget.cleanup()


def test_host_context_skips_the_dialog(tmp_path, monkeypatch) -> None:
    from src.tools.ground_reaction_plots.plate_import_form import context_from_form

    widget, opened = _widget(monkeypatch, w.write_swing(tmp_path), None)
    widget.set_plate_context(context_from_form(_texts()))
    widget.plate_button.click()
    assert opened == ["file"]
    assert _FakeDialog.seen_texts == []
    widget.cleanup()
