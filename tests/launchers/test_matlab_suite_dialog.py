"""Tests for ``src.launchers.matlab_suite_dialog``.

The dialog renders one button per known MATLAB / Simscape model and
delegates launching back to the parent launcher's ``_launch_matlab_app``
method.  Because the dialog touches QPushButton.styleSheet but is
otherwise pure Qt construction, we instantiate a real dialog and inspect
the resulting widget tree.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QPushButton, QWidget


class _StubParent(QWidget):
    """Minimal real QWidget so QDialog accepts it as a parent."""

    def __init__(self) -> None:
        super().__init__()
        self._launch_matlab_app = MagicMock()


from src.launchers import matlab_suite_dialog


def test_mock_model_config_round_trips_dict() -> None:
    data = matlab_suite_dialog.MATLAB_MODELS[0]
    cfg = matlab_suite_dialog.MockModelConfig(data)
    assert cfg.id == data["id"]
    assert cfg.name == data["name"]
    assert cfg.description == data["description"]
    assert cfg.type == data["type"]
    assert cfg.path == data["path"]
    assert cfg.source_root is None
    assert cfg.provider is None
    assert cfg.working_dir is None


def test_dialog_creates_button_per_model(qapp) -> None:
    parent = _StubParent()
    dlg = matlab_suite_dialog.MatlabSuiteDialog(parent)
    buttons = dlg.findChildren(QPushButton)
    # one button per model + a Close button
    assert len(buttons) == len(matlab_suite_dialog.MATLAB_MODELS) + 1
    assert dlg.windowTitle() == "Matlab Simscape Models"
    dlg.deleteLater()


def test_dialog_launch_model_invokes_parent(qapp) -> None:
    parent = _StubParent()
    dlg = matlab_suite_dialog.MatlabSuiteDialog(parent)
    target = matlab_suite_dialog.MATLAB_MODELS[0]
    dlg.launch_model(target)
    parent._launch_matlab_app.assert_called_once()
    forwarded = parent._launch_matlab_app.call_args.args[0]
    assert forwarded.id == target["id"]
    dlg.deleteLater()


def test_dialog_buttons_route_through_launch_model(qapp) -> None:
    parent = _StubParent()
    dlg = matlab_suite_dialog.MatlabSuiteDialog(parent)
    # Find a button whose text contains the first model's name and click it.
    target = matlab_suite_dialog.MATLAB_MODELS[0]
    for btn in dlg.findChildren(QPushButton):
        if target["name"] in btn.text():
            btn.click()
            break
    parent._launch_matlab_app.assert_called_once()
    dlg.deleteLater()
