"""Regression tests for grip contact data export handling."""

from __future__ import annotations

import builtins
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.grip_modelling_tab import (
    GripModellingTab,
)


@pytest.mark.unit
def test_export_contact_data_shows_error_dialog_when_write_fails() -> None:
    """Write failures should use the existing export error dialog path."""
    exporter = SimpleNamespace(
        timesteps=[object()],
        export_to_dict=lambda: {"contacts": []},
    )
    tab = SimpleNamespace(contact_exporter=exporter)
    write_error = OSError("disk is read-only")

    with (
        patch(
            "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.grip_modelling_tab.QtWidgets.QFileDialog.getSaveFileName",
            return_value=("contacts.json", ""),
        ),
        patch.object(builtins, "open", side_effect=write_error),
        patch(
            "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.grip_modelling_tab.QtWidgets.QMessageBox.critical"
        ) as critical,
        patch(
            "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.grip_modelling_tab.QtWidgets.QMessageBox.information"
        ) as information,
    ):
        GripModellingTab._export_contact_data(tab)  # type: ignore[arg-type]

    critical.assert_called_once_with(
        tab,
        "Export Failed",
        "Failed to export: disk is read-only",
    )
    information.assert_not_called()
