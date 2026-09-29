"""Architecture contracts for the launcher settings-dialog decomposition."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.launchers import settings_dialog
from src.launchers._settings_auxiliary_tabs import SettingsAuxiliaryTabsMixin
from src.launchers.settings_runtime import (
    RuntimeDependencyCheckFailure,
    RuntimeDependencyCheckWorker,
    RuntimeDependencyReport,
    WslScriptDialog,
    compare_version_strings,
)

pytestmark = pytest.mark.unit


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_PATH = _REPO_ROOT / "src" / "launchers" / "settings_dialog.py"


def test_settings_dialog_preserves_runtime_compatibility_exports() -> None:
    """Existing launcher imports keep resolving through the slim facade."""
    assert settings_dialog.RuntimeDependencyReport is RuntimeDependencyReport
    assert (
        settings_dialog.RuntimeDependencyCheckFailure is RuntimeDependencyCheckFailure
    )
    assert settings_dialog.RuntimeDependencyCheckWorker is RuntimeDependencyCheckWorker
    assert settings_dialog.WslScriptDialog is WslScriptDialog
    assert settings_dialog._compare_version_strings is compare_version_strings


def test_settings_widget_delegates_auxiliary_tabs_to_focused_mixin() -> None:
    """Diagnostics and process behavior have one implementation owner."""
    assert issubclass(settings_dialog.SettingsWidget, SettingsAuxiliaryTabsMixin)
    assert (
        settings_dialog.SettingsWidget._create_diagnostics_tab
        is SettingsAuxiliaryTabsMixin._create_diagnostics_tab
    )
    assert (
        settings_dialog.SettingsWidget.refresh_processes_ui
        is SettingsAuxiliaryTabsMixin.refresh_processes_ui
    )


def test_settings_dialog_is_under_budget_without_size_exceptions() -> None:
    """The decomposition retires both settings-dialog size exceptions."""
    assert len(_SETTINGS_PATH.read_text(encoding="utf-8").splitlines()) <= 1200

    for config_name in (
        "file_size_budget.json",
        "module_size_budget_baseline.json",
    ):
        config_path = _REPO_ROOT / "scripts" / "config" / config_name
        config = json.loads(config_path.read_text(encoding="utf-8"))
        paths = {entry["path"] for entry in config["exceptions"]}
        assert "src/launchers/settings_dialog.py" not in paths
