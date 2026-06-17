"""Import contracts for the shared dashboard package."""

from __future__ import annotations

import importlib
import sys

import pytest

pytestmark = pytest.mark.unit


def _drop_modules(*module_names: str) -> None:
    for module_name in module_names:
        sys.modules.pop(module_name, None)


def test_dashboard_package_import_is_qt_free() -> None:
    """API services import dashboard recorder without the GUI stack."""
    _drop_modules(
        "src.shared.python.dashboard",
        "src.shared.python.dashboard.launcher",
        "src.shared.python.dashboard.window",
    )

    dashboard = importlib.import_module("src.shared.python.dashboard")
    recorder = importlib.import_module("src.shared.python.dashboard.recorder")

    assert dashboard.__all__ == ["UnifiedDashboardWindow", "launch_dashboard"]
    assert hasattr(recorder, "GenericPhysicsRecorder")
    assert "src.shared.python.dashboard.window" not in sys.modules
    assert "src.shared.python.dashboard.launcher" not in sys.modules


def test_launcher_manifest_import_is_qt_free() -> None:
    """API launcher routes load the manifest without importing GUI dialogs."""
    _drop_modules(
        "src.config.launcher_manifest_loader",
        "src.launchers",
        "src.launchers.about_dialog",
    )

    manifest_loader = importlib.import_module("src.config.launcher_manifest_loader")

    assert hasattr(manifest_loader, "LauncherManifest")
    assert "src.launchers.about_dialog" not in sys.modules
