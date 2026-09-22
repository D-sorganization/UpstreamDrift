"""One QSettings namespace, window geometry and single config root (#8907).

All QSettings stores are redirected to INI files under ``tmp_path`` so the
real user profile (registry / ~/.config) is never read or written.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtCore import QRect, QSettings  # noqa: E402
from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

from src.launchers import launcher_settings_store as store  # noqa: E402
from src.shared.python.data_io.user_config_root import user_config_dir  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def ini_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Redirect every launcher QSettings pair to a temp INI file."""

    def _open(organization: str, application: str) -> QSettings:
        path = tmp_path / f"{organization}__{application}.ini"
        return QSettings(str(path), QSettings.Format.IniFormat)

    monkeypatch.setattr(store, "_open_settings", _open)
    return _open


@pytest.fixture
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


# ── one QSettings namespace ─────────────────────────────────────────────────


def test_canonical_pair_is_the_dominant_launcher_pair() -> None:
    assert (store.SETTINGS_ORG, store.SETTINGS_APP) == ("UpstreamDrift", "Launcher")
    assert ("D-sorganization", "UpstreamDrift") in store.LEGACY_SETTINGS_PAIRS


def test_legacy_key_is_read_when_canonical_missing(ini_settings: Any) -> None:
    ini_settings("D-sorganization", "UpstreamDrift").setValue("font_family", "Fira")

    assert store.launcher_settings().value("font_family") == "Fira"


def test_canonical_value_wins_over_legacy(ini_settings: Any) -> None:
    ini_settings("D-sorganization", "UpstreamDrift").setValue("font_family", "Old")
    ini_settings("UpstreamDrift", "Launcher").setValue("font_family", "New")

    assert store.launcher_settings().value("font_family") == "New"
    # The legacy store is never modified.
    legacy = ini_settings("D-sorganization", "UpstreamDrift")
    assert legacy.value("font_family") == "Old"


def test_settings_scope_aliases_before_returning_pair(ini_settings: Any) -> None:
    ini_settings("D-sorganization", "UpstreamDrift").setValue("font_family", "Fira")

    org, app_name = store.launcher_settings_scope()

    assert ini_settings(org, app_name).value("font_family") == "Fira"


# ── window geometry ────────────────────────────────────────────────────────


def test_persist_window_geometry_preconditions(app: QApplication) -> None:
    with pytest.raises(ValueError):
        store.persist_window_geometry(QWidget(), "")
    with pytest.raises(ValueError):
        store.persist_window_geometry(object(), "x")


def _settings_dialog() -> QWidget:
    from src.launchers.settings_dialog import SettingsDialog

    return SettingsDialog(parent=None, launcher=QWidget())


def _cross_engine_dashboard() -> QWidget:
    from src.launchers import cross_engine_dashboard as ced

    return ced._build_qt_window()


def _integrations_health_window() -> QWidget:
    from src.launchers.integrations_health_window import (
        open_integrations_health_window,
    )

    return open_integrations_health_window()


# The offscreen screen is 800x800 and restoreGeometry() clamps to it, so each
# target fits on screen, differs from the window's hard-coded default size
# and respects its minimum size (the dashboard's minimum width is 900).
@pytest.mark.parametrize(
    ("factory", "target"),
    [
        (_settings_dialog, QRect(30, 40, 700, 560)),
        (_cross_engine_dashboard, QRect(0, 30, 900, 740)),
        (_integrations_health_window, QRect(50, 60, 610, 470)),
    ],
    ids=["SettingsDialog", "CrossEngineDashboard", "IntegrationsHealthWindow"],
)
def test_window_geometry_round_trips(
    factory: Any, target: QRect, app: QApplication, ini_settings: Any
) -> None:
    first = factory()
    first.show()
    first.setGeometry(target)
    app.processEvents()
    expected_size = first.geometry().size()
    first.close()
    app.processEvents()

    second = factory()
    second.show()
    app.processEvents()
    try:
        assert second.geometry().size() == expected_size
        # Width may be widened by the window's style-dependent minimum size;
        # the height proves the saved (non-default) geometry was restored.
        assert second.geometry().height() == target.height()
    finally:
        second.close()


# ── one config root for every launcher writer ──────────────────────────────


def test_launcher_writers_use_the_single_config_root() -> None:
    from src.launchers import launcher_constants, launcher_diagnostics
    from src.launchers.launcher_process_manager import ProcessManager
    from src.launchers.library_widget import DB_PATH
    from src.launchers.onboarding_dialog import ONBOARDING_CONFIG_PATH
    from src.shared.python.ui.preferences_dialog import PREFS_FILE
    from src.shared.python.ui.recent_models import RECENT_FILE

    root = user_config_dir()
    assert root == launcher_constants.CONFIG_DIR
    assert launcher_constants.user_config_dir() == root
    assert launcher_diagnostics.LAYOUT_CONFIG_FILE == (
        launcher_constants.LAYOUT_CONFIG_FILE
    )
    for path in (
        PREFS_FILE,
        RECENT_FILE,
        DB_PATH,
        ONBOARDING_CONFIG_PATH,
        ProcessManager.get_log_path(),
    ):
        assert root in path.parents, path


def test_diagnostics_tab_reads_log_from_new_root(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.launchers._settings_auxiliary_tabs import SettingsAuxiliaryTabsMixin

    seen: list[Path] = []

    def _record(log_path: Path, *_args: object) -> bool:
        seen.append(log_path)
        return False

    monkeypatch.setattr(
        SettingsAuxiliaryTabsMixin, "_load_log_file", staticmethod(_record)
    )
    tab = SettingsAuxiliaryTabsMixin()
    tab._log_viewer = tab._proc_log_viewer = _NullViewer()
    tab._load_app_log()
    tab._load_process_log()

    root = user_config_dir()
    assert root / "launcher.log" in seen
    assert root / "process_output.log" in seen
    assert not any(".golf_modeling_suite" in p.parts for p in seen)


class _NullViewer:
    def setPlainText(self, _text: str) -> None:  # noqa: N802
        return None
