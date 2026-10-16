"""Settings-opening menu actions must not raise (issue #12992).

``SettingsWidget`` read ``launcher.layout_manager.view_mode`` (a name that
does not exist), so File > Preferences, Tools > Environment Manager,
Tools > Diagnostics and Window > Open MCP Servers Settings were all dead.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("PyQt6")

from src.launchers import upstream_drift_launcher  # noqa: E402
from src.launchers.launcher_constants import ViewMode  # noqa: E402


@pytest.fixture
def qapp_or_skip():
    """Return a QApplication, creating one if the suite has not already."""
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


_DEAD_ACTIONS = (
    "Preferences...",
    "Environment Manager...",
    "Diagnostics...",
    "Open MCP Servers Settings",
)


def _walk(menu, out):
    for action in menu.actions():
        submenu = action.menu()
        if submenu is not None:
            _walk(submenu, out)
        else:
            out.append(action)
    return out


@pytest.fixture
def launcher(qapp_or_skip):
    win = upstream_drift_launcher.UpstreamDriftLauncher()
    yield win
    pm = getattr(win, "process_manager", None)
    cleanup = getattr(pm, "cleanup_all", None)
    if callable(cleanup):
        cleanup()
    win.deleteLater()


@pytest.mark.unit
def test_layout_view_state_is_exposed_without_internals(launcher) -> None:
    """The owner answers the view-state query (Law of Demeter)."""
    state = launcher.get_layout_view_state()
    assert state is not None
    mode, scale = state
    assert isinstance(mode, ViewMode)
    assert scale > 0


@pytest.mark.unit
@pytest.mark.parametrize("label", _DEAD_ACTIONS)
def test_settings_menu_action_opens_settings_tab(launcher, label, monkeypatch) -> None:
    """Triggering the QAction raises nothing and creates the Settings tab."""
    errors: list[BaseException] = []
    monkeypatch.setattr(sys, "excepthook", lambda t, v, tb: errors.append(v))

    by_text = {a.text().replace("&", ""): a for a in _walk(launcher.menu_bar, [])}
    action = by_text.get(label)
    assert action is not None, f"{label!r} missing; have {sorted(by_text)}"
    action.trigger()

    assert not errors, errors
    tabs = launcher.workspace_tabs
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert "Settings" in titles


@pytest.mark.unit
@pytest.mark.parametrize("tab", range(5))
def test_open_settings_every_tab_index(launcher, tab) -> None:
    """``_open_settings(tab=n)`` succeeds for each sub-tab."""
    launcher._open_settings(tab=tab)
    tabs = launcher.workspace_tabs
    assert "Settings" in [tabs.tabText(i) for i in range(tabs.count())]
