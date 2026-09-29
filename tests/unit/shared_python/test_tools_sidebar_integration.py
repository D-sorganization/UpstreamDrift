"""Tests for optional Unified Tools Sidebar host integration."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import src.shared.python.gui_launcher.tools_sidebar_integration as _tsi_module
from src.shared.python.gui_launcher.tools_sidebar_integration import (
    install_tools_sidebar,
    is_tools_sidebar_available,
)

# ── Helpers ───────────────────────────────────────────────────────────

#: All module names that _import_sidebar_module() tries in order.
_ALL_CANDIDATES = (
    "sidekick.ui.tools_sidebar",
    "sidekick.ui.tools_sidebar",
    "shared.python.sidekick.ui.tools_sidebar",
)

# The fully-qualified name of the private loader function we need to patch.
_IMPORT_FN = (
    "src.shared.python.gui_launcher.tools_sidebar_integration._import_sidebar_module"
)


class FakeSignal:
    def __init__(self) -> None:
        self.slots: list[Any] = []

    def connect(self, slot: Any) -> None:
        self.slots.append(slot)

    def emit(self, value: Any) -> None:
        for slot in self.slots:
            slot(value)


class FakeDock:
    def __init__(self) -> None:
        self.file_open_requested = FakeSignal()
        self.widget_value = None

    def setWidget(self, widget: Any) -> None:
        self.widget_value = widget

    def toggleViewAction(self) -> object:
        return object()


class FakeMainWindow:
    def __init__(self) -> None:
        self.docks: list[tuple[Any, Any]] = []
        self.opened_paths: list[Any] = []

    def addDockWidget(self, area: Any, dock: Any) -> None:
        self.docks.append((area, dock))

    def open_file(self, path: Any) -> None:
        self.opened_paths.append(path)


def test_install_tools_sidebar_returns_not_installed_with_fallback_disabled(
    monkeypatch: Any,
) -> None:
    """With ``install_fallback=False``, status is not-installed when no module found."""
    monkeypatch.setattr(_tsi_module, "_import_sidebar_module", lambda: None)

    status = install_tools_sidebar(FakeMainWindow(), install_fallback=False)

    assert status.installed is False
    assert "not available" in status.reason


def test_install_tools_sidebar_installs_null_fallback_when_module_missing(
    monkeypatch: Any,
) -> None:
    """With ``install_fallback=True`` (default), a NullToolsSidebar is docked."""
    from src.shared.python.gui_launcher.tools_sidebar_integration import (
        NullToolsSidebar,
    )

    monkeypatch.setattr(_tsi_module, "_import_sidebar_module", lambda: None)
    window = FakeMainWindow()
    status = install_tools_sidebar(window)

    assert status.installed is True
    assert "null sidebar fallback" in status.reason
    assert isinstance(status.sidebar, NullToolsSidebar)
    # dock may be sidebar itself if Qt is unavailable in test context
    assert status.dock is not None


def test_install_tools_sidebar_adds_shared_dock_and_connects_file_open(
    monkeypatch: Any, tmp_path: Path
) -> None:
    module = ModuleType("sidekick.ui.tools_sidebar")
    created: dict[str, Any] = {}

    class ToolsSidebar(FakeDock):
        def __init__(
            self,
            *,
            parent: Any,
            project_root: Path,
            context_provider: Any,
        ) -> None:
            super().__init__()
            created["parent"] = parent
            created["project_root"] = project_root
            created["context_provider"] = context_provider

    module.ToolsSidebar = ToolsSidebar  # type: ignore[attr-defined]
    monkeypatch.setattr(_tsi_module, "_import_sidebar_module", lambda: module)

    window = FakeMainWindow()
    status = install_tools_sidebar(
        window,
        project_root=tmp_path,
        context_provider=lambda: {"host": "test"},
    )

    assert status.installed is True
    assert status.file_open_connected is True
    assert status.module_name == module.__name__
    assert len(window.docks) == 1
    assert created["parent"] is window
    assert created["project_root"] == tmp_path.resolve()

    status.sidebar.file_open_requested.emit("model.urdf")

    assert window.opened_paths == ["model.urdf"]


def test_install_tools_sidebar_passes_sidekick_tokens_when_supported(
    monkeypatch: Any,
) -> None:
    module = ModuleType("sidekick.ui.tools_sidebar")
    created: dict[str, Any] = {}

    class ToolsSidebar(FakeDock):
        def __init__(
            self,
            *,
            parent: Any,
            sidekick_tokens: dict[str, str],
        ) -> None:
            super().__init__()
            created["parent"] = parent
            created["sidekick_tokens"] = sidekick_tokens

    module.ToolsSidebar = ToolsSidebar  # type: ignore[attr-defined]
    monkeypatch.setattr(_tsi_module, "_import_sidebar_module", lambda: module)

    window = FakeMainWindow()
    status = install_tools_sidebar(window)

    assert status.installed is True
    assert created["parent"] is window
    assert created["sidekick_tokens"]["sidekick.color.surface"]
    assert created["sidekick_tokens"]["sidekick.radius.chat"] == "8px"


def test_install_tools_sidebar_uses_shared_installer_status(monkeypatch: Any) -> None:
    module = ModuleType("sidekick.ui.tools_sidebar")
    dock = FakeDock()

    def shared_installer(
        main_window: Any,
        *,
        project_root: Path | None,
        context_provider: Any,
    ) -> FakeDock:
        main_window.addDockWidget("right", dock)
        return dock

    module.install_tools_sidebar = shared_installer  # type: ignore[attr-defined]
    monkeypatch.setattr(_tsi_module, "_import_sidebar_module", lambda: module)

    window = FakeMainWindow()
    status = install_tools_sidebar(window)

    assert status.installed is True
    assert status.reason == "installed by shared module"
    assert status.dock is dock
    assert status.file_open_connected is True


def test_install_tools_sidebar_shared_installer_can_accept_sidekick_tokens(
    monkeypatch: Any,
) -> None:
    module = ModuleType("sidekick.ui.tools_sidebar")
    observed: dict[str, Any] = {}
    dock = FakeDock()

    def shared_installer(
        main_window: Any,
        *,
        sidekick_tokens: dict[str, str],
    ) -> FakeDock:
        observed["tokens"] = sidekick_tokens
        main_window.addDockWidget("right", dock)
        return dock

    module.install_tools_sidebar = shared_installer  # type: ignore[attr-defined]
    monkeypatch.setattr(_tsi_module, "_import_sidebar_module", lambda: module)

    status = install_tools_sidebar(FakeMainWindow())

    assert status.installed is True
    assert observed["tokens"]["sidekick.color.canvas"]


def test_install_tools_sidebar_rejects_non_dock_hosts(monkeypatch: Any) -> None:
    module = ModuleType("sidekick.ui.tools_sidebar")
    module.ToolsSidebar = FakeDock  # type: ignore[attr-defined]
    monkeypatch.setattr(_tsi_module, "_import_sidebar_module", lambda: module)

    status = install_tools_sidebar(object())

    assert status.installed is False
    assert "dock widgets" in status.reason


def test_install_tools_sidebar_passes_sidekick_tokens_via_factory() -> None:
    """Success-path pin: tokens reach a ``create_tools_sidebar`` factory.

    Uses ``patch.dict`` so the fake sidebar module is auto-cleaned after the
    test (per CLAUDE.md's test-pollution guidance).
    """
    module_name = "sidekick.ui.tools_sidebar"
    module = ModuleType(module_name)
    recorded: dict[str, Any] = {}

    def create_tools_sidebar(
        *,
        parent: Any,
        sidekick_tokens: dict[str, str],
        project_root: Path | None = None,
        context_provider: Any = None,
    ) -> FakeDock:
        recorded["parent"] = parent
        recorded["project_root"] = project_root
        recorded["context_provider"] = context_provider
        recorded["sidekick_tokens"] = sidekick_tokens
        return FakeDock()

    module.create_tools_sidebar = create_tools_sidebar  # type: ignore[attr-defined]

    with patch.object(_tsi_module, "_import_sidebar_module", return_value=module):
        window = FakeMainWindow()
        status = install_tools_sidebar(window)

    assert status.installed is True
    assert status.module_name == module_name
    assert recorded["parent"] is window
    tokens = recorded["sidekick_tokens"]
    assert isinstance(tokens, dict)
    assert tokens, "Sidekick tokens must be a non-empty dict"
    assert "sidekick.color.canvas" in tokens
    assert tokens["sidekick.color.canvas"]


def test_is_tools_sidebar_available_true_when_stub_registered() -> None:
    module = ModuleType("sidekick.ui.tools_sidebar")

    with patch.object(_tsi_module, "_import_sidebar_module", return_value=module):
        assert is_tools_sidebar_available() is True


def test_is_tools_sidebar_available_false_when_no_module_present() -> None:
    with patch.object(_tsi_module, "_import_sidebar_module", return_value=None):
        assert is_tools_sidebar_available() is False


# ── NullToolsSidebar unit tests ───────────────────────────────────────


def test_null_tools_sidebar_stores_tokens() -> None:
    """NullToolsSidebar stores the sidekick_tokens mapping on construction."""
    from src.shared.python.gui_launcher.tools_sidebar_integration import (
        NullToolsSidebar,
    )

    tokens = {"sidekick.color.surface": "#1e1e2e", "sidekick.radius.chat": "8px"}
    sidebar = NullToolsSidebar(sidekick_tokens=tokens)
    assert sidebar.sidekick_tokens == tokens


def test_null_tools_sidebar_empty_tokens_when_none_passed() -> None:
    """NullToolsSidebar defaults to an empty token dict."""
    from src.shared.python.gui_launcher.tools_sidebar_integration import (
        NullToolsSidebar,
    )

    sidebar = NullToolsSidebar()
    assert sidebar.sidekick_tokens == {}


def test_null_tools_sidebar_setwidget_noop() -> None:
    """setWidget is accepted without error (protocol conformance)."""
    from src.shared.python.gui_launcher.tools_sidebar_integration import (
        NullToolsSidebar,
    )

    sidebar = NullToolsSidebar()
    sidebar.setWidget(object())  # must not raise


def test_null_tools_sidebar_toggle_view_action_returns_object() -> None:
    """toggleViewAction returns something (duck-typed dock-widget protocol)."""
    from src.shared.python.gui_launcher.tools_sidebar_integration import (
        NullToolsSidebar,
    )

    sidebar = NullToolsSidebar()
    action = sidebar.toggleViewAction()
    assert action is not None
