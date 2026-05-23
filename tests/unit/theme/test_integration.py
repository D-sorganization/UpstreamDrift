"""Headless unit coverage for theme integration helpers."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest


class FakeSignal:
    def __init__(self) -> None:
        self._callbacks: list[Any] = []

    def connect(self, callback: Any) -> None:
        self._callbacks.append(callback)

    def emit(self, *args: Any) -> None:
        for callback in list(self._callbacks):
            callback(*args)


class FakeAction:
    def __init__(self, text: str, parent: Any) -> None:
        self.text = text
        self.parent = parent
        self.triggered = FakeSignal()
        self.checkable = False
        self.checked = False
        self._data: Any = None

    def setCheckable(self, value: bool) -> None:
        self.checkable = value

    def setChecked(self, value: bool) -> None:
        self.checked = value

    def setData(self, value: Any) -> None:
        self._data = value

    def data(self) -> Any:
        return self._data


class FakeActionGroup:
    def __init__(self, parent: Any) -> None:
        self.parent = parent
        self.exclusive = False
        self._actions: list[FakeAction] = []

    def setExclusive(self, value: bool) -> None:
        self.exclusive = value

    def addAction(self, action: FakeAction) -> None:
        self._actions.append(action)

    def actions(self) -> list[FakeAction]:
        return self._actions


class FakeMenu:
    def __init__(self, title: str = "", parent: Any = None) -> None:
        self.title = title
        self.parent = parent
        self.actions: list[Any] = []
        self.menus: list[FakeMenu] = []
        self.separators = 0

    def addAction(self, action: FakeAction) -> None:
        self.actions.append(action)

    def addMenu(self, menu: FakeMenu) -> None:
        self.menus.append(menu)

    def addSeparator(self) -> None:
        self.separators += 1


class FakeMenuBar(FakeMenu):
    pass


class FakeThemeManager:
    next_manager: FakeThemeManager
    instance_calls: list[dict[str, Any]] = []

    def __init__(
        self,
        *,
        current_theme: str = "Dark",
        available_themes: tuple[str, ...] = ("Light", "Dark", "High Contrast"),
    ) -> None:
        self.current_theme = current_theme
        self.available_themes = list(available_themes)
        self.apply_calls = 0
        self.changed_themes: list[str] = []
        self.themeChanged = FakeSignal()

    @classmethod
    def instance(
        cls,
        *,
        main_window: Any = None,
        settings_org: str,
        settings_app: str,
    ) -> FakeThemeManager:
        cls.instance_calls.append(
            {
                "main_window": main_window,
                "settings_org": settings_org,
                "settings_app": settings_app,
            }
        )
        return cls.next_manager

    def apply_theme(self) -> None:
        self.apply_calls += 1

    def change_theme(self, theme_name: str) -> None:
        self.changed_themes.append(theme_name)
        self.current_theme = theme_name
        self.themeChanged.emit(theme_name)

    def get_current_theme_name(self) -> str:
        return self.current_theme

    def get_available_themes(self) -> list[str]:
        return self.available_themes


class FakeDialog:
    instances: list[FakeDialog] = []

    def __init__(self, manager: FakeThemeManager, window: Any) -> None:
        self.manager = manager
        self.window = window
        self.exec_calls = 0
        self.__class__.instances.append(self)

    def exec(self) -> None:
        self.exec_calls += 1


class FakeWindow:
    def __init__(self, menubar: FakeMenuBar | None = None) -> None:
        self._menubar = menubar

    def menuBar(self) -> FakeMenuBar | None:
        return self._menubar


class FakeThemedWindow(FakeWindow):
    pass


@pytest.fixture()
def integration_module(monkeypatch: pytest.MonkeyPatch):
    FakeThemeManager.instance_calls = []
    FakeThemeManager.next_manager = FakeThemeManager()

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QApplication = type("QApplication", (), {})
    qt_widgets.QMenu = FakeMenu
    qt_widgets.QMenuBar = FakeMenuBar

    qt_gui = types.ModuleType("PyQt6.QtGui")
    qt_gui.QAction = FakeAction
    qt_gui.QActionGroup = FakeActionGroup

    pyqt = types.ModuleType("PyQt6")
    pyqt.QtWidgets = qt_widgets
    pyqt.QtGui = qt_gui

    theme_manager = types.ModuleType("src.shared.python.theme.theme_manager")
    theme_manager.ThemeManager = FakeThemeManager

    dialogs = types.ModuleType("src.shared.python.theme.dialogs")
    dialogs.CustomThemeEditor = type("CustomThemeEditor", (FakeDialog,), {})
    dialogs.ThemeManagerDialog = type("ThemeManagerDialog", (FakeDialog,), {})
    dialogs.CustomThemeEditor.instances = []
    dialogs.ThemeManagerDialog.instances = []

    theme_package = types.ModuleType("src.shared.python.theme")
    theme_package.__path__ = [
        str(Path(__file__).resolve().parents[3] / "src" / "shared" / "python" / "theme")
    ]

    monkeypatch.setitem(sys.modules, "PyQt6", pyqt)
    monkeypatch.setitem(sys.modules, "PyQt6.QtWidgets", qt_widgets)
    monkeypatch.setitem(sys.modules, "PyQt6.QtGui", qt_gui)
    monkeypatch.setitem(sys.modules, "src.shared.python.theme", theme_package)
    monkeypatch.setitem(
        sys.modules, "src.shared.python.theme.theme_manager", theme_manager
    )
    monkeypatch.setitem(sys.modules, "src.shared.python.theme.dialogs", dialogs)
    monkeypatch.delitem(
        sys.modules, "src.shared.python.theme.integration", raising=False
    )

    return importlib.import_module("src.shared.python.theme.integration")


def test_module_imports_with_stubbed_pyqt_without_qapplication(
    integration_module: types.ModuleType,
) -> None:
    assert integration_module.__all__ == [
        "ThemedWindowMixin",
        "apply_theme_to_window",
        "create_theme_menu",
        "get_theme_manager",
        "setup_themed_app",
    ]


def test_get_theme_manager_delegates_to_singleton_with_settings(
    integration_module: types.ModuleType,
) -> None:
    window = FakeWindow()

    manager = integration_module.get_theme_manager(
        window,
        settings_org="Org",
        settings_app="App",
    )

    assert manager is FakeThemeManager.next_manager
    assert FakeThemeManager.instance_calls == [
        {"main_window": window, "settings_org": "Org", "settings_app": "App"}
    ]


def test_get_theme_manager_rejects_missing_settings_org(
    integration_module: types.ModuleType,
) -> None:
    with pytest.raises(ValueError, match="settings_org must be provided"):
        integration_module.get_theme_manager(settings_org=None)


def test_apply_theme_to_window_uses_current_theme_when_name_omitted(
    integration_module: types.ModuleType,
) -> None:
    manager = FakeThemeManager.next_manager

    integration_module.apply_theme_to_window(FakeWindow())

    assert manager.apply_calls == 1
    assert manager.changed_themes == []


def test_apply_theme_to_window_changes_explicit_theme(
    integration_module: types.ModuleType,
) -> None:
    manager = FakeThemeManager.next_manager

    integration_module.apply_theme_to_window(FakeWindow(), "High Contrast")

    assert manager.apply_calls == 0
    assert manager.changed_themes == ["High Contrast"]


def test_apply_theme_to_window_rejects_missing_window(
    integration_module: types.ModuleType,
) -> None:
    with pytest.raises(ValueError, match="window must be provided"):
        integration_module.apply_theme_to_window(None)


def test_create_theme_menu_builds_checked_actions_and_updates_on_signal(
    integration_module: types.ModuleType,
) -> None:
    menubar = FakeMenuBar()
    manager = FakeThemeManager.next_manager

    menu = integration_module.create_theme_menu(FakeWindow(), menubar)

    assert menu.title == "&Theme"
    assert menubar.menus == [menu]
    assert [action.text for action in menu.actions] == [
        "Light",
        "Dark",
        "High Contrast",
    ]
    assert [action.checked for action in menu.actions] == [False, True, False]

    menu.actions[0].triggered.emit(True)

    assert manager.changed_themes == ["Light"]
    assert [action.checked for action in menu.actions] == [True, False, False]


def test_create_theme_menu_can_attach_to_parent_menu(
    integration_module: types.ModuleType,
) -> None:
    parent_menu = FakeMenu("Parent")

    menu = integration_module.create_theme_menu(FakeWindow(), parent_menu)

    assert parent_menu.menus == [menu]


def test_create_theme_menu_adds_custom_dialog_actions(
    integration_module: types.ModuleType,
) -> None:
    manager = FakeThemeManager.next_manager

    menu = integration_module.create_theme_menu(
        FakeWindow(),
        show_custom_options=True,
    )
    create_action, manage_action = menu.actions[-2:]

    create_action.triggered.emit()
    manage_action.triggered.emit()

    assert menu.separators == 1
    assert create_action.text == "Create Custom Theme..."
    assert manage_action.text == "Manage Themes..."
    assert (
        sys.modules["src.shared.python.theme.dialogs"]
        .CustomThemeEditor.instances[0]
        .manager
        is manager
    )
    assert (
        sys.modules["src.shared.python.theme.dialogs"]
        .ThemeManagerDialog.instances[0]
        .exec_calls
        == 1
    )


def test_create_theme_menu_rejects_missing_window(
    integration_module: types.ModuleType,
) -> None:
    with pytest.raises(ValueError, match="window must be provided"):
        integration_module.create_theme_menu(None)


def test_setup_themed_app_applies_theme_and_adds_menu_with_class_default_app_name(
    integration_module: types.ModuleType,
) -> None:
    manager = FakeThemeManager.next_manager
    window = FakeThemedWindow(FakeMenuBar())

    integration_module.setup_themed_app(object(), window, settings_org="Org")

    assert manager.apply_calls == 1
    assert window.menuBar().menus[0].title == "&Theme"  # type: ignore[union-attr]
    assert FakeThemeManager.instance_calls[0] == {
        "main_window": window,
        "settings_org": "Org",
        "settings_app": "FakeThemedWindow",
    }


def test_setup_themed_app_can_skip_menu_and_use_explicit_app_name(
    integration_module: types.ModuleType,
) -> None:
    manager = FakeThemeManager.next_manager
    menubar = FakeMenuBar()
    window = FakeWindow(menubar)

    integration_module.setup_themed_app(
        object(),
        window,
        add_menu=False,
        settings_app="ExplicitApp",
    )

    assert manager.apply_calls == 1
    assert menubar.menus == []
    assert FakeThemeManager.instance_calls[-1]["settings_app"] == "ExplicitApp"


def test_setup_themed_app_tolerates_window_without_menubar(
    integration_module: types.ModuleType,
) -> None:
    manager = FakeThemeManager.next_manager
    window = FakeWindow(menubar=None)

    integration_module.setup_themed_app(object(), window)

    assert manager.apply_calls == 1
    assert FakeThemeManager.instance_calls == [
        {
            "main_window": window,
            "settings_org": "D-sorganization",
            "settings_app": "FakeWindow",
        }
    ]


def test_setup_themed_app_rejects_missing_app(
    integration_module: types.ModuleType,
) -> None:
    with pytest.raises(ValueError, match="app must be provided"):
        integration_module.setup_themed_app(None, FakeWindow())


def test_themed_window_mixin_setup_and_change_helpers(
    integration_module: types.ModuleType,
) -> None:
    class Window(integration_module.ThemedWindowMixin, FakeWindow):
        def __init__(self) -> None:
            FakeWindow.__init__(self, FakeMenuBar())
            self.seen_themes: list[str] = []

        def _on_theme_changed(self, theme_name: str) -> None:
            self.seen_themes.append(theme_name)

    manager = FakeThemeManager.next_manager
    window = Window()

    window.setup_theme_support(settings_org="Org", settings_app="MixinApp")
    window.change_theme("Light")

    assert window.get_theme_manager() is manager
    assert window.get_current_theme() == "Light"
    assert window.seen_themes == ["Light"]
    assert window.menuBar().menus[0].title == "&Theme"  # type: ignore[union-attr]
    assert FakeThemeManager.instance_calls[0] == {
        "main_window": window,
        "settings_org": "Org",
        "settings_app": "MixinApp",
    }


def test_themed_window_mixin_uses_class_name_and_handles_missing_menubar(
    integration_module: types.ModuleType,
) -> None:
    class Window(integration_module.ThemedWindowMixin, FakeWindow):
        def __init__(self) -> None:
            FakeWindow.__init__(self, menubar=None)

    manager = FakeThemeManager.next_manager
    window = Window()

    window.setup_theme_support()

    assert manager.apply_calls == 1
    assert FakeThemeManager.instance_calls == [
        {
            "main_window": window,
            "settings_org": "D-sorganization",
            "settings_app": "Window",
        }
    ]


def test_themed_window_mixin_fallbacks_and_validation(
    integration_module: types.ModuleType,
) -> None:
    window = integration_module.ThemedWindowMixin()

    assert window.get_theme_manager() is None
    assert window.get_current_theme() == "Light"
    window.change_theme("Dark")
    with pytest.raises(ValueError, match="add_menu must be provided"):
        window.setup_theme_support(add_menu=None)


def test_dialog_helpers_reject_missing_window(
    integration_module: types.ModuleType,
) -> None:
    with pytest.raises(ValueError, match="window must be provided"):
        integration_module._open_custom_theme_editor(
            FakeThemeManager.next_manager, None
        )

    with pytest.raises(ValueError, match="window must be provided"):
        integration_module._open_theme_manager_dialog(
            FakeThemeManager.next_manager, None
        )
