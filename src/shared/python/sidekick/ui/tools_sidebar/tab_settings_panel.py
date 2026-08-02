"""Qt helpers for Sidekick tab settings panels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .qt_compat import QtCore, QtWidgets
from .settings import SidebarTabSettingsStore

__all__ = [
    "ConfigureTabsDialog",
    "SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME",
    "TabSettingsMixin",
    "build_tab_settings_dialog",
    "build_tab_settings_toolbar",
]

if TYPE_CHECKING:
    from .state import SidebarState
    from .tab_definition import SidebarTabDefinition

SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME = "SidekickActiveTabSettings"


class TabSettingsMixin:
    """Methods that keep tab settings out of the main sidebar controller.

    Designed to be mixed into ``UnifiedToolsSidebar``, which supplies the
    annotations below as real instance attributes. Declaring them here at
    class level gives mypy enough information to resolve the chained method
    calls (``self._settings_store.settings_for(...)``) without resorting to
    blanket ``# type: ignore`` headers.
    """

    # Attributes supplied by the host class (UnifiedToolsSidebar)
    _settings_store: SidebarTabSettingsStore
    _settings_dialog: QtWidgets.QDialog | None
    _settings_button: QtWidgets.QToolButton
    _tab_definitions: dict[str, SidebarTabDefinition]
    _tab_ids: list[str]
    _state: SidebarState

    # Methods supplied by the host class
    def _emit_context(self) -> None:  # pragma: no cover - host provides
        raise NotImplementedError

    def active_tab_id(self) -> str:  # pragma: no cover - host provides
        raise NotImplementedError

    def tab_display_name(self, tab_id: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def register_settings_button(  # pragma: no cover - host provides
        self, button: QtWidgets.QToolButton
    ) -> None:
        raise NotImplementedError

    def _configure_tab_settings(self) -> None:
        self._settings_store = SidebarTabSettingsStore(
            cast(Any, self._tab_definitions.values()),
            self._state,
        )
        self._refresh_settings_button()

    def tab_settings(self, tab_id: str) -> dict[str, Any]:
        """Return materialized settings for one tab instance."""
        # Explicit local annotation: CI runs mypy with --follow-imports=skip,
        # so the imported SidebarTabSettingsStore.settings_for() return type
        # is seen as Any. The store IS declared as returning dict[str, Any]
        # in settings.py:91; this annotation documents that contract at the
        # call site so the boundary stays clear.
        result: dict[str, Any] = self._settings_store.settings_for(tab_id)
        return result

    def update_tab_settings(
        self,
        tab_id: str,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist settings for a known tab instance."""
        updated: dict[str, Any] = self._settings_store.update_settings(tab_id, values)
        self._state.tab_settings = self._settings_store.raw_settings()
        self._emit_context()
        return updated

    def open_active_tab_settings(self) -> bool:
        """Open the selected tab settings panel when the tab declares one."""
        tab_id = self.active_tab_id()
        definition = self._tab_definitions.get(tab_id)
        if definition is None or definition.settings is None:
            return False
        factory = definition.settings.widget_factory
        content = factory(self, tab_id) if factory is not None else None
        self._settings_dialog = build_tab_settings_dialog(self, tab_id, content)
        self._settings_dialog.exec()
        return True

    def open_configure_tabs(self) -> None:
        """Open the Configure Tabs dialog."""
        dialog = ConfigureTabsDialog(self)
        dialog.exec()

    def _refresh_settings_button(self, *_args: Any) -> None:
        if not hasattr(self, "_settings_button"):
            return
        if not getattr(self, "_tab_ids", []):
            self._settings_button.setEnabled(False)
            return
        definition = self._tab_definitions.get(self.active_tab_id())
        enabled = definition is not None and definition.settings is not None
        self._settings_button.setEnabled(enabled)

    def _tab_settings_payload(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, dict[str, Any]] = (
            self._settings_store.materialized_settings()
        )
        return payload


class ConfigureTabsDialog(QtWidgets.QDialog):
    """Dialog allowing the user to select which tabs are visible."""

    def __init__(self, sidebar: Any, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent or sidebar)
        self.setWindowTitle("Configure Tabs")
        self.setObjectName("SidekickConfigureTabsDialog")
        self.sidebar = sidebar
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("Select which tabs to display in the sidebar:", self)
        layout.addWidget(label)

        self.checkboxes = {}
        definitions = self.sidebar._tab_definitions
        visible_ids = self.sidebar.visible_tab_ids()

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget(scroll)
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)

        for tab_id, definition in definitions.items():
            cb = QtWidgets.QCheckBox(definition.title, scroll_content)
            cb.setChecked(tab_id in visible_ids)
            cb.toggled.connect(
                lambda checked, tid=tab_id, checkbox=cb: self._on_toggle(
                    tid, checked, checkbox
                )
            )
            scroll_layout.addWidget(cb)
            self.checkboxes[tab_id] = cb

        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        btn_box.accepted.connect(self.accept)
        layout.addWidget(btn_box)

    def _on_toggle(
        self, tab_id: str, checked: bool, checkbox: QtWidgets.QCheckBox
    ) -> None:
        success = self.sidebar.set_tab_visible(tab_id, checked)
        if not success:
            checkbox.blockSignals(True)
            checkbox.setChecked(not checked)
            checkbox.blockSignals(False)


def build_tab_settings_toolbar(sidebar: Any) -> QtWidgets.QToolBar:
    """Build the compact selected-tab settings action surface."""
    toolbar = QtWidgets.QToolBar(sidebar)
    toolbar.setObjectName("SidekickSettingsToolbar")
    toolbar.setIconSize(QtCore.QSize(16, 16))

    button = QtWidgets.QToolButton(toolbar)
    button.setObjectName(SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME)
    button.setText("⚙")
    button.setToolTip("Active tab settings")
    button.clicked.connect(sidebar.open_active_tab_settings)
    toolbar.addWidget(button)
    sidebar.register_settings_button(button)

    config_button = QtWidgets.QToolButton(toolbar)
    config_button.setObjectName("SidekickConfigureTabsButton")
    config_button.setText("☰")
    config_button.setToolTip("Configure visible tabs")
    config_button.clicked.connect(sidebar.open_configure_tabs)
    toolbar.addWidget(config_button)

    spacer = QtWidgets.QWidget()
    spacer.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred
    )
    toolbar.addWidget(spacer)

    close_btn = QtWidgets.QToolButton(toolbar)
    close_btn.setText("×")
    close_btn.setToolTip("Close Sidekick")
    close_btn.clicked.connect(lambda: sidebar.setVisible(False))
    toolbar.addWidget(close_btn)

    return toolbar


def build_tab_settings_dialog(
    sidebar: Any,
    tab_id: str,
    content: QtWidgets.QWidget | None,
) -> QtWidgets.QDialog:
    """Create a simple settings dialog for the active tab."""
    dialog = QtWidgets.QDialog(sidebar)
    dialog.setObjectName(f"SidekickTabSettingsDialog_{tab_id}")
    dialog.setWindowTitle(f"{sidebar.tab_display_name(tab_id)} Settings")
    layout = QtWidgets.QVBoxLayout(dialog)
    if content is None:
        payload = sidebar.tab_settings(tab_id)
        label = QtWidgets.QLabel(str(payload["values"]), dialog)
        label.setWordWrap(True)
        content = label
    layout.addWidget(content)
    return dialog
