"""AI Assistant Settings Dialog (slim coordinator).

This module historically held ~1100 lines mixing the ``AISettings``
dataclass, keyring helpers, a switch-by-enum ``ProviderConfigWidget``,
and a 3-tab dialog. Tools issue #2762 split those concerns:

* :class:`AISettings` lives in :mod:`src.shared.python.ai._settings_model`.
* Keyring helpers live in :mod:`src.shared.python.ai.gui._api_keys` and
  delegate where possible to ``chat.credentials.CredentialManager``.
* Per-provider widgets are individual classes in
  :mod:`src.shared.python.ai.gui._provider_config_widgets` and looked up
  through :class:`ProviderConfigRegistry`.
* Each tab is its own widget in ``_general_tab.py`` / ``_providers_tab.py``
  / ``_rag_tab.py``.

The module re-exports the previous public names so downstream callers
(UpstreamDrift, Gasification_Model, ``assistant_panel`` here) keep
working unchanged.
"""

from __future__ import annotations

import contextlib
from typing import Any

from PyQt6.QtCore import (  # noqa: F401  QSettings re-exported for monkeypatch in tests
    QSettings,
    pyqtSignal,
)
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.ai._settings_model import AISettings
from src.shared.python.ai.access_policy import coerce_access_mode
from src.shared.python.ai.config import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    KEY_ACCESS_MODE,
    KEY_AUTO_INDEX_ON_OPEN,
    KEY_CHAT_MODE,
    KEY_EXPERTISE,
    KEY_MODEL,
    KEY_OLLAMA_HOST,
    KEY_PROVIDER,
    KEY_RAG_ENABLED,
    KEY_RESPONSE_STYLE,
    KEY_STREAMING,
    SETTINGS_APP,
    SETTINGS_ORG,
    get_ollama_host,
)
from src.shared.python.ai.gui._api_keys import (
    delete_api_key,
    get_api_key,
    set_api_key,
)
from src.shared.python.ai.gui._general_tab import GeneralPreferencesTab
from src.shared.python.ai.gui._provider_config_registry import ProviderConfigRegistry
from src.shared.python.ai.gui._provider_config_widgets import (
    AnthropicConfigWidget,
    BitnetConfigWidget,
    ClaudeCliConfigWidget,
    ClineConfigWidget,
    CodexCliConfigWidget,
    GeminiConfigWidget,
    OllamaConfigWidget,
    OpenAIConfigWidget,
)
from src.shared.python.ai.gui._provider_registry_data import (
    BITNET_ROOT_ENV,
    DEFAULT_CLINE_HOST,
    PROVIDER_INFO,
    AIProvider,
    populate_model_combo,
    populate_provider_combo,
    provider_default_model,
    provider_display_name,
    provider_model_names,
)
from src.shared.python.ai.gui._providers_tab import ProvidersTab
from src.shared.python.ai.gui._rag_tab import RagTab
from src.shared.python.ai.mcp.gui import McpServersTab
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Backward-compatibility shim.
# Legacy code instantiated ``ProviderConfigWidget(AIProvider.OLLAMA)``. We
# keep that constructor signature working by routing to the registry.
# ---------------------------------------------------------------------------
def ProviderConfigWidget(  # noqa: N802 - keep historical class-shaped name
    provider: AIProvider,
    parent: QWidget | None = None,
) -> QWidget:
    """Return the registered config widget for ``provider``.

    Compatibility wrapper for code (and tests) that still call this as
    ``ProviderConfigWidget(provider)``. New code should use
    :meth:`ProviderConfigRegistry.get_widget` directly.
    """
    if provider is None:
        raise ValueError("provider must be provided")
    return ProviderConfigRegistry.get_widget(provider, parent)


class AISettingsDialog(QDialog):
    """Settings dialog for AI Assistant configuration."""

    settings_changed = pyqtSignal(AISettings)
    rebuild_index_requested = pyqtSignal()

    _DARK_STYLESHEET = """
        QDialog, QWidget {
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        QTabWidget::pane {
            border: 1px solid #3c3c3c;
            background-color: #1e1e1e;
        }
        QTabBar::tab {
            background-color: #2d2d2d;
            color: #e0e0e0;
            padding: 8px 16px;
            border: 1px solid #3c3c3c;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #1e1e1e;
            border-bottom: 2px solid #FF8800;
            font-weight: bold;
        }
        QGroupBox {
            border: 1px solid #3c3c3c;
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 10px;
            font-weight: bold;
            color: #FF8800;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 4px;
            left: 8px;
        }
        QLabel { color: #e0e0e0; }
        QLineEdit, QComboBox {
            background-color: #252526;
            color: #e0e0e0;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            padding: 4px;
        }
        QLineEdit:focus, QComboBox:focus { border: 1px solid #FF8800; }
        QPushButton {
            background-color: #0e639c;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
        }
        QPushButton:hover { background-color: #1177bb; }
        QCheckBox { color: #e0e0e0; }
        QDialogButtonBox QPushButton { min-width: 60px; }
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI Assistant Settings")
        self.setMinimumSize(500, 400)
        self.setStyleSheet(self._DARK_STYLESHEET)

        self._settings = AISettings.load()
        self._build_ui()
        self._load_settings_into_ui()

    # ---- construction --------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        self._providers_tab = ProvidersTab()
        self._providers_tab.provider_combo.currentIndexChanged.connect(
            self._on_provider_changed
        )
        tabs.addTab(self._providers_tab, "Provider")

        self._general_tab = GeneralPreferencesTab()
        tabs.addTab(self._general_tab, "Preferences")

        self._rag_tab = RagTab()
        self._rag_tab.rebuild_index_requested.connect(self.rebuild_index_requested)
        tabs.addTab(self._rag_tab, "Knowledge Base")

        self._mcp_tab = McpServersTab()
        tabs.addTab(self._mcp_tab, "MCP Servers")

        layout.addWidget(tabs)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    # ---- settings <-> UI sync -----------------------------------------

    def _load_settings_into_ui(self) -> None:
        # Provider combo
        provider_combo = self._providers_tab.provider_combo
        for i in range(provider_combo.count()):
            if provider_combo.itemData(i) == self._settings.provider:
                provider_combo.setCurrentIndex(i)
                break

        self._on_provider_changed(provider_combo.currentIndex())

        model_combo = self._providers_tab.model_combo
        for i in range(model_combo.count()):
            if model_combo.itemText(i) == self._settings.model:
                model_combo.setCurrentIndex(i)
                break

        self._general_tab.select_response_style(self._settings.response_style)
        self._general_tab.streaming_check.setChecked(self._settings.streaming_enabled)

        self._rag_tab.rag_enabled_check.setChecked(self._settings.rag_enabled)
        self._rag_tab.auto_index_check.setChecked(self._settings.auto_index_on_open)
        am_combo = self._rag_tab.access_mode_combo
        for i in range(am_combo.count()):
            if am_combo.itemData(i) == self._settings.access_mode:
                am_combo.setCurrentIndex(i)
                break

    def _on_provider_changed(self, index: int) -> None:
        if index is None:
            raise ValueError("index must be provided")
        provider_combo = self._providers_tab.provider_combo
        provider_data = provider_combo.itemData(index)
        if provider_data is None or not isinstance(provider_data, AIProvider):
            return
        provider: AIProvider = provider_data

        populate_model_combo(
            self._providers_tab.model_combo, provider, self._settings.model
        )

        for p, widget in self._providers_tab.provider_configs.items():
            widget.setVisible(p == provider)

        if provider == AIProvider.OLLAMA:
            ollama_widget = self._providers_tab.provider_configs.get(AIProvider.OLLAMA)
            if ollama_widget is not None and hasattr(ollama_widget, "models_refreshed"):
                with contextlib.suppress(TypeError):
                    ollama_widget.models_refreshed.disconnect(
                        self._update_ollama_models
                    )
                ollama_widget.models_refreshed.connect(self._update_ollama_models)

    def _accept(self) -> None:
        self._settings.provider = self._providers_tab.provider_combo.currentData()
        self._settings.model = self._providers_tab.model_combo.currentText()

        style = self._general_tab.current_response_style()
        self._settings.response_style = style
        self._settings.expertise_level = (
            GeneralPreferencesTab.STYLE_TO_LEGACY_LEVEL.get(
                style, self._settings.expertise_level
            )
        )
        self._settings.streaming_enabled = self._general_tab.streaming_check.isChecked()
        self._settings.rag_enabled = self._rag_tab.rag_enabled_check.isChecked()
        self._settings.auto_index_on_open = self._rag_tab.auto_index_check.isChecked()
        self._settings.access_mode = coerce_access_mode(
            self._rag_tab.access_mode_combo.currentData()
        )

        if self._settings.provider == AIProvider.OLLAMA:
            ollama_widget = self._providers_tab.provider_configs.get(AIProvider.OLLAMA)
            if ollama_widget is not None and hasattr(ollama_widget, "get_host"):
                self._settings.ollama_host = ollama_widget.get_host()

        self._settings.save()
        self.settings_changed.emit(self._settings)
        self.accept()

    # ---- public helpers preserved from the legacy implementation -------

    def get_settings(self) -> AISettings:
        return self._settings

    def _update_ollama_models(self, models: list[str]) -> None:
        if not models:
            return
        model_combo = self._providers_tab.model_combo
        current = model_combo.currentText()
        model_combo.clear()
        for model in models:
            model_combo.addItem(model)
        idx = model_combo.findText(current)
        if idx >= 0:
            model_combo.setCurrentIndex(idx)
        elif models:
            model_combo.setCurrentIndex(0)

    def bind_to_chat_dock(self, chat_dock: Any) -> None:
        """Wire ``chat_dock.models_refreshed`` to repopulate the model combo.

        See Tools #2547 / PR #2566 for the signal contract.
        """
        signal = getattr(chat_dock, "models_refreshed", None)
        if signal is None or not hasattr(signal, "connect"):
            return
        with contextlib.suppress(TypeError):
            signal.disconnect(self._on_chat_models_refreshed)
        signal.connect(self._on_chat_models_refreshed)

    def _on_chat_models_refreshed(self, models: list[Any]) -> None:
        names: list[str] = []
        for entry in models:
            if isinstance(entry, str):
                if entry:
                    names.append(entry)
            elif isinstance(entry, dict):
                name = entry.get("name")
                if isinstance(name, str) and name:
                    names.append(name)
        self._update_ollama_models(names)


# Public re-exports kept stable for downstream importers.
__all__ = [
    # constants
    "BITNET_ROOT_ENV",
    "DEFAULT_CLINE_HOST",
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_OLLAMA_MODEL",
    "KEY_ACCESS_MODE",
    "KEY_AUTO_INDEX_ON_OPEN",
    "KEY_CHAT_MODE",
    "KEY_EXPERTISE",
    "KEY_MODEL",
    "KEY_OLLAMA_HOST",
    "KEY_PROVIDER",
    "KEY_RAG_ENABLED",
    "KEY_RESPONSE_STYLE",
    "KEY_STREAMING",
    "PROVIDER_INFO",
    "SETTINGS_APP",
    "SETTINGS_ORG",
    # core types
    "AIProvider",
    "AISettings",
    "AISettingsDialog",
    # widgets
    "AnthropicConfigWidget",
    "BitnetConfigWidget",
    "ClaudeCliConfigWidget",
    "ClineConfigWidget",
    "CodexCliConfigWidget",
    "GeminiConfigWidget",
    "OllamaConfigWidget",
    "OpenAIConfigWidget",
    "ProviderConfigRegistry",
    "ProviderConfigWidget",
    # keyring helpers
    "delete_api_key",
    "get_api_key",
    "get_ollama_host",
    "set_api_key",
    # combo helpers
    "populate_model_combo",
    "populate_provider_combo",
    "provider_default_model",
    "provider_display_name",
    "provider_model_names",
]
