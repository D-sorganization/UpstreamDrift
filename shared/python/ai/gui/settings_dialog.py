"""AI Assistant Settings Dialog.

Provides configuration for AI provider selection, API key management,
and user preferences. API keys are stored securely in the OS keyring.

Security:
    - API keys stored in Windows Credential Manager / macOS Keychain
    - Keys never logged or transmitted to developers
    - Local-only mode (Ollama) requires no keys
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from PyQt6.QtCore import QSettings, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    # Import removed to avoid circular dependency
    pass

logger = logging.getLogger(__name__)

# Settings keys
SETTINGS_ORG = "GolfModelingSuite"
SETTINGS_APP = "AIAssistant"
KEY_PROVIDER = "ai/provider"
KEY_MODEL = "ai/model"
KEY_EXPERTISE = "ai/expertise_level"
KEY_OLLAMA_HOST = "ai/ollama_host"
KEY_STREAMING = "ai/streaming_enabled"


class AIProvider(Enum):
    """Available AI providers."""

    OLLAMA = auto()  # Free, local
    OPENAI = auto()  # GPT-4
    ANTHROPIC = auto()  # Claude


# Provider display info - explicitly typed for mypy
PROVIDER_INFO: dict[AIProvider, dict[str, str | bool | list[str]]] = {
    AIProvider.OLLAMA: {
        "name": "Ollama (Local - FREE)",
        "description": "Run AI locally on your computer. No API key needed.",
        "requires_key": False,
        "default_model": "llama3.1:8b",
        "models": ["llama3.1:8b", "llama3.1:70b", "mistral", "codellama"],
    },
    AIProvider.OPENAI: {
        "name": "OpenAI (GPT-4)",
        "description": "Cloud-based GPT-4. Requires OpenAI API key.",
        "requires_key": True,
        "key_service": "golf_suite_openai_key",
        "default_model": "gpt-4-turbo-preview",
        "models": ["gpt-4-turbo-preview", "gpt-4", "gpt-4o", "gpt-3.5-turbo"],
    },
    AIProvider.ANTHROPIC: {
        "name": "Anthropic (Claude)",
        "description": "Cloud-based Claude 3. Requires Anthropic API key.",
        "requires_key": True,
        "key_service": "golf_suite_anthropic_key",
        "default_model": "claude-3-sonnet-20240229",
        "models": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
    },
}


@dataclass
class AISettings:
    """AI configuration settings.

    Attributes:
        provider: Selected AI provider.
        model: Model name for the provider.
        expertise_level: User's expertise level (1-4).
        ollama_host: Ollama server URL.
        streaming_enabled: Whether to stream responses.
        api_keys: In-memory API keys (not persisted directly).
    """

    provider: AIProvider = AIProvider.OLLAMA
    model: str = "llama3.1:8b"
    expertise_level: int = 1
    ollama_host: str = "http://localhost:11434"
    streaming_enabled: bool = True
    api_keys: dict[AIProvider, str] = field(default_factory=dict)

    def save(self) -> None:
        """Save settings to persistent storage."""
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        settings.setValue(KEY_PROVIDER, self.provider.name)
        settings.setValue(KEY_MODEL, self.model)
        settings.setValue(KEY_EXPERTISE, self.expertise_level)
        settings.setValue(KEY_OLLAMA_HOST, self.ollama_host)
        settings.setValue(KEY_STREAMING, self.streaming_enabled)
        # Note: API keys are stored separately in keyring
        logger.info("Saved AI settings: provider=%s", self.provider.name)

    @classmethod
    def load(cls) -> AISettings:
        """Load settings from persistent storage."""
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)

        provider_name = settings.value(KEY_PROVIDER, "OLLAMA")
        try:
            provider = AIProvider[provider_name]
        except KeyError:
            provider = AIProvider.OLLAMA

        return cls(
            provider=provider,
            model=settings.value(KEY_MODEL, "llama3.1:8b"),
            expertise_level=int(settings.value(KEY_EXPERTISE, 1)),
            ollama_host=settings.value(KEY_OLLAMA_HOST, "http://localhost:11434"),
            streaming_enabled=settings.value(KEY_STREAMING, True, type=bool),
        )


def get_api_key(provider: AIProvider) -> str | None:
    """Get API key from secure storage.

    Args:
        provider: Provider to get key for.

    Returns:
        API key if found, None otherwise.
    """
    info = PROVIDER_INFO.get(provider)
    if not info or not info.get("requires_key"):
        return None

    service_name = info.get("key_service", "")
    if not service_name or not isinstance(service_name, str):
        return None

    try:
        import keyring

        result = keyring.get_password(service_name, "api_key")
        return result if isinstance(result, str) else None
    except ImportError:
        logger.warning("keyring package not installed for secure key storage")
        return None
    except Exception as e:
        logger.warning("Failed to get API key from keyring: %s", e)
        return None


def set_api_key(provider: AIProvider, key: str) -> bool:
    """Store API key in secure storage.

    Args:
        provider: Provider to set key for.
        key: API key to store.

    Returns:
        True if successful, False otherwise.
    """
    info = PROVIDER_INFO.get(provider)
    if not info or not info.get("requires_key"):
        return False

    service_name = info.get("key_service", "")
    if not service_name:
        return False

    try:
        import keyring

        keyring.set_password(service_name, "api_key", key)
        logger.info("Stored API key for %s", provider.name)
        return True
    except ImportError:
        logger.warning("keyring package not installed for secure key storage")
        return False
    except Exception as e:
        logger.warning("Failed to store API key: %s", e)
        return False


def delete_api_key(provider: AIProvider) -> bool:
    """Delete API key from secure storage.

    Args:
        provider: Provider to delete key for.

    Returns:
        True if successful, False otherwise.
    """
    info = PROVIDER_INFO.get(provider)
    if not info or not info.get("requires_key"):
        return False

    service_name = info.get("key_service", "")
    if not service_name:
        return False

    try:
        import keyring

        keyring.delete_password(service_name, "api_key")
        logger.info("Deleted API key for %s", provider.name)
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.warning("Failed to delete API key: %s", e)
        return False


class ProviderConfigWidget(QWidget):
    """Widget for configuring a single AI provider."""

    key_changed = pyqtSignal(str)  # Emits new key value

    def __init__(
        self,
        provider: AIProvider,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize provider config widget.

        Args:
            provider: The provider this widget configures.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._provider = provider
        self._info = PROVIDER_INFO[provider]
        self._setup_ui()
        self._load_current_key()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Description
        desc = self._info.get("description", "")
        desc_label = QLabel(str(desc))
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # API Key section (if required)
        if self._info["requires_key"]:
            key_layout = QHBoxLayout()

            self._key_input = QLineEdit()
            self._key_input.setPlaceholderText("Enter API key...")
            self._key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self._key_input.textChanged.connect(self._on_key_changed)
            key_layout.addWidget(self._key_input)

            self._show_key_btn = QPushButton("Show")
            self._show_key_btn.setCheckable(True)
            self._show_key_btn.toggled.connect(self._toggle_key_visibility)
            key_layout.addWidget(self._show_key_btn)

            self._save_key_btn = QPushButton("Save")
            self._save_key_btn.clicked.connect(self._save_key)
            key_layout.addWidget(self._save_key_btn)

            layout.addLayout(key_layout)

            # Key status
            self._key_status = QLabel()
            layout.addWidget(self._key_status)
        else:
            # Ollama host configuration
            host_layout = QFormLayout()
            self._host_input = QLineEdit("http://localhost:11434")
            host_layout.addRow("Ollama Host:", self._host_input)
            layout.addLayout(host_layout)

            # Test connection button
            self._test_btn = QPushButton("Test Connection")
            self._test_btn.clicked.connect(self._test_ollama_connection)
            layout.addWidget(self._test_btn)

            self._status_label = QLabel()
            layout.addWidget(self._status_label)

        layout.addStretch()

    def _load_current_key(self) -> None:
        """Load current API key from keyring."""
        if not self._info["requires_key"]:
            return

        key = get_api_key(self._provider)
        if key:
            # Show masked key
            self._key_input.setText(key)
            self._key_status.setText("✓ API key configured")
            self._key_status.setStyleSheet("color: green;")
        else:
            self._key_status.setText("⚠ No API key configured")
            self._key_status.setStyleSheet("color: orange;")

    def _on_key_changed(self, text: str) -> None:
        """Handle key input changes."""
        self.key_changed.emit(text)

    def _toggle_key_visibility(self, show: bool) -> None:
        """Toggle API key visibility."""
        if show:
            self._key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self._show_key_btn.setText("Hide")
        else:
            self._key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self._show_key_btn.setText("Show")

    def _save_key(self) -> None:
        """Save API key to secure storage."""
        key = self._key_input.text().strip()
        if not key:
            QMessageBox.warning(self, "Error", "Please enter an API key.")
            return

        if set_api_key(self._provider, key):
            self._key_status.setText("✓ API key saved securely")
            self._key_status.setStyleSheet("color: green;")
            QMessageBox.information(
                self,
                "Success",
                f"API key saved to {self._get_keyring_location()}",
            )
        else:
            self._key_status.setText("✗ Failed to save key")
            self._key_status.setStyleSheet("color: red;")
            QMessageBox.warning(
                self,
                "Error",
                "Failed to save API key. The keyring package may not be installed.",
            )

    def _get_keyring_location(self) -> str:
        """Get description of where keys are stored."""
        import platform

        system = platform.system()
        if system == "Windows":
            return "Windows Credential Manager"
        elif system == "Darwin":
            return "macOS Keychain"
        else:
            return "System keyring"

    def _test_ollama_connection(self) -> None:
        """Test connection to Ollama server."""
        self._status_label.setText("Testing connection...")
        self._status_label.setStyleSheet("")

        try:
            from shared.python.ai.adapters.ollama_adapter import OllamaAdapter

            host = self._host_input.text().strip()
            adapter = OllamaAdapter(host=host)
            success, message = adapter.validate_connection()

            if success:
                self._status_label.setText(f"✓ {message}")
                self._status_label.setStyleSheet("color: green;")
            else:
                self._status_label.setText(f"✗ {message}")
                self._status_label.setStyleSheet("color: red;")

        except Exception as e:
            self._status_label.setText(f"✗ Error: {e}")
            self._status_label.setStyleSheet("color: red;")

    def get_host(self) -> str:
        """Get Ollama host if applicable."""
        if hasattr(self, "_host_input"):
            return str(self._host_input.text().strip())
        return "http://localhost:11434"


class AISettingsDialog(QDialog):
    """Settings dialog for AI Assistant configuration."""

    settings_changed = pyqtSignal(AISettings)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize settings dialog.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("AI Assistant Settings")
        self.setMinimumSize(500, 400)
        self._settings = AISettings.load()
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)

        # Create tabs
        tabs = QTabWidget()

        # Provider tab
        provider_tab = self._create_provider_tab()
        tabs.addTab(provider_tab, "Provider")

        # Preferences tab
        prefs_tab = self._create_preferences_tab()
        tabs.addTab(prefs_tab, "Preferences")

        layout.addWidget(tabs)

        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _create_provider_tab(self) -> QWidget:
        """Create the provider configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Provider selection
        provider_group = QGroupBox("Select AI Provider")
        provider_layout = QVBoxLayout(provider_group)

        self._provider_combo = QComboBox()
        for provider in AIProvider:
            info = PROVIDER_INFO[provider]
            name = str(info.get("name", provider.name))
            self._provider_combo.addItem(name, provider)
        self._provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        provider_layout.addWidget(self._provider_combo)

        # Cost note
        cost_label = QLabel(
            "<b>💡 Tip:</b> Ollama is completely FREE and runs locally. "
            "No API keys or cloud costs!"
        )
        cost_label.setWordWrap(True)
        provider_layout.addWidget(cost_label)

        layout.addWidget(provider_group)

        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)

        self._model_combo = QComboBox()
        model_layout.addRow("Model:", self._model_combo)

        layout.addWidget(model_group)

        # Provider-specific config
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)

        self._provider_configs: dict[AIProvider, ProviderConfigWidget] = {}
        for provider in AIProvider:
            config_widget = ProviderConfigWidget(provider)
            config_widget.hide()
            self._provider_configs[provider] = config_widget
            config_layout.addWidget(config_widget)

        layout.addWidget(config_group)
        layout.addStretch()

        return widget

    def _create_preferences_tab(self) -> QWidget:
        """Create the preferences tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Expertise level
        expertise_group = QGroupBox("Expertise Level")
        expertise_layout = QVBoxLayout(expertise_group)

        self._expertise_combo = QComboBox()
        self._expertise_combo.addItem("Beginner - Clear explanations, no jargon", 1)
        self._expertise_combo.addItem("Intermediate - Some technical terms", 2)
        self._expertise_combo.addItem("Advanced - Full technical depth", 3)
        self._expertise_combo.addItem("Expert - Research-level precision", 4)
        expertise_layout.addWidget(self._expertise_combo)

        expertise_desc = QLabel(
            "This setting adjusts how the AI explains concepts. "
            "Beginners get analogies and simple language; "
            "experts get equations and technical details."
        )
        expertise_desc.setWordWrap(True)
        expertise_layout.addWidget(expertise_desc)

        layout.addWidget(expertise_group)

        # Response settings
        response_group = QGroupBox("Response Settings")
        response_layout = QVBoxLayout(response_group)

        self._streaming_check = QCheckBox("Enable streaming responses")
        self._streaming_check.setToolTip(
            "Show responses as they're generated (more responsive)"
        )
        response_layout.addWidget(self._streaming_check)

        layout.addWidget(response_group)

        layout.addStretch()
        return widget

    def _load_settings(self) -> None:
        """Load current settings into UI."""
        # Provider
        for i in range(self._provider_combo.count()):
            if self._provider_combo.itemData(i) == self._settings.provider:
                self._provider_combo.setCurrentIndex(i)
                break

        self._on_provider_changed(self._provider_combo.currentIndex())

        # Model
        for i in range(self._model_combo.count()):
            if self._model_combo.itemText(i) == self._settings.model:
                self._model_combo.setCurrentIndex(i)
                break

        # Expertise
        for i in range(self._expertise_combo.count()):
            if self._expertise_combo.itemData(i) == self._settings.expertise_level:
                self._expertise_combo.setCurrentIndex(i)
                break

        # Streaming
        self._streaming_check.setChecked(self._settings.streaming_enabled)

    def _on_provider_changed(self, index: int) -> None:
        """Handle provider selection change."""
        provider_data = self._provider_combo.itemData(index)
        if provider_data is None or not isinstance(provider_data, AIProvider):
            return
        provider: AIProvider = provider_data

        # Update model combo
        info = PROVIDER_INFO[provider]
        self._model_combo.clear()
        models = info.get("models", [])
        if isinstance(models, list):
            for model in models:
                self._model_combo.addItem(str(model))

        # Select default model
        default_model = info.get("default_model", "")
        if isinstance(default_model, str):
            idx = self._model_combo.findText(default_model)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)

        # Show/hide provider configs
        for p, widget in self._provider_configs.items():
            widget.setVisible(p == provider)

    def _accept(self) -> None:
        """Accept dialog and save settings."""
        # Update settings from UI
        self._settings.provider = self._provider_combo.currentData()
        self._settings.model = self._model_combo.currentText()
        self._settings.expertise_level = self._expertise_combo.currentData()
        self._settings.streaming_enabled = self._streaming_check.isChecked()

        # Get Ollama host if applicable
        if self._settings.provider == AIProvider.OLLAMA:
            config = self._provider_configs[AIProvider.OLLAMA]
            self._settings.ollama_host = config.get_host()

        # Save and emit
        self._settings.save()
        self.settings_changed.emit(self._settings)
        self.accept()

    def get_settings(self) -> AISettings:
        """Get current settings.

        Returns:
            Current AISettings.
        """
        return self._settings
