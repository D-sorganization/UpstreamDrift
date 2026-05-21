"""Per-provider configuration widgets used by the AI settings dialog.

Each provider gets its own ``QWidget`` subclass instead of a single
hardcoded-by-enum-branch widget. They're registered in
``_provider_config_registry.ProviderConfigRegistry`` so adding a new
provider only requires a new widget class + a registry call.

Tools issue #2762.
"""

from __future__ import annotations

import os
from typing import Any

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.ai.config import get_ollama_host
from src.shared.python.ai.gui._api_keys import get_api_key, set_api_key
from src.shared.python.ai.gui._provider_registry_data import (
    BITNET_ROOT_ENV,
    DEFAULT_CLINE_HOST,
    PROVIDER_INFO,
    AIProvider,
)
from src.shared.python.theme.style_constants import Styles


class _BaseProviderConfigWidget(QWidget):
    """Common scaffolding for per-provider config widgets."""

    key_changed = pyqtSignal(str)
    models_refreshed = pyqtSignal(list)

    PROVIDER: AIProvider  # subclasses override

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._provider = self.PROVIDER
        self._info = PROVIDER_INFO[self._provider]
        self._build()

    # ---- subclass hooks -------------------------------------------------

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        desc = self._info.get("description", "")
        desc_label = QLabel(str(desc))
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        self._build_body(layout)
        layout.addStretch()

    def _build_body(self, layout: QVBoxLayout) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- public API mirrored from the legacy ProviderConfigWidget -------

    def get_host(self) -> str:
        return ""


class _ApiKeyProviderConfigWidget(_BaseProviderConfigWidget):
    """Reusable body for cloud providers that require an API key."""

    def _build_body(self, layout: QVBoxLayout) -> None:
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

        self._key_status = QLabel()
        layout.addWidget(self._key_status)

        self._load_current_key()

    def _load_current_key(self) -> None:
        key = get_api_key(self._provider)
        if key:
            self._key_input.setText(key)
            self._key_status.setText("✓ API key configured")
            self._key_status.setStyleSheet(Styles.COLOR_GREEN)
        else:
            self._key_status.setText("⚠ No API key configured")
            self._key_status.setStyleSheet(Styles.COLOR_ORANGE)

    def _on_key_changed(self, text: str) -> None:
        self.key_changed.emit(text)

    def _toggle_key_visibility(self, show: bool) -> None:
        if show:
            self._key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self._show_key_btn.setText("Hide")
        else:
            self._key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self._show_key_btn.setText("Show")

    def _save_key(self) -> None:
        key = self._key_input.text().strip()
        if not key:
            QMessageBox.warning(self, "Error", "Please enter an API key.")
            return
        if set_api_key(self._provider, key):
            self._key_status.setText("✓ API key saved securely")
            self._key_status.setStyleSheet(Styles.COLOR_GREEN)
            QMessageBox.information(
                self,
                "Success",
                f"API key saved to {self._get_keyring_location()}",
            )
        else:
            self._key_status.setText("✗ Failed to save key")
            self._key_status.setStyleSheet(Styles.COLOR_RED)
            QMessageBox.warning(
                self,
                "Error",
                "Failed to save API key. The keyring package may not be installed.",
            )

    def _get_keyring_location(self) -> str:
        import platform

        system = platform.system()
        if system == "Windows":
            return "Windows Credential Manager"
        if system == "Darwin":
            return "macOS Keychain"
        return "System keyring"


class OpenAIConfigWidget(_ApiKeyProviderConfigWidget):
    PROVIDER = AIProvider.OPENAI


class AnthropicConfigWidget(_ApiKeyProviderConfigWidget):
    PROVIDER = AIProvider.ANTHROPIC


class GeminiConfigWidget(_ApiKeyProviderConfigWidget):
    PROVIDER = AIProvider.GEMINI


class OllamaConfigWidget(_BaseProviderConfigWidget):
    PROVIDER = AIProvider.OLLAMA

    def _build_body(self, layout: QVBoxLayout) -> None:
        host_layout = QFormLayout()
        self._host_input = QLineEdit(get_ollama_host())
        host_layout.addRow("Ollama Host:", self._host_input)
        layout.addLayout(host_layout)

        self._test_btn = QPushButton("Test Connection")
        self._test_btn.clicked.connect(self._test_connection)
        layout.addWidget(self._test_btn)

        self._refresh_models_btn = QPushButton("\U0001f504 Refresh Available Models")
        self._refresh_models_btn.setToolTip(
            "Fetch the list of installed models from your local Ollama instance"
        )
        self._refresh_models_btn.clicked.connect(self._refresh_models)
        layout.addWidget(self._refresh_models_btn)

        self._status_label = QLabel()
        layout.addWidget(self._status_label)

        self._model_count_label = QLabel()
        self._model_count_label.setStyleSheet(Styles.TEXT_MUTED)
        layout.addWidget(self._model_count_label)

    def get_host(self) -> str:
        return str(self._host_input.text().strip())

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        QTimer.singleShot(100, self._refresh_models)

    def _test_connection(self) -> None:
        self._status_label.setText("Testing connection...")
        self._status_label.setStyleSheet(Styles.COLOR_RESET)
        try:
            from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter

            adapter = OllamaAdapter(host=self._host_input.text().strip())
            success, message = adapter.validate_connection()
            if success:
                self._status_label.setText(f"✓ {message}")
                self._status_label.setStyleSheet(Styles.COLOR_GREEN)
                self._refresh_models()
            else:
                self._status_label.setText(f"✗ {message}")
                self._status_label.setStyleSheet(Styles.COLOR_RED)
        except ImportError as e:
            self._status_label.setText(f"✗ Error: {e}")
            self._status_label.setStyleSheet(Styles.COLOR_RED)

    def _refresh_models(self) -> None:
        self._status_label.setText("Fetching available models...")
        self._status_label.setStyleSheet(Styles.COLOR_RESET)
        try:
            from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter

            adapter = OllamaAdapter(host=self._host_input.text().strip())
            models = adapter.list_available_models()
            if models:
                preview = ", ".join(models[:5])
                suffix = "..." if len(models) > 5 else ""
                self._model_count_label.setText(
                    f"✓ Found {len(models)} model(s): {preview}{suffix}"
                )
                self._model_count_label.setStyleSheet(Styles.COLOR_GREEN)
                self.models_refreshed.emit(models)
            else:
                self._model_count_label.setText(
                    "⚠ No models found. Pull one with: ollama pull llama3.1:8b"
                )
                self._model_count_label.setStyleSheet(Styles.COLOR_ORANGE)
                self.models_refreshed.emit([])
        except Exception as e:  # noqa: BLE001 - GUI status display
            self._model_count_label.setText(f"✗ Failed to fetch models: {e}")
            self._model_count_label.setStyleSheet(Styles.COLOR_RED)
            self.models_refreshed.emit([])


class ClineConfigWidget(_BaseProviderConfigWidget):
    PROVIDER = AIProvider.CLINE_CLI

    def _build_body(self, layout: QVBoxLayout) -> None:
        host_layout = QFormLayout()
        self._host_input = QLineEdit(DEFAULT_CLINE_HOST)
        host_layout.addRow("Cline Host:", self._host_input)
        layout.addLayout(host_layout)

        self._test_btn = QPushButton("Test Connection")
        self._test_btn.clicked.connect(self._test_connection)
        layout.addWidget(self._test_btn)

        self._status_label = QLabel()
        layout.addWidget(self._status_label)

    def get_host(self) -> str:
        if hasattr(self, "_host_input"):
            return str(self._host_input.text().strip())
        return DEFAULT_CLINE_HOST

    def _test_connection(self) -> None:
        self._status_label.setText("Testing connection...")
        self._status_label.setStyleSheet(Styles.COLOR_RESET)
        try:
            from src.shared.python.ai.adapters.cline_adapter import ClineAdapter

            adapter = ClineAdapter(host=self._host_input.text().strip())
            success, message = adapter.validate_connection()
            if success:
                self._status_label.setText(f"✓ {message}")
                self._status_label.setStyleSheet(Styles.COLOR_GREEN)
            else:
                self._status_label.setText(f"✗ {message}")
                self._status_label.setStyleSheet(Styles.COLOR_RED)
        except ImportError as e:
            self._status_label.setText(f"✗ Error: {e}")
            self._status_label.setStyleSheet(Styles.COLOR_RED)


class BitnetConfigWidget(_BaseProviderConfigWidget):
    PROVIDER = AIProvider.BITNET

    def _build_body(self, layout: QVBoxLayout) -> None:
        root_layout = QFormLayout()
        self._bitnet_root_input = QLineEdit(os.environ.get(BITNET_ROOT_ENV, ""))
        self._bitnet_root_input.setPlaceholderText(
            "Optional path to the BitNet installation root"
        )
        root_layout.addRow("BitNet Root:", self._bitnet_root_input)
        layout.addLayout(root_layout)

        note = QLabel(
            "Use the main model selector to choose a GGUF model. "
            "Leave BitNet Root blank to rely on PATH or BITNET_ROOT."
        )
        note.setWordWrap(True)
        note.setStyleSheet(Styles.TEXT_MUTED)
        layout.addWidget(note)

    def get_host(self) -> str:
        if hasattr(self, "_bitnet_root_input"):
            return str(self._bitnet_root_input.text().strip())
        return os.environ.get(BITNET_ROOT_ENV, "")


class _CliNoteConfigWidget(_BaseProviderConfigWidget):
    """Honest no-op widget for CLI-backed providers (Claude CLI, Codex CLI)."""

    def _build_body(self, layout: QVBoxLayout) -> None:
        note = QLabel(
            "This provider uses your installed CLI tooling and does not need "
            "extra connection settings in this dialog."
        )
        note.setWordWrap(True)
        note.setStyleSheet(Styles.TEXT_MUTED)
        layout.addWidget(note)


class ClaudeCliConfigWidget(_CliNoteConfigWidget):
    PROVIDER = AIProvider.CLAUDE_CLI


class CodexCliConfigWidget(_CliNoteConfigWidget):
    PROVIDER = AIProvider.CODEX_CLI


__all__ = [
    "AnthropicConfigWidget",
    "BitnetConfigWidget",
    "ClaudeCliConfigWidget",
    "ClineConfigWidget",
    "CodexCliConfigWidget",
    "GeminiConfigWidget",
    "OllamaConfigWidget",
    "OpenAIConfigWidget",
]
