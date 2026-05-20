"""Adapter lifecycle manager for the AI assistant.

Owns provider/key resolution and adapter construction. Returns the new
adapter (or ``None``) so the panel can install it; emits a
``system_message`` signal for user-visible status text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QObject, pyqtSignal

if TYPE_CHECKING:
    from src.shared.python.ai._settings_model import AISettings

from src.shared.python.ai.gui._provider_registry_data import AIProvider
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class AdapterLifecycleManager(QObject):
    """Constructs adapter instances from ``AISettings``.

    Emits ``adapter_changed(adapter, adapter_id)`` with the freshly built
    adapter (or ``None`` when construction failed). The ``system_message``
    signal carries human-readable status the panel can display.
    """

    adapter_changed = pyqtSignal(object, str)
    system_message = pyqtSignal(str)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def auto_load(self) -> AISettings | None:
        """Reload settings from disk and rebuild the adapter."""
        from src.shared.python.ai._settings_model import AISettings

        try:
            settings = AISettings.load()
        except ImportError as exc:
            logger.warning("Failed to auto-load AI settings: %s", exc)
            return None
        self.build(settings)
        return settings

    def build(self, settings: AISettings) -> Any:
        """Construct an adapter for ``settings`` and emit signals."""
        adapter = self._construct(settings)
        adapter_id = type(adapter).__name__ if adapter is not None else ""
        if adapter is not None:
            self.adapter_changed.emit(adapter, adapter_id)
            self.system_message.emit(
                f"✓ Connected to {settings.provider.name} ({settings.model})"
            )
        else:
            self.adapter_changed.emit(None, "")
            self.system_message.emit(
                f"⚠️ Could not connect to {settings.provider.name}. "
                "Please check your settings."
            )
        return adapter

    # ------------------------------------------------------------------
    # Provider construction
    # ------------------------------------------------------------------
    def _construct(self, settings: AISettings) -> Any:
        provider = settings.provider
        if provider == AIProvider.OLLAMA:
            return self._build_ollama(settings)
        if provider == AIProvider.OPENAI:
            return self._build_openai(settings)
        if provider == AIProvider.ANTHROPIC:
            return self._build_anthropic(settings)
        if provider == AIProvider.GEMINI:
            return self._build_gemini(settings)
        return None

    def _build_ollama(self, settings: AISettings) -> Any:
        try:
            from src.shared.python.ai.adapters.rust_adapter import RustAgentAdapter

            adapter = RustAgentAdapter(
                api_key="ollama",
                base_url=settings.ollama_host,
                model=settings.model,
                chat_path="/v1/chat/completions",
                embed_path="/v1/embeddings",
            )
            self.system_message.emit("🚀 Using high-performance Rust AI backend.")
            return adapter
        except ImportError:
            from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter

            return OllamaAdapter(host=settings.ollama_host, model=settings.model)

    @staticmethod
    def _build_openai(settings: AISettings) -> Any:
        from src.shared.python.ai.gui.settings_dialog import get_api_key

        api_key = get_api_key(AIProvider.OPENAI)
        if not api_key:
            return None
        from src.shared.python.ai.adapters.openai_adapter import OpenAIAdapter

        return OpenAIAdapter(api_key=api_key, model=settings.model)

    @staticmethod
    def _build_anthropic(settings: AISettings) -> Any:
        from src.shared.python.ai.gui.settings_dialog import get_api_key

        api_key = get_api_key(AIProvider.ANTHROPIC)
        if not api_key:
            return None
        from src.shared.python.ai.adapters.anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(api_key=api_key, model=settings.model)

    @staticmethod
    def _build_gemini(settings: AISettings) -> Any:
        from src.shared.python.ai.gui.settings_dialog import get_api_key

        api_key = get_api_key(AIProvider.GEMINI)
        if not api_key:
            return None
        from src.shared.python.ai.adapters.gemini_adapter import GeminiAdapter

        return GeminiAdapter(api_key=api_key, model=settings.model)
