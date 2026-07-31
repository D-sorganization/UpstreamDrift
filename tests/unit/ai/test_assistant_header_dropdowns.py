"""Tests for the AI Assistant header UI.

Tests the Python-level behavior of the header helper methods extracted into
the _MockPanel test double:
- _get_models_for_provider() returns ChatModelInfo lists per provider
- _get_thinking_capabilities_for_model() returns ThinkingCapabilities
- _on_provider_changed() calls _model_combo.clear()
- _on_model_changed() calls _thinking_combo.setEnabled()

Also verifies that a constructed AIAssistantPanel (from the post-split
assistant sub-package) exposes the expected header attributes.

These tests use a lightweight mock panel object to avoid Qt widget construction
issues in headless/mocked environments.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class _MockPanel:
    """Minimal stand-in for AIAssistantPanel exposing only the new methods.

    This avoids full Qt widget construction while testing the pure-Python
    business logic added to AIAssistantPanel.
    """

    def __init__(self) -> None:
        # Mock the combo attributes that the methods access
        self._provider_combo = MagicMock()
        self._model_combo = MagicMock()
        self._thinking_combo = MagicMock()
        self._model_combo.blockSignals = MagicMock()
        self._model_combo.clear = MagicMock()
        self._model_combo.addItem = MagicMock()
        self._model_combo.currentData = MagicMock(return_value=None)
        self._thinking_combo.setEnabled = MagicMock()

    # -- Methods copied from AIAssistantPanel (must stay in sync) --

    def _get_models_for_provider(self, provider_label: str) -> list:
        from src.shared.python.ai.types import ChatModelInfo

        _static: dict = {
            "Ollama": [
                ChatModelInfo("llama3.1:8b"),
                ChatModelInfo("llama3.1:70b"),
                ChatModelInfo("mistral"),
                ChatModelInfo("codellama"),
                ChatModelInfo("deepseek-coder"),
                ChatModelInfo("phi3"),
            ],
            "OpenAI": [
                ChatModelInfo("gpt-4o", "GPT-4o"),
                ChatModelInfo("gpt-4o-mini", "GPT-4o Mini"),
                ChatModelInfo("gpt-4-turbo", "GPT-4 Turbo"),
                ChatModelInfo("o1", "o1"),
                ChatModelInfo("o3-mini", "o3-mini"),
            ],
            "Anthropic": [
                ChatModelInfo("claude-sonnet-4-6", "Claude Sonnet 4.6"),
                ChatModelInfo("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"),
                ChatModelInfo("claude-3-5-haiku-20241022", "Claude 3.5 Haiku"),
                ChatModelInfo("claude-3-opus-20240229", "Claude 3 Opus"),
            ],
            "Gemini": [
                ChatModelInfo("gemini-2.0-flash", "Gemini 2.0 Flash"),
                ChatModelInfo("gemini-1.5-pro", "Gemini 1.5 Pro"),
                ChatModelInfo("gemini-1.5-flash", "Gemini 1.5 Flash"),
            ],
        }
        return _static.get(provider_label, [])

    def _get_thinking_capabilities_for_model(self, model_id: str):
        from src.shared.python.ai.types import ThinkingCapabilities, ThinkingLevel

        _thinking_models = {
            "claude-sonnet-4-6",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "o1",
            "o3-mini",
            "o3",
        }
        supports = any(m in model_id for m in _thinking_models)
        if supports:
            return ThinkingCapabilities(
                supports_levels=True,
                available_levels=[
                    ThinkingLevel.OFF,
                    ThinkingLevel.LOW,
                    ThinkingLevel.MEDIUM,
                    ThinkingLevel.HIGH,
                ],
            )
        return ThinkingCapabilities(
            supports_levels=False,
            available_levels=[ThinkingLevel.OFF],
        )

    def _on_provider_changed(self, provider_label: str) -> None:
        if not hasattr(self, "_model_combo"):
            return
        models = self._get_models_for_provider(provider_label)
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for m in models:
            self._model_combo.addItem(m.display_name or m.model_id, m.model_id)
        self._model_combo.blockSignals(False)
        if models and hasattr(self, "_thinking_combo"):
            self._on_model_changed(models[0].model_id)

    def _on_model_changed(self, model_text: str) -> None:
        if not hasattr(self, "_thinking_combo"):
            return
        if hasattr(self, "_model_combo"):
            model_id = self._model_combo.currentData() or model_text
        else:
            model_id = model_text
        caps = self._get_thinking_capabilities_for_model(model_id)
        self._thinking_combo.setEnabled(caps.supports_levels)


class TestGetModelsForProvider:
    """_get_models_for_provider must return non-empty lists for main providers."""

    def test_returns_nonempty_list_for_ollama(self) -> None:
        panel = _MockPanel()
        models = panel._get_models_for_provider("Ollama")
        assert len(models) >= 1
        assert all(hasattr(m, "model_id") for m in models)

    def test_returns_nonempty_list_for_openai(self) -> None:
        panel = _MockPanel()
        models = panel._get_models_for_provider("OpenAI")
        assert len(models) >= 1

    def test_returns_nonempty_list_for_anthropic(self) -> None:
        panel = _MockPanel()
        models = panel._get_models_for_provider("Anthropic")
        assert len(models) >= 1

    def test_returns_nonempty_list_for_gemini(self) -> None:
        panel = _MockPanel()
        models = panel._get_models_for_provider("Gemini")
        assert len(models) >= 1

    def test_returns_empty_list_for_unknown_provider(self) -> None:
        panel = _MockPanel()
        models = panel._get_models_for_provider("UnknownProvider")
        assert models == []

    def test_all_entries_are_chat_model_info(self) -> None:
        from src.shared.python.ai.types import ChatModelInfo

        panel = _MockPanel()
        for provider in ("Ollama", "OpenAI", "Anthropic", "Gemini"):
            models = panel._get_models_for_provider(provider)
            assert all(
                isinstance(m, ChatModelInfo) for m in models
            ), f"All models for {provider} must be ChatModelInfo instances"


class TestGetThinkingCapabilitiesForModel:
    """_get_thinking_capabilities_for_model returns correct ThinkingCapabilities."""

    def test_claude_sonnet_supports_thinking(self) -> None:
        panel = _MockPanel()
        caps = panel._get_thinking_capabilities_for_model("claude-3-5-sonnet-20241022")
        assert caps.supports_levels is True

    def test_llama_does_not_support_thinking(self) -> None:
        panel = _MockPanel()
        caps = panel._get_thinking_capabilities_for_model("llama3.1:8b")
        assert caps.supports_levels is False

    def test_o1_supports_thinking(self) -> None:
        panel = _MockPanel()
        caps = panel._get_thinking_capabilities_for_model("o1")
        assert caps.supports_levels is True

    def test_gemini_pro_supports_thinking(self) -> None:
        panel = _MockPanel()
        caps = panel._get_thinking_capabilities_for_model("gemini-1.5-pro")
        assert caps.supports_levels is True

    def test_returns_thinking_capabilities_dataclass(self) -> None:
        from src.shared.python.ai.types import ThinkingCapabilities

        panel = _MockPanel()
        caps = panel._get_thinking_capabilities_for_model("llama3.1:8b")
        assert isinstance(caps, ThinkingCapabilities)
        assert isinstance(caps.supports_levels, bool)
        assert isinstance(caps.available_levels, list)


class TestOnProviderChanged:
    """_on_provider_changed must clear and repopulate the model combo."""

    def test_clears_model_combo_on_provider_change(self) -> None:
        panel = _MockPanel()
        panel._on_provider_changed("Ollama")
        panel._model_combo.clear.assert_called()

    def test_adds_items_for_known_provider(self) -> None:
        panel = _MockPanel()
        panel._on_provider_changed("Anthropic")
        assert (
            panel._model_combo.addItem.called
        ), "addItem must be called for each model when provider is known"

    def test_provider_change_triggers_thinking_update(self) -> None:
        """After provider change, _on_model_changed is called for first model."""
        panel = _MockPanel()
        panel._on_provider_changed("Anthropic")
        # Claude Sonnet is the first Anthropic model and supports thinking
        panel._thinking_combo.setEnabled.assert_called_with(True)


class TestOnModelChanged:
    """_on_model_changed must enable/disable thinking combo."""

    def test_disables_thinking_for_non_thinking_model(self) -> None:
        from src.shared.python.ai.types import ThinkingCapabilities

        panel = _MockPanel()
        no_thinking = ThinkingCapabilities(supports_levels=False, available_levels=[])

        with patch.object(
            panel, "_get_thinking_capabilities_for_model", return_value=no_thinking
        ):
            panel._on_model_changed("llama3.1:8b")

        panel._thinking_combo.setEnabled.assert_called_with(False)

    def test_enables_thinking_for_thinking_model(self) -> None:
        from src.shared.python.ai.types import ThinkingCapabilities, ThinkingLevel

        panel = _MockPanel()
        thinking = ThinkingCapabilities(
            supports_levels=True,
            available_levels=[ThinkingLevel.OFF, ThinkingLevel.HIGH],
        )

        with patch.object(
            panel, "_get_thinking_capabilities_for_model", return_value=thinking
        ):
            panel._on_model_changed("claude-3-5-sonnet-20241022")

        panel._thinking_combo.setEnabled.assert_called_with(True)


@pytest.mark.headless_safe
class TestPanelHasRequiredCombos:
    """Verify AIAssistantPanel has the expected header attributes after construction.

    The header was simplified during the #5493 split: the triple-dropdown
    (_provider_combo, _model_combo, _thinking_combo) was replaced with a
    lightweight icon + label pair so the panel remains under the 500-line
    class limit.  These tests reflect the current (post-split) API.

    Uses a single panel construction per test class to minimise Qt init costs.
    Requires a running QApplication — skipped when the display server is absent.
    """

    _panel = None

    @classmethod
    def _get_panel(cls) -> object:
        if cls._panel is None:
            try:
                from PyQt6.QtWidgets import QApplication

                if QApplication.instance() is None:
                    pytest.skip(
                        "No QApplication available — skipping widget construction test"
                    )
            except (ImportError, OSError) as exc:
                pytest.skip(f"PyQt6 QApplication not available: {exc}")

            from src.shared.python.ai.gui.assistant.panel import AIAssistantPanel

            mock_session_mgr = MagicMock()
            mock_session_mgr.list_sessions.return_value = []
            mock_session_mgr.session_loaded = MagicMock()
            mock_session_mgr.session_loaded.connect = MagicMock()

            mock_theme_mgr = MagicMock()
            mock_theme_mgr.get_current_colors.return_value = {}
            mock_theme_mgr.instance.return_value = mock_theme_mgr

            with (
                patch(
                    "src.shared.python.ai.gui.assistant.panel.ChatSessionManager",
                    return_value=mock_session_mgr,
                ),
                patch(
                    "src.shared.python.ai.gui.assistant.panel.AIAssistantPanel._auto_load_settings"
                ),
                patch(
                    "src.shared.python.ai.gui.assistant.panel.AIAssistantPanel.refresh_theme"
                ),
                patch(
                    "src.shared.python.theme.theme_manager.get_theme_manager",
                    return_value=mock_theme_mgr,
                ),
                patch(
                    "src.shared.python.ai.gui.history_sidebar.get_theme_manager",
                    return_value=mock_theme_mgr,
                    create=True,
                ),
            ):
                cls._panel = AIAssistantPanel()

        return cls._panel

    def test_has_provider_icon(self) -> None:
        """Panel has _provider_icon QLabel in the header."""
        panel = self._get_panel()
        assert hasattr(
            panel, "_provider_icon"
        ), "_provider_icon must be created in _add_header_title_widgets"

    def test_has_model_label(self) -> None:
        """Panel has _model_label QLabel in the header."""
        panel = self._get_panel()
        assert hasattr(
            panel, "_model_label"
        ), "_model_label must be created in _add_header_title_widgets"

    def test_has_mode_combo(self) -> None:
        """Panel has _mode_combo in the header."""
        panel = self._get_panel()
        assert hasattr(
            panel, "_mode_combo"
        ), "_mode_combo must be created in _add_header_mode_and_status"

    def test_has_status_label(self) -> None:
        """Panel has _status_label in the header."""
        panel = self._get_panel()
        assert hasattr(
            panel, "_status_label"
        ), "_status_label must be created in _add_header_mode_and_status"
