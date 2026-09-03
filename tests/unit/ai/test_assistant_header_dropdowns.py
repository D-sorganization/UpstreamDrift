"""Tests for the AI Assistant header's provider/model dropdowns.

These exercise the **production** helpers in
``src.shared.python.ai.gui._provider_registry_data`` -- ``PROVIDER_INFO`` and
the ``populate_*``/``provider_*`` functions that ``_panel_header.AIPanelHeader``
actually calls -- plus the post-split panel attributes.

History (#9474): this file used to define a ``_MockPanel`` test double that
**copied** ``_get_models_for_provider`` and ``_get_thinking_capabilities_for_model``
out of ``AIAssistantPanel``, with a comment reading "Methods copied from
AIAssistantPanel (must stay in sync)". They did not stay in sync. The panel was
split in #5493 and both methods were deleted from production; the copies lived
on, importing ``ChatModelInfo``/``ThinkingCapabilities`` from
``src.shared.python.ai.types``, where neither name has ever existed. All sixteen
tests raised ImportError, so the file asserted nothing about the product for
months -- it tested its own copy, and then stopped even doing that.

The duplication is removed rather than repaired: every assertion below runs
against the single source of truth, so the drift that caused this cannot recur.
No coverage is lost -- ``thinking_capabilities()`` is covered per adapter in
``tests/unit/ai/test_adapter_capabilities.py``, which is where the behaviour
now lives.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.ai.gui._provider_registry_data import (
    PROVIDER_INFO,
    AIProvider,
    populate_model_combo,
    populate_provider_combo,
    provider_default_model,
    provider_display_name,
    provider_model_names,
)


class _FakeCombo:
    """Minimal stand-in for the QComboBox subset the populate helpers use.

    Using a fake rather than a real widget keeps these tests headless and
    Qt-free; the subject under test is the production function, not the widget.
    """

    def __init__(self) -> None:
        self.items: list[tuple[str, object]] = []
        self.current_index = -1
        self.clear_calls = 0

    def clear(self) -> None:
        self.clear_calls += 1
        self.items = []
        self.current_index = -1

    def addItem(self, text: str, data: object = None) -> None:  # noqa: N802
        self.items.append((text, data))

    def findText(self, text: str) -> int:  # noqa: N802
        for index, (label, _data) in enumerate(self.items):
            if label == text:
                return index
        return -1

    def count(self) -> int:
        return len(self.items)

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802
        self.current_index = index

    def currentText(self) -> str:  # noqa: N802
        if 0 <= self.current_index < len(self.items):
            return self.items[self.current_index][0]
        return ""


ALL_PROVIDERS = list(AIProvider)


class TestProviderRegistry:
    """``PROVIDER_INFO`` must describe every provider completely."""

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    def test_every_provider_has_a_nonempty_model_catalogue(
        self, provider: AIProvider
    ) -> None:
        models = provider_model_names(provider)

        assert models, f"{provider.name} has no models"
        assert all(isinstance(name, str) and name.strip() for name in models)

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    def test_default_model_is_one_of_the_listed_models(
        self, provider: AIProvider
    ) -> None:
        """The default must be selectable, or the dropdown opens on the wrong row."""
        assert provider_default_model(provider) in provider_model_names(provider)

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    def test_every_provider_has_a_display_name(self, provider: AIProvider) -> None:
        assert provider_display_name(provider).strip()

    def test_registry_covers_the_whole_enum(self) -> None:
        """A new ``AIProvider`` member without registry data must fail here."""
        assert set(PROVIDER_INFO) == set(AIProvider)


class TestPopulateProviderCombo:
    """``populate_provider_combo`` fills a combo from the registry."""

    def test_adds_one_entry_per_provider_carrying_the_enum_member(self) -> None:
        combo = _FakeCombo()

        populate_provider_combo(combo)

        assert [label for label, _ in combo.items] == [
            provider_display_name(p) for p in AIProvider
        ]
        assert [data for _, data in combo.items] == list(AIProvider)

    def test_clears_before_populating(self) -> None:
        combo = _FakeCombo()
        combo.addItem("stale", None)

        populate_provider_combo(combo)

        assert combo.clear_calls == 1
        assert ("stale", None) not in combo.items


class TestPopulateModelCombo:
    """``populate_model_combo`` is what the header calls on provider change."""

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    def test_lists_exactly_the_registry_models(self, provider: AIProvider) -> None:
        combo = _FakeCombo()

        populate_model_combo(combo, provider)

        assert [label for label, _ in combo.items] == provider_model_names(provider)

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    def test_selects_the_default_model_when_none_requested(
        self, provider: AIProvider
    ) -> None:
        combo = _FakeCombo()

        populate_model_combo(combo, provider)

        assert combo.currentText() == provider_default_model(provider)

    def test_keeps_a_still_valid_selection_across_repopulation(self) -> None:
        combo = _FakeCombo()
        keep = provider_model_names(AIProvider.ANTHROPIC)[-1]

        populate_model_combo(combo, AIProvider.ANTHROPIC, keep)

        assert combo.currentText() == keep

    def test_falls_back_to_the_default_for_a_foreign_selection(self) -> None:
        """Switching providers must not leave the previous provider's model."""
        combo = _FakeCombo()

        populate_model_combo(combo, AIProvider.OLLAMA, "gpt-4o")

        assert combo.currentText() == provider_default_model(AIProvider.OLLAMA)

    def test_clears_the_previous_providers_models(self) -> None:
        combo = _FakeCombo()

        populate_model_combo(combo, AIProvider.OPENAI)
        populate_model_combo(combo, AIProvider.OLLAMA)

        assert [label for label, _ in combo.items] == provider_model_names(
            AIProvider.OLLAMA
        )
        assert combo.clear_calls == 2


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
        assert hasattr(panel, "_provider_icon"), (
            "_provider_icon must be created in _add_header_title_widgets"
        )

    def test_has_model_label(self) -> None:
        """Panel has _model_label QLabel in the header."""
        panel = self._get_panel()
        assert hasattr(panel, "_model_label"), (
            "_model_label must be created in _add_header_title_widgets"
        )

    def test_has_mode_combo(self) -> None:
        """Panel has _mode_combo in the header."""
        panel = self._get_panel()
        assert hasattr(panel, "_mode_combo"), (
            "_mode_combo must be created in _add_header_mode_and_status"
        )

    def test_has_status_label(self) -> None:
        """Panel has _status_label in the header."""
        panel = self._get_panel()
        assert hasattr(panel, "_status_label"), (
            "_status_label must be created in _add_header_mode_and_status"
        )
