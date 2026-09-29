"""Tests that the chat header dropdown surfaces CLI Agents.

Qt is patched out — we verify the populator's calls against a mock
combo to keep the test headless-safe.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.shared.python.ai.cli_providers import registry as registry_mod
from src.shared.python.ai.cli_providers.contracts import CliProviderDescriptor


@pytest.fixture
def fake_panel(qapp):
    """Build a panel object that exposes only the populator method."""
    from src.shared.python.ai.gui.assistant_panel import AIAssistantPanel

    panel = AIAssistantPanel()
    return panel


def _patch_discovery(monkeypatch: pytest.MonkeyPatch, descriptors: list) -> None:
    monkeypatch.setattr(registry_mod, "discover_cli_providers", lambda: descriptors)


def test_populator_adds_cli_section_when_providers_found(
    monkeypatch: pytest.MonkeyPatch, fake_panel
) -> None:
    descriptors = [
        CliProviderDescriptor(
            id="claude-cli", name="Claude CLI", executable_path="/x/claude"
        ),
        CliProviderDescriptor(
            id="codex-cli", name="Codex CLI", executable_path="/x/codex"
        ),
    ]
    _patch_discovery(monkeypatch, descriptors)

    combo = MagicMock()
    combo.count = MagicMock(side_effect=[2, 3, 4, 5])

    fake_panel._populate_cli_provider_entries(combo)

    combo.insertSeparator.assert_called_once()
    added = [c.args for c in combo.addItem.call_args_list]
    labels = [args[0] for args in added]
    assert "— CLI Agents —" in labels
    assert "Claude CLI" in labels
    assert "Codex CLI" in labels


def test_populator_no_op_when_no_providers(
    monkeypatch: pytest.MonkeyPatch, fake_panel
) -> None:
    _patch_discovery(monkeypatch, [])
    combo = MagicMock()

    fake_panel._populate_cli_provider_entries(combo)

    combo.insertSeparator.assert_not_called()
    combo.addItem.assert_not_called()


def test_populator_rejects_none_combo(fake_panel) -> None:
    with pytest.raises(ValueError):
        fake_panel._populate_cli_provider_entries(None)


def test_populator_stores_entries_for_later_lookup(
    monkeypatch: pytest.MonkeyPatch, fake_panel
) -> None:
    descriptors = [
        CliProviderDescriptor(
            id="claude-cli", name="Claude CLI", executable_path="/x/claude"
        )
    ]
    _patch_discovery(monkeypatch, descriptors)

    combo = MagicMock()
    combo.count = MagicMock(return_value=4)
    fake_panel._populate_cli_provider_entries(combo)

    assert hasattr(fake_panel, "_cli_provider_entries")
    assert "Claude CLI" in fake_panel._cli_provider_entries
