"""Tests for the CLI provider registry and the unified AllProviders view."""

from __future__ import annotations

import pytest

from src.shared.python.ai.cli_providers import registry as registry_mod
from src.shared.python.ai.cli_providers.contracts import CliProviderDescriptor
from src.shared.python.ai.cli_providers.registry import (
    AllProviders,
    CliProviderRegistry,
)


def _fake_descriptors() -> list[CliProviderDescriptor]:
    return [
        CliProviderDescriptor(
            id="claude-cli", name="Claude CLI", executable_path="/x/claude"
        ),
        CliProviderDescriptor(
            id="codex-cli", name="Codex CLI", executable_path="/x/codex"
        ),
        CliProviderDescriptor(
            id="cline",
            name="Cline",
            transport="socket",
            working_dir_aware=False,
        ),
    ]


def test_registry_lists_discovered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_mod, "discover_cli_providers", _fake_descriptors)
    reg = CliProviderRegistry()
    ids = [d.id for d in reg.list()]
    assert ids == ["claude-cli", "codex-cli", "cline"]


def test_registry_get_returns_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_mod, "discover_cli_providers", _fake_descriptors)
    reg = CliProviderRegistry()
    desc = reg.get("codex-cli")
    assert desc is not None
    assert desc.name == "Codex CLI"


def test_registry_get_missing_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_mod, "discover_cli_providers", _fake_descriptors)
    assert CliProviderRegistry().get("not-real") is None


def test_registry_get_empty_id_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_mod, "discover_cli_providers", _fake_descriptors)
    with pytest.raises(ValueError):
        CliProviderRegistry().get("")


def test_all_providers_merges_http_and_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_mod, "discover_cli_providers", _fake_descriptors)
    http = [("openai", "OpenAI"), ("anthropic", "Anthropic")]
    view = AllProviders(http_providers=http)
    entries = view.list()
    categories = [e.category for e in entries]
    assert categories[:2] == [AllProviders.HTTP_CATEGORY] * 2
    assert categories[2:] == [AllProviders.CLI_CATEGORY] * 3


def test_all_providers_cli_entries_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_mod, "discover_cli_providers", _fake_descriptors)
    view = AllProviders(http_providers=[("openai", "OpenAI")])
    cli_only = view.cli_entries()
    assert {e.id for e in cli_only} == {"claude-cli", "codex-cli", "cline"}


def test_all_providers_namespaces_collisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        registry_mod,
        "discover_cli_providers",
        lambda: [
            CliProviderDescriptor(
                id="anthropic", name="Anthropic CLI", executable_path="/x"
            )
        ],
    )
    view = AllProviders(http_providers=[("anthropic", "Anthropic")])
    entries = view.list()
    cli_entry = next(e for e in entries if e.category == "cli")
    assert cli_entry.id == "cli:anthropic"


def test_set_http_providers_replaces_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_mod, "discover_cli_providers", list)
    view = AllProviders()
    view.set_http_providers([("openai", "OpenAI")])
    assert [e.id for e in view.list()] == ["openai"]
