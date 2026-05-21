"""Tests for the Integrations Health dashboard (TDD — written before implementation).

Issue #5643: feat(launcher): integrations health dashboard — one pane of glass
for clients, MCP, CLI, API.
"""

from __future__ import annotations

import os
from datetime import datetime
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# 1. IntegrationRecord dataclass shape
# ---------------------------------------------------------------------------


def test_integration_record_has_required_fields() -> None:
    """IntegrationRecord dataclass must carry: kind, name, status, last_checked, last_error."""
    from src.launchers.integrations_health_data import IntegrationRecord

    rec = IntegrationRecord(
        kind="cli",
        name="claude",
        status="healthy",
        last_checked=datetime(2026, 5, 16, 12, 0, 0),
        last_error=None,
    )

    assert rec.kind == "cli"
    assert rec.name == "claude"
    assert rec.status == "healthy"
    assert rec.last_checked == datetime(2026, 5, 16, 12, 0, 0)
    assert rec.last_error is None


# ---------------------------------------------------------------------------
# 2. aggregate_cli_agents — probes known CLI tools
# ---------------------------------------------------------------------------


def test_aggregate_cli_agents_probes_known_tools() -> None:
    """aggregate_cli_agents() returns one record per known CLI tool."""
    from src.launchers.integrations_health_data import aggregate_cli_agents

    # Pretend only 'gh' is on PATH
    def fake_which(name: str) -> str | None:
        return "/usr/bin/gh" if name == "gh" else None

    with patch("shutil.which", side_effect=fake_which):
        records = aggregate_cli_agents()

    # Must include records for all known tools, not just the found ones
    names = [r.name for r in records]
    for tool in ("claude", "codex", "cline", "gh"):
        assert tool in names, f"Expected '{tool}' in CLI records"

    # Kind must be "cli" for every record
    assert all(r.kind == "cli" for r in records)

    # The 'gh' record should be healthy; others should not be healthy
    gh_rec = next(r for r in records if r.name == "gh")
    assert gh_rec.status == "healthy"

    claude_rec = next(r for r in records if r.name == "claude")
    assert claude_rec.status != "healthy"


# ---------------------------------------------------------------------------
# 3. aggregate_api_adapters — checks known env vars
# ---------------------------------------------------------------------------


def test_aggregate_api_adapters_checks_env_vars() -> None:
    """aggregate_api_adapters() returns one record per known API provider."""
    from src.launchers.integrations_health_data import aggregate_api_adapters

    env_clear = dict.fromkeys(
        ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "OLLAMA_HOST")
    )
    with patch.dict(os.environ, {}, clear=False):
        for k in env_clear:
            os.environ.pop(k, None)
        records = aggregate_api_adapters()

    provider_names = [r.name for r in records]
    for provider in ("anthropic", "openai", "gemini", "ollama"):
        assert provider in provider_names, f"Expected '{provider}' in API records"

    assert all(r.kind == "api" for r in records)


# ---------------------------------------------------------------------------
# 4. aggregate_api_adapters — "healthy" when env var is set
# ---------------------------------------------------------------------------


def test_aggregate_api_adapters_present_when_env_var_set() -> None:
    """When an API key env var is set the corresponding record is 'configured'."""
    from src.launchers.integrations_health_data import aggregate_api_adapters

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-1234"}, clear=False):
        records = aggregate_api_adapters()

    anthropic_rec = next(r for r in records if r.name == "anthropic")
    assert anthropic_rec.status in (
        "healthy",
        "configured",
    ), f"Expected 'healthy' or 'configured', got '{anthropic_rec.status}'"


# ---------------------------------------------------------------------------
# 5. aggregate_api_adapters — "unconfigured" when env var absent
# ---------------------------------------------------------------------------


def test_aggregate_api_adapters_missing_when_env_var_absent() -> None:
    """When an API key env var is absent, the record status is 'unconfigured'."""
    from src.launchers.integrations_health_data import aggregate_api_adapters

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENAI_API_KEY", None)
        records = aggregate_api_adapters()

    openai_rec = next(r for r in records if r.name == "openai")
    assert openai_rec.status == "unconfigured"


# ---------------------------------------------------------------------------
# 6. copy_diagnostics — no secrets in output
# ---------------------------------------------------------------------------


def test_diagnostics_export_contains_no_secrets() -> None:
    """copy_diagnostics() must not include raw API key values or Bearer tokens."""
    from src.launchers.integrations_health_data import (
        IntegrationRecord,
        copy_diagnostics,
    )

    secret_value = "sk-super-secret-key-that-must-not-appear"
    records = [
        IntegrationRecord(
            kind="api",
            name="openai",
            status="configured",
            last_checked=datetime(2026, 5, 16, 12, 0, 0),
            last_error=None,
            detail=f"Bearer {secret_value}",
        )
    ]
    output = copy_diagnostics(records)

    assert "Bearer " not in output, "copy_diagnostics must strip 'Bearer ' tokens"
    assert secret_value not in output, "copy_diagnostics must not expose raw key values"


# ---------------------------------------------------------------------------
# 7. copy_diagnostics — markdown format
# ---------------------------------------------------------------------------


def test_diagnostics_export_markdown_format() -> None:
    """copy_diagnostics() output starts with '# Integration Health'."""
    from src.launchers.integrations_health_data import (
        IntegrationRecord,
        copy_diagnostics,
    )

    records = [
        IntegrationRecord(kind="cli", name="gh", status="healthy"),
    ]
    output = copy_diagnostics(records)

    assert output.startswith("# Integration Health"), (
        f"Expected markdown header, got: {output[:60]!r}"
    )


# ---------------------------------------------------------------------------
# 8. collect_all — combines sub-aggregators
# ---------------------------------------------------------------------------


def test_collect_all_integrations_returns_list() -> None:
    """collect_all() returns a non-empty list combining all sub-aggregator results."""
    from src.launchers.integrations_health_data import collect_all

    def fake_cli() -> list:
        from src.launchers.integrations_health_data import IntegrationRecord

        return [IntegrationRecord(kind="cli", name="gh", status="healthy")]

    def fake_api() -> list:
        from src.launchers.integrations_health_data import IntegrationRecord

        return [IntegrationRecord(kind="api", name="openai", status="unconfigured")]

    def fake_mcp() -> list:
        from src.launchers.integrations_health_data import IntegrationRecord

        return [IntegrationRecord(kind="mcp", name="test-server", status="unknown")]

    with (
        patch(
            "src.launchers.integrations_health_data.aggregate_cli_agents",
            side_effect=fake_cli,
        ),
        patch(
            "src.launchers.integrations_health_data.aggregate_api_adapters",
            side_effect=fake_api,
        ),
        patch(
            "src.launchers.integrations_health_data.aggregate_mcp_servers",
            side_effect=fake_mcp,
        ),
    ):
        result = collect_all()

    assert isinstance(result, list)
    assert len(result) == 3

    kinds = {r.kind for r in result}
    assert kinds == {"cli", "api", "mcp"}


# ---------------------------------------------------------------------------
# 9. DbC — copy_diagnostics raises TypeError for non-list input
# ---------------------------------------------------------------------------


def test_copy_diagnostics_raises_type_error_for_non_list() -> None:
    """copy_diagnostics raises TypeError when records is not a list (DbC)."""
    from src.launchers.integrations_health_data import copy_diagnostics

    with pytest.raises(TypeError, match="records"):
        copy_diagnostics("not a list")  # type: ignore[arg-type]
