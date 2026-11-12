"""Agent API capability inventory for ``ud-sim-mcp`` (#15020)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.shared.python.engine_core.engine_availability import EngineStatus
from src.sim_run.agent_api import (
    build_capability_inventory,
    read_auth_token,
    read_private_mode,
)
from src.sim_run.mcp import SecurityPolicy


@pytest.mark.unit
def test_missing_engine_reports_unavailable_not_failed() -> None:
    """Inventory must surface optional engines as unavailable, never failed."""
    with patch(
        "src.sim_run.agent_api.get_engine_status",
        return_value=EngineStatus.NOT_INSTALLED,
    ):
        with patch(
            "src.sim_run.agent_api.get_engine_error",
            return_value=ImportError("No module named 'pydrake'"),
        ):
            inventory = build_capability_inventory()
    drake = next(e for e in inventory["engines"] if e["engine_type"] == "drake")
    assert drake["runtime_status"] == "unavailable"
    assert "failed" not in drake["runtime_status"]
    assert "optional engine dependency" in (drake.get("unavailable_reason") or "")

    blob = str(inventory).lower()
    assert "failed" not in blob


@pytest.mark.unit
def test_canonical_auth_env_with_legacy_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UPSTREAMDRIFT_AGENT_API_TOKEN", raising=False)
    monkeypatch.delenv("UD_SIM_AUTH_TOKEN", raising=False)
    assert read_auth_token() is None

    monkeypatch.setenv("UD_SIM_AUTH_TOKEN", "legacy")
    assert read_auth_token() == "legacy"

    monkeypatch.setenv("UPSTREAMDRIFT_AGENT_API_TOKEN", "canonical")
    assert read_auth_token() == "canonical"


@pytest.mark.unit
def test_canonical_private_mode_with_legacy_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "UPSTREAMDRIFT_PRIVATE_MODE",
        "UD_PRIVATE_MODE",
    ):
        monkeypatch.delenv(name, raising=False)

    assert not read_private_mode()

    monkeypatch.setenv("UD_PRIVATE_MODE", "1")
    assert read_private_mode()

    monkeypatch.delenv("UD_PRIVATE_MODE", raising=False)
    monkeypatch.setenv("UPSTREAMDRIFT_PRIVATE_MODE", "yes")
    assert read_private_mode()


@pytest.mark.unit
def test_security_policy_from_env_honors_canonical_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UPSTREAMDRIFT_AGENT_API_TOKEN", "agent-secret")
    monkeypatch.setenv("UPSTREAMDRIFT_PRIVATE_MODE", "true")
    policy = SecurityPolicy.from_env()
    assert policy.auth_token == "agent-secret"
    assert policy.private_mode is True


@pytest.mark.unit
def test_inventory_generator_idempotent() -> None:
    first = build_capability_inventory()
    second = build_capability_inventory()
    assert first == second
