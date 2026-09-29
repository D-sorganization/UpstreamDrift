"""Tests for the Sidekick canonical-core tool adapter (#6811 / CC-38)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from sidekick.agent import (
    CANONICAL_ACTION_IDS,
    CanonicalOperationResult,
    CanonicalToolAdapter,
    MemoryActionAudit,
    SidekickActionPolicy,
    SidekickActionService,
)

pytestmark = pytest.mark.unit


class _CanonicalPort:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def configure(self, request: Mapping[str, Any]) -> CanonicalOperationResult:
        self.calls.append(("configure", dict(request)))
        return CanonicalOperationResult(
            ok=True,
            value={"configured": request["request"]},
        )

    def validate(self, request: Mapping[str, Any]) -> CanonicalOperationResult:
        self.calls.append(("validate", dict(request)))
        payload = request["payload"]
        if payload.get("units") != "SI":
            return CanonicalOperationResult(ok=False, error="units must be SI")
        return CanonicalOperationResult(ok=True, value={"valid": True})

    def run(self, request: Mapping[str, Any]) -> CanonicalOperationResult:
        self.calls.append(("run", dict(request)))
        return CanonicalOperationResult(
            ok=True,
            value={"run_id": "run-1", "status": "submitted"},
            provenance={
                "engine": "canonical-core",
                "convention": "canonical-v2",
                "units": {"length": "m", "time": "s"},
            },
        )

    def compare(self, request: Mapping[str, Any]) -> CanonicalOperationResult:
        self.calls.append(("compare", dict(request)))
        return CanonicalOperationResult(ok=True, value={"passed": True})

    def interpret(self, request: Mapping[str, Any]) -> CanonicalOperationResult:
        self.calls.append(("interpret", dict(request)))
        return CanonicalOperationResult(ok=True, value={"summary": "stable"})


def _service(
    port: _CanonicalPort,
    *,
    policy: SidekickActionPolicy | None = None,
    audit: MemoryActionAudit | None = None,
) -> SidekickActionService:
    service = SidekickActionService(
        policy=policy,
        audit_sink=audit,
    )
    service.register(CanonicalToolAdapter(port=port))
    return service


def test_adapter_exposes_only_allowlisted_canonical_actions() -> None:
    port = _CanonicalPort()
    service = _service(port)
    action_ids = {descriptor.action_id for descriptor in service.list_actions()}

    assert action_ids == CANONICAL_ACTION_IDS
    assert all(action_id.startswith("canonical.") for action_id in action_ids)


def test_run_schema_and_side_effects_require_confirmable_request() -> None:
    port = _CanonicalPort()
    service = _service(port)
    descriptors = {
        descriptor.action_id: descriptor for descriptor in service.list_actions()
    }

    run_descriptor = descriptors["canonical.run"]

    assert run_descriptor.side_effects == "destructive"
    assert run_descriptor.params_schema["required"] == ["request"]
    assert run_descriptor.params_schema["properties"]["request"]["type"] == "object"
    assert run_descriptor.params_schema["properties"]["_confirmed"]["type"] == "boolean"


def test_schema_validation_blocks_malformed_validate_request() -> None:
    port = _CanonicalPort()
    service = _service(port)

    result = service.invoke(
        "canonical.validate",
        {"artifact_type": "state", "payload": "not-an-object"},
    )

    assert result.ok is False
    assert result.error is not None
    assert "params validation failed" in result.error
    assert port.calls == []


def test_validate_reports_canonical_contract_errors() -> None:
    port = _CanonicalPort()
    service = _service(port)

    result = service.invoke(
        "canonical.validate",
        {"artifact_type": "run_request", "payload": {"units": "imperial"}},
    )

    assert result.ok is False
    assert result.error == "units must be SI"
    assert port.calls == [
        (
            "validate",
            {"artifact_type": "run_request", "payload": {"units": "imperial"}},
        )
    ]


def test_dry_run_returns_preview_without_calling_port() -> None:
    port = _CanonicalPort()
    service = _service(port)

    result = service.invoke(
        "canonical.run",
        {"request": {"target": "trial-1"}},
        dry_run=True,
    )

    assert result.ok is True
    assert result.metadata["would_call"] == "canonical.run"
    assert port.calls == []


def test_run_requires_user_confirmation_at_handler_boundary() -> None:
    port = _CanonicalPort()
    service = _service(port)

    result = service.invoke("canonical.run", {"request": {"target": "trial-1"}})

    assert result.ok is False
    assert result.error is not None
    assert "_confirmed=True" in result.error
    assert port.calls == []


def test_policy_denies_unallowlisted_run_before_handler() -> None:
    port = _CanonicalPort()
    service = _service(port, policy=SidekickActionPolicy())

    result = service.invoke(
        "canonical.run",
        {"request": {"target": "trial-1"}, "_confirmed": True},
    )

    assert result.ok is False
    assert result.error is not None
    assert "forbidden by policy" in result.error
    assert port.calls == []


def test_confirmed_allowlisted_run_returns_provenance_and_audits() -> None:
    port = _CanonicalPort()
    audit = MemoryActionAudit()
    policy = SidekickActionPolicy(
        allow_destructive=frozenset({"canonical.run"}),
    )
    service = _service(port, policy=policy, audit=audit)

    result = service.invoke(
        "canonical.run",
        {"request": {"target": "trial-1"}, "_confirmed": True},
    )

    assert result.ok is True
    assert result.value["run_id"] == "run-1"
    assert result.metadata["provenance"]["convention"] == "canonical-v2"
    assert port.calls == [("run", {"request": {"target": "trial-1"}})]
    assert audit.records[-1].action_id == "canonical.run"


def test_read_actions_execute_without_destructive_allowlist() -> None:
    port = _CanonicalPort()
    service = _service(port, policy=SidekickActionPolicy())

    result = service.invoke("canonical.interpret", {"result": {"run_id": "run-1"}})

    assert result.ok is True
    assert result.value == {"summary": "stable"}
    assert port.calls == [("interpret", {"result": {"run_id": "run-1"}})]
