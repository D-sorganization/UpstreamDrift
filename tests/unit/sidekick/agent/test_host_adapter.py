"""Focused coverage for Sidekick host action adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from sidekick.agent.host_adapter import (
    HostAdapter,
    HostCapability,
    HostInvocationResult,
)

pytestmark = pytest.mark.unit


class _Port:
    host_id = "test"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_capabilities(self) -> Sequence[HostCapability]:
        return (
            HostCapability(
                capability_id="host.test.open",
                summary="Open a thing.",
                params_schema={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            ),
            HostCapability(
                capability_id="host.test.delete",
                summary="Delete a thing.",
                params_schema={"type": "object", "properties": {}},
                requires_confirmation=True,
            ),
        )

    def invoke(
        self, capability_id: str, params: Mapping[str, Any]
    ) -> HostInvocationResult:
        self.calls.append((capability_id, dict(params)))
        return HostInvocationResult(
            ok=True, value={"id": capability_id}, metadata={"seen": True}
        )


def test_host_capability_and_result_validate_contracts() -> None:
    with pytest.raises(ValueError, match="host"):
        HostCapability("bad.open", "Open", {"type": "object"})
    with pytest.raises(ValueError, match="summary"):
        HostCapability("host.test.open", "", {"type": "object"})
    with pytest.raises(ValueError, match="error"):
        HostInvocationResult(ok=False)


def test_adapter_exposes_capabilities_and_invokes_port() -> None:
    port = _Port()
    adapter = HostAdapter(port=port)

    descriptors = adapter.describe()
    result = adapter.invoke("host.test.open", {"name": "alpha"})

    assert [descriptor.action_id for descriptor in descriptors] == [
        "host.test.open",
        "host.test.delete",
    ]
    assert descriptors[1].side_effects == "destructive"
    assert result.ok is True
    assert result.metadata == {"seen": True}
    assert port.calls == [("host.test.open", {"name": "alpha"})]


def test_adapter_requires_confirmation_and_strips_private_flag() -> None:
    port = _Port()
    adapter = HostAdapter(port=port)

    blocked = adapter.invoke("host.test.delete", {})
    allowed = adapter.invoke("host.test.delete", {"_confirmed": True, "extra": 1})

    assert blocked.ok is False
    assert "requires confirmation" in str(blocked.error)
    assert allowed.ok is True
    assert port.calls == [("host.test.delete", {"extra": 1})]


def test_adapter_handles_absent_unknown_and_bad_port_results() -> None:
    adapter = HostAdapter()
    assert adapter.invoke("host.test.open", {}).ok is False
    assert adapter.describe() == ()

    port = _Port()
    adapter.set_port(port)
    assert adapter.invoke("host.test.missing", {}).ok is False

    port.invoke = lambda capability_id, params: object()  # type: ignore[method-assign]
    result = adapter.invoke("host.test.open", {})

    assert result.ok is False
    assert "expected HostInvocationResult" in str(result.error)
