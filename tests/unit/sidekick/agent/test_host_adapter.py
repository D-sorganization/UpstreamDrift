"""Tests for sidekick.agent.host_adapter (epic #5967 / S4 / #5973).

TDD: contract pinned before implementation. The host adapter lets
Sidekick call out to its embedding host (launcher, Pose Studio, etc.)
through a published HostActionPort Protocol — Sidekick never imports
launcher modules directly. Dependency direction is host → sidekick.agent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from sidekick.agent.action_service import SidekickActionService
from sidekick.agent.host_adapter import (
    HostActionPort,
    HostAdapter,
    HostCapability,
    HostInvocationResult,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fake port
# ---------------------------------------------------------------------------


class _FakeHostPort:
    """Pretends to be a launcher exposing two capabilities."""

    host_id = "fake-launcher"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._caps: tuple[HostCapability, ...] = (
            HostCapability(
                capability_id="host.launcher.open_tile",
                summary="Open a launcher tile by id.",
                params_schema={
                    "type": "object",
                    "properties": {"tile_id": {"type": "string"}},
                    "required": ["tile_id"],
                },
                requires_confirmation=False,
            ),
            HostCapability(
                capability_id="host.launcher.set_theme",
                summary="Switch the active theme.",
                params_schema={
                    "type": "object",
                    "properties": {"theme": {"type": "string"}},
                    "required": ["theme"],
                },
                requires_confirmation=True,
            ),
        )

    def list_capabilities(self) -> Sequence[HostCapability]:
        return self._caps

    def invoke(
        self, capability_id: str, params: Mapping[str, Any]
    ) -> HostInvocationResult:
        self.calls.append((capability_id, dict(params)))
        if capability_id == "host.launcher.open_tile":
            return HostInvocationResult(ok=True, value={"opened": params["tile_id"]})
        if capability_id == "host.launcher.set_theme":
            return HostInvocationResult(ok=True, value=None)
        return HostInvocationResult(ok=False, error=f"unknown {capability_id}")


# ---------------------------------------------------------------------------
# Capability DbC
# ---------------------------------------------------------------------------


def test_capability_id_must_be_host_namespace() -> None:
    with pytest.raises(ValueError, match="namespace"):
        HostCapability(
            capability_id="launcher.open_tile",  # missing host. prefix
            summary="s",
            params_schema={"type": "object"},
            requires_confirmation=False,
        )


def test_capability_summary_required() -> None:
    with pytest.raises(ValueError, match="summary"):
        HostCapability(
            capability_id="host.x.y",
            summary="",
            params_schema={"type": "object"},
            requires_confirmation=False,
        )


def test_capability_params_schema_must_be_jsonschema_shaped() -> None:
    with pytest.raises(ValueError, match="params_schema"):
        HostCapability(
            capability_id="host.x.y",
            summary="s",
            params_schema={"not": "a schema"},  # missing 'type'
            requires_confirmation=False,
        )


def test_invocation_result_ok_implies_no_error() -> None:
    with pytest.raises(ValueError, match="ok=True"):
        HostInvocationResult(ok=True, error="x")


def test_invocation_result_not_ok_requires_error() -> None:
    with pytest.raises(ValueError, match="error"):
        HostInvocationResult(ok=False)


# ---------------------------------------------------------------------------
# Adapter without a port — graceful absence
# ---------------------------------------------------------------------------


def test_adapter_without_port_publishes_no_actions() -> None:
    adapter = HostAdapter()
    assert adapter.describe() == ()


def test_adapter_without_port_returns_error_on_invoke() -> None:
    adapter = HostAdapter()
    result = adapter.invoke("host.launcher.open_tile", {"tile_id": "x"})
    assert result.ok is False
    assert "no host" in (result.error or "")


# ---------------------------------------------------------------------------
# Adapter with port — descriptors mirror capabilities
# ---------------------------------------------------------------------------


def test_adapter_publishes_one_action_per_capability() -> None:
    port = _FakeHostPort()
    adapter = HostAdapter(port=port)
    ids = [d.action_id for d in adapter.describe()]
    assert ids == [
        "host.launcher.open_tile",
        "host.launcher.set_theme",
    ]


def test_set_port_after_construction_is_supported() -> None:
    adapter = HostAdapter()
    assert adapter.describe() == ()
    adapter.set_port(_FakeHostPort())
    assert len(adapter.describe()) == 2


def test_set_port_to_none_clears_actions() -> None:
    adapter = HostAdapter(port=_FakeHostPort())
    adapter.set_port(None)
    assert adapter.describe() == ()


def test_destructive_confirmation_is_translated_to_side_effect() -> None:
    """A capability requiring confirmation is exposed as 'destructive'."""
    port = _FakeHostPort()
    adapter = HostAdapter(port=port)
    by_id = {d.action_id: d for d in adapter.describe()}
    assert by_id["host.launcher.open_tile"].side_effects == "write"
    assert by_id["host.launcher.set_theme"].side_effects == "destructive"


# ---------------------------------------------------------------------------
# Confirmation gating
# ---------------------------------------------------------------------------


def test_destructive_action_requires_confirmation() -> None:
    port = _FakeHostPort()
    service = SidekickActionService()
    service.register(HostAdapter(port=port))
    result = service.invoke("host.launcher.set_theme", {"theme": "macchiato"})
    assert result.ok is False
    assert result.error is not None
    assert "confirm" in result.error.lower()
    # Critical: port must NOT have been called.
    assert port.calls == []


def test_destructive_action_succeeds_with_confirmation() -> None:
    port = _FakeHostPort()
    service = SidekickActionService()
    service.register(HostAdapter(port=port))
    result = service.invoke(
        "host.launcher.set_theme",
        {"theme": "macchiato", "_confirmed": True},
    )
    assert result.ok is True
    # The _confirmed flag must not be forwarded to the port.
    assert port.calls == [("host.launcher.set_theme", {"theme": "macchiato"})]


def test_non_destructive_action_does_not_need_confirmation() -> None:
    port = _FakeHostPort()
    service = SidekickActionService()
    service.register(HostAdapter(port=port))
    result = service.invoke("host.launcher.open_tile", {"tile_id": "model_explorer"})
    assert result.ok is True
    assert result.value == {"opened": "model_explorer"}


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------


def test_unknown_capability_returns_error_result() -> None:
    class _OneCapPort:
        host_id = "p"

        def list_capabilities(self) -> Sequence[HostCapability]:
            return (
                HostCapability(
                    capability_id="host.p.known",
                    summary="s",
                    params_schema={"type": "object"},
                    requires_confirmation=False,
                ),
            )

        def invoke(
            self, capability_id: str, params: Mapping[str, Any]
        ) -> HostInvocationResult:
            return HostInvocationResult(ok=False, error="not me")

    adapter = HostAdapter(port=_OneCapPort())
    result = adapter.invoke("host.p.unknown", {})
    assert result.ok is False
    assert "unknown" in (result.error or "").lower()


def test_port_returns_malformed_value_is_translated() -> None:
    class _BadPort:
        host_id = "bad"

        def list_capabilities(self) -> Sequence[HostCapability]:
            return (
                HostCapability(
                    capability_id="host.bad.x",
                    summary="s",
                    params_schema={"type": "object"},
                    requires_confirmation=False,
                ),
            )

        def invoke(
            self, capability_id: str, params: Mapping[str, Any]
        ) -> HostInvocationResult:
            return "not an invocation result"  # type: ignore[return-value]

    adapter = HostAdapter(port=_BadPort())
    result = adapter.invoke("host.bad.x", {})
    assert result.ok is False
    assert "host" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# Protocol runtime check
# ---------------------------------------------------------------------------


def test_fake_port_satisfies_protocol() -> None:
    assert isinstance(_FakeHostPort(), HostActionPort)


def test_arbitrary_object_does_not_satisfy_protocol() -> None:
    assert not isinstance(object(), HostActionPort)


# ---------------------------------------------------------------------------
# LOD: sidekick must never import launcher modules
# ---------------------------------------------------------------------------


def test_host_adapter_module_imports_only_sidekick() -> None:
    """Static hygiene: the host_adapter module must not import any
    sidekick consumer (launchers, tools, etc.)."""
    import ast
    from pathlib import Path

    src = Path("src/shared/python/sidekick/agent/host_adapter.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("src.launchers"), (
                f"host_adapter must not import from launchers: {module}"
            )
            assert "tools." not in module, (
                f"host_adapter must not import tools: {module}"
            )
