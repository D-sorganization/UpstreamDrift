"""Tests for sidekick.agent.subtab_adapter (epic #5967 / S3 / #5972).

TDD: contract pinned before implementation. The adapter exposes the
Sidekick subtab surface (tools_sidebar) through SidekickActionService
without any direct PyQt6 calls — all UI side effects route through a
SubtabActionPort Protocol injected at construction. This keeps tests
fast, headless, and decoupled from the actual widget code (which may
be in an inconsistent state during multi-agent edits).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from sidekick.agent.action_service import SidekickActionService
from sidekick.agent.subtab_adapter import (
    CalculatorRun,
    StateProfile,
    SubtabActionPort,
    SubtabAdapter,
    WorkspaceSnapshot,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fake port — captures every call without touching real widgets
# ---------------------------------------------------------------------------


class _FakePort:
    """In-memory ``SubtabActionPort`` for unit tests."""

    def __init__(
        self, *, available_tabs: Sequence[str] = ("calculator", "workspace")
    ) -> None:
        self._tabs: list[str] = list(available_tabs)
        self._visible: set[str] = set(self._tabs)
        self._active: str | None = self._tabs[0] if self._tabs else None
        self._workspace: dict[str, Any] = {"y": 99}
        self._profiles: dict[str, dict[str, Any]] = {}
        # Records every method call as (name, args).
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    # SubtabActionPort surface ---------------------------------------------

    def list_tabs(self) -> Sequence[str]:
        self.calls.append(("list_tabs", ()))
        return tuple(self._tabs)

    def active_tab(self) -> str | None:
        return self._active

    def focus(self, tab_id: str) -> None:
        self.calls.append(("focus", (tab_id,)))
        if tab_id not in self._tabs:
            raise KeyError(tab_id)
        self._active = tab_id

    def set_visible(self, tab_id: str, visible: bool) -> None:
        self.calls.append(("set_visible", (tab_id, visible)))
        if tab_id not in self._tabs:
            raise KeyError(tab_id)
        if visible:
            self._visible.add(tab_id)
        else:
            self._visible.discard(tab_id)

    def workspace_snapshot(self) -> WorkspaceSnapshot:
        self.calls.append(("workspace_snapshot", ()))
        return WorkspaceSnapshot(values=dict(self._workspace))

    def workspace_set_variable(self, name: str, value: Any) -> Any:
        self.calls.append(("workspace_set_variable", (name, value)))
        prior = self._workspace.get(name)
        self._workspace[name] = value
        return prior

    def calculator_run(
        self, calculator_id: str, inputs: Mapping[str, Any]
    ) -> CalculatorRun:
        self.calls.append(("calculator_run", (calculator_id, dict(inputs))))
        if calculator_id == "broken":
            raise RuntimeError("simulated failure")
        # Mirror Calculator protocol output shape.
        return CalculatorRun(
            values={"answer": float(inputs.get("x", 0)) * 2.0},
            units={"answer": "dimensionless"},
            warnings=(),
            metadata={"calculator_id": calculator_id},
        )

    def state_profile_save(self, name: str, payload: Mapping[str, Any]) -> None:
        self.calls.append(("state_profile_save", (name, dict(payload))))
        self._profiles[name] = dict(payload)

    def state_profile_load(self, name: str) -> StateProfile:
        self.calls.append(("state_profile_load", (name,)))
        payload = self._profiles.get(name)
        if payload is None:
            raise KeyError(name)
        return StateProfile(name=name, payload=dict(payload))


# ---------------------------------------------------------------------------
# Construction + descriptor surface
# ---------------------------------------------------------------------------


def test_adapter_namespace_is_subtab() -> None:
    adapter = SubtabAdapter(port=_FakePort())
    assert adapter.namespace == "subtab"


def test_adapter_rejects_non_port() -> None:
    with pytest.raises(TypeError):
        SubtabAdapter(port="not-a-port")  # type: ignore[arg-type]


def test_adapter_publishes_all_actions() -> None:
    adapter = SubtabAdapter(port=_FakePort())
    ids = {d.action_id for d in adapter.describe()}
    assert ids == {
        "subtab.list",
        "subtab.focus",
        "subtab.show",
        "subtab.hide",
        "subtab.calculator.run",
        "subtab.workspace.snapshot",
        "subtab.workspace.set_variable",
        "subtab.state_profile.save",
        "subtab.state_profile.load",
    }


def test_side_effects_classification() -> None:
    adapter = SubtabAdapter(port=_FakePort())
    by_id = {d.action_id: d for d in adapter.describe()}
    assert by_id["subtab.list"].side_effects == "read"
    assert by_id["subtab.workspace.snapshot"].side_effects == "read"
    assert by_id["subtab.focus"].side_effects == "write"
    assert by_id["subtab.workspace.set_variable"].side_effects == "write"
    assert by_id["subtab.state_profile.save"].side_effects == "write"
    assert by_id["subtab.state_profile.load"].side_effects == "write"


def test_reversible_flags_match_actual_undo_support() -> None:
    adapter = SubtabAdapter(port=_FakePort())
    by_id = {d.action_id: d for d in adapter.describe()}
    assert by_id["subtab.focus"].reversible is True
    assert by_id["subtab.show"].reversible is True
    assert by_id["subtab.hide"].reversible is True
    assert by_id["subtab.workspace.set_variable"].reversible is True
    assert by_id["subtab.state_profile.save"].reversible is True
    # calculator.run is not reversible — a calculation has no inverse.
    assert by_id["subtab.calculator.run"].reversible is False


# ---------------------------------------------------------------------------
# Action invocation via the service
# ---------------------------------------------------------------------------


def _build_service(
    port: SubtabActionPort,
) -> tuple[SidekickActionService, SubtabAdapter]:
    service = SidekickActionService()
    adapter = SubtabAdapter(port=port)
    service.register(adapter)
    return service, adapter


def test_subtab_list_returns_port_tabs() -> None:
    port = _FakePort()
    service, _ = _build_service(port)
    result = service.invoke("subtab.list", {})
    assert result.ok is True
    assert result.value == ["calculator", "workspace"]
    assert ("list_tabs", ()) in port.calls


def test_subtab_focus_records_undo_token() -> None:
    port = _FakePort(available_tabs=("a", "b"))
    service, _ = _build_service(port)
    result = service.invoke("subtab.focus", {"tab_id": "b"})
    assert result.ok is True
    assert result.undo_token  # opaque; just must be present
    assert port.active_tab() == "b"


def test_subtab_focus_unknown_tab_returns_error() -> None:
    port = _FakePort(available_tabs=("a",))
    service, _ = _build_service(port)
    result = service.invoke("subtab.focus", {"tab_id": "nonexistent"})
    assert result.ok is False
    assert "nonexistent" in (result.error or "")


def test_subtab_show_and_hide_round_trip() -> None:
    port = _FakePort(available_tabs=("a", "b"))
    service, _ = _build_service(port)
    r1 = service.invoke("subtab.hide", {"tab_id": "a"})
    assert r1.ok is True
    r2 = service.invoke("subtab.show", {"tab_id": "a"})
    assert r2.ok is True


def test_calculator_run_returns_calculation_result_shape() -> None:
    port = _FakePort()
    service, _ = _build_service(port)
    result = service.invoke(
        "subtab.calculator.run",
        {"calculator_id": "doubler", "inputs": {"x": 21}},
    )
    assert result.ok is True
    # Same shape as sidekick.protocols.CalculationResult.
    assert result.value["values"] == {"answer": 42.0}
    assert result.value["units"] == {"answer": "dimensionless"}


def test_calculator_run_failure_translates_to_error_result() -> None:
    port = _FakePort()
    service, _ = _build_service(port)
    result = service.invoke(
        "subtab.calculator.run",
        {"calculator_id": "broken", "inputs": {}},
    )
    assert result.ok is False
    assert result.error is not None


def test_workspace_snapshot_returns_values_dict() -> None:
    port = _FakePort()
    service, _ = _build_service(port)
    result = service.invoke("subtab.workspace.snapshot", {})
    assert result.ok is True
    assert result.value == {"y": 99}


def test_workspace_set_variable_emits_undo_token() -> None:
    port = _FakePort()
    service, _ = _build_service(port)
    result = service.invoke(
        "subtab.workspace.set_variable",
        {"name": "z", "value": 7},
    )
    assert result.ok is True
    assert result.undo_token is not None


def test_state_profile_save_and_load_round_trip() -> None:
    port = _FakePort()
    service, _ = _build_service(port)
    save = service.invoke(
        "subtab.state_profile.save",
        {"name": "trial", "payload": {"x": 1}},
    )
    assert save.ok is True
    load = service.invoke("subtab.state_profile.load", {"name": "trial"})
    assert load.ok is True
    assert load.value == {"x": 1}


def test_state_profile_load_unknown_returns_error() -> None:
    port = _FakePort()
    service, _ = _build_service(port)
    result = service.invoke("subtab.state_profile.load", {"name": "missing"})
    assert result.ok is False
    assert "missing" in (result.error or "")


# ---------------------------------------------------------------------------
# DbC / LOD: no direct widget access from outside
# ---------------------------------------------------------------------------


def test_invalid_params_do_not_reach_port() -> None:
    port = _FakePort()
    service, _ = _build_service(port)
    # missing required tab_id
    result = service.invoke("subtab.focus", {})
    assert result.ok is False
    assert all(call[0] != "focus" for call in port.calls)


def test_port_protocol_runtime_checkable() -> None:
    port = _FakePort()
    assert isinstance(port, SubtabActionPort)
    assert not isinstance("string", SubtabActionPort)


# ---------------------------------------------------------------------------
# Result dataclass invariants
# ---------------------------------------------------------------------------


def test_calculator_run_dataclass_rejects_negative_values_for_units() -> None:
    # The dataclass should not invent invariants the adapter doesn't have;
    # but it should enforce that values/units keys agree if units present.
    with pytest.raises(ValueError, match="units"):
        CalculatorRun(
            values={"a": 1.0},
            units={"b": "kg"},  # 'b' not in values
            warnings=(),
            metadata={},
        )


def test_workspace_snapshot_is_frozen() -> None:
    import dataclasses

    snap = WorkspaceSnapshot(values={"a": 1})
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.values = {"b": 2}  # type: ignore[misc]
