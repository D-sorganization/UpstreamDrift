"""Focused coverage for Sidekick subtab adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from sidekick.agent.subtab_adapter import (
    CalculatorRun,
    StateProfile,
    SubtabAdapter,
    WorkspaceSnapshot,
)

pytestmark = pytest.mark.unit


class _Port:
    def __init__(self) -> None:
        self.tabs = ["workspace", "calculator"]
        self.active = "workspace"
        self.visible: dict[str, bool] = dict.fromkeys(self.tabs, True)
        self.workspace = {"mass": 1.0}
        self.profiles: dict[str, dict[str, Any]] = {}
        self.focused: list[str] = []

    def list_tabs(self) -> Sequence[str]:
        return tuple(self.tabs)

    def active_tab(self) -> str | None:
        return self.active

    def focus(self, tab_id: str) -> None:
        if tab_id not in self.tabs:
            raise KeyError(tab_id)
        self.active = tab_id
        self.focused.append(tab_id)

    def set_visible(self, tab_id: str, visible: bool) -> None:
        if tab_id not in self.tabs:
            raise KeyError(tab_id)
        self.visible[tab_id] = visible

    def workspace_snapshot(self) -> WorkspaceSnapshot:
        return WorkspaceSnapshot(values=dict(self.workspace))

    def workspace_set_variable(self, name: str, value: Any) -> Any:
        prior = self.workspace.get(name)
        self.workspace[name] = value
        return prior

    def calculator_run(
        self, calculator_id: str, inputs: Mapping[str, Any]
    ) -> CalculatorRun:
        if calculator_id == "missing":
            raise KeyError(calculator_id)
        return CalculatorRun(values={"out": float(inputs["x"])}, units={"out": "kg"})

    def state_profile_save(self, name: str, payload: Mapping[str, Any]) -> None:
        self.profiles[name] = dict(payload)

    def state_profile_load(self, name: str) -> StateProfile:
        if name not in self.profiles:
            raise KeyError(name)
        return StateProfile(name=name, payload=dict(self.profiles[name]))

    def state_profile_delete(self, name: str) -> None:
        self.profiles.pop(name, None)


def test_value_objects_validate_and_project() -> None:
    run = CalculatorRun(values={"out": 1.0}, units={"out": "kg"}, warnings=("warn",))
    assert run.as_dict()["warnings"] == ["warn"]
    with pytest.raises(ValueError, match="units keys"):
        CalculatorRun(values={"out": 1.0}, units={"other": "kg"})


def test_describe_and_core_tab_actions() -> None:
    port = _Port()
    adapter = SubtabAdapter(port=port)

    assert [descriptor.action_id for descriptor in adapter.describe()] == [
        "subtab.list",
        "subtab.focus",
        "subtab.show",
        "subtab.hide",
        "subtab.calculator.run",
        "subtab.workspace.snapshot",
        "subtab.workspace.set_variable",
        "subtab.state_profile.save",
        "subtab.state_profile.load",
        "subtab.state_profile.delete",
    ]
    assert adapter.invoke("subtab.list", {}).value == ["workspace", "calculator"]

    focused = adapter.invoke("subtab.focus", {"tab_id": "calculator"})
    hidden = adapter.invoke("subtab.hide", {"tab_id": "calculator"})
    shown = adapter.invoke("subtab.show", {"tab_id": "calculator"})

    assert focused.ok is True
    assert focused.metadata["_undo"]["params"] == {"tab_id": "workspace"}
    assert hidden.metadata["_undo"]["action_id"] == "subtab.show"
    assert shown.metadata["_undo"]["action_id"] == "subtab.hide"


def test_workspace_calculator_and_profile_actions() -> None:
    port = _Port()
    adapter = SubtabAdapter(port=port)

    assert adapter.invoke("subtab.workspace.snapshot", {}).value == {"mass": 1.0}
    set_result = adapter.invoke(
        "subtab.workspace.set_variable", {"name": "mass", "value": 2.0}
    )
    calc_result = adapter.invoke(
        "subtab.calculator.run", {"calculator_id": "calc", "inputs": {"x": 3}}
    )
    save_new = adapter.invoke(
        "subtab.state_profile.save", {"name": "p1", "payload": {"a": 1}}
    )
    save_existing = adapter.invoke(
        "subtab.state_profile.save", {"name": "p1", "payload": {"a": 2}}
    )
    loaded = adapter.invoke("subtab.state_profile.load", {"name": "p1"})
    deleted = adapter.invoke("subtab.state_profile.delete", {"name": "p1"})

    assert set_result.metadata["_undo"]["params"] == {"name": "mass", "value": 1.0}
    assert calc_result.value["values"] == {"out": 3.0}
    assert save_new.metadata["_undo"]["action_id"] == "subtab.state_profile.delete"
    assert save_existing.metadata["_undo"]["params"] == {
        "name": "p1",
        "payload": {"a": 1},
    }
    assert loaded.value == {"a": 2}
    assert deleted.ok is True


def test_error_paths_are_action_results() -> None:
    adapter = SubtabAdapter(port=_Port())

    assert adapter.invoke("subtab.missing", {}).ok is False
    assert adapter.invoke("subtab.focus", {"tab_id": "missing"}).ok is False
    assert adapter.invoke("subtab.workspace.set_variable", {"name": "mass"}).ok is False
    assert (
        adapter.invoke(
            "subtab.calculator.run", {"calculator_id": "missing", "inputs": {}}
        ).ok
        is False
    )
    assert adapter.invoke("subtab.state_profile.load", {"name": "missing"}).ok is False
