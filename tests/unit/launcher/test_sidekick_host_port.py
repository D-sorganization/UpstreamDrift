"""Headless tests for :mod:`src.launchers.sidekick_host_port`.

The port is exercised against a fake :class:`TabHost`, proving the
launcher-side Sidekick bridge needs no Qt to be validated. End-to-end
adapter behaviour (descriptors, undo metadata) is covered by the vendor
suite ``tests/unit/sidekick/agent/test_subtab_adapter.py``; here we pin
the host-port semantics.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from sidekick.agent.subtab_adapter import (
    CalculatorRun,
    SubtabActionPort,
    SubtabAdapter,
)

from src.launchers.sidekick_host_port import (
    LauncherSubtabPort,
    create_launcher_subtab_adapter,
)

pytestmark = [pytest.mark.unit]


class _FakeHost:
    """In-memory TabHost double mirroring EmbeddedHostWidget semantics."""

    def __init__(self) -> None:
        self.known = {"alpha", "beta", "gamma"}
        self.open: list[str] = []
        self.backgrounded: set[str] = set()
        self.popped: set[str] = set()
        self.active: str | None = None

    def open_tool_ids(self) -> list[str]:
        return list(self.open)

    def active_tool_id(self) -> str | None:
        return self.active

    def focus_tab(self, tool_id: str) -> None:
        if tool_id not in set(self.open) | self.backgrounded | self.popped:
            raise KeyError(tool_id)
        self.open_tab(tool_id)

    def open_tab(self, tool_id: str) -> int:
        if tool_id not in self.known:
            raise ValueError(f"tool_id {tool_id!r} is not registered")
        self.backgrounded.discard(tool_id)
        self.popped.discard(tool_id)
        if tool_id not in self.open:
            self.open.append(tool_id)
        self.active = tool_id
        return self.open.index(tool_id)

    def close_tab(self, target: int | str, *, destroy: bool = True) -> bool:
        if not isinstance(target, str) or target not in self.open:
            return False
        self.open.remove(target)
        if not destroy:
            self.backgrounded.add(target)
        return True

    def backgrounded_tools(self) -> set[str]:
        return set(self.backgrounded)

    def popped_out_tools(self) -> set[str]:
        return set(self.popped)


@pytest.fixture
def fake_host() -> _FakeHost:
    return _FakeHost()


@pytest.fixture
def port(fake_host: _FakeHost) -> LauncherSubtabPort:
    return LauncherSubtabPort(fake_host)


class TestConstruction:
    def test_satisfies_subtab_action_port(self, port: LauncherSubtabPort) -> None:
        assert isinstance(port, SubtabActionPort)

    def test_rejects_non_host(self) -> None:
        with pytest.raises(TypeError, match="TabHost"):
            LauncherSubtabPort("not-a-host")  # type: ignore[arg-type]

    def test_factory_builds_registered_adapter(self, fake_host: _FakeHost) -> None:
        adapter = create_launcher_subtab_adapter(fake_host)
        assert isinstance(adapter, SubtabAdapter)
        result = adapter.invoke("subtab.list", {})
        assert result.ok


class TestTabs:
    def test_list_tabs_open_then_hidden(
        self, fake_host: _FakeHost, port: LauncherSubtabPort
    ) -> None:
        fake_host.open = ["beta", "alpha"]
        fake_host.backgrounded = {"gamma"}
        assert list(port.list_tabs()) == ["beta", "alpha", "gamma"]

    def test_active_tab(self, fake_host: _FakeHost, port: LauncherSubtabPort) -> None:
        assert port.active_tab() is None
        fake_host.open_tab("alpha")
        assert port.active_tab() == "alpha"

    def test_focus_unknown_raises(self, port: LauncherSubtabPort) -> None:
        with pytest.raises(KeyError):
            port.focus("missing")

    def test_set_visible_true_opens_registered_tool(
        self, fake_host: _FakeHost, port: LauncherSubtabPort
    ) -> None:
        port.set_visible("alpha", True)
        assert fake_host.open == ["alpha"]

    def test_set_visible_true_unknown_raises_keyerror(
        self, port: LauncherSubtabPort
    ) -> None:
        # The host raises ValueError for unregistered ids; the Protocol
        # promises KeyError — the port must translate.
        with pytest.raises(KeyError):
            port.set_visible("missing", True)

    def test_set_visible_false_backgrounds_open_tab(
        self, fake_host: _FakeHost, port: LauncherSubtabPort
    ) -> None:
        fake_host.open_tab("alpha")
        port.set_visible("alpha", False)
        assert fake_host.open == []
        assert "alpha" in fake_host.backgrounded

    def test_set_visible_false_hidden_is_idempotent(
        self, fake_host: _FakeHost, port: LauncherSubtabPort
    ) -> None:
        fake_host.backgrounded = {"alpha"}
        port.set_visible("alpha", False)  # no raise
        assert "alpha" in fake_host.backgrounded

    def test_set_visible_false_unknown_raises(self, port: LauncherSubtabPort) -> None:
        with pytest.raises(KeyError):
            port.set_visible("missing", False)


class TestWorkspace:
    def test_snapshot_roundtrip(self, port: LauncherSubtabPort) -> None:
        assert port.workspace_set_variable("x", 1) is None
        assert port.workspace_set_variable("x", 2) == 1
        snap = port.workspace_snapshot()
        assert dict(snap.values) == {"x": 2}

    def test_injected_registry_is_used(self, fake_host: _FakeHost) -> None:
        class _Reg:
            def __init__(self) -> None:
                self.data: dict[str, Any] = {"seed": 42}

            def list(self) -> list[str]:
                return sorted(self.data)

            def get(self, name: str, default: Any = None) -> Any:
                return self.data.get(name, default)

            def set(self, name: str, value: Any) -> None:
                self.data[name] = value

        reg = _Reg()
        port = LauncherSubtabPort(fake_host, workspace=reg)
        assert dict(port.workspace_snapshot().values) == {"seed": 42}
        port.workspace_set_variable("y", "z")
        assert reg.data["y"] == "z"


class TestCalculators:
    @staticmethod
    def _double(inputs: Mapping[str, Any]) -> CalculatorRun:
        return CalculatorRun(values={"out": float(inputs["x"]) * 2})

    def test_unknown_calculator_raises_keyerror(self, port: LauncherSubtabPort) -> None:
        with pytest.raises(KeyError):
            port.calculator_run("nope", {})

    def test_registered_calculator_runs(self, fake_host: _FakeHost) -> None:
        port = LauncherSubtabPort(fake_host, calculators={"double": self._double})
        run = port.calculator_run("double", {"x": 3})
        assert run.values == {"out": 6.0}

    def test_register_calculator_validates(self, port: LauncherSubtabPort) -> None:
        port.register_calculator("double", self._double)
        with pytest.raises(ValueError, match="already registered"):
            port.register_calculator("double", self._double)
        with pytest.raises(ValueError, match="non-empty"):
            port.register_calculator("  ", self._double)


class TestStateProfiles:
    def test_in_memory_save_load_delete(self, port: LauncherSubtabPort) -> None:
        port.state_profile_save("p1", {"a": 1})
        profile = port.state_profile_load("p1")
        assert profile.payload == {"a": 1}
        port.state_profile_delete("p1")
        with pytest.raises(KeyError):
            port.state_profile_load("p1")
        port.state_profile_delete("p1")  # idempotent

    def test_persistence_roundtrip(self, fake_host: _FakeHost, tmp_path) -> None:
        path = tmp_path / "profiles.json"
        port = LauncherSubtabPort(fake_host, profile_path=path)
        port.state_profile_save("p1", {"a": 1})
        assert json.loads(path.read_text(encoding="utf-8")) == {"p1": {"a": 1}}

        reloaded = LauncherSubtabPort(fake_host, profile_path=path)
        assert reloaded.state_profile_load("p1").payload == {"a": 1}

    def test_corrupt_profile_store_is_ignored(
        self, fake_host: _FakeHost, tmp_path
    ) -> None:
        path = tmp_path / "profiles.json"
        path.write_text("not json", encoding="utf-8")
        port = LauncherSubtabPort(fake_host, profile_path=path)
        with pytest.raises(KeyError):
            port.state_profile_load("anything")
