"""AIP JSON-RPC must not claim success without work (#14940)."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from src.api.aip.dispatcher import dispatch
from src.api.aip.errors import AIP_METHOD_NOT_IMPLEMENTED, AIP_NOT_IMPLEMENTED_CODE
from src.api.aip.methods import create_registry, methods_passing_self_check
from src.shared.python.engine_core.engine_manager import EngineManager

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "method",
    [
        "simulation.start",
        "simulation.stop",
        "simulation.step",
        "simulation.status",
        "analysis.export",
        "model.query",
    ],
)
@pytest.mark.asyncio
async def test_stub_methods_are_not_implemented(method: str) -> None:
    p = {"engine_type": "mujoco"} if method == "simulation.start" else {}
    res = await dispatch(
        create_registry(), {"jsonrpc": "2.0", "method": method, "params": p, "id": 1}
    )
    assert res and "error" in res
    assert res["error"]["code"] == AIP_METHOD_NOT_IMPLEMENTED
    assert res["error"]["data"]["code"] == AIP_NOT_IMPLEMENTED_CODE
    assert res["error"]["data"]["method"] == method


def test_capabilities_lists_only_self_checked_methods() -> None:
    registry = create_registry()
    handler = registry.get_method("system.capabilities")
    assert handler is not None
    supported = handler()["supported_methods"]
    assert {"system.ping", "model.list"}.issubset(supported)
    assert "simulation.start" not in supported
    assert supported == methods_passing_self_check(registry)


def test_aip_methods_do_not_use_hasattr_or_missing_engine_manager_api() -> None:
    source = (Path(__file__).resolve().parents[3] / "src/api/aip/methods.py").read_text(
        encoding="utf-8"
    )
    assert "get_active_engine" not in source
    assert "hasattr(engine_manager" not in source
    assert "hasattr(engine," not in source

    members = inspect.getmembers(EngineManager, inspect.isroutine)
    public_mgr = {m for m, _ in members if not m.startswith("_")}
    assert not {"get_active_engine", "load_engine"} & public_mgr
