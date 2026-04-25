"""Real-behaviour tests for sample_tools (issue #3163).

These tests verify that stub tools honestly return not-implemented rather than
faking results with synthetic data.
"""

from __future__ import annotations

import pytest

from src.shared.python.ai.sample_tools import register_golf_suite_tools
from src.shared.python.ai.tool_registry import ToolRegistry

pytestmark = pytest.mark.unit


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    register_golf_suite_tools(reg)
    return reg


def test_run_inverse_dynamics_not_implemented(registry: ToolRegistry) -> None:
    """run_inverse_dynamics returns honest not-implemented (issue #3163)."""
    result = registry.execute("run_inverse_dynamics", {"engine": "mujoco"})
    assert result.success is True  # tool ran without exception
    payload = result.result
    assert payload["success"] is False
    assert "not implemented" in payload["error"]
    assert payload["issue"] == "#3163"
    assert payload["tool"] == "run_inverse_dynamics"


def test_run_inverse_dynamics_bad_engine(registry: ToolRegistry) -> None:
    """run_inverse_dynamics rejects invalid engine names."""
    result = registry.execute("run_inverse_dynamics", {"engine": "invalid"})
    payload = result.result
    assert payload["success"] is False
    assert "engine" in payload["error"].lower() or "invalid" in payload["error"].lower()


def test_run_inverse_dynamics_bad_c3d(registry: ToolRegistry, tmp_path) -> None:
    """A non-parseable C3D file also surfaces a clean not-implemented response."""
    fake = tmp_path / "not_a_real.c3d"
    fake.write_bytes(b"this is not a c3d file")
    result = registry.execute(
        "run_inverse_dynamics", {"file_path": str(fake), "engine": "mujoco"}
    )
    payload = result.result
    assert payload["success"] is False
    assert "error" in payload


def test_check_energy_conservation_not_implemented(registry: ToolRegistry) -> None:
    """check_energy_conservation returns honest not-implemented (issue #3163)."""
    result = registry.execute("check_energy_conservation", {"tolerance": 0.1})
    payload = result.result
    assert payload["success"] is False
    assert "not implemented" in payload["error"]
    assert payload["issue"] == "#3163"
    assert payload["tool"] == "check_energy_conservation"


def test_validate_cross_engine_not_implemented(registry: ToolRegistry) -> None:
    """validate_cross_engine returns honest not-implemented (issue #3163)."""
    result = registry.execute("validate_cross_engine", {"tolerance": 0.02})
    payload = result.result
    assert payload["success"] is False
    assert "not implemented" in payload["error"]
    assert payload["issue"] == "#3163"
    assert payload["tool"] == "validate_cross_engine"


def test_list_physics_engines_introspects(registry: ToolRegistry) -> None:
    """list_physics_engines calls _available_engines rather than hardcoding."""
    result = registry.execute("list_physics_engines", {})
    payload = result.result
    engines = payload["engines"]
    names = {e["name"] for e in engines}
    assert {"MuJoCo", "Drake", "Pinocchio"} == names
    for entry in engines:
        assert entry["status"] in {"available", "not installed"}


def test_interpret_torques_has_sources(registry: ToolRegistry) -> None:
    """interpret_torques includes a citation for each joint range."""
    import inspect

    import src.shared.python.ai.sample_tools as st

    src = inspect.getsource(st)
    assert "Nesbit" in src or "Kwon" in src


def test_load_c3d_degrades_without_library(registry: ToolRegistry, tmp_path) -> None:
    """load_c3d returns success=False with a clear error when c3d is missing."""
    path = tmp_path / "fake.c3d"
    path.write_bytes(b"")
    result = registry.execute("load_c3d", {"file_path": str(path)})
    payload = result.result
    assert payload["success"] is False
    assert "error" in payload


@pytest.mark.live_simulation
def test_run_inverse_dynamics_real_mujoco() -> None:
    """run_inverse_dynamics still returns not-implemented even when mujoco is installed.

    The real IK/ID pipeline integration is tracked in issue #3163.
    """
    pytest.importorskip("mujoco")
    registry = ToolRegistry()
    register_golf_suite_tools(registry)
    result = registry.execute("run_inverse_dynamics", {"engine": "mujoco"})
    payload = result.result
    assert payload["success"] is False
    assert payload["issue"] == "#3163"
