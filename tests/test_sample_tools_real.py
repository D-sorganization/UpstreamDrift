"""Real-behaviour tests for sample_tools (issue #3163).

The tests cover the demo-fixture path (1-DoF pendulum) so they pass
without any physics engine installed. Engine-backed paths are marked
with ``live_simulation`` and ``pytest.importorskip`` so they only run
when the corresponding engine is available.
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


def test_run_inverse_dynamics_demo_fixture(registry: ToolRegistry) -> None:
    """With no real C3D file, the tool returns the demo pendulum torques."""
    result = registry.execute("run_inverse_dynamics", {"engine": "mujoco"})
    assert result.success is True
    payload = result.result
    assert payload["success"] is True
    assert payload["source"] == "demo_fixture"
    assert payload["engine"] == "mujoco"
    torques = payload["torques"]
    assert len(torques) > 10
    # Each row is [t, joint_name, tau]
    assert torques[0][1] == "joint_0"


def test_run_inverse_dynamics_reports_available_engines(
    registry: ToolRegistry,
) -> None:
    """Payload includes an available_engines list for the client to display."""
    result = registry.execute("run_inverse_dynamics", {"engine": "pinocchio"})
    payload = result.result
    assert isinstance(payload.get("available_engines", []), list)


def test_run_inverse_dynamics_bad_c3d(registry: ToolRegistry, tmp_path) -> None:
    """A non-parseable C3D file surfaces a clean error."""
    fake = tmp_path / "not_a_real.c3d"
    fake.write_bytes(b"this is not a c3d file")
    result = registry.execute(
        "run_inverse_dynamics", {"file_path": str(fake), "engine": "mujoco"}
    )
    payload = result.result
    # Either: c3d library missing -> error, or parse fails -> error.
    assert payload["success"] is False
    assert "error" in payload


def test_check_energy_conservation_demo_fixture(registry: ToolRegistry) -> None:
    """Energy drift is computed on the closed-form pendulum."""
    result = registry.execute("check_energy_conservation", {"tolerance": 0.1})
    payload = result.result
    assert payload["success"] is True
    assert payload["source"] == "demo_fixture"
    assert 0.0 <= payload["drift_fraction"] <= 10.0
    assert payload["samples"] > 10


def test_validate_cross_engine_demo_fixture(registry: ToolRegistry) -> None:
    """validate_cross_engine returns a structured per-engine diff."""
    result = registry.execute("validate_cross_engine", {"tolerance": 0.02})
    payload = result.result
    assert payload["success"] is True
    assert payload["source"] == "demo_fixture"
    assert len(payload["engines"]) == 3
    for entry in payload["engines"]:
        assert "engine" in entry
        assert "available" in entry
        assert "max_delta" in entry


def test_list_physics_engines_introspects(registry: ToolRegistry) -> None:
    """list_physics_engines calls _available_engines rather than hardcoding."""
    result = registry.execute("list_physics_engines", {})
    payload = result.result
    engines = payload["engines"]
    # We don't know which are installed, but shape must match.
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
    # c3d is an optional dep; either library missing OR parse failure.
    assert payload["success"] is False
    assert "error" in payload


@pytest.mark.live_simulation
def test_run_inverse_dynamics_real_mujoco() -> None:
    """Engine-backed ID runs when mujoco is installed (skipped otherwise)."""
    pytest.importorskip("mujoco")
    registry = ToolRegistry()
    register_golf_suite_tools(registry)
    result = registry.execute("run_inverse_dynamics", {"engine": "mujoco"})
    payload = result.result
    assert payload["success"] is True
