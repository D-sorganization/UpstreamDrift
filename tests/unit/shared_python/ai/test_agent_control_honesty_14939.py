"""Agent control tools must not report success without real work (#14939)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.ai.sample_tools import register_golf_suite_tools
from src.shared.python.ai.tool_registry import ToolRegistry
from src.shared.python.ai.tools.agent_control import (
    AgentController,
    create_agent_tools_for_registry,
)

pytestmark = pytest.mark.unit

_NOT_IMPLEMENTED = "not_implemented"

# Registered chat tools that must refuse until backed by a collaborator (#14939).
_REGISTERED_STUB_TOOL_NAMES = frozenset(
    {
        "import_file",
        "load_model",
        "start_simulation",
        "stop_simulation",
        "set_settings",
    }
)

# Stub AgentController methods (registered or not) cited in #14939.
_STUB_CONTROLLER_METHODS = _REGISTERED_STUB_TOOL_NAMES | frozenset(
    {
        "export_model",
        "pause_simulation",
        "resume_simulation",
    }
)


def _registered_stub_handlers() -> list[tuple[str, object]]:
    tools = create_agent_tools_for_registry()
    stubs = []
    for tool_def in tools:
        name = tool_def["name"]
        if name in _REGISTERED_STUB_TOOL_NAMES:
            stubs.append((name, tool_def["handler"]))
    missing = _REGISTERED_STUB_TOOL_NAMES - {name for name, _ in stubs}
    assert not missing, (
        f"Registered stub tools missing from registry: {sorted(missing)}"
    )
    return stubs


def _action_success(result: object) -> bool:
    if isinstance(result, dict):
        return bool(result.get("success"))
    return bool(getattr(result, "success", False))


def _action_status(result: object) -> str | None:
    if isinstance(result, dict):
        return result.get("status")
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data.get("status")
    return None


@pytest.mark.parametrize("tool_name,handler", _registered_stub_handlers())
def test_stub_registered_handlers_do_not_report_success(
    tool_name: str,
    handler: object,
    tmp_path: Path,
) -> None:
    import_file = tmp_path / "sample.urdf"
    import_file.write_text("<robot name='x'/>\n", encoding="utf-8")

    kwargs: dict[str, object]
    if tool_name == "import_file":
        kwargs = {"file_path": str(import_file)}
    elif tool_name == "start_simulation":
        kwargs = {"model_name": "humanoid", "duration": 1.0}
    elif tool_name == "stop_simulation":
        kwargs = {}
    elif tool_name == "set_settings":
        kwargs = {"settings": {"theme": "light"}}
    elif tool_name == "load_model":
        kwargs = {"model_name": "humanoid_golf"}
    else:
        pytest.fail(f"Unhandled stub tool {tool_name}")

    raw = handler(**kwargs)  # type: ignore[operator]
    assert not _action_success(raw), (
        f"{tool_name} reported success without a backing collaborator"
    )
    assert _action_status(raw) == _NOT_IMPLEMENTED


@pytest.mark.parametrize("method_name", sorted(_STUB_CONTROLLER_METHODS))
def test_stub_controller_methods_do_not_report_success(
    method_name: str,
    tmp_path: Path,
) -> None:
    controller = AgentController(repo_root=tmp_path)
    import_file = tmp_path / "sample.urdf"
    import_file.write_text("<robot name='x'/>\n", encoding="utf-8")

    if method_name == "import_file":
        result = controller.import_file(str(import_file))
    elif method_name == "export_model":
        result = controller.export_model("m", str(tmp_path / "out.urdf"))
    elif method_name == "start_simulation":
        result = controller.start_simulation("m", duration=1.0)
    elif method_name == "stop_simulation":
        result = controller.stop_simulation()
    elif method_name == "pause_simulation":
        result = controller.pause_simulation()
    elif method_name == "resume_simulation":
        result = controller.resume_simulation()
    elif method_name == "set_settings":
        result = controller.set_settings({"theme": "dark"})
    elif method_name == "load_model":
        result = controller.load_model("humanoid_golf")
    else:
        pytest.fail(f"Unhandled controller stub {method_name}")

    assert not result.success, (
        f"AgentController.{method_name} reported success without a collaborator"
    )
    assert result.data.get("status") == _NOT_IMPLEMENTED
    assert result.error == _NOT_IMPLEMENTED


def test_agent_control_tools_via_registry_are_honest(tmp_path: Path) -> None:
    """End-to-end: registered agent tools must not surface success via the registry."""
    reg = ToolRegistry()
    register_golf_suite_tools(reg)

    import_file = tmp_path / "motion.urdf"
    import_file.write_text("<robot/>", encoding="utf-8")

    res = reg.execute(
        "import_file",
        {"file_path": str(import_file)},
    )
    assert res.success is True  # ToolResult wrapper succeeded
    payload = res.result
    assert not _action_success(payload)
    assert _action_status(payload) == _NOT_IMPLEMENTED
