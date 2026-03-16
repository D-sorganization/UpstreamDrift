from __future__ import annotations

import pytest

from src.shared.python.ai.exceptions import ToolExecutionError
from src.shared.python.ai.tool_registry import (
    Tool,
    ToolCategory,
    ToolParameter,
    ToolRegistry,
    get_global_registry,
)


def test_tool_parameter_to_json():
    tp = ToolParameter("p1", "desc", type="string", required=True, default="a")
    js = tp.to_json_schema()
    assert js["type"] == "string"
    assert js["description"] == "desc"
    assert js["default"] == "a"
    assert "enum" not in js

    tp2 = ToolParameter("p2", "desc", enum=["x", "y"])
    js2 = tp2.to_json_schema()
    assert js2["enum"] == ["x", "y"]


def test_tool_to_json_schema():
    def handler(a):
        return a

    t = Tool(
        name="my_tool",
        description="does things",
        handler=handler,
        parameters=[ToolParameter("a", "alpha", type="integer", required=True)],
    )
    js = t.to_json_schema()
    assert js["name"] == "my_tool"
    assert js["description"] == "does things"
    assert "a" in js["parameters"]["properties"]
    assert "a" in js["parameters"]["required"]


def test_tool_formats():
    def handler(a):
        return a

    t = Tool(name="tool", description="desc", handler=handler)

    openai = t.to_openai_format()
    assert openai["type"] == "function"
    assert openai["function"]["name"] == "tool"

    anthropic = t.to_anthropic_format()
    assert anthropic["name"] == "tool"
    assert "input_schema" in anthropic


def test_validate_arguments():
    def handler(a, b=1):
        return a + b

    t = Tool(
        name="tool",
        description="desc",
        handler=handler,
        parameters=[
            ToolParameter("a", "desc", required=True),
            ToolParameter("b", "desc", required=False, enum=["x", "y", 1]),
        ],
    )

    # Missing required
    errs = t.validate_arguments({})
    assert len(errs) == 1
    assert "Missing required parameter" in errs[0]

    # Unknown parameter
    errs = t.validate_arguments({"a": 1, "c": 2})
    assert len(errs) == 1
    assert "Unknown parameter" in errs[0]

    # Invalid enum
    errs = t.validate_arguments({"a": 1, "b": "z"})
    assert len(errs) == 1
    assert "Invalid value" in errs[0]

    # Valid
    assert t.validate_arguments({"a": 1, "b": "x"}) == []


def test_execute():
    def handler(a):
        if a == 0:
            raise ValueError("bad a")
        return a * 2

    t = Tool("tool", "desc", handler, parameters=[ToolParameter("a", "desc")])

    # Success
    res = t.execute({"a": 2})
    assert res.success
    assert res.result == 4

    # Validation failure
    res_val = t.execute({})
    assert not res_val.success
    assert "Missing required parameter" in res_val.error

    # Exception
    res_exc = t.execute({"a": 0})
    assert not res_exc.success
    assert "bad a" in res_exc.error


def test_registry_extract_parameters():
    reg = ToolRegistry()

    def my_func(a: int, b: str = "yes") -> None:
        pass

    params = reg._extract_parameters(my_func)
    assert len(params) == 2
    assert params[0].name == "a"
    assert params[0].type == "integer"
    assert params[0].required is True

    assert params[1].name == "b"
    assert params[1].type == "string"
    assert params[1].required is False


def test_registry_register_and_execute():
    reg = ToolRegistry()

    @reg.register("add", "adds")
    def add(a: int, b: int) -> int:
        return a + b

    assert "add" in reg
    assert len(reg) == 1

    res = reg.execute("add", {"a": 2, "b": 3})
    assert res.success
    assert res.result == 5

    with pytest.raises(ToolExecutionError):
        reg.execute("missing", {})


def test_registry_list_tools():
    reg = ToolRegistry()

    @reg.register("t1", "desc1", category=ToolCategory.ANALYSIS, expertise_level=1)
    def t1():
        pass

    @reg.register("t2", "desc2", category=ToolCategory.SIMULATION, expertise_level=3)
    def t2():
        pass

    all_t = reg.list_tools()
    assert len(all_t) == 2

    cat_t = reg.list_tools(category=ToolCategory.ANALYSIS)
    assert len(cat_t) == 1
    assert cat_t[0].name == "t1"

    exp_t = reg.list_tools(max_expertise=2)
    assert len(exp_t) == 1
    assert exp_t[0].name == "t1"


def test_get_tools_for_provider():
    reg = ToolRegistry()

    @reg.register("tool", "desc")
    def tool():
        pass

    assert "type" in reg.get_tools_for_provider("openai")[0]
    assert "input_schema" in reg.get_tools_for_provider("anthropic")[0]
    assert "type" not in reg.get_tools_for_provider("json")[0]  # JSON schema format


def test_global_registry():
    r1 = get_global_registry()
    r2 = get_global_registry()
    assert r1 is r2


def test_python_type_to_json():
    reg = ToolRegistry()
    assert reg._python_type_to_json(str) == "string"
    assert reg._python_type_to_json(int) == "integer"
    assert reg._python_type_to_json(float) == "number"
    assert reg._python_type_to_json(list) == "array"
    assert reg._python_type_to_json(dict) == "object"
    assert reg._python_type_to_json(tuple) == "string"  # fallback
    assert reg._python_type_to_json(list[int]) == "array"
    assert reg._python_type_to_json(dict[str, str]) == "object"
