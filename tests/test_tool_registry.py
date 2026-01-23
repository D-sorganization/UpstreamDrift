import pytest
from shared.python.ai.tool_registry import ToolCategory, ToolRegistry


class TestToolRegistry:
    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    def test_decorator_registration(self, registry):
        @registry.register("test_tool", "Description")
        def my_tool(arg1: int, arg2: str = "default") -> str:
            return f"{arg1}-{arg2}"

        assert "test_tool" in registry
        tool = registry.get_tool("test_tool")
        assert tool.name == "test_tool"
        assert len(tool.parameters) == 2

        # Check int parameter
        p1 = next(p for p in tool.parameters if p.name == "arg1")
        assert p1.type == "integer"
        assert p1.required is True

        # Check str parameter
        p2 = next(p for p in tool.parameters if p.name == "arg2")
        assert p2.type == "string"
        assert p2.required is False
        assert p2.default == "default"

    def test_execution_success(self, registry):
        @registry.register("add", "Add nums")
        def add(a: int, b: int) -> int:
            return a + b

        result = registry.execute("add", {"a": 5, "b": 3})
        assert result.success is True
        assert result.result == 8

    def test_execution_param_validation(self, registry):
        @registry.register("echo", "Echo")
        def echo(msg: str):
            return msg

        # Missing required
        res1 = registry.execute("echo", {})
        assert res1.success is False
        assert "Missing required parameter" in res1.error

        # Unknown param
        res2 = registry.execute("echo", {"msg": "hi", "bad": 1})
        assert res2.success is False
        assert "Unknown parameter" in res2.error

    def test_json_schema_generation(self, registry):
        @registry.register("complex", "Complex tool")
        def complex_tool(req: int, opt: str | None = None):
            pass

        tool = registry.get_tool("complex")
        schema = tool.to_json_schema()

        assert schema["name"] == "complex"
        assert "req" in schema["parameters"]["properties"]
        assert "req" in schema["parameters"]["required"]
        assert "opt" not in schema["parameters"]["required"]

    def test_provider_formats(self, registry):
        @registry.register("tool", "desc")
        def tool(a: int):
            pass

        # OpenAI
        oa = registry.get_tools_for_provider("openai")[0]
        assert oa["type"] == "function"
        assert "name" in oa["function"]

        # Anthropic
        anth = registry.get_tools_for_provider("anthropic")[0]
        assert "input_schema" in anth
        assert "name" in anth

    def test_list_filtering(self, registry):
        @registry.register("t1", "desc", category=ToolCategory.ANALYSIS)
        def t1():
            pass

        @registry.register("t2", "desc", category=ToolCategory.VISUALIZATION)
        def t2():
            pass

        assert len(registry.list_tools(category=ToolCategory.ANALYSIS)) == 1
        assert registry.list_tools(category=ToolCategory.ANALYSIS)[0].name == "t1"
