"""Unit tests for ToolRegistry."""

from __future__ import annotations

import pytest

from shared.python.ai.tool_registry import (
    Tool,
    ToolCategory,
    ToolParameter,
    ToolRegistry,
    get_global_registry,
)


class TestToolParameter:
    """Tests for ToolParameter."""

    def test_parameter_to_json_schema(self) -> None:
        """Test JSON Schema conversion."""
        param = ToolParameter(
            name="file_path",
            description="Path to the file",
            type="string",
            required=True,
        )
        schema = param.to_json_schema()
        assert schema["type"] == "string"
        assert schema["description"] == "Path to the file"

    def test_parameter_with_enum(self) -> None:
        """Test parameter with enum values."""
        param = ToolParameter(
            name="engine",
            description="Physics engine to use",
            type="string",
            enum=["mujoco", "drake", "pinocchio"],
        )
        schema = param.to_json_schema()
        assert schema["enum"] == ["mujoco", "drake", "pinocchio"]


class TestTool:
    """Tests for Tool class."""

    def test_to_json_schema(self) -> None:
        """Test complete JSON Schema generation."""
        tool = Tool(
            name="add_numbers",
            description="Add two numbers",
            handler=lambda a, b: a + b,
            parameters=[
                ToolParameter(name="a", description="First number", type="integer"),
                ToolParameter(name="b", description="Second number", type="integer"),
            ],
        )
        schema = tool.to_json_schema()
        assert schema["name"] == "add_numbers"
        assert "a" in schema["parameters"]["properties"]
        assert "b" in schema["parameters"]["properties"]

    def test_to_openai_format(self) -> None:
        """Test OpenAI function format."""
        tool = Tool(
            name="test_tool",
            description="A test",
            handler=lambda: None,
        )
        openai_format = tool.to_openai_format()
        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "test_tool"

    def test_to_anthropic_format(self) -> None:
        """Test Anthropic tool format."""
        tool = Tool(
            name="test_tool",
            description="A test",
            handler=lambda: None,
        )
        anthropic_format = tool.to_anthropic_format()
        assert anthropic_format["name"] == "test_tool"
        assert "input_schema" in anthropic_format

    def test_validate_arguments_missing_required(self) -> None:
        """Test validation catches missing required params."""
        tool = Tool(
            name="test",
            description="test",
            handler=lambda x: x,
            parameters=[
                ToolParameter(name="x", description="required", required=True),
            ],
        )
        errors = tool.validate_arguments({})
        assert len(errors) == 1
        assert "Missing required" in errors[0]

    def test_validate_arguments_unknown_param(self) -> None:
        """Test validation catches unknown params."""
        tool = Tool(
            name="test",
            description="test",
            handler=lambda: None,
            parameters=[],
        )
        errors = tool.validate_arguments({"unknown": 42})
        assert len(errors) == 1
        assert "Unknown parameter" in errors[0]

    def test_validate_arguments_invalid_enum(self) -> None:
        """Test validation catches invalid enum values."""
        tool = Tool(
            name="test",
            description="test",
            handler=lambda engine: engine,
            parameters=[
                ToolParameter(
                    name="engine",
                    description="engine",
                    enum=["a", "b"],
                ),
            ],
        )
        errors = tool.validate_arguments({"engine": "c"})
        assert len(errors) == 1
        assert "Invalid value" in errors[0]

    def test_execute_success(self) -> None:
        """Test successful tool execution."""

        def add(a: int, b: int) -> int:
            return a + b

        tool = Tool(
            name="add",
            description="Add numbers",
            handler=add,
            parameters=[
                ToolParameter(name="a", description="a", type="integer"),
                ToolParameter(name="b", description="b", type="integer"),
            ],
        )
        result = tool.execute({"a": 2, "b": 3})
        assert result.success is True
        assert result.result == 5

    def test_execute_failure(self) -> None:
        """Test tool execution with error."""

        def failing_tool() -> None:
            raise ValueError("Intentional error")

        tool = Tool(
            name="failing",
            description="Always fails",
            handler=failing_tool,
        )
        result = tool.execute({})
        assert result.success is False
        assert "Intentional error" in result.error  # type: ignore[operator]


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_with_decorator(self) -> None:
        """Test registering tool with decorator."""
        registry = ToolRegistry()

        @registry.register("my_tool", "My description")
        def my_tool(x: int) -> int:
            return x * 2

        assert "my_tool" in registry
        assert len(registry) == 1

    def test_register_tool_object(self) -> None:
        """Test registering pre-built Tool object."""
        registry = ToolRegistry()
        tool = Tool(
            name="test_tool",
            description="Test",
            handler=lambda: None,
        )
        registry.register_tool(tool)
        assert registry.get_tool("test_tool") is tool

    def test_list_tools_filter_by_category(self) -> None:
        """Test listing tools by category."""
        registry = ToolRegistry()
        registry.register_tool(
            Tool(
                name="tool1",
                description="",
                handler=lambda: None,
                category=ToolCategory.DATA_LOADING,
            )
        )
        registry.register_tool(
            Tool(
                name="tool2",
                description="",
                handler=lambda: None,
                category=ToolCategory.SIMULATION,
            )
        )

        data_tools = registry.list_tools(category=ToolCategory.DATA_LOADING)
        assert len(data_tools) == 1
        assert data_tools[0].name == "tool1"

    def test_list_tools_filter_by_expertise(self) -> None:
        """Test filtering tools by expertise level."""
        registry = ToolRegistry()
        registry.register_tool(
            Tool(
                name="beginner_tool",
                description="",
                handler=lambda: None,
                expertise_level=1,
            )
        )
        registry.register_tool(
            Tool(
                name="expert_tool",
                description="",
                handler=lambda: None,
                expertise_level=4,
            )
        )

        beginner_tools = registry.list_tools(max_expertise=1)
        assert len(beginner_tools) == 1
        assert beginner_tools[0].name == "beginner_tool"

    def test_get_tools_for_provider(self) -> None:
        """Test getting tools in provider format."""
        registry = ToolRegistry()

        @registry.register("test", "Test tool")
        def test() -> None:
            pass

        openai_tools = registry.get_tools_for_provider("openai")
        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"

        anthropic_tools = registry.get_tools_for_provider("anthropic")
        assert len(anthropic_tools) == 1
        assert "input_schema" in anthropic_tools[0]

    def test_execute(self) -> None:
        """Test executing tool by name."""
        registry = ToolRegistry()

        @registry.register("multiply", "Multiply numbers")
        def multiply(x: int, y: int) -> int:
            return x * y

        result = registry.execute("multiply", {"x": 3, "y": 4})
        assert result.success is True
        assert result.result == 12

    def test_execute_not_found(self) -> None:
        """Test executing non-existent tool raises error."""
        from shared.python.ai.exceptions import ToolExecutionError

        registry = ToolRegistry()
        with pytest.raises(ToolExecutionError):
            registry.execute("nonexistent", {})


class TestGlobalRegistry:
    """Tests for global registry singleton."""

    def test_get_global_registry(self) -> None:
        """Test getting global registry."""
        registry = get_global_registry()
        assert isinstance(registry, ToolRegistry)

    def test_global_registry_is_singleton(self) -> None:
        """Test that global registry returns same instance."""
        registry1 = get_global_registry()
        registry2 = get_global_registry()
        assert registry1 is registry2
