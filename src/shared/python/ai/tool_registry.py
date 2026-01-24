"""Tool Registry for AI-callable capabilities.

This module provides a registry of tools that AI assistants can invoke
to interact with the Golf Modeling Suite. Each tool is self-describing
with JSON Schema parameters and validation.

The registry follows the JSON-RPC 2.0 convention for tool definitions,
making it compatible with OpenAI, Anthropic, and other providers.

Example:
    >>> from shared.python.ai.tool_registry import ToolRegistry
    >>> registry = ToolRegistry()
    >>> @registry.register("load_c3d", "Load a C3D motion capture file")
    ... def load_c3d(file_path: str) -> dict:
    ...     # Implementation
    ...     ...
"""

from __future__ import annotations

import inspect
from src.shared.python.logging_config import get_logger
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, get_type_hints

from src.shared.python.ai.exceptions import ToolExecutionError
from src.shared.python.ai.types import ToolResult

logger = get_logger(__name__)


class ToolCategory(Enum):
    """Categories for organizing tools in the UI.

    Tools are grouped by category to help users discover
    relevant functionality.
    """

    DATA_LOADING = auto()  # C3D, motion capture, etc.
    SIMULATION = auto()  # Physics engine operations
    ANALYSIS = auto()  # Inverse dynamics, energy, etc.
    VISUALIZATION = auto()  # Plotting, rendering
    VALIDATION = auto()  # Cross-engine checks
    CONFIGURATION = auto()  # Settings, preferences
    EDUCATIONAL = auto()  # Learning content


@dataclass
class ToolParameter:
    """Definition of a single tool parameter.

    Attributes:
        name: Parameter name.
        description: What the parameter is for.
        type: JSON Schema type (string, number, boolean, array, object).
        required: Whether the parameter is required.
        default: Default value if not provided.
        enum: List of allowed values (for enum parameters).
    """

    name: str
    description: str
    type: str = "string"
    required: bool = True
    default: Any = None
    enum: list[str] | None = None

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format.

        Returns:
            JSON Schema property definition.
        """
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class Tool:
    """A registered tool that AI can invoke.

    Attributes:
        name: Unique tool identifier (snake_case).
        description: What the tool does (AI-consumable, <500 chars).
        handler: Callable that executes the tool.
        parameters: List of parameter definitions.
        category: Tool category for UI organization.
        requires_confirmation: Whether user must confirm before execution.
        expertise_level: Minimum expertise level to see this tool.
        examples: Example invocations for few-shot learning.
    """

    name: str
    description: str
    handler: Callable[..., Any]
    parameters: list[ToolParameter] = field(default_factory=list)
    category: ToolCategory = ToolCategory.ANALYSIS
    requires_confirmation: bool = False
    expertise_level: int = 1  # 1=beginner, 4=expert
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema for AI providers.

        Returns:
            Complete JSON Schema tool definition.
        """
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format.

        Returns:
            OpenAI-compatible function definition.
        """
        schema = self.to_json_schema()
        return {
            "type": "function",
            "function": schema,
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool format.

        Returns:
            Anthropic-compatible tool definition.
        """
        schema = self.to_json_schema()
        return {
            "name": schema["name"],
            "description": schema["description"],
            "input_schema": schema["parameters"],
        }

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """Validate arguments against parameter definitions.

        Args:
            arguments: Arguments to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in arguments:
                errors.append(f"Missing required parameter: {param.name}")

        # Check unknown parameters
        known_params = {p.name for p in self.parameters}
        for arg_name in arguments:
            if arg_name not in known_params:
                errors.append(f"Unknown parameter: {arg_name}")

        # Check enum constraints
        for param in self.parameters:
            if param.enum and param.name in arguments:
                if arguments[param.name] not in param.enum:
                    errors.append(
                        f"Invalid value for {param.name}: {arguments[param.name]}. "
                        f"Must be one of: {param.enum}"
                    )

        return errors

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool with given arguments.

        Args:
            arguments: Arguments to pass to the handler.

        Returns:
            ToolResult with execution outcome.
        """
        import time

        start_time = time.perf_counter()

        # Validate arguments
        errors = self.validate_arguments(arguments)
        if errors:
            return ToolResult(
                tool_call_id="",
                success=False,
                error="; ".join(errors),
                execution_time=time.perf_counter() - start_time,
            )

        # Execute handler
        try:
            result = self.handler(**arguments)
            return ToolResult(
                tool_call_id="",
                success=True,
                result=result,
                execution_time=time.perf_counter() - start_time,
            )
        except Exception as e:
            logger.exception("Tool execution failed: %s", self.name)
            return ToolResult(
                tool_call_id="",
                success=False,
                error=str(e),
                execution_time=time.perf_counter() - start_time,
            )


class ToolRegistry:
    """Registry of all AI-callable tools.

    Provides registration, discovery, and execution of tools
    that AI assistants can invoke.

    Example:
        >>> registry = ToolRegistry()
        >>> @registry.register("add", "Add two numbers")
        ... def add(a: int, b: int) -> int:
        ...     return a + b
        >>> result = registry.execute("add", {"a": 1, "b": 2})
        >>> result.result
        3
    """

    def __init__(self) -> None:
        """Initialize empty tool registry."""
        self._tools: dict[str, Tool] = {}
        logger.info("Initialized ToolRegistry")

    def register(
        self,
        name: str,
        description: str,
        category: ToolCategory = ToolCategory.ANALYSIS,
        requires_confirmation: bool = False,
        expertise_level: int = 1,
        examples: list[dict[str, Any]] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a function as a tool.

        Args:
            name: Unique tool identifier.
            description: What the tool does.
            category: Tool category.
            requires_confirmation: Whether to confirm before execution.
            expertise_level: Minimum expertise level (1-4).
            examples: Example invocations.

        Returns:
            Decorator function.

        Example:
            >>> @registry.register("load_c3d", "Load C3D file")
            ... def load_c3d(file_path: str) -> dict:
            ...     ...
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            # Extract parameters from function signature
            parameters = self._extract_parameters(func)

            tool = Tool(
                name=name,
                description=description,
                handler=func,
                parameters=parameters,
                category=category,
                requires_confirmation=requires_confirmation,
                expertise_level=expertise_level,
                examples=examples or [],
            )

            self._tools[name] = tool
            logger.debug("Registered tool: %s", name)
            return func

        return decorator

    def register_tool(self, tool: Tool) -> None:
        """Register a pre-built Tool object.

        Args:
            tool: Tool to register.
        """
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def _extract_parameters(self, func: Callable[..., Any]) -> list[ToolParameter]:
        """Extract parameter definitions from function signature.

        Uses type hints and docstrings to build parameter metadata.

        Args:
            func: Function to extract parameters from.

        Returns:
            List of ToolParameter definitions.
        """
        parameters: list[ToolParameter] = []
        sig = inspect.signature(func)

        # Try to get type hints
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        # Extract parameter info
        for param_name, param in sig.parameters.items():
            # Skip self/cls
            if param_name in ("self", "cls"):
                continue

            # Determine type
            json_type = "string"  # default
            if param_name in hints:
                hint = hints[param_name]
                json_type = self._python_type_to_json(hint)

            # Check if required
            required = param.default is inspect.Parameter.empty

            # Get default value
            default = None if required else param.default

            # Create parameter definition
            parameters.append(
                ToolParameter(
                    name=param_name,
                    description=f"Parameter: {param_name}",  # Enhanced by docstring
                    type=json_type,
                    required=required,
                    default=default,
                )
            )

        return parameters

    def _python_type_to_json(self, python_type: type) -> str:
        """Convert Python type to JSON Schema type.

        Args:
            python_type: Python type annotation.

        Returns:
            JSON Schema type string.
        """
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        # Handle basic types
        if python_type in type_mapping:
            return type_mapping[python_type]

        # Handle Optional, Union, etc.
        origin = getattr(python_type, "__origin__", None)
        if origin is list:
            return "array"
        if origin is dict:
            return "object"

        return "string"

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name.

        Args:
            name: Tool name.

        Returns:
            Tool if found, None otherwise.
        """
        return self._tools.get(name)

    def list_tools(
        self,
        category: ToolCategory | None = None,
        max_expertise: int = 4,
    ) -> list[Tool]:
        """List available tools.

        Args:
            category: Filter by category (None for all).
            max_expertise: Maximum expertise level to include.

        Returns:
            List of matching tools.
        """
        tools = list(self._tools.values())

        if category is not None:
            tools = [t for t in tools if t.category == category]

        tools = [t for t in tools if t.expertise_level <= max_expertise]

        return sorted(tools, key=lambda t: t.name)

    def get_tools_for_provider(
        self,
        provider_format: str = "openai",
        max_expertise: int = 4,
    ) -> list[dict[str, Any]]:
        """Get tools in provider-specific format.

        Args:
            provider_format: "openai" or "anthropic".
            max_expertise: Maximum expertise level to include.

        Returns:
            List of tool definitions in provider format.
        """
        tools = self.list_tools(max_expertise=max_expertise)

        if provider_format == "openai":
            return [t.to_openai_format() for t in tools]
        elif provider_format == "anthropic":
            return [t.to_anthropic_format() for t in tools]
        else:
            return [t.to_json_schema() for t in tools]

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        tool_call_id: str = "",
    ) -> ToolResult:
        """Execute a tool by name.

        Args:
            name: Tool name to execute.
            arguments: Arguments to pass.
            tool_call_id: ID linking to the AI's tool call.

        Returns:
            ToolResult with execution outcome.

        Raises:
            ToolExecutionError: If tool not found.
        """
        tool = self.get_tool(name)
        if tool is None:
            raise ToolExecutionError(
                f"Tool not found: {name}",
                tool_name=name,
                parameters=arguments,
            )

        result = tool.execute(arguments)
        result.tool_call_id = tool_call_id
        return result

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools


# Global registry instance (optional singleton pattern)
_global_registry: ToolRegistry | None = None


def get_global_registry() -> ToolRegistry:
    """Get or create the global tool registry.

    Returns:
        Global ToolRegistry instance.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry
