"""End-to-end verification tests for tool registry and chat feature discovery.

Issue #5316 — chat and launcher feature-discovery verification.
Parent epic: https://github.com/D-sorganization/UpstreamDrift/issues/5309

Verifies:
- The AI tool registry discovers all expected categories and tools.
- The chat system has a 'list tools' capability (responds to availability query).
- Failures are classified as: dependency resolution, contract validation,
  or backend availability errors.

Design: TDD, DbC (public APIs validated against contracts), DRY (shared
helpers), LOD (no method chains deeper than two levels).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_tool_registry():
    """Import ToolRegistry or skip if unavailable.

    Returns:
        The ToolRegistry class.
    """
    try:
        from src.shared.python.ai.tool_registry import ToolRegistry

        return ToolRegistry
    except ImportError as exc:
        pytest.skip(f"ToolRegistry not importable (dependency resolution): {exc}")


def _build_golf_suite_registry():
    """Build a populated ToolRegistry with all Golf Suite tools.

    Returns:
        A :class:`ToolRegistry` pre-loaded with Golf Suite tools.
    """
    ToolRegistry = _import_tool_registry()
    try:
        from src.shared.python.ai.sample_tools import register_golf_suite_tools
    except ImportError as exc:
        pytest.skip(
            f"register_golf_suite_tools not importable (dependency resolution): {exc}"
        )

    registry = ToolRegistry()
    register_golf_suite_tools(registry)
    return registry


# ---------------------------------------------------------------------------
# Tool registry — feature discovery
# ---------------------------------------------------------------------------


class TestToolRegistryDiscovery:
    """Launcher discovers all registered tools through the AI registry."""

    def test_registry_instantiates(self) -> None:
        """ToolRegistry must be instantiable without errors."""
        ToolRegistry = _import_tool_registry()
        registry = ToolRegistry()
        assert registry is not None

    def test_registry_is_initially_empty(self) -> None:
        """A fresh ToolRegistry must contain no tools."""
        ToolRegistry = _import_tool_registry()
        registry = ToolRegistry()
        assert len(registry) == 0

    def test_golf_suite_registration_produces_tools(self) -> None:
        """register_golf_suite_tools must register at least one tool."""
        registry = _build_golf_suite_registry()
        assert (
            len(registry) > 0
        ), "register_golf_suite_tools registered 0 tools — contract validation failure"

    def test_expected_core_tools_are_present(self) -> None:
        """Core tools expected by the chat system must be registered."""
        registry = _build_golf_suite_registry()

        # These tools are referenced in the system prompt and chat contract.
        expected_tools = {
            "list_sample_files",
            "load_c3d",
            "explain_concept",
            "list_glossary_terms",
            "summarize_simulation_run",
        }
        registered_names = {tool.name for tool in registry.list_tools()}
        missing = expected_tools - registered_names
        assert not missing, (
            f"Missing expected tools (contract validation failure): {sorted(missing)}\n"
            f"Registered: {sorted(registered_names)}"
        )

    def test_tools_have_non_empty_descriptions(self) -> None:
        """Every registered tool must have a non-empty description."""
        registry = _build_golf_suite_registry()
        empty_desc = [
            tool.name
            for tool in registry.list_tools()
            if not tool.description or not tool.description.strip()
        ]
        assert (
            not empty_desc
        ), f"Tools missing descriptions (contract validation): {empty_desc}"

    def test_tools_have_callable_handlers(self) -> None:
        """Every registered tool must have a callable handler."""
        registry = _build_golf_suite_registry()
        non_callable = [
            tool.name for tool in registry.list_tools() if not callable(tool.handler)
        ]
        assert (
            not non_callable
        ), f"Tools with non-callable handlers (contract validation): {non_callable}"

    def test_expected_categories_present(self) -> None:
        """At least ANALYSIS, DATA_LOADING, EDUCATIONAL categories must exist."""
        registry = _build_golf_suite_registry()
        from src.shared.python.ai.tool_registry import ToolCategory

        category_set = {tool.category for tool in registry.list_tools()}
        expected_categories = {
            ToolCategory.ANALYSIS,
            ToolCategory.DATA_LOADING,
            ToolCategory.EDUCATIONAL,
        }
        missing = expected_categories - category_set
        assert (
            not missing
        ), f"Missing expected tool categories: {missing}\nFound: {category_set}"

    def test_list_tools_by_category_filters_correctly(self) -> None:
        """list_tools(category=X) must return only tools in that category."""
        registry = _build_golf_suite_registry()
        from src.shared.python.ai.tool_registry import ToolCategory

        educational = registry.list_tools(category=ToolCategory.EDUCATIONAL)
        assert all(
            t.category == ToolCategory.EDUCATIONAL for t in educational
        ), "list_tools category filter returned tools from wrong categories"

    def test_tools_in_schema_format_are_valid(self) -> None:
        """get_tools_for_provider must return non-empty list with 'name' keys."""
        registry = _build_golf_suite_registry()
        tools_json = registry.get_tools_for_provider("openai")
        assert isinstance(tools_json, list)
        assert len(tools_json) > 0
        for entry in tools_json:
            assert (
                "function" in entry
            ), f"OpenAI format missing 'function' key: {list(entry.keys())}"
            func = entry["function"]
            assert "name" in func, "OpenAI function definition missing 'name'"
            assert (
                "description" in func
            ), "OpenAI function definition missing 'description'"

    def test_sidekick_analytics_tool_registered(self) -> None:
        """The Sidekick summarize_simulation_run tool must appear in the registry."""
        registry = _build_golf_suite_registry()
        tool = registry.get_tool("summarize_simulation_run")
        assert tool is not None, (
            "'summarize_simulation_run' not registered — "
            "Sidekick analytics backend not wired (backend availability failure)"
        )
        assert callable(tool.handler)


# ---------------------------------------------------------------------------
# Chat system — availability query response
# ---------------------------------------------------------------------------


class TestChatSystemToolAvailability:
    """Chat agent can respond to 'what tools are available?' queries."""

    def test_system_prompt_mentions_tool_capabilities(self) -> None:
        """build_system_prompt must describe at least one callable tool."""
        try:
            from src.shared.python.ai.system_prompts import build_system_prompt
        except ImportError as exc:
            pytest.skip(f"system_prompts not importable: {exc}")

        prompt = build_system_prompt(app_context="upstream_drift")
        assert "tool" in prompt.lower() or "capabilit" in prompt.lower(), (
            "System prompt does not mention tools/capabilities — "
            "chat cannot answer 'what tools are available?' correctly"
        )

    def test_system_prompt_mentions_summarize_simulation_run(self) -> None:
        """The system prompt must reference the analytics tool (issue #5464)."""
        try:
            from src.shared.python.ai.system_prompts import build_system_prompt
        except ImportError as exc:
            pytest.skip(f"system_prompts not importable: {exc}")

        prompt = build_system_prompt(app_context="upstream_drift")
        assert "summarize_simulation_run" in prompt, (
            "System prompt does not reference 'summarize_simulation_run' — "
            "chat will not invoke the analytics tool (contract validation)"
        )

    def test_tool_registry_lists_tools_as_structured_response(self) -> None:
        """list_tools() returns a list that can be formatted as a chat response."""
        registry = _build_golf_suite_registry()
        tools = registry.list_tools()

        # Simulate what the chat agent would do to answer "what tools exist?"
        response_lines = [f"- {tool.name}: {tool.description[:80]}" for tool in tools]
        response_text = "\n".join(response_lines)

        assert response_text, "Tool list is empty — cannot answer discovery query"
        assert (
            "summarize_simulation_run" in response_text
        ), "'summarize_simulation_run' not in tool list response"

    def test_tool_bridge_importable(self) -> None:
        """ChatToolBridge must be importable for chat-to-registry wiring."""
        try:
            from src.shared.python.ai.tool_bridge import ChatToolBridge

            assert ChatToolBridge is not None
        except ImportError as exc:
            pytest.skip(f"ChatToolBridge not importable (dependency resolution): {exc}")

    def test_tool_bridge_exposes_get_tools_for_provider(self) -> None:
        """ChatToolBridge.get_tools_for_provider must be callable."""
        try:
            from src.shared.python.ai.tool_bridge import ChatToolBridge
        except ImportError as exc:
            pytest.skip(f"ChatToolBridge not importable: {exc}")

        bridge = ChatToolBridge()
        assert hasattr(bridge, "get_tools_for_provider"), (
            "ChatToolBridge missing get_tools_for_provider — "
            "chat cannot expose tool list to AI provider"
        )
        assert callable(bridge.get_tools_for_provider)


# ---------------------------------------------------------------------------
# Feature-discovery contract — launcher manifest
# ---------------------------------------------------------------------------


class TestLauncherManifestFeatureDiscovery:
    """Launcher manifest exposes all expected feature categories."""

    def test_launcher_manifest_importable(self) -> None:
        """LauncherManifest must be importable from the launchers package."""
        try:
            from src.launchers.launcher_model_registry import ModelHandlerRegistry

            assert ModelHandlerRegistry is not None
        except ImportError:
            # Try alternate import path
            try:
                from src.shared.python.capabilities import KNOWN_CAPABILITIES

                assert KNOWN_CAPABILITIES is not None
            except ImportError as exc:
                pytest.skip(
                    f"Launcher manifest not importable (dependency resolution): {exc}"
                )

    def test_gui_registry_lists_tools(self) -> None:
        """GUIRegistry.list_tools() must return a structured list."""
        try:
            from src.shared.python.gui_launcher.registry import GUIRegistry
        except ImportError as exc:
            pytest.skip(f"GUIRegistry not importable: {exc}")

        registry = GUIRegistry()
        # Fresh registry has no tools, but list_tools must be callable
        tools = registry.list_tools()
        assert isinstance(tools, list)

    def test_gui_registry_list_categories_returns_list(self) -> None:
        """GUIRegistry.list_categories() must return a list."""
        try:
            from src.shared.python.gui_launcher.registry import GUIRegistry
        except ImportError as exc:
            pytest.skip(f"GUIRegistry not importable: {exc}")

        registry = GUIRegistry()
        cats = registry.list_categories()
        assert isinstance(cats, list)

    def test_tool_registry_contains_is_correct(self) -> None:
        """ToolRegistry.__contains__ contract: registered name returns True."""
        registry = _build_golf_suite_registry()
        tools = registry.list_tools()
        if not tools:
            pytest.skip("No tools registered")

        first_tool_name = tools[0].name
        assert (
            first_tool_name in registry
        ), f"__contains__ returned False for registered tool '{first_tool_name}'"
        assert "definitely_not_a_tool_xyzzy" not in registry
