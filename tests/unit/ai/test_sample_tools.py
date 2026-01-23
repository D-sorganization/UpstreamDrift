"""Unit tests for sample tools."""

from __future__ import annotations

from src.shared.python.ai.sample_tools import (
    register_golf_suite_tools,
)
from src.shared.python.ai.tool_registry import ToolCategory, ToolRegistry


class TestRegisterGolfSuiteTools:
    """Tests for tool registration."""

    def test_register_all_tools(self) -> None:
        """Test registering all Golf Suite tools."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        # Should have registered multiple tools
        assert len(registry) >= 10

    def test_data_loading_tools_registered(self) -> None:
        """Test that data loading tools are registered."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        tools = registry.list_tools(category=ToolCategory.DATA_LOADING)
        tool_names = [t.name for t in tools]

        assert "list_sample_files" in tool_names
        assert "load_c3d" in tool_names

    def test_analysis_tools_registered(self) -> None:
        """Test that analysis tools are registered."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        # Check simulation category
        sim_tools = registry.list_tools(category=ToolCategory.SIMULATION)
        assert any(t.name == "run_inverse_dynamics" for t in sim_tools)

        # Check analysis category
        analysis_tools = registry.list_tools(category=ToolCategory.ANALYSIS)
        assert any(t.name == "interpret_torques" for t in analysis_tools)

    def test_education_tools_registered(self) -> None:
        """Test that education tools are registered."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        tools = registry.list_tools(category=ToolCategory.EDUCATIONAL)
        tool_names = [t.name for t in tools]

        assert "explain_concept" in tool_names
        assert "list_glossary_terms" in tool_names
        assert "search_glossary" in tool_names

    def test_validation_tools_registered(self) -> None:
        """Test that validation tools are registered."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        tools = registry.list_tools(category=ToolCategory.VALIDATION)
        tool_names = [t.name for t in tools]

        assert "validate_cross_engine" in tool_names
        assert "check_energy_conservation" in tool_names


class TestDataTools:
    """Tests for data loading tools."""

    def test_list_sample_files(self) -> None:
        """Test listing sample files returns valid structure."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        result = registry.execute("list_sample_files", {})
        assert result.success is True
        assert "files" in result.result
        assert "message" in result.result

    def test_load_c3d_file_not_found(self) -> None:
        """Test loading non-existent C3D file."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        result = registry.execute("load_c3d", {"file_path": "/nonexistent.c3d"})
        assert result.success is True  # Tool executed successfully
        assert result.result["success"] is False  # But operation failed
        assert "not found" in result.result["error"].lower()

    def test_load_c3d_wrong_extension(self) -> None:
        """Test loading file with wrong extension returns error."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        # Note: File doesn't exist, so we get "file not found" first
        # In real usage with existing non-.c3d file, we'd get the extension error
        result = registry.execute("load_c3d", {"file_path": "file.txt"})
        assert result.success is True  # Tool executed
        assert result.result["success"] is False  # But operation failed


class TestAnalysisTools:
    """Tests for analysis tools."""

    def test_run_inverse_dynamics_invalid_engine(self) -> None:
        """Test inverse dynamics with invalid engine."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        result = registry.execute(
            "run_inverse_dynamics",
            {"file_path": "test.c3d", "engine": "invalid"},
        )
        assert result.success is True
        assert result.result["success"] is False
        assert "invalid" in result.result["error"].lower()

    def test_run_inverse_dynamics_valid_engine(self) -> None:
        """Test inverse dynamics with valid engine."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        result = registry.execute(
            "run_inverse_dynamics",
            {"file_path": "test.c3d", "engine": "mujoco"},
        )
        assert result.success is True
        assert result.result["success"] is True
        assert result.result["engine"] == "mujoco"

    def test_interpret_torques(self) -> None:
        """Test torque interpretation."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        result = registry.execute(
            "interpret_torques",
            {"shoulder_torque": 100.0, "hip_torque": 150.0, "wrist_torque": 30.0},
        )
        assert result.success is True
        assert "shoulder" in result.result
        assert "hip" in result.result
        assert "wrist" in result.result

    def test_interpret_torques_classification(self) -> None:
        """Test that torques are classified correctly."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        # Low values
        result = registry.execute(
            "interpret_torques",
            {"shoulder_torque": 10.0, "hip_torque": 20.0, "wrist_torque": 5.0},
        )
        assert "below" in result.result["shoulder"]["classification"].lower()

        # High values
        result = registry.execute(
            "interpret_torques",
            {"shoulder_torque": 200.0, "hip_torque": 300.0, "wrist_torque": 100.0},
        )
        assert "above" in result.result["shoulder"]["classification"].lower()


class TestEducationTools:
    """Tests for education tools."""

    def test_explain_concept_known_term(self) -> None:
        """Test explaining a known concept."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        result = registry.execute(
            "explain_concept",
            {"term": "inverse_dynamics", "expertise_level": 1},
        )
        assert result.success is True
        assert "explanation" in result.result
        assert len(result.result["explanation"]) > 50  # Non-trivial explanation

    def test_explain_concept_unknown_term(self) -> None:
        """Test explaining an unknown concept."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        result = registry.execute(
            "explain_concept",
            {"term": "nonexistent_term", "expertise_level": 1},
        )
        assert result.success is True
        assert "not found" in result.result["explanation"].lower()

    def test_explain_concept_expertise_levels(self) -> None:
        """Test that expertise level affects explanation."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        beginner = registry.execute(
            "explain_concept",
            {"term": "inverse_dynamics", "expertise_level": 1},
        )
        expert = registry.execute(
            "explain_concept",
            {"term": "inverse_dynamics", "expertise_level": 4},
        )

        # Expert explanation should be more technical
        assert beginner.result["level"] == "beginner"
        assert expert.result["level"] == "expert"
        # Expert should have formula
        assert "formula" in expert.result

    def test_list_glossary_terms(self) -> None:
        """Test listing glossary terms."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        result = registry.execute("list_glossary_terms", {})
        assert result.success is True
        assert len(result.result["terms"]) > 0
        assert len(result.result["categories"]) > 0

    def test_list_glossary_terms_filtered(self) -> None:
        """Test listing terms filtered by category."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        result = registry.execute("list_glossary_terms", {"category": "dynamics"})
        assert result.success is True
        assert result.result["filter"] == "dynamics"
        assert "inverse_dynamics" in result.result["terms"]

    def test_search_glossary(self) -> None:
        """Test searching the glossary."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        result = registry.execute("search_glossary", {"query": "force"})
        assert result.success is True
        assert result.result["count"] > 0


class TestValidationTools:
    """Tests for validation tools."""

    def test_validate_cross_engine(self) -> None:
        """Test cross-engine validation queuing."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        result = registry.execute(
            "validate_cross_engine",
            {"file_path": "test.c3d", "tolerance": 0.02},
        )
        assert result.success is True
        assert "engines" in result.result
        assert len(result.result["engines"]) == 3

    def test_check_energy_conservation(self) -> None:
        """Test energy conservation check."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        result = registry.execute(
            "check_energy_conservation",
            {"tolerance": 0.01},
        )
        assert result.success is True
        assert result.result["tolerance"] == 0.01

    def test_list_physics_engines(self) -> None:
        """Test listing physics engines."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        result = registry.execute("list_physics_engines", {})
        assert result.success is True
        assert len(result.result["engines"]) == 3
        assert "available_count" in result.result


class TestToolsForProvider:
    """Tests for getting tools in provider format."""

    def test_tools_to_openai_format(self) -> None:
        """Test converting registered tools to OpenAI format."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        openai_tools = registry.get_tools_for_provider("openai")

        assert len(openai_tools) >= 10
        for tool in openai_tools:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]

    def test_tools_to_anthropic_format(self) -> None:
        """Test converting registered tools to Anthropic format."""
        registry = ToolRegistry()
        register_golf_suite_tools(registry)

        anthropic_tools = registry.get_tools_for_provider("anthropic")

        assert len(anthropic_tools) >= 10
        for tool in anthropic_tools:
            assert "name" in tool
            assert "input_schema" in tool
