"""Tests for Issue #5423: Global Report Templates and Agentic Summaries.

Coverage:
- JinjaReportTemplate construction, validation, and rendering (7 tests)
- AgenticSummaryGenerator with mocked AI client (8 tests)
- AgenticSummaryGenerator fallback behaviour (5 tests)
- GLOBAL_REPORT_REGISTRY structure and content (5 tests)

Total: 25 tests
"""

from __future__ import annotations

import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.reporting import (
    GLOBAL_REPORT_REGISTRY,
    AgenticSummaryGenerator,
    JinjaReportTemplate,
)
from src.shared.python.reporting._agentic_summary import AIClient
from src.shared.python.simulation_store import SimulationDataStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_store(tmp_path):
    """Return a SimulationDataStore backed by a temporary directory."""
    return SimulationDataStore(base_dir=tmp_path / "sims")


@pytest.fixture()
def store_with_run(tmp_store):
    """Return a store pre-populated with a single run."""
    tmp_store.save_run(
        "run_001",
        {
            "engine": "drake",
            "peak_chs_mph": 108.5,
            "ball_speed_mph": 161.2,
            "launch_angle_deg": 12.3,
        },
    )
    return tmp_store


class _MockAIClient:
    """Minimal AIClient implementation for testing."""

    def __init__(self, response: str = "## Insights\n\nTest insight.") -> None:
        self._response = response
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


class _FailingAIClient:
    """AIClient that always raises an exception."""

    def complete(self, prompt: str) -> str:
        raise RuntimeError("AI unavailable")


# ===========================================================================
# JinjaReportTemplate tests
# ===========================================================================


class TestJinjaReportTemplateConstruction:
    def test_valid_construction(self):
        tpl = JinjaReportTemplate("swing_analysis", "# {{ title }}")
        assert tpl.report_type == "swing_analysis"

    def test_template_text_preserved(self):
        text = "# {{ title }}\n\n{{ body }}"
        tpl = JinjaReportTemplate("test", text)
        assert tpl.template_text == text

    def test_empty_report_type_raises(self):
        with pytest.raises((ValueError, PreconditionError)):
            JinjaReportTemplate("", "# content")

    def test_whitespace_report_type_raises(self):
        with pytest.raises((ValueError, PreconditionError)):
            JinjaReportTemplate("   ", "# content")

    def test_empty_template_text_raises(self):
        with pytest.raises((ValueError, PreconditionError)):
            JinjaReportTemplate("swing", "")

    def test_whitespace_template_text_raises(self):
        with pytest.raises((ValueError, PreconditionError)):
            JinjaReportTemplate("swing", "   ")


class TestJinjaReportTemplateRendering:
    def test_render_returns_string(self):
        tpl = JinjaReportTemplate("test", "Hello {{ name }}")
        result = tpl.render({"name": "World"})
        assert isinstance(result, str)
        assert result  # non-empty

    def test_render_substitutes_variable(self):
        tpl = JinjaReportTemplate("test", "Run: {{ run_id }}")
        result = tpl.render({"run_id": "abc_123"})
        assert "abc_123" in result

    def test_render_missing_variable_does_not_crash(self):
        tpl = JinjaReportTemplate("test", "Val: {{ missing }}")
        # Jinja2 Undefined renders as empty string; fallback replaces with ""
        result = tpl.render({})
        assert isinstance(result, str)

    def test_render_context_must_be_dict(self):
        tpl = JinjaReportTemplate("test", "{{ x }}")
        with pytest.raises((TypeError, PreconditionError)):
            tpl.render("not a dict")  # type: ignore[arg-type]

    def test_render_multiple_variables(self):
        tpl = JinjaReportTemplate("test", "{{ a }}-{{ b }}-{{ c }}")
        result = tpl.render({"a": "X", "b": "Y", "c": "Z"})
        assert "X" in result
        assert "Y" in result
        assert "Z" in result

    def test_render_numeric_values(self):
        tpl = JinjaReportTemplate("test", "CHS={{ chs }}")
        result = tpl.render({"chs": 108.5})
        assert "108.5" in result

    def test_render_escapes_html_context(self):
        tpl = JinjaReportTemplate("test", "Notes: {{ notes }}")
        result = tpl.render({"notes": "<script>alert(1)</script>"})
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


# ===========================================================================
# AgenticSummaryGenerator tests
# ===========================================================================


class TestAgenticSummaryGeneratorConstruction:
    def test_construction_without_ai_client(self, tmp_store):
        gen = AgenticSummaryGenerator(tmp_store)
        assert not gen.has_ai_client

    def test_construction_with_ai_client(self, tmp_store):
        gen = AgenticSummaryGenerator(tmp_store, ai_client=_MockAIClient())
        assert gen.has_ai_client

    def test_invalid_store_raises(self):
        with pytest.raises((TypeError, PreconditionError)):
            AgenticSummaryGenerator("not a store")  # type: ignore[arg-type]


class TestAgenticSummaryGeneratorWithAI:
    def test_generate_calls_ai_client(self, store_with_run):
        client = _MockAIClient("## Insights\n\nGreat swing.")
        gen = AgenticSummaryGenerator(store_with_run, ai_client=client)
        gen.generate("run_001")
        assert len(client.calls) == 1

    def test_generate_includes_run_id_in_prompt(self, store_with_run):
        client = _MockAIClient()
        gen = AgenticSummaryGenerator(store_with_run, ai_client=client)
        gen.generate("run_001")
        assert "run_001" in client.calls[0]

    def test_generate_includes_ai_insights_in_output(self, store_with_run):
        client = _MockAIClient("## Insights\n\nExcellent hip rotation.")
        gen = AgenticSummaryGenerator(store_with_run, ai_client=client)
        result = gen.generate("run_001")
        assert "Excellent hip rotation" in result

    def test_generate_result_is_non_empty_string(self, store_with_run):
        gen = AgenticSummaryGenerator(store_with_run, ai_client=_MockAIClient())
        result = gen.generate("run_001")
        assert isinstance(result, str) and bool(result.strip())

    def test_generate_result_contains_run_id(self, store_with_run):
        gen = AgenticSummaryGenerator(store_with_run, ai_client=_MockAIClient())
        result = gen.generate("run_001")
        assert "run_001" in result


class TestAgenticSummaryGeneratorPreconditions:
    def test_empty_run_id_raises(self, tmp_store):
        gen = AgenticSummaryGenerator(tmp_store)
        with pytest.raises((ValueError, PreconditionError)):
            gen.generate("")

    def test_invalid_run_id_characters_raises(self, tmp_store):
        gen = AgenticSummaryGenerator(tmp_store)
        with pytest.raises((ValueError, PreconditionError)):
            gen.generate("run/001")

    def test_nonexistent_run_id_raises_key_error(self, tmp_store):
        gen = AgenticSummaryGenerator(tmp_store)
        with pytest.raises(KeyError):
            gen.generate("nonexistent-run")


class TestAgenticSummaryGeneratorFallback:
    def test_no_ai_client_uses_fallback(self, store_with_run):
        gen = AgenticSummaryGenerator(store_with_run)
        result = gen.generate("run_001")
        # Fallback template contains "template-based summary"
        assert "template" in result.lower() or "run_001" in result

    def test_failing_ai_client_falls_back_gracefully(self, store_with_run):
        gen = AgenticSummaryGenerator(store_with_run, ai_client=_FailingAIClient())
        result = gen.generate("run_001")
        # Should return non-empty string (fallback)
        assert isinstance(result, str) and bool(result.strip())

    def test_fallback_includes_run_id(self, store_with_run):
        gen = AgenticSummaryGenerator(store_with_run)
        result = gen.generate("run_001")
        assert "run_001" in result

    def test_fallback_includes_data_keys(self, store_with_run):
        gen = AgenticSummaryGenerator(store_with_run)
        result = gen.generate("run_001")
        # At least one key from the stored data should appear
        assert "engine" in result or "drake" in result or "chs" in result.lower()

    def test_failing_ai_result_is_non_empty(self, store_with_run):
        gen = AgenticSummaryGenerator(store_with_run, ai_client=_FailingAIClient())
        result = gen.generate("run_001")
        assert bool(result.strip())


# ===========================================================================
# GLOBAL_REPORT_REGISTRY tests
# ===========================================================================


class TestGlobalReportRegistry:
    EXPECTED_KEYS = {"swing_analysis", "ball_flight", "biomechanics", "equipment"}

    def test_registry_has_all_expected_keys(self):
        assert self.EXPECTED_KEYS.issubset(set(GLOBAL_REPORT_REGISTRY.keys()))

    def test_all_values_are_report_templates(self):
        for key, value in GLOBAL_REPORT_REGISTRY.items():
            assert isinstance(value, JinjaReportTemplate), (
                f"Entry {key!r} is not a JinjaReportTemplate"
            )

    def test_report_type_matches_key(self):
        for key, tpl in GLOBAL_REPORT_REGISTRY.items():
            assert tpl.report_type == key

    def test_swing_analysis_template_renders(self):
        tpl = GLOBAL_REPORT_REGISTRY["swing_analysis"]
        result = tpl.render({"run_id": "r1", "title": "Test Swing"})
        assert "r1" in result

    def test_ball_flight_template_renders(self):
        tpl = GLOBAL_REPORT_REGISTRY["ball_flight"]
        result = tpl.render({"run_id": "r2", "title": "Driver"})
        assert "r2" in result

    def test_biomechanics_template_renders(self):
        tpl = GLOBAL_REPORT_REGISTRY["biomechanics"]
        result = tpl.render({"run_id": "r3"})
        assert "r3" in result

    def test_equipment_template_renders(self):
        tpl = GLOBAL_REPORT_REGISTRY["equipment"]
        result = tpl.render({"run_id": "r4"})
        assert "r4" in result

    def test_ai_client_protocol_satisfied_by_mock(self):
        client = _MockAIClient()
        assert isinstance(client, AIClient)
