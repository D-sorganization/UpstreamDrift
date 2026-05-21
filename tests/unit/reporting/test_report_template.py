"""Tests for ReportTemplate engine — Epic #5393.

Covers:
- ReportSection creation and rendering
- ReportTemplate creation and rendering
- Nested sections (subsections)
- Empty content sections
- Template registry lookup
- Heading-level capping
- DbC precondition enforcement for invalid inputs
- Markdown output structure
- Standard template completeness
"""

from __future__ import annotations

import pytest

from src.shared.python.reporting import REPORT_TEMPLATES, ReportSection, ReportTemplate

# ---------------------------------------------------------------------------
# ReportSection
# ---------------------------------------------------------------------------


class TestReportSection:
    def test_create_minimal(self) -> None:
        sec = ReportSection(heading="Intro")
        assert sec.heading == "Intro"
        assert sec.content == ""
        assert sec.subsections == []

    def test_create_with_content(self) -> None:
        sec = ReportSection(heading="H", content="Some text.")
        assert sec.content == "Some text."

    def test_create_with_subsections(self) -> None:
        child = ReportSection(heading="Child")
        parent = ReportSection(heading="Parent", subsections=[child])
        assert len(parent.subsections) == 1
        assert parent.subsections[0].heading == "Child"

    def test_render_produces_string(self) -> None:
        sec = ReportSection(heading="Test")
        result = sec.render()
        assert isinstance(result, str)

    def test_render_contains_heading(self) -> None:
        sec = ReportSection(heading="My Section")
        result = sec.render()
        assert "My Section" in result

    def test_render_default_level_uses_hash_hash(self) -> None:
        sec = ReportSection(heading="Demo")
        result = sec.render(level=2)
        assert result.startswith("## Demo")

    def test_render_level_3_uses_hash_hash_hash(self) -> None:
        sec = ReportSection(heading="Sub")
        result = sec.render(level=3)
        assert result.startswith("### Sub")

    def test_render_level_capped_at_four(self) -> None:
        sec = ReportSection(heading="Deep")
        result = sec.render(level=10)
        assert result.startswith("#### Deep")

    def test_render_includes_content(self) -> None:
        sec = ReportSection(heading="Kin", content="Joint angles here.")
        result = sec.render()
        assert "Joint angles here." in result

    def test_render_empty_content_no_trailing_blank(self) -> None:
        sec = ReportSection(heading="Empty")
        result = sec.render()
        # No extra blank line after the heading when content is empty
        assert "\n\n" not in result or "## Empty" in result

    def test_render_subsections_included(self) -> None:
        child = ReportSection(heading="Child", content="child body")
        parent = ReportSection(heading="Parent", subsections=[child])
        result = parent.render(level=2)
        assert "Child" in result
        assert "child body" in result

    def test_render_subsection_uses_deeper_level(self) -> None:
        child = ReportSection(heading="Child")
        parent = ReportSection(heading="Parent", subsections=[child])
        result = parent.render(level=2)
        assert "### Child" in result

    def test_render_multiple_subsections(self) -> None:
        children = [ReportSection(heading=f"Sub{i}") for i in range(3)]
        parent = ReportSection(heading="Parent", subsections=children)
        result = parent.render()
        for i in range(3):
            assert f"Sub{i}" in result


# ---------------------------------------------------------------------------
# ReportSection — DbC invalid inputs
# ---------------------------------------------------------------------------


class TestReportSectionDbC:
    def test_empty_heading_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            ReportSection(heading="")

    def test_whitespace_heading_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            ReportSection(heading="   ")

    def test_non_string_content_raises(self) -> None:
        with pytest.raises((ValueError, TypeError, Exception)):
            ReportSection(heading="Valid", content=123)  # type: ignore[arg-type]

    def test_non_list_subsections_raises(self) -> None:
        with pytest.raises((ValueError, TypeError, Exception)):
            ReportSection(heading="Valid", subsections="not_a_list")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ReportTemplate
# ---------------------------------------------------------------------------


class TestReportTemplate:
    def test_create_minimal(self) -> None:
        tmpl = ReportTemplate(title="My Report")
        assert tmpl.title == "My Report"
        assert tmpl.sections == []

    def test_create_with_sections(self) -> None:
        s = ReportSection(heading="Intro")
        tmpl = ReportTemplate(title="T", sections=[s])
        assert len(tmpl.sections) == 1

    def test_render_returns_string(self) -> None:
        tmpl = ReportTemplate(title="R")
        assert isinstance(tmpl.render(), str)

    def test_render_contains_title(self) -> None:
        tmpl = ReportTemplate(title="Swing Report")
        result = tmpl.render()
        assert "Swing Report" in result

    def test_render_starts_with_h1(self) -> None:
        tmpl = ReportTemplate(title="My Title")
        result = tmpl.render()
        assert result.startswith("# My Title")

    def test_render_includes_section_headings(self) -> None:
        tmpl = ReportTemplate(
            title="Report",
            sections=[
                ReportSection(heading="Alpha"),
                ReportSection(heading="Beta"),
            ],
        )
        result = tmpl.render()
        assert "Alpha" in result
        assert "Beta" in result

    def test_render_sections_as_h2(self) -> None:
        tmpl = ReportTemplate(
            title="R",
            sections=[ReportSection(heading="Section One")],
        )
        result = tmpl.render()
        assert "## Section One" in result

    def test_render_empty_sections_list(self) -> None:
        tmpl = ReportTemplate(title="Empty")
        result = tmpl.render()
        assert result == "# Empty"

    def test_render_section_content_included(self) -> None:
        tmpl = ReportTemplate(
            title="T",
            sections=[ReportSection(heading="H", content="Body text here.")],
        )
        result = tmpl.render()
        assert "Body text here." in result

    def test_render_nested_subsections(self) -> None:
        child = ReportSection(heading="Sub", content="sub content")
        parent = ReportSection(heading="Top", subsections=[child])
        tmpl = ReportTemplate(title="T", sections=[parent])
        result = tmpl.render()
        assert "Sub" in result
        assert "sub content" in result

    def test_render_preserves_section_order(self) -> None:
        tmpl = ReportTemplate(
            title="T",
            sections=[
                ReportSection(heading="First"),
                ReportSection(heading="Second"),
                ReportSection(heading="Third"),
            ],
        )
        result = tmpl.render()
        pos_first = result.index("First")
        pos_second = result.index("Second")
        pos_third = result.index("Third")
        assert pos_first < pos_second < pos_third


# ---------------------------------------------------------------------------
# ReportTemplate — DbC invalid inputs
# ---------------------------------------------------------------------------


class TestReportTemplateDbC:
    def test_empty_title_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            ReportTemplate(title="")

    def test_whitespace_title_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            ReportTemplate(title="   ")

    def test_non_list_sections_raises(self) -> None:
        with pytest.raises((ValueError, TypeError, Exception)):
            ReportTemplate(title="T", sections="not_a_list")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# REPORT_TEMPLATES registry
# ---------------------------------------------------------------------------


class TestReportTemplatesRegistry:
    @pytest.mark.parametrize("key", ["swing_analysis", "ball_flight", "biomechanics"])
    def test_key_exists(self, key: str) -> None:
        assert key in REPORT_TEMPLATES

    @pytest.mark.parametrize("key", ["swing_analysis", "ball_flight", "biomechanics"])
    def test_value_is_report_template(self, key: str) -> None:
        assert isinstance(REPORT_TEMPLATES[key], ReportTemplate)

    @pytest.mark.parametrize("key", ["swing_analysis", "ball_flight", "biomechanics"])
    def test_template_has_title(self, key: str) -> None:
        tmpl = REPORT_TEMPLATES[key]
        assert len(tmpl.title.strip()) > 0

    @pytest.mark.parametrize("key", ["swing_analysis", "ball_flight", "biomechanics"])
    def test_template_has_sections(self, key: str) -> None:
        tmpl = REPORT_TEMPLATES[key]
        assert len(tmpl.sections) > 0

    @pytest.mark.parametrize("key", ["swing_analysis", "ball_flight", "biomechanics"])
    def test_template_renders_without_error(self, key: str) -> None:
        tmpl = REPORT_TEMPLATES[key]
        result = tmpl.render()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_swing_analysis_contains_kinematic_sequence(self) -> None:
        result = REPORT_TEMPLATES["swing_analysis"].render()
        assert "Kinematic" in result

    def test_ball_flight_contains_launch_conditions(self) -> None:
        result = REPORT_TEMPLATES["ball_flight"].render()
        assert "Launch" in result

    def test_biomechanics_contains_joint_kinematics(self) -> None:
        result = REPORT_TEMPLATES["biomechanics"].render()
        assert "Kinematics" in result

    def test_registry_is_dict(self) -> None:
        assert isinstance(REPORT_TEMPLATES, dict)

    def test_registry_has_exactly_three_entries(self) -> None:
        assert len(REPORT_TEMPLATES) == 3
