"""Tests for src.shared.python.assessment.reporting and constants."""

from __future__ import annotations

import pytest

from src.shared.python.assessment.constants import (
    CATEGORIES,
    GROUP_MAPPING,
    GROUP_WEIGHTS,
    PRAGMATIC_PRINCIPLES,
)
from src.shared.python.assessment.reporting import (
    generate_issue_document,
    generate_markdown_report,
)


@pytest.mark.unit
def test_generate_markdown_report_creates_file(tmp_path):
    out = generate_markdown_report(
        category_id="A",
        category_name="Code Quality",
        grade=7.5,
        details="Some detail.",
        recommendations=["Refactor", "Add tests"],
        output_dir=tmp_path,
    )
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Code Quality" in text
    assert "7.5/10" in text
    assert "1. Refactor" in text
    assert "2. Add tests" in text
    # Filename contains category id + name with spaces replaced by underscores
    assert out.name == "Assessment_A_Code_Quality.md"


@pytest.mark.unit
def test_generate_markdown_report_creates_parent_dir(tmp_path):
    target = tmp_path / "nested" / "dir"
    out = generate_markdown_report(
        category_id="B",
        category_name="X",
        grade=5.0,
        details="d",
        recommendations=[],
        output_dir=target,
    )
    assert out.exists()
    assert out.parent == target


@pytest.mark.unit
def test_generate_markdown_report_requires_category_id(tmp_path):
    with pytest.raises(ValueError, match="category_id must be provided"):
        generate_markdown_report(
            category_id=None,  # type: ignore[arg-type]
            category_name="x",
            grade=1.0,
            details="d",
            recommendations=[],
            output_dir=tmp_path,
        )


@pytest.mark.unit
def test_generate_issue_document(tmp_path):
    out = generate_issue_document(
        category_id="C",
        category_name="Test Coverage",
        grade=2.1,
        details="Low coverage in module foo.",
        output_dir=tmp_path,
    )
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Test Coverage" in text
    assert "2.1/10" in text
    assert "jules:assessment" in text
    assert out.name == "ISSUE_Assessment_C_Test_Coverage.md"


@pytest.mark.unit
def test_generate_issue_document_requires_category_id(tmp_path):
    with pytest.raises(ValueError, match="category_id must be provided"):
        generate_issue_document(
            category_id=None,  # type: ignore[arg-type]
            category_name="x",
            grade=1.0,
            details="d",
            output_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_categories_are_complete():
    # Sanity: a few well-known keys are present
    assert CATEGORIES["A"]
    assert "Documentation" in CATEGORIES.values()
    assert len(CATEGORIES) == 15


@pytest.mark.unit
def test_group_weights_sum_to_one():
    total = sum(GROUP_WEIGHTS.values())
    assert total == pytest.approx(1.0)


@pytest.mark.unit
def test_group_mapping_targets_are_valid_groups():
    valid_groups = set(GROUP_WEIGHTS.keys())
    for cat, group in GROUP_MAPPING.items():
        assert group in valid_groups, f"{cat} -> {group} not in {valid_groups}"


@pytest.mark.unit
def test_pragmatic_principles_have_required_fields():
    for key, info in PRAGMATIC_PRINCIPLES.items():
        assert "name" in info
        assert "description" in info
        assert "weight" in info
        assert isinstance(info["weight"], (int, float))
        assert info["weight"] > 0, key
