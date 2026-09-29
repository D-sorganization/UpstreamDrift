"""Tests for src.shared.python.assessment.reporting (Issues #1949, #1744)."""

from __future__ import annotations

from pathlib import Path

from src.shared.python.assessment.reporting import (
    generate_issue_document,
    generate_markdown_report,
)


class TestGenerateMarkdownReport:
    def test_creates_file(self, tmp_path: Path) -> None:
        p = generate_markdown_report(
            category_id="CAT1",
            category_name="Test Category",
            grade=8.5,
            details="Some details",
            recommendations=["Fix A", "Fix B"],
            output_dir=tmp_path,
        )
        assert p.exists()

    def test_reporting_returns_path(self, tmp_path: Path) -> None:
        p = generate_markdown_report(
            category_id="CAT1",
            category_name="Test Category",
            grade=7.0,
            details="Details here",
            recommendations=["Rec 1"],
            output_dir=tmp_path,
        )
        assert isinstance(p, Path)

    def test_filename_contains_category_id(self, tmp_path: Path) -> None:
        p = generate_markdown_report(
            category_id="MYID",
            category_name="Name",
            grade=6.0,
            details="d",
            recommendations=[],
            output_dir=tmp_path,
        )
        assert "MYID" in p.name

    def test_content_includes_grade(self, tmp_path: Path) -> None:
        generate_markdown_report(
            category_id="G1",
            category_name="Grade Test",
            grade=9.5,
            details="x",
            recommendations=[],
            output_dir=tmp_path,
        )
        content = (tmp_path / next(tmp_path.iterdir()).name).read_text()
        assert "9.5" in content

    def test_content_includes_recommendations(self, tmp_path: Path) -> None:
        generate_markdown_report(
            category_id="R1",
            category_name="Rec Test",
            grade=5.0,
            details="y",
            recommendations=["Do this", "Do that"],
            output_dir=tmp_path,
        )
        content = (tmp_path / next(tmp_path.iterdir()).name).read_text()
        assert "Do this" in content
        assert "Do that" in content

    def test_reporting_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b"
        generate_markdown_report(
            category_id="X1",
            category_name="Nested",
            grade=7.0,
            details="z",
            recommendations=[],
            output_dir=nested,
        )
        assert nested.exists()


class TestGenerateIssueDocument:
    def test_creates_file(self, tmp_path: Path) -> None:
        p = generate_issue_document(
            category_id="ISS1",
            category_name="Issue Category",
            grade=3.0,
            details="Issue details",
            output_dir=tmp_path,
        )
        assert p.exists()

    def test_filename_contains_issue_prefix(self, tmp_path: Path) -> None:
        p = generate_issue_document(
            category_id="ID2",
            category_name="Low Score",
            grade=2.0,
            details="Bad",
            output_dir=tmp_path,
        )
        assert "ISSUE" in p.name

    def test_content_includes_category_name(self, tmp_path: Path) -> None:
        p = generate_issue_document(
            category_id="ID3",
            category_name="SpecialCategory",
            grade=4.0,
            details="Low",
            output_dir=tmp_path,
        )
        content = p.read_text()
        assert "SpecialCategory" in content
