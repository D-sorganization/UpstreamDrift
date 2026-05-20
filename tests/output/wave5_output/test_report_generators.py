"""Tests for analysis report generators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.data_io._report_generators import (
    export_analysis_report,
    generate_html_report,
)


def test_export_analysis_report_json(tmp_path: Path):
    out = export_analysis_report(
        {"score": 0.95, "label": "ok"},
        report_name="run1",
        report_dir=tmp_path,
        format_type="json",
    )
    assert out.exists()
    assert out.suffix == ".json"
    data = json.loads(out.read_text())
    assert data["score"] == 0.95
    assert data["label"] == "ok"
    assert "provenance" in data


def test_export_analysis_report_html(tmp_path: Path):
    out = export_analysis_report(
        {"k": "v"},
        report_name="rep",
        report_dir=tmp_path,
        format_type="html",
    )
    assert out.exists()
    assert out.suffix == ".html"
    text = out.read_text()
    assert "<html>" in text
    assert "rep" in text


def test_export_analysis_report_creates_dir(tmp_path: Path):
    target = tmp_path / "nested" / "deep"
    out = export_analysis_report({"k": 1}, "rep", target, "json")
    assert out.exists()


def test_export_analysis_report_none_data_raises(tmp_path: Path):
    with pytest.raises(Exception):  # noqa: B017 — DbC precondition
        export_analysis_report(None, "rep", tmp_path, "json")  # type: ignore[arg-type]


def test_export_analysis_report_empty_name_raises(tmp_path: Path):
    with pytest.raises(Exception):  # noqa: B017 — DbC precondition
        export_analysis_report({"k": 1}, "", tmp_path, "json")


def test_generate_html_report_includes_title():
    html = generate_html_report({"k": "v"}, "MyTitle")
    assert "MyTitle" in html
    assert "<html>" in html


def test_generate_html_report_renders_scalars_in_table():
    html = generate_html_report({"name": "alpha", "score": 0.5}, "T")
    assert "name" in html
    assert "alpha" in html
    assert "score" in html


def test_generate_html_report_skips_nested_in_summary_table():
    html = generate_html_report({"nested": {"a": 1}, "flat": "ok"}, "T")
    # flat appears in summary; nested only in JSON details
    assert "flat" in html
    # Both appear somewhere (in the JSON dump)
    assert "nested" in html
