"""Compact report tables preserve numeric tokens and exact source mappings."""

from pathlib import Path
import json
import re
import shutil

import pytest

from scripts.research.shot_pattern_analysis.build_final_report import (
    _resolved_selected_comparison_lines,
)
from src.tools.shot_pattern_analysis.matched_formatting import format_sg_interval_record

pytestmark = pytest.mark.unit


def _cells():
    return json.loads(
        (
            Path(__file__).parent / "fixtures/shot_pattern_selected_synthesis.json"
        ).read_text()
    )["cells"]


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="Pandoc is optional")
def test_resolved_table_keeps_every_numeric_token_intact_in_pdf(tmp_path):
    pytest.importorskip("reportlab")
    pypdf = pytest.importorskip("pypdf")
    from scripts.research.shot_pattern_analysis.build_pdf import render_markdown_to_pdf

    cells = _cells()
    source = tmp_path / "preview.md"
    source.write_text(
        "# Partial Preview\n\n## Selected Results\n\n"
        + "\n".join(_resolved_selected_comparison_lines(cells))
    )
    output = tmp_path / "preview.pdf"
    render_markdown_to_pdf(source, output)
    text = "\n".join(page.extract_text() for page in pypdf.PdfReader(output).pages)
    for cell in cells:
        for pattern in ("Draw", "Fade"):
            interval = format_sg_interval_record(
                cell["course_scoring"]["paired_benefit_vs_straight"][pattern]
            )
            tokens = re.findall(r"[+-]\d+(?:\.\d+)?(?:e[+-]\d+)?", interval)
            for token in tokens:
                assert token in text, f"numeric token split across lines: {token}"


def test_resolved_table_maps_all_cells_and_independent_populations():
    cells = _cells()
    text = "\n".join(_resolved_selected_comparison_lines(cells))
    for cell in cells:
        assert cell["cell"] in text
        assert cell["source_protocol"] in text
    assert "C01" in text and "C24" in text
    assert "Individual Supported" in text and "Common-Three Scored" in text
    assert (
        max(line.count("|") - 1 for line in text.splitlines() if line.startswith("|"))
        <= 5
    )


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="Pandoc is optional")
def test_keyed_effects_preserve_order_identity_intervals_and_counts(tmp_path):
    pytest.importorskip("reportlab")
    pypdf = pytest.importorskip("pypdf")
    from scripts.research.shot_pattern_analysis.build_pdf import render_markdown_to_pdf
    from scripts.research.shot_pattern_analysis.report_tables import keyed_effect_table

    cells = _cells()
    effects = []
    for i in range(12):
        a, b = cells[i * 2 : i * 2 + 2]
        effects.append(
            {
                "comparison_kind": "test_ordered_contrast",
                "source_cells": (a["cell"], b["cell"]),
                "source_protocols": (a["source_protocol"], b["source_protocol"]),
                "common_six_n": 9997,
                "total_n_per_population": 10000,
                "draw_minus_straight_change": a["course_scoring"][
                    "paired_benefit_vs_straight"
                ]["Draw"],
                "fade_minus_straight_change": a["course_scoring"][
                    "paired_benefit_vs_straight"
                ]["Fade"],
            }
        )
    text = keyed_effect_table(effects)
    assert "E01" in text and "E12" in text
    for effect in effects:
        a, b = effect["source_cells"]
        assert text.index(a) < text.index(b)
        for protocol in effect["source_protocols"]:
            assert protocol in text
    source = tmp_path / "effects.md"
    source.write_text("# Partial Preview\n\n" + text)
    output = tmp_path / "effects.pdf"
    render_markdown_to_pdf(source, output)
    extracted = "\n".join(p.extract_text() for p in pypdf.PdfReader(output).pages)
    for effect in effects:
        for key in ("draw_minus_straight_change", "fade_minus_straight_change"):
            interval = format_sg_interval_record(effect[key])
            for token in re.findall(r"[+-]\d+(?:\.\d+)?(?:e[+-]\d+)?", interval):
                assert token in extracted
    assert "9,997 / 10,000" in extracted
