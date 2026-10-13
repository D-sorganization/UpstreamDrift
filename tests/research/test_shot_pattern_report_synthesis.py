"""Report-only scientific claims must depend on their qualified numeric inputs."""

import copy
import json
from pathlib import Path

import pytest

from scripts.research.shot_pattern_analysis.build_final_report import (
    _conclusions_and_synthesis_lines,
    _resolved_selected_comparison_lines,
)
from scripts.research.shot_pattern_analysis.final_report_inputs import (
    assemble_resolved_selected_cells,
)
from src.tools.shot_pattern_analysis.matched_formatting import format_sg_value

pytestmark = pytest.mark.unit


def _cells():
    path = Path(__file__).parent / "fixtures/shot_pattern_selected_synthesis.json"
    return json.loads(path.read_text())["cells"]


def test_synthesis_rejects_missing_data_instead_of_inventing_claims():
    with pytest.raises(ValueError, match="24|selection|resolved"):
        _conclusions_and_synthesis_lines({})


def test_synthesis_actual_numeric_ranges_and_study_sd():
    cells = _cells()
    text = "\n".join(
        _conclusions_and_synthesis_lines({"resolved_selected_cells": cells})
    )
    values = [
        c["course_scoring"]["paired_benefit_vs_straight"]["Fade"]["estimate"]
        for c in cells
        if c["config"]["club"]["club_id"] == "seven_iron"
    ]
    assert format_sg_value(min(values)) in text
    assert format_sg_value(max(values)) in text
    assert "SD 0.5" not in text
    assert "-0.046 SG" not in text
    assert "open face angles to reduced" not in text
    assert "shot_pattern_analysis.tex" in text
    assert "1°" in text and "2°" in text


def test_synthesis_numbers_change_with_actual_input():
    cells = _cells()
    before = _conclusions_and_synthesis_lines({"resolved_selected_cells": cells})
    changed = copy.deepcopy(cells)
    for c in changed:
        if c["config"]["club"]["club_id"] == "seven_iron":
            c["course_scoring"]["paired_benefit_vs_straight"]["Fade"] = {
                "estimate": -0.123,
                "lower_95": -0.124,
                "upper_95": -0.122,
            }
    after = _conclusions_and_synthesis_lines({"resolved_selected_cells": changed})
    assert before != after
    assert "-0.123" in "\n".join(after)


def test_resolved_sg_table_uses_scored_population_not_geometric_count():
    cell = copy.deepcopy(_cells()[0])
    cell["paired_comparisons"]["paired_n"] = 10000
    cell["course_scoring"]["paired_n"] = 9997
    text = "\n".join(_resolved_selected_comparison_lines([cell]))
    assert "9,997" in text
    assert "Common-Three Scored" in text


def test_synthesis_rejects_duplicate_identity():
    cells = _cells()
    cells[-1] = cells[0]
    with pytest.raises(ValueError, match="duplicate|selection|identity"):
        _conclusions_and_synthesis_lines({"resolved_selected_cells": cells})


def test_resolved_selection_rejects_wrong_supplement_protocol_and_count():
    cells = _cells()
    primary = []
    entries = []
    for cell in cells:
        if cell["source_protocol"] == "pw-primary-face-bounds/1":
            primary.append({**cell, "status": "unavailable"})
            entries.append(
                {
                    "cell": cell["cell"],
                    "status": "completed",
                    "source_protocol": cell["source_protocol"],
                    "summary": cell,
                }
            )
        else:
            primary.append(cell)
    supplement = {"report": {"unique_cells": entries}}
    assert len(assemble_resolved_selected_cells({"cells": primary}, supplement)) == 24
    bad = copy.deepcopy(supplement)
    bad["report"]["unique_cells"][0]["source_protocol"] = "unreviewed/1"
    with pytest.raises(ValueError, match="protocol"):
        assemble_resolved_selected_cells({"cells": primary}, bad)
    bad = copy.deepcopy(supplement)
    bad["report"]["unique_cells"][0]["summary"]["course_scoring"]["paired_n"] = 10001
    with pytest.raises(ValueError, match="count|population"):
        assemble_resolved_selected_cells({"cells": primary}, bad)


@pytest.mark.parametrize(
    "field,value",
    [
        ("match_face_bounds_deg", None),
        ("match_face_bounds_deg", [-7.0, 6.0]),
        ("match_speed_factors", [0.75, 1.3]),
    ],
)
def test_resolved_selection_rejects_wrong_or_missing_bounds(field, value):
    from scripts.research.shot_pattern_analysis.report_synthesis import (
        validate_resolved_selection,
    )

    cells = _cells()
    cell = next(c for c in cells if c["source_protocol"] == "pw-primary-face-bounds/1")
    cell["config"][field] = value
    with pytest.raises(ValueError, match="bounds"):
        validate_resolved_selection(cells)


def test_resolved_table_discloses_individual_counts_and_attempts():
    cell = copy.deepcopy(_cells()[0])
    cell["course_scoring"]["paired_n"] = 9997
    cell["summary"]["Straight"]["supported_n"] = 9999
    text = "\n".join(_resolved_selected_comparison_lines([cell]))
    assert "9,999 / 10,000" in text
    assert "9,997 / 10,000" in text
    assert "Individual Supported" in text


def test_report_source_receipt_binds_synthesis_module():
    from scripts.research.shot_pattern_analysis.build_final_report import (
        _report_source_hashes,
    )

    assert (
        "scripts/research/shot_pattern_analysis/report_synthesis.py"
        in _report_source_hashes()
    )
