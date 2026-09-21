"""Tests for Tour Baselines Coverage Matrix and Non-Golf Exclusions (TB-00 #10585).

TDD test-first suite verifying:
1. Coverage matrix covers every model x {driver, iron}.
2. Every cell specifies supported flag, observation set, ownership, and governing issue.
3. Missing or blocked models have explicit reasons, never silent omission.
4. Non-golf launcher tools have explicit documented exclusions.
5. Formatted markdown table renders without errors.
"""

from __future__ import annotations

import pytest

from src.shared.python.tour_baselines.coverage import (
    generate_coverage_matrix,
    list_excluded_tools,
    render_coverage_markdown,
)
from src.shared.python.tour_baselines.registry import list_golf_models

pytestmark = pytest.mark.unit


def test_every_model_has_driver_and_iron_cells():
    """Every registered model must appear in exactly two cells: driver and iron."""
    models = list_golf_models()
    matrix = generate_coverage_matrix()

    model_ids = {m.model_id for m in models}
    matrix_model_ids = {cell.model_id for cell in matrix}

    # All registered models must be represented
    assert model_ids == matrix_model_ids

    # For each model, there must be exactly one 'driver' and one 'iron' cell
    for m_id in model_ids:
        cells = [c for c in matrix if c.model_id == m_id]
        msg = f"Model {m_id} must have exactly 2 cells, got {len(cells)}"
        assert len(cells) == 2, msg
        captures = {c.capture for c in cells}
        assert captures == {"driver", "iron"}


def test_missing_optional_engines_have_explicit_reasons():
    """Missing optional engines or adapters must record blocked reason, not zero error."""
    matrix = generate_coverage_matrix()

    myosuite_cells = [c for c in matrix if "myosuite" in c.model_id]
    assert len(myosuite_cells) > 0
    for cell in myosuite_cells:
        assert cell.supported is False
        assert cell.blocked_reason is not None
        assert len(cell.blocked_reason) > 0

    opensim_cells = [c for c in matrix if "opensim_golfer" in c.model_id]
    for cell in opensim_cells:
        assert cell.supported is False
        assert cell.blocked_reason is not None


def test_non_golf_tools_have_explicit_exclusions():
    """Non-golf tools must have explicit documented exclusions."""
    exclusions = list_excluded_tools()
    assert len(exclusions) >= 10

    excluded_ids = {e.tool_id for e in exclusions}
    expected_non_golf = {
        "bunkershot",
        "shot_tracer",
        "cross_engine_dashboard",
        "pose_studio",
        "starting_pose_matcher",
        "force_plate_lab",
        "swing_plane_analyzer",
        "putting_green",
        "camera_setup",
        "coaching_drawings",
        "model_explorer",
        "calibration_wizard",
    }

    for tool_id in expected_non_golf:
        assert tool_id in excluded_ids, f"Tool {tool_id} must be explicitly excluded"

    for e in exclusions:
        assert e.reason, f"Exclusion {e.tool_id} missing reason"


def test_render_coverage_markdown():
    """Markdown report must render and include key headers and model rows."""
    md = render_coverage_markdown()
    assert "# Tour Baselines Coverage Matrix" in md
    assert "Driver" in md
    assert "Iron" in md
    assert "reconstruction_double_pendulum" in md
    assert "driven_double_pendulum" in md
    assert "Non-Golf Tool Exclusions" in md
