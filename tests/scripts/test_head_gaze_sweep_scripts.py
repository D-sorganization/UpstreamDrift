"""Helpers of the OSV-3b sweep and engine-render scripts (no heavy runs)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.render_head_gaze_engines import head_glyph_arrows
from scripts.sweep_gaze_weight import _fixed_scales, row_from

pytestmark = pytest.mark.unit


def test_row_from_converts_metres_to_millimetres() -> None:
    ik = {
        "reference": {"marker_rms_m": 0.03, "closure_error_max_m": 0.004},
        "face_orientation": {"reference": {"rms_deg": 0.8}},
    }
    row = row_from(ik, {"gaze_weight": 3.0})
    assert row["marker_rms_mm"] == pytest.approx(30.0)
    assert row["closure_error_max_mm"] == pytest.approx(4.0)
    assert row["face_fit_deg"] == {"rms_deg": 0.8}


def test_row_from_without_face_block_reports_empty_not_zero() -> None:
    ik = {"reference": {"marker_rms_m": 0.03, "closure_error_max_m": 0.004}}
    assert row_from(ik, {})["face_fit_deg"] == {}


def test_fixed_scales_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        _fixed_scales(0.0, 1.0)


def test_head_glyphs_point_along_forward_and_at_the_ball() -> None:
    forward, sight = head_glyph_arrows([0, 0, 1.5], [1, 0, 0], [0.5, 0, 0.02])
    assert forward.label == "head_forward" and sight.label == "line_of_sight"
    assert np.allclose(forward.tail_m, [0, 0, 1.5])
    assert forward.tip_m[0] > 0.5 and forward.tip_m[2] == pytest.approx(1.5)
    assert np.allclose(sight.tip_m, [0.5, 0, 0.02])


def test_head_glyphs_reject_degenerate_input() -> None:
    with pytest.raises(ValueError):
        head_glyph_arrows([0, 0, 1], [0, 0, 0], [1, 0, 0])
    with pytest.raises(ValueError):
        head_glyph_arrows([0, 0, 1], [1, 0, 0], [0, 0, 1])
