"""Analytic controls for the declared motion-limit audit (#15494)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from scripts.research.nullspace_motion_limits import audit_motion_limits

pytestmark = pytest.mark.unit


def _audit(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "times_s": [0.0, 0.5, 2.0],
        "q_rad": [[0.0, 0.0], [0.5, -0.25], [0.2, 0.5]],
        "coordinate_order": ["lead_wrist_flexion", "trail_wrist_flexion"],
        "lower_rad": [-1.0, -0.5],
        "upper_rad": [1.0, 1.0],
        "max_speed_rad_s": [2.0, 1.0],
        "limit_source": "Synthetic model-specific envelope; not human ROM",
    }
    args.update(overrides)
    return audit_motion_limits(**args)


def test_rom_and_segment_speed_margins_match_hand_calculation() -> None:
    r = _audit()
    assert r["declared_limits_satisfied"] is True
    assert r["rom"]["minimum_margin"] == pytest.approx([0.5, 0.25])
    assert r["rom"]["narrowest"] == {
        "sample": 1,
        "axis": 1,
        "margin": 0.25,
    }
    assert r["speed"]["minimum_margin"] == pytest.approx([1.0, 0.5])
    assert r["peak_speed_rad_s"] == pytest.approx([1.0, 0.5])
    assert r["coupled_posture"] is None
    assert r["forward_replay_verified"] is False
    assert r["physiological_feasibility_verified"] is False
    assert r["interpolation"] == "piecewise_linear_unwrapped_radians"
    assert r["coordinate_order"][0] == "lead_wrist_flexion"
    assert r["limit_source"].startswith("Synthetic")
    json.dumps(r, allow_nan=False)


def test_rom_violation_is_reported_without_clipping_or_mutation() -> None:
    q = np.array([[0.0, 0.0], [1.25, -0.75], [0.2, 0.5]])
    before = q.copy()
    r = _audit(q_rad=q)
    assert r["declared_limits_satisfied"] is False
    assert r["rom"]["violations"] == [
        {"sample": 1, "axis": 0, "margin": -0.25},
        {"sample": 1, "axis": 1, "margin": -0.25},
    ]
    np.testing.assert_array_equal(q, before)


def test_speed_detects_fast_segment_despite_admissible_positions() -> None:
    r = _audit(times_s=[0.0, 0.1, 2.0])
    assert r["rom"]["violations"] == []
    assert r["speed"]["violations"] == [
        {"segment": 0, "axis": 0, "margin": -3.0},
        {"segment": 0, "axis": 1, "margin": -1.5},
    ]
    assert r["declared_limits_satisfied"] is False


def test_coupling_rejects_pose_inside_every_independent_rom_limit() -> None:
    r = _audit(coupled_matrix=[[1.0, 1.0]], coupled_upper_rad=[0.5])
    assert r["rom"]["violations"] == []
    assert r["coupled_posture"]["minimum_margin"] == pytest.approx([-0.2 / np.sqrt(2)])
    assert r["coupled_posture"]["violations"][0]["sample"] == 2
    assert r["declared_limits_satisfied"] is False


def test_declared_interior_margin_rejects_a_boundary_pose() -> None:
    assert _audit(q_rad=[[0, 0], [1, 0], [0, 0]])["declared_limits_satisfied"]
    r = _audit(q_rad=[[0, 0], [1, 0], [0, 0]], safety_margin_rad=0.1)
    assert r["rom"]["minimum_margin"][0] == pytest.approx(-0.1)
    assert r["declared_limits_satisfied"] is False


def test_coupled_margin_is_normalized_and_row_scale_independent() -> None:
    a = _audit(coupled_matrix=[[1, 1]], coupled_upper_rad=[1], safety_margin_rad=0.1)
    b = _audit(coupled_matrix=[[7, 7]], coupled_upper_rad=[7], safety_margin_rad=0.1)
    assert a["coupled_posture"]["minimum_margin"] == pytest.approx(
        b["coupled_posture"]["minimum_margin"]
    )
    assert a["coupled_posture"]["minimum_margin"] == pytest.approx(
        [0.3 / np.sqrt(2) - 0.1]
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"times_s": [0, 0, 1]},
        {"times_s": [0, 2, 1]},
        {"times_s": [0, np.nan, 2]},
        {"times_s": [[0, 1, 2]]},
        {"times_s": [0]},
        {"q_rad": [[0, 0]]},
        {"q_rad": [0, 1, 2]},
        {"q_rad": [[], [], []]},
        {"q_rad": [[0, 0], [np.inf, 0], [0, 0]]},
        {"lower_rad": [1, 0]},
        {"upper_rad": [1]},
        {"upper_rad": [np.nan, 1]},
        {"max_speed_rad_s": [0, 1]},
        {"max_speed_rad_s": [np.inf, 1]},
        {"max_speed_rad_s": [1]},
        {"coordinate_order": ["wrist", "wrist"]},
        {"coordinate_order": ["wrist"]},
        {"coordinate_order": ["wrist", " "]},
        {"coordinate_order": "ab"},
        {"limit_source": " "},
        {"limit_source": None},
        {"safety_margin_rad": -0.1},
        {"safety_margin_rad": np.nan},
        {"safety_margin_rad": 0.8},
        {"coupled_matrix": [[1, 1]]},
        {"coupled_upper_rad": [1]},
        {"coupled_matrix": [[0, 0]], "coupled_upper_rad": [1]},
        {"coupled_matrix": [[1]], "coupled_upper_rad": [1]},
        {"coupled_matrix": [[1, np.inf]], "coupled_upper_rad": [1]},
        {"coupled_matrix": [[1, 1]], "coupled_upper_rad": [np.nan]},
        {"times_s": [0, 1e-320, 2]},
    ],
)
def test_invalid_or_incomplete_declarations_fail_closed(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        _audit(**overrides)
