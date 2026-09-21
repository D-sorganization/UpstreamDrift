"""Unit tests for pipeline address solving, closure fitting, and posture summary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


def test_scaled_offsets_scales_femur_and_tibia() -> None:
    from src.shared.python.motion_matching.pipeline.address import scaled_offsets

    offsets = {
        "RKneeOut": ("femur_r", (0.0, -0.4, 0.06)),
        "RAnkleOut": ("tibia_r", (-0.01, -0.44, 0.055)),
        "RToeIn": ("calcn_r", (0.19, 0.03, -0.03)),
    }
    scaled = scaled_offsets(offsets, femur=1.1, tibia=1.2)

    # Femur should be scaled by 1.1
    np.testing.assert_allclose(scaled["RKneeOut"][1], (0.0, -0.44, 0.066), rtol=1e-5)
    # Tibia should be scaled by 1.2
    np.testing.assert_allclose(
        scaled["RAnkleOut"][1], (-0.012, -0.528, 0.066), rtol=1e-5
    )
    # Calcaneus unchanged (scale 1.0)
    np.testing.assert_allclose(scaled["RToeIn"][1], (0.19, 0.03, -0.03), rtol=1e-5)


def test_scaled_offsets_validates_inputs() -> None:
    from src.shared.python.motion_matching.pipeline.address import scaled_offsets

    with pytest.raises(ValueError, match="positive scale factors"):
        scaled_offsets({}, femur=-1.0, tibia=1.0)
    with pytest.raises(ValueError, match="positive scale factors"):
        scaled_offsets({}, femur=1.0, tibia=0.0)


def test_search_segment_scales_finds_minimum_rms() -> None:
    from unittest.mock import MagicMock
    from src.shared.python.motion_matching.pipeline.address import search_segment_scales

    lane = MagicMock()

    # Pinned RMS returns 0.01 for femur=1.0, tibia=1.0, else 0.05
    def mock_pinned_rms(doc_bytes: bytes, attachments: dict, q: np.ndarray) -> float:
        femur_y = attachments["RKneeOut"][1][1]
        tibia_y = attachments["RAnkleOut"][1][1]
        if abs(femur_y - (-0.4)) < 1e-4 and abs(tibia_y - (-0.44)) < 1e-4:
            return 0.010
        return 0.050

    lane.pinned_rms.side_effect = mock_pinned_rms
    hip_spec = {
        "bodies": [
            {"name": "femur_r", "solids": [], "contact_spheres": []},
            {"name": "femur_l", "solids": [], "contact_spheres": []},
            {"name": "tibia_r", "solids": [], "contact_spheres": []},
            {"name": "tibia_l", "solids": [], "contact_spheres": []},
        ],
        "joints": [],
        "marker_attachments": {},
        "contact": {"spheres": []},
    }
    fixed = {"Waist": ("pelvis", (0.0, 0.0, 0.0))}
    offsets = {
        "RKneeOut": ("femur_r", (0.0, -0.4, 0.06)),
        "RAnkleOut": ("tibia_r", (-0.01, -0.44, 0.055)),
    }
    q_address = np.zeros(10)

    best_spec, femur_scale, tibia_scale, table = search_segment_scales(
        lane, hip_spec, fixed, offsets, q_address, grid=(0.95, 1.00)
    )

    assert femur_scale == 1.00
    assert tibia_scale == 1.00
    assert len(table) == 4
    assert any(entry["pinned_rms_m"] == 0.010 for entry in table)


def test_search_segment_scales_validates_inputs() -> None:
    from unittest.mock import MagicMock
    from src.shared.python.motion_matching.pipeline.address import search_segment_scales

    lane = MagicMock()
    with pytest.raises(ValueError, match="grid must not be empty"):
        search_segment_scales(lane, {}, {}, {}, np.zeros(5), grid=())
    with pytest.raises(ValueError, match="grid scale values must be positive"):
        search_segment_scales(lane, {}, {}, {}, np.zeros(5), grid=(0.0,))


def test_calibrated_address_summary_builds_expected_dict() -> None:
    from unittest.mock import MagicMock
    from src.shared.python.motion_matching.pipeline.address import (
        calibrated_address_summary,
    )

    sim = MagicMock()
    sim.centre_of_mass.return_value = (np.array([0.0, 0.0, 0.9]), 75.0)
    kin = MagicMock()
    kin.coordinate_order = ["tx", "ty", "tz", "rx", "ry", "rz", "knee_r"]
    kin.support_offset.return_value = 0.02
    kin.sphere_ground_points.return_value = {"s1": np.array([0.1, 0.2, 0.0])}
    kin.marker_positions.return_value = np.zeros((5, 3))
    address = MagicMock()
    address.q = np.zeros(7)
    address.marker_rms_m = 0.012
    address.closure_error_m = 0.003
    address.lowest_sphere_height_m = 0.001
    lane = MagicMock()
    lane.points = np.zeros((1, 5, 3))
    lane.valid = np.ones((1, 5), dtype=bool)
    lane.ground = MagicMock()
    lane.ground.height_m = 0.0
    adapter = MagicMock()
    adapter.upper_body_coordinates = 6

    labels = ("m1", "m2", "m3", "m4", "m5")
    from unittest.mock import patch

    with patch(
        "src.shared.python.motion_matching.pipeline.address.posture_summary",
        return_value={"spine_bend_deg": 10.0},
    ):
        res = calibrated_address_summary(sim, kin, address, lane, labels, adapter)

    assert res["marker_rms_m"] == 0.012
    assert res["closure_error_m"] == 0.003
    assert "leg_angles_deg" in res
    assert res["posture"] == {"spine_bend_deg": 10.0}


def test_solve_address_stage_requires_static_seeds_for_fit_closure() -> None:
    from unittest.mock import MagicMock
    from src.shared.python.motion_matching.pipeline.address import (
        AddressStageInputs,
        solve_address_stage,
    )

    lane = MagicMock()
    inputs = AddressStageInputs(
        lane=lane,
        base_spec={},
        hip_spec={},
        hip_bytes=b"{}",
        upper={},
        fixed={},
        seeds_all={},
        labels=(),
        static_seeds=False,
        fit_closure=True,
    )
    with pytest.raises(ValueError, match="--fit-closure needs --static-seeds"):
        solve_address_stage(inputs)
