"""Anatomical leg-marker constraints for the ground-support calibration (#11737).

A single lateral marker per segment lets the alternating calibration trade a
segment's twist about its long axis against the marker's azimuth around that
axis, so hip rotation and foot yaw become gauge modes held only by the offset
prior. These tests pin the projection that removes that freedom.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching import marker_calibration
from src.shared.python.motion_matching.pipeline.constants import (
    LEG_SEEDS,
    square_forefoot_seeds,
)
from src.shared.python.motion_matching.pipeline.leg_marker_constraints import (
    anatomical_leg_constraint,
    preserve_azimuth,
    square_forefoot,
)
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

pytestmark = pytest.mark.unit


def _azimuth_deg(offset: tuple[float, float, float]) -> float:
    return float(np.degrees(np.arctan2(offset[0], offset[2])))


def test_preserve_azimuth_keeps_the_axial_component_and_the_seed_direction() -> None:
    seed = (0.0, -0.40, 0.06)
    out = preserve_azimuth((0.106, -0.386, 0.068), seed)
    assert out[1] == pytest.approx(-0.386)
    assert _azimuth_deg(out) == pytest.approx(_azimuth_deg(seed), abs=1e-9)
    # Least-squares projection onto the seed ray: the radial length is the
    # component of the free (x, z) placement along the seed direction.
    assert out[2] == pytest.approx(0.068)
    assert out[0] == pytest.approx(0.0, abs=1e-12)


def test_preserve_azimuth_clamps_a_placement_behind_the_axis_to_the_axis() -> None:
    out = preserve_azimuth((0.0, -0.40, -0.05), (0.0, -0.40, 0.06))
    assert out == pytest.approx((0.0, -0.40, 0.0))


def test_preserve_azimuth_rejects_a_seed_on_the_segment_axis() -> None:
    with pytest.raises(ValueError, match="azimuth"):
        preserve_azimuth((0.01, -0.4, 0.02), (0.0, -0.4, 0.0))


def test_square_forefoot_shares_one_forward_offset_and_keeps_the_midpoint() -> None:
    offsets = {
        "LToeIn": ("calcn_l", (0.1355, -0.0014, 0.0883)),
        "LToeOut": ("calcn_l", (0.1828, -0.0033, 0.0005)),
    }
    out = square_forefoot(offsets, "L")
    x_in, x_out = out["LToeIn"][1][0], out["LToeOut"][1][0]
    assert x_in == pytest.approx(x_out)
    assert x_in == pytest.approx(0.5 * (0.1355 + 0.1828))
    assert out["LToeIn"][1][1:] == offsets["LToeIn"][1][1:]
    assert out["LToeIn"][0] == "calcn_l"


def test_square_forefoot_can_pin_the_lateral_midpoint() -> None:
    # A toe pair shifted sideways in the calcn frame is foot yaw in disguise:
    # 6 cm at 18 cm forward is about 18 degrees. Pinning the midpoint's z keeps
    # the spacing and removes the shift.
    offsets = {
        "LToeIn": ("calcn_l", (0.1355, -0.0014, 0.0883)),
        "LToeOut": ("calcn_l", (0.1828, -0.0033, 0.0005)),
    }
    out = square_forefoot(offsets, "L", mid_z=-0.015)
    z_in, z_out = out["LToeIn"][1][2], out["LToeOut"][1][2]
    assert 0.5 * (z_in + z_out) == pytest.approx(-0.015)
    assert z_in - z_out == pytest.approx(0.0883 - 0.0005)


def test_square_forefoot_requires_both_toe_markers() -> None:
    with pytest.raises(ValueError, match="RToeOut"):
        square_forefoot({"RToeIn": ("calcn_r", (0.2, 0.0, 0.0))}, "R")


def test_anatomical_constraint_projects_every_leg_marker_and_leaves_others() -> None:
    seeds = square_forefoot_seeds(LEG_SEEDS)
    drifted = {
        "RKneeOut": ("femur_r", (0.106, -0.386, 0.068)),
        "LKneeOut": ("femur_l", (0.0137, -0.3975, -0.0005)),
        "RAnkleOut": ("tibia_r", (-0.0095, -0.3799, 0.0622)),
        "LAnkleOut": ("tibia_l", (0.0048, -0.4134, -0.0625)),
        "RToeIn": ("calcn_r", (0.1738, 0.0003, -0.0245)),
        "RToeOut": ("calcn_r", (0.1589, -0.0109, 0.0884)),
        "LToeIn": ("calcn_l", (0.1355, -0.0014, 0.0883)),
        "LToeOut": ("calcn_l", (0.1828, -0.0033, 0.0005)),
        "LShoulder": ("LS", (0.0, 0.04, 0.0)),
    }
    out = anatomical_leg_constraint(seeds)(drifted)
    assert set(out) == set(drifted)
    assert out["LShoulder"] == drifted["LShoulder"]
    for label in ("RKneeOut", "LKneeOut", "RAnkleOut", "LAnkleOut"):
        assert _azimuth_deg(out[label][1]) == pytest.approx(
            _azimuth_deg(seeds[label][1]), abs=1e-9
        )
    for side in ("R", "L"):
        toe_in, toe_out = out[f"{side}ToeIn"][1], out[f"{side}ToeOut"][1]
        assert toe_in[0] == pytest.approx(toe_out[0])
        seed_mid = 0.5 * (seeds[f"{side}ToeIn"][1][2] + seeds[f"{side}ToeOut"][1][2])
        assert 0.5 * (toe_in[2] + toe_out[2]) == pytest.approx(seed_mid)


def test_anatomical_constraint_rejects_a_marker_moved_to_another_body() -> None:
    seeds = square_forefoot_seeds(LEG_SEEDS)
    constraint = anatomical_leg_constraint(seeds)
    with pytest.raises(ValueError, match="body"):
        constraint({"RKneeOut": ("tibia_r", (0.0, -0.4, 0.06))})


def _one_body_rig() -> tuple[TourCapture, dict[str, str]]:
    labels = ("M1", "M2")
    points = np.array([[[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]]] * 3)
    capture = TourCapture(np.arange(3) / 360.0, labels, points, np.ones((3, 2), bool))
    return capture, {"M1": "A", "M2": "A"}


def test_calibration_applies_the_constraint_to_every_placement() -> None:
    capture, bodies = _one_body_rig()
    seen: list[dict[str, tuple[str, tuple[float, float, float]]]] = []

    def constrain(
        offsets: dict[str, tuple[str, tuple[float, float, float]]],
    ) -> dict[str, tuple[str, tuple[float, float, float]]]:
        seen.append(offsets)
        return {k: (b, (0.0, o[1], o[2])) for k, (b, o) in offsets.items()}

    result = marker_calibration.calibrate_marker_offsets(
        capture,
        bodies,
        lambda q: {"A": (np.eye(3), np.zeros(3))},
        lambda offsets, cap: np.zeros((cap.frames, 1)),
        initial_q=np.zeros(1),
        iterations=2,
        constrain=constrain,
    )
    assert len(seen) == 2
    assert all(offset[0] == 0.0 for _, offset in result.offsets.values())


def test_calibration_rejects_a_constraint_that_drops_a_marker() -> None:
    capture, bodies = _one_body_rig()
    with pytest.raises(ValueError, match="constrain"):
        marker_calibration.calibrate_marker_offsets(
            capture,
            bodies,
            lambda q: {"A": (np.eye(3), np.zeros(3))},
            lambda offsets, cap: np.zeros((cap.frames, 1)),
            initial_q=np.zeros(1),
            iterations=1,
            constrain=lambda offsets: {"M1": offsets["M1"]},
        )
