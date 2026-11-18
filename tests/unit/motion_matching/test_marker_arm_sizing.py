"""Arm scale estimated from capture marker distances (#15258).

The shoulder, elbow and wrist markers sit on the skin outside the joints, so
the marker-to-marker distance is longer than the joint-to-joint length.
``arm_scale_from_markers`` removes the stated skin-to-joint offsets and
returns the scale against the de Leva arm lengths.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.anthropometry import segment_parameters
from src.shared.python.motion_matching.marker_arm_sizing import (
    ArmMarkerSettings,
    arm_scale_from_markers,
)

pytestmark = pytest.mark.unit
STATURE_M, MASS_KG = 1.71, 78.0
LABELS = (
    "LShoulderTop",
    "LElbowOut",
    "LWristTop",
    "RShoulderTop",
    "RElbowOut",
    "RWristTop",
)


def _capture(scale: float, frames: int = 40, settings: ArmMarkerSettings | None = None):
    """Markers on two straight arms of a known scale, swung through space."""
    s = settings or ArmMarkerSettings()
    upper = segment_parameters(STATURE_M, MASS_KG, "upper_arm").length_m * scale
    fore = segment_parameters(STATURE_M, MASS_KG, "forearm").length_m * scale
    rng = np.random.default_rng(3)
    points = np.zeros((frames, 6, 3))
    for f in range(frames):
        turn = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        for arm, side in enumerate((1.0, -1.0)):
            shoulder = np.array([0.0, 0.2 * side, 1.4])
            local = [
                np.array([0.0, 0.0, s.shoulder_marker_above_joint_m]),
                np.array([0.0, s.elbow_marker_from_joint_m * side, -upper]),
                np.array([s.wrist_marker_from_joint_m, 0.0, -upper - fore]),
            ]
            for k, offset in enumerate(local):
                points[f, 3 * arm + k] = shoulder + turn @ offset
    return points, np.ones((frames, 6), dtype=bool)


@pytest.mark.parametrize("scale", [0.95, 1.0, 1.08])
def test_a_known_arm_scale_is_recovered(scale: float) -> None:
    points, valid = _capture(scale)
    sizing = arm_scale_from_markers(points, valid, LABELS, STATURE_M, MASS_KG)
    assert sizing.arm_scale == pytest.approx(scale, abs=2e-3)
    for side in ("L", "R"):
        assert sizing.upper_arm_m[side] > sizing.forearm_m[side] > 0.2


def test_marker_distance_alone_overstates_the_arm() -> None:
    points, valid = _capture(1.0)
    none = ArmMarkerSettings(
        shoulder_marker_above_joint_m=0.0,
        elbow_marker_from_joint_m=0.0,
        wrist_marker_from_joint_m=0.0,
    )
    naive = arm_scale_from_markers(points, valid, LABELS, STATURE_M, MASS_KG, none)
    assert naive.arm_scale > 1.05


def test_the_reported_range_brackets_the_estimate() -> None:
    points, valid = _capture(1.02)
    sizing = arm_scale_from_markers(points, valid, LABELS, STATURE_M, MASS_KG)
    low, high = sizing.arm_scale_range
    assert low < sizing.arm_scale < high
    assert high - low > 0.02  # centimetre doubts in the offsets matter


def test_invalid_frames_are_left_out() -> None:
    points, valid = _capture(1.0)
    points[:10] = np.nan
    valid[:10] = False
    sizing = arm_scale_from_markers(points, valid, LABELS, STATURE_M, MASS_KG)
    assert sizing.arm_scale == pytest.approx(1.0, abs=2e-3)
    assert sizing.frames_used == 30


def test_bad_input_is_refused() -> None:
    points, valid = _capture(1.0)
    with pytest.raises(ValueError, match="RWristTop"):
        arm_scale_from_markers(points[:, :5], valid[:, :5], LABELS[:5], 1.71, 78.0)
    with pytest.raises(ValueError, match="shape"):
        arm_scale_from_markers(points, valid[:, :5], LABELS, 1.71, 78.0)
    with pytest.raises(ValueError, match="frame"):
        arm_scale_from_markers(points, np.zeros_like(valid), LABELS, 1.71, 78.0)
    with pytest.raises(ValueError, match="offset"):
        ArmMarkerSettings(elbow_marker_from_joint_m=-0.01)
