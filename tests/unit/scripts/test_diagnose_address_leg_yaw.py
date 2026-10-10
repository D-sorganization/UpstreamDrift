"""Leg-chain heading math of ``scripts.diagnose_address_leg_yaw`` (#12109)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.diagnose_address_leg_yaw import heading_deg, leg_headings, wrap_deg
from src.shared.python.motion_matching.hip_calibration import (
    ANKLE_OUT_LATERAL_M,
    KNEE_OUT_LATERAL_M,
)

pytestmark = pytest.mark.unit


def _rz(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _leg(side: str, leg_yaw: float, foot_yaw: float) -> tuple[dict, np.ndarray]:
    """Flexed leg in world (x forward, y left, z up) with markers on its surface.

    ``leg_yaw`` turns the whole leg about the hip's vertical; ``foot_yaw``
    turns only the forefoot about the ankle (tibial torsion / toe-out).
    """
    sign = -1.0 if side == "R" else 1.0  # lateral direction along y
    lateral = np.array([0.0, sign, 0.0])
    hip = np.array([0.0, 0.1 * sign, 1.0])
    rot = _rz(leg_yaw)
    knee = hip + rot @ np.array([0.12, 0.0, -0.44])
    ankle = hip + rot @ np.array([0.0, 0.0, -0.88])
    toe_mid = ankle + rot @ _rz(foot_yaw) @ np.array([0.16, 0.0, -0.06])
    half = rot @ _rz(foot_yaw) @ (0.045 * lateral)
    markers = {
        f"{side}KneeOut": knee + KNEE_OUT_LATERAL_M * (rot @ lateral),
        f"{side}AnkleOut": ankle + ANKLE_OUT_LATERAL_M * (rot @ lateral),
        f"{side}ToeOut": toe_mid + half,
        f"{side}ToeIn": toe_mid - half,
    }
    return markers, hip


def _both(leg_yaw=(0.0, 0.0), foot_yaw=(0.0, 0.0)):
    m_r, h_r = _leg("R", leg_yaw[0], foot_yaw[0])
    m_l, h_l = _leg("L", leg_yaw[1], foot_yaw[1])
    return leg_headings({**m_r, **m_l}, {"right": h_r, "left": h_l})


def test_square_leg_has_zero_headings() -> None:
    out = _both()
    for side in ("right", "left"):
        assert out[side]["knee_forward_heading_deg"] == pytest.approx(0.0, abs=1e-6)
        assert out[side]["forefoot_heading_deg"] == pytest.approx(0.0, abs=1e-6)
        assert out[side]["forefoot_minus_knee_deg"] == pytest.approx(0.0, abs=1e-6)


def test_whole_leg_yaw_moves_knee_and_foot_together() -> None:
    out = _both(leg_yaw=(-20.0, 15.0))
    assert out["right"]["knee_forward_heading_deg"] == pytest.approx(-20.0, abs=1e-6)
    assert out["right"]["forefoot_minus_knee_deg"] == pytest.approx(0.0, abs=1e-6)
    assert out["left"]["knee_forward_heading_deg"] == pytest.approx(15.0, abs=1e-6)
    assert out["left"]["forefoot_minus_knee_deg"] == pytest.approx(0.0, abs=1e-6)


def test_foot_only_yaw_is_isolated_to_the_shank_to_foot_link() -> None:
    out = _both(foot_yaw=(-12.0, 9.0))
    assert out["right"]["knee_forward_heading_deg"] == pytest.approx(0.0, abs=1e-6)
    assert out["right"]["forefoot_minus_knee_deg"] == pytest.approx(-12.0, abs=1e-6)
    assert out["left"]["forefoot_minus_knee_deg"] == pytest.approx(9.0, abs=1e-6)


def test_heading_rejects_vertical_vector() -> None:
    with pytest.raises(ValueError, match="horizontal"):
        heading_deg((0.0, 0.0, 1.0))


@pytest.mark.parametrize(("raw", "wrapped"), [(190.0, -170.0), (-180.0, 180.0)])
def test_wrap_deg(raw: float, wrapped: float) -> None:
    assert wrap_deg(raw) == pytest.approx(wrapped)
