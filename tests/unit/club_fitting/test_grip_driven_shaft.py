"""Grip-driven shaft simulation on a synthetic planar-arc grip motion (#15684)."""

from __future__ import annotations

import dataclasses
import functools
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("mujoco")

from src.shared.python.club_fitting.grip_driven_shaft import (  # noqa: E402
    HeadGeometry,
    simulate_grip_driven_shaft,
)
from src.shared.python.club_fitting.grip_motion import (  # noqa: E402
    ClubPoseTrack,
    grip_point_in_club,
    load_settings,
)
from src.shared.python.motion_matching import club_models as cm  # noqa: E402
from src.shared.python.motion_matching.club_shaft_chain import (  # noqa: E402
    build_shaft_chain,
)

pytestmark = pytest.mark.unit

OMEGA = 12.0  # rad/s about world z, constant
IMPACT_S = 0.0
SAMPLE_DT = 0.002
HEAD = HeadGeometry(np.array([0.0, 0.0, cm.HEAD_HEIGHT_M]), np.array([0.0, 0.0, 1.0]))
FAST = dataclasses.replace(
    load_settings(),
    lead_in_s=0.06,
    post_impact_s=0.005,
    integration_dt_s=2e-4,
    record_every=1,
)


def arc_track(accel: float = 0.0) -> ClubPoseTrack:
    """Club swung about the vertical through the grip at a constant rate.

    The shaft stays horizontal (club +y along the world x axis at t=0) and the
    grip point moves along +x with constant ``accel`` so the arc is planar.
    """
    t = np.arange(-0.6, 0.05, SAMPLE_DT)
    base = Rotation.from_euler("z", -90.0, degrees=True)
    rot = Rotation.from_rotvec(np.outer(OMEGA * t, [0.0, 0.0, 1.0])) * base
    grip = np.stack([0.5 * accel * t**2, 0.0 * t, 1.0 + 0.0 * t], axis=1)
    return ClubPoseTrack(t, grip, rot.as_matrix())


@functools.cache
def run(scale: float, flex: str = "Stiff"):
    chain = build_shaft_chain(cm.DRIVER, flex=flex, stiffness_scale=scale)
    return simulate_grip_driven_shaft(arc_track(), chain, HEAD, IMPACT_S, FAST)


@pytest.mark.slow
def test_rigid_limit_head_speed_is_omega_r() -> None:
    result = run(scale=4.0)
    # grip to face centre lies along the horizontal shaft (same height)
    expected = OMEGA * -grip_point_in_club(cm.DRIVER)[1]
    assert result.head_speed_mps[-1] == pytest.approx(expected, rel=0.01)
    assert result.rigid_head_speed_mps[-1] == pytest.approx(expected, rel=0.01)
    assert result.peak_tip_deflection_m < 0.4 * run(1.0, "Stiff").peak_tip_deflection_m


@pytest.mark.slow
def test_softer_flex_deflects_more() -> None:
    soft = run(1.0, "Ladies").peak_tip_deflection_m
    stiff = run(1.0, "X-Stiff").peak_tip_deflection_m
    assert soft > stiff > 0.0


@pytest.mark.slow
def test_series_are_aligned_and_finite() -> None:
    r = run(1.0, "Stiff")
    n = len(r.times_s)
    for name in ("tip_deflection_m", "head_velocity_mps", "head_centre_m"):
        arr = getattr(r, name)
        assert arr.shape == (n, 3) and np.all(np.isfinite(arr))
    for name in ("face_delta_deg", "loft_delta_deg", "lie_delta_deg"):
        assert getattr(r, name).shape == (n,)
    assert r.times_s[-1] == pytest.approx(IMPACT_S + FAST.post_impact_s, abs=1e-9)


def test_input_contracts() -> None:
    chain = build_shaft_chain(cm.DRIVER)
    track = arc_track()
    with pytest.raises(ValueError, match="horizontal"):
        simulate_grip_driven_shaft(track, chain, HEAD, 0.0, FAST, (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="cover"):
        simulate_grip_driven_shaft(track, chain, HEAD, -0.55, FAST)
    with pytest.raises(ValueError, match="finite"):
        simulate_grip_driven_shaft(track, chain, HEAD, float("nan"), FAST)
    with pytest.raises(ValueError):
        ClubPoseTrack(track.times_s[::-1], track.grip_m, track.rotation)
    with pytest.raises(ValueError):
        HeadGeometry(np.zeros(3), np.zeros(3))
