"""WP-1 clubhead orientation and displacement-derived travel audit tests (#15805)."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from bunkershot3d.geometry.delivery import DeliveryCondition, entry_velocity_m_s
from bunkershot3d.geometry.orientation_audit import (
    ClubheadOrientationSample,
    OrientationAuditResult,
    OrthographicVerificationFrame,
    audit_clubhead_orientation,
    build_orthographic_verification_frame,
    derive_world_travel_vectors,
)
from bunkershot3d.geometry.presets import get_preset

pytestmark = pytest.mark.unit
_SPEED, _ATTACK = 25.0, -6.0


def _run(
    face_deg: float = 0.0, *, att: float = _ATTACK, rev: bool = False
) -> OrientationAuditResult:
    """Run orientation audit on acushnet_example_1 along a linear entry path."""
    geo = get_preset("acushnet_example_1").geometry
    vel = entry_velocity_m_s(speed_m_s=_SPEED, attack_angle_deg=att).copy()
    if rev:
        vel[0] = -vel[0]
    pos = np.outer(np.arange(5) * 0.001, vel)
    cond = DeliveryCondition(face_open_deg=face_deg, attack_angle_deg=_ATTACK)
    return audit_clubhead_orientation(
        geo, pos, delivery=cond, velocities_m_s=np.tile(vel, (5, 1))
    )


def test_derive_travel_independent_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Derive travel from positions without TRAVEL_AXIS_BODY and reject bad input."""
    import bunkershot3d.geometry.delivery as dm

    monkeypatch.setattr(dm, "TRAVEL_AXIS_BODY", (999.0, 0.0, 0.0))
    pos = np.array([[0.0, 0.0, 0.0], [-0.03, 0.04, -0.005], [-0.06, 0.08, -0.010]])
    v3 = derive_world_travel_vectors(pos)
    vh = derive_world_travel_vectors(pos, horizontal_only=True)
    e3 = np.array([-0.03, 0.04, -0.005]) / math.sqrt(0.03**2 + 0.04**2 + 0.005**2)
    eh = np.array([-0.6, 0.8, 0.0])
    assert v3.shape == (2, 3) and vh.shape == (2, 3)
    assert np.allclose(v3[0], e3, atol=1e-12) and np.allclose(vh[1], eh, atol=1e-12)
    nan_row = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    for item in (np.zeros((1, 3)), np.zeros((3, 2)), np.ones((2, 3)), nan_row):
        with pytest.raises(ValueError):
            derive_world_travel_vectors(item)
    with pytest.raises(ValueError):
        vert = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -0.01]])
        derive_world_travel_vectors(vert, horizontal_only=True)


@pytest.mark.parametrize("face_open_deg", [0.0, 10.0])
def test_forward_orientation_and_attack_angle(face_open_deg: float) -> None:
    """Verify leading-edge lead, forward/upward face normal, and -6 deg attack."""
    vel = entry_velocity_m_s(speed_m_s=_SPEED, attack_angle_deg=_ATTACK)
    horiz = math.hypot(float(vel[0]), float(vel[1]))
    assert math.degrees(math.atan2(float(vel[2]), horiz)) == pytest.approx(_ATTACK)
    res = _run(face_open_deg)
    assert isinstance(res, OrientationAuditResult)
    assert res.passed and not res.is_backward_travel
    res.require_passed()
    for s in res.samples:
        assert isinstance(s, ClubheadOrientationSample)
        lead_v = s.leading_edge_world_m - s.trailing_edge_world_m
        assert float(np.dot(lead_v, s.horizontal_travel_unit)) > 0.0
        assert float(np.dot(s.face_normal_world, s.horizontal_travel_unit)) > 0.0
        assert float(s.face_normal_world[2]) > 0.0
        assert s.attack_angle_deg == pytest.approx(_ATTACK, abs=1e-6)
        assert s.velocity_attack_angle_deg == pytest.approx(_ATTACK, abs=1e-6)


def test_detects_backward_travel_reversal_and_attack_mismatch() -> None:
    """Detect trailing-edge-first (+x) travel, reversals, and attack mismatch."""
    back = _run(10.0, rev=True)
    assert not back.passed and back.is_backward_travel
    assert back.samples[0].leading_ahead_margin_m < 0.0
    assert back.samples[0].face_forward_alignment < 0.0
    with pytest.raises(ValueError, match="backward"):
        back.require_passed()
    geo = get_preset("acushnet_example_1").geometry
    cond = DeliveryCondition(face_open_deg=10.0, attack_angle_deg=_ATTACK)
    dz = -0.025 * math.tan(math.radians(abs(_ATTACK)))
    rev_p = np.array([[0.0, 0.0, 0.0], [-0.025, 0.0, dz], [0.0, 0.0, 2.0 * dz]])
    rev = audit_clubhead_orientation(geo, rev_p, delivery=cond)
    assert not rev.passed and rev.is_backward_travel
    assert not _run(10.0, att=-12.0).passed


@pytest.mark.parametrize("face_open_deg", [0.0, 10.0])
def test_orthographic_frame_headless_and_frozen(
    face_open_deg: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build 2D orthographic frame, deterministic RGBA buffer, and frozen types."""
    monkeypatch.delenv("DISPLAY", raising=False)
    audit = _run(face_open_deg)
    s = audit.samples[0]
    fa = build_orthographic_verification_frame(s, render=True)
    fb = build_orthographic_verification_frame(s, render=True)
    foff = build_orthographic_verification_frame(s, render=False)
    assert isinstance(fa, OrthographicVerificationFrame)
    assert fa.plane == "travel_vertical"
    assert fa.leading_edge_2d[0] > fa.trailing_edge_2d[0]
    assert fa.face_normal_2d[0] > 0.0 and fa.face_normal_2d[1] > 0.0
    assert fa.travel_vector_2d[0] > 0.0 and fa.travel_vector_2d[1] < 0.0
    assert foff.image_buffer is None
    assert fa.image_buffer is not None and fb.image_buffer is not None
    assert fa.image_buffer.shape == (120, 160, 4)
    assert fa.image_buffer.dtype == np.uint8
    assert np.array_equal(fa.image_buffer, fb.image_buffer)
    assert np.unique(fa.image_buffer.reshape(-1, 4), axis=0).shape[0] >= 4
    targets: tuple[tuple[object, str, object], ...] = (
        (audit, "passed", False),
        (s, "step_index", 99),
        (fa, "plane", "x"),
    )
    for obj, attr, val in targets:
        with pytest.raises(FrozenInstanceError):
            setattr(obj, attr, val)
