"""Shared curl-to-contact posing (epic #15150), tested on a synthetic finger.

The finger here is three rigid planar links with a skin offset, so these tests
do not depend on any hand asset: any backend that offers ``PosableHand`` gets
the same behaviour.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pytest

from src.shared.python.model_appearance.hand_backends.posing import (
    CoupledJoint,
    CurlChain,
    CurlJoint,
    CurlSettings,
    FlatHandLandmarks,
    club_axis_in_hand,
    curl_to_contact,
    cylinder_distance,
    flat_hand_rotation,
    wrap_to_contact,
)

pytestmark = pytest.mark.unit

_RADIUS = 0.0125
_HALF_THICKNESS = 0.008
_LINKS = (0.040, 0.026, 0.022)
_LIMIT = math.radians(110.0)
_SETTINGS = CurlSettings(
    step_rad=math.radians(2.0),
    bisection_steps=24,
    contact_tolerance_m=0.0008,
    penetration_allowance_m=0.0001,
)


class _PlanarFinger:
    """Three links curling in the xy-plane about ``z``; skin on the -y side."""

    bones = ("prox", "mid", "dist")

    def __init__(self, base_x: float) -> None:
        self._base = np.array([base_x, _RADIUS + _HALF_THICKNESS, 0.0])
        self._samples = 25

    def points(self, angles: Mapping[str, float]) -> np.ndarray:
        origin = self._base.copy()
        heading = 0.0
        out = []
        for bone, length in zip(self.bones, _LINKS, strict=True):
            heading -= angles.get(bone, 0.0)  # flexion curls toward -y
            along = np.array([math.cos(heading), math.sin(heading), 0.0])
            normal = np.array([-along[1], along[0], 0.0])
            s = np.linspace(0.0, length, self._samples)[:, None]
            out.append(origin + s * along - _HALF_THICKNESS * normal)
            origin = origin + length * along
        return np.vstack(out)

    def segment_ids(self, bone: str) -> np.ndarray:
        k = self.bones.index(bone)
        return np.arange(k * self._samples, (k + 1) * self._samples)

    def moved_ids(self, bone: str) -> np.ndarray:
        k = self.bones.index(bone)
        return np.arange(k * self._samples, len(self.bones) * self._samples)


def _chain(limit: float = _LIMIT) -> CurlChain:
    return CurlChain(
        "finger",
        tuple(CurlJoint(bone, 0.0, limit, 0.0) for bone in _PlanarFinger.bones),
    )


def test_cylinder_distance_is_signed_and_axis_aligned() -> None:
    distance = cylinder_distance(_RADIUS)
    pts = np.array([[_RADIUS, 0.0, 5.0], [0.0, 0.0, -1.0], [0.0, 2 * _RADIUS, 0.0]])
    np.testing.assert_allclose(distance(pts), [0.0, -_RADIUS, _RADIUS], atol=1e-12)
    with pytest.raises(ValueError):
        cylinder_distance(0.0)


def test_curl_brings_every_segment_into_contact_without_penetration() -> None:
    finger = _PlanarFinger(base_x=-0.012)
    distance = cylinder_distance(_RADIUS)
    result = curl_to_contact(finger, (_chain(),), distance, _SETTINGS, {})
    posed = finger.points(result.angles)
    assert float(distance(posed).min()) >= -_SETTINGS.penetration_allowance_m - 1e-9
    for bone in finger.bones:
        gap = result.gaps[bone]
        assert gap >= -_SETTINGS.penetration_allowance_m - 1e-9
        assert gap <= _SETTINGS.contact_tolerance_m, f"{bone} gap {gap * 1e3:.2f} mm"
        assert 0.0 <= result.angles[bone] <= _LIMIT
    assert result.angles["mid"] > math.radians(30.0), "finger did not wrap"


def test_curl_is_proximal_to_distal_and_deterministic() -> None:
    finger = _PlanarFinger(base_x=-0.012)
    distance = cylinder_distance(_RADIUS)
    first = curl_to_contact(finger, (_chain(),), distance, _SETTINGS, {})
    second = curl_to_contact(finger, (_chain(),), distance, _SETTINGS, {})
    assert first.angles == second.angles
    assert first.order == ("prox", "mid", "dist")


def test_joint_limit_is_respected_when_contact_is_out_of_reach() -> None:
    finger = _PlanarFinger(base_x=-0.012)
    far = cylinder_distance(_RADIUS * 0.2)
    limit = math.radians(20.0)
    result = curl_to_contact(finger, (_chain(limit),), far, _SETTINGS, {})
    assert all(angle <= limit + 1e-12 for angle in result.angles.values())
    assert result.gaps["prox"] > _SETTINGS.contact_tolerance_m


def test_coupled_joint_follows_its_driver_up_to_its_cap() -> None:
    finger = _PlanarFinger(base_x=-0.012)
    cap = math.radians(6.0)
    joints = list(_chain().joints)
    joints[0] = CurlJoint("prox", 0.0, _LIMIT, 0.0, (CoupledJoint("arch", 0.5, cap),))
    result = curl_to_contact(
        finger,
        (CurlChain("finger", tuple(joints)),),
        cylinder_distance(_RADIUS),
        _SETTINGS,
        {},
    )
    expected = min(cap, 0.5 * result.angles["prox"])
    assert result.angles["arch"] == pytest.approx(expected)


def test_curl_rejects_bad_settings() -> None:
    with pytest.raises(ValueError):
        CurlSettings(0.0, 10, 1e-3, 1e-4)
    with pytest.raises(ValueError):
        CurlJoint("prox", 1.0, 0.5, 0.0)


def _landmarks() -> FlatHandLandmarks:
    return FlatHandLandmarks(
        index_mcp=np.array([0.025, 0.097, 0.0]),
        index_pip=np.array([0.029, 0.122, 0.0]),
        little_mcp=np.array([-0.037, 0.081, 0.0]),
        radial=np.array([1.0, 0.0, 0.0]),
        distal=np.array([0.0, 1.0, 0.0]),
        palmar=np.array([0.0, 0.0, 1.0]),
    )


def test_club_axis_sits_on_the_index_first_segment_and_below_the_pinky_base() -> None:
    """Dieter's grip spec: club across the finger bases at a slight angle."""
    marks = _landmarks()
    point, direction = club_axis_in_hand(
        marks, cross_angle_deg=40.0, index_station=0.5, pinky_palm_depth_m=0.0
    )
    np.testing.assert_allclose(point, 0.5 * (marks.index_mcp + marks.index_pip))
    assert direction @ marks.radial == pytest.approx(math.cos(math.radians(40.0)))
    assert direction @ marks.distal == pytest.approx(math.sin(math.radians(40.0)))
    # Where the axis passes the little-finger ray it is proximal of the knuckle.
    t = (marks.little_mcp - point) @ marks.radial / (direction @ marks.radial)
    crossing = point + t * direction
    assert crossing @ marks.distal < marks.little_mcp @ marks.distal


def test_pinky_palm_depth_overrides_the_cross_angle() -> None:
    marks = _landmarks()
    depth = 0.012
    point, direction = club_axis_in_hand(
        marks, cross_angle_deg=40.0, index_station=0.5, pinky_palm_depth_m=depth
    )
    target = marks.little_mcp - depth * marks.distal
    off_axis = (target - point) - ((target - point) @ direction) * direction
    assert np.linalg.norm(off_axis) < 1e-12


def test_flat_hand_rotation_puts_shaft_on_z_and_palm_on_the_grip() -> None:
    marks = _landmarks()
    _, direction = club_axis_in_hand(
        marks, cross_angle_deg=40.0, index_station=0.5, pinky_palm_depth_m=0.0
    )
    azimuth = math.radians(55.0)
    rot = flat_hand_rotation(marks, direction, azimuth)
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rot) == pytest.approx(1.0)
    np.testing.assert_allclose(rot @ direction, [0.0, 0.0, 1.0], atol=1e-12)
    radial_out = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
    np.testing.assert_allclose(rot @ marks.palmar, -radial_out, atol=1e-12)


def test_wrap_search_frees_a_finger_that_lands_on_its_tip() -> None:
    """Base well above the grip: the greedy curl hooks, the wrap search wraps."""
    finger = _PlanarFinger(base_x=-0.012)
    finger._base = finger._base + np.array([0.0, 0.012, 0.0])
    distance = cylinder_distance(_RADIUS)
    chain = CurlChain("finger", _chain().joints, must_touch=("mid", "dist"))
    greedy = curl_to_contact(finger, (chain,), distance, _SETTINGS, {})
    wrapped = wrap_to_contact(finger, (chain,), distance, _SETTINGS, {}, samples=12)
    posed = finger.points(wrapped.angles)
    assert float(distance(posed).min()) >= -_SETTINGS.penetration_allowance_m - 1e-9

    def shortfall(result) -> float:
        return sum(
            max(0.0, result.gaps[b] - _SETTINGS.contact_tolerance_m)
            for b in chain.must_touch
        )

    assert shortfall(wrapped) <= shortfall(greedy) + 1e-12
    assert max(wrapped.angles.values()) <= max(greedy.angles.values()) + 1e-12
    for joint in chain.joints:
        assert joint.lower_rad <= wrapped.angles[joint.bone] <= joint.upper_rad
    with pytest.raises(ValueError):
        wrap_to_contact(finger, (chain,), distance, _SETTINGS, {}, samples=1)
    with pytest.raises(ValueError):
        CurlChain("finger", _chain().joints, must_touch=("thumb",))


def test_a_pre_curled_joint_carries_on_from_where_it_is() -> None:
    finger = _PlanarFinger(base_x=-0.012)
    far = cylinder_distance(_RADIUS * 0.05)
    limit = math.radians(40.0)
    start = {"mid": math.radians(25.0)}
    result = curl_to_contact(finger, (_chain(limit),), far, _SETTINGS, start)
    assert result.angles["mid"] >= math.radians(25.0)
