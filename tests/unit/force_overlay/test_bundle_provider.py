"""Overlay frame provider built from same-input bundles (NV-3, #11676)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.force_overlay import WrenchKind
from src.shared.python.force_overlay.bundle_provider import (
    BundleOverlayProvider,
    JointFrame,
)
from src.shared.python.motion_matching.contact_law import ContactSample
from src.shared.python.motion_matching.same_input import InputBundle

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

COORDS = ("TranslationInputX", "HipInputX", "HipInputY", "KneeInput")
MASS = 80.0
G = 9.80665


def _bundle(steps: int = 4) -> InputBundle:
    spec = {
        "coordinate_order": list(COORDS),
        "gravity_m_s2": [0.0, 0.0, -G],
        "contact": {
            "spheres": [
                {"name": "heel_r", "body": "calcn_r"},
                {"name": "toe_r", "body": "calcn_r"},
                {"name": "heel_l", "body": "calcn_l"},
            ]
        },
    }
    nv = len(COORDS)
    q = np.zeros((steps + 1, nv))
    efforts = np.tile([5.0, 30.0, 40.0, -12.0], (steps, 1))
    return InputBundle(
        spec_bytes=json.dumps(spec).encode(),
        coordinate_order=COORDS,
        dt_s=0.01,
        q0=q[0],
        v0=q[0],
        efforts=efforts,
        reference_q=q,
        reference_v=q.copy(),
        reference_engine="fake",
    )


def _sample(point, normal, friction=(0.0, 0.0, 0.0)) -> ContactSample:
    return ContactSample(
        0.001, 0.0, np.array(point, float), np.array(normal, float), np.array(friction)
    )


class _Contact:
    def evaluate_contact_samples(self, coordinates, rates):
        return {
            "heel_r": _sample((0.0, 0.0, 0.0), (0, 0, 300.0), (10.0, 0, 0)),
            "toe_r": _sample((0.2, 0.0, 0.0), (0, 0, 100.0)),
            "heel_l": _sample((1.0, 1.0, 0.0), (0, 0, 0.0)),  # not in contact
        }


class _Kinematics:
    total_mass_kg = MASS

    def joint_frames(self, coordinates):
        # HipInputX/Y share the hip anchor, KneeInput has its own; the root
        # translation is not a rotational joint and must be omitted.
        hip = (0.0, 0.0, 1.0)
        return {
            "HipInputX": JointFrame("pelvis", hip, (1.0, 0.0, 0.0)),
            "HipInputY": JointFrame("pelvis", hip, (0.0, 1.0, 0.0)),
            "KneeInput": JointFrame("tibia_r", (0.0, 0.1, 0.5), (0.0, 1.0, 0.0)),
        }

    def center_of_mass_m(self, coordinates):
        return (0.1, 0.0, 0.95)


def _provider(**kw) -> BundleOverlayProvider:
    return BundleOverlayProvider(
        _bundle(), _Contact(), _Kinematics(), engine="fake", **kw
    )


def _contact_by_label(frame) -> dict:
    return {w.label: w for w in frame.by_kind(WrenchKind.CONTACT)}


def test_frame_has_expected_wrench_counts() -> None:
    frame = _provider().frame_at(0)
    # right foot + net, each with force, free moment and moment about the CoM;
    # the unloaded left foot is omitted (GCV-2 breakdown)
    assert sorted(_contact_by_label(frame)) == [
        "contact:free_moment_net",
        "contact:free_moment_right",
        "contact:grf_net",
        "contact:grf_right",
        "contact:moment_com_net",
        "contact:moment_com_right",
    ]
    assert len(frame.by_kind(WrenchKind.JOINT_ACTUATOR)) == 2  # hip (merged) + knee
    assert len(frame.by_kind(WrenchKind.GRAVITY)) == 1
    assert len(frame.wrenches) == 9
    assert frame.engine == "fake"


def test_grf_is_summed_force_at_the_wrench_derived_cop() -> None:
    contacts = _contact_by_label(_provider().frame_at(0))
    grf = contacts["contact:grf_right"]
    np.testing.assert_allclose(grf.force_n, (10.0, 0.0, 400.0))
    # CoP from the wrench: the 300 N at x=0 and 100 N at x=0.2 give x = 0.05
    np.testing.assert_allclose(grf.point_m, (0.05, 0.0, 0.0), atol=1e-12)
    assert grf.torque_nm is None
    # one loaded foot: the net equals the foot
    net = contacts["contact:grf_net"]
    np.testing.assert_allclose(net.force_n, grf.force_n)
    np.testing.assert_allclose(net.point_m, grf.point_m, atol=1e-12)


def test_com_moment_is_taken_about_the_kinematics_centre_of_mass() -> None:
    m = _contact_by_label(_provider().frame_at(0))["contact:moment_com_net"]
    np.testing.assert_allclose(m.point_m, (0.1, 0.0, 0.95))
    # M_c = M_O - c x F with M_O = sum p x F about the origin
    force = np.array([10.0, 0.0, 400.0])
    m_origin = np.array([0.0, -20.0, 0.0])
    expected = m_origin - np.cross([0.1, 0.0, 0.95], force)
    np.testing.assert_allclose(m.torque_nm, expected, atol=1e-9)


def test_joint_torques_use_bundle_efforts_and_merge_shared_anchor() -> None:
    acts = {
        w.body: w for w in _provider().frame_at(0).by_kind(WrenchKind.JOINT_ACTUATOR)
    }
    np.testing.assert_allclose(acts["pelvis"].torque_nm, (30.0, 40.0, 0.0))
    np.testing.assert_allclose(acts["pelvis"].point_m, (0.0, 0.0, 1.0))
    np.testing.assert_allclose(acts["tibia_r"].torque_nm, (0.0, -12.0, 0.0))
    assert acts["pelvis"].force_n is None  # never fabricated as zero


def test_weight_is_mass_times_gravity_at_com() -> None:
    (w,) = _provider().frame_at(0).by_kind(WrenchKind.GRAVITY)
    np.testing.assert_allclose(w.force_n, (0.0, 0.0, -MASS * G))
    np.testing.assert_allclose(w.point_m, (0.1, 0.0, 0.95))


def test_options_drop_weight_and_small_torques() -> None:
    p = _provider(include_weight=False, torque_floor_nm=20.0)
    frame = p.frame_at(1)
    assert not frame.by_kind(WrenchKind.GRAVITY)
    assert [w.body for w in frame.by_kind(WrenchKind.JOINT_ACTUATOR)] == ["pelvis"]


def test_last_frame_reuses_final_effort_and_times_are_on_dt_grid() -> None:
    p = _provider()
    last = p.frame_at(4)
    assert last.time_s == pytest.approx(0.04)
    assert len(last.by_kind(WrenchKind.JOINT_ACTUATOR)) == 2


def test_series_is_strictly_increasing_and_skips_nothing() -> None:
    series = _provider().series(stride=2)
    assert series.times_s == pytest.approx((0.0, 0.02, 0.04))
    assert series.engine == "fake"


def test_rollout_override_is_used_for_states() -> None:
    seen = []

    class _Spy(_Contact):
        def evaluate_contact_samples(self, coordinates, rates):
            seen.append(coordinates["KneeInput"])
            return super().evaluate_contact_samples(coordinates, rates)

    q = np.zeros((5, 4))
    q[:, 3] = 0.7
    p = BundleOverlayProvider(
        _bundle(), _Spy(), _Kinematics(), engine="fake", q=q, v=np.zeros((5, 4))
    )
    p.frame_at(2)
    assert seen == [0.7]


def test_preconditions() -> None:
    with pytest.raises(IndexError):
        _provider().frame_at(99)
    with pytest.raises(ValueError, match="stride"):
        _provider().series(stride=0)
    with pytest.raises(ValueError, match="shape"):
        BundleOverlayProvider(
            _bundle(),
            _Contact(),
            _Kinematics(),
            engine="fake",
            q=np.zeros((3, 4)),
            v=None,
        )
    with pytest.raises(TypeError):
        BundleOverlayProvider("bundle", _Contact(), _Kinematics(), engine="fake")  # type: ignore[arg-type]
