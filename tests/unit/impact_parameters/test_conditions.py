"""Contract tests for the optimiser-loop impact-conditions evaluator (SHOW-C2, #14558).

The evaluator is a thin facade over ``clubhead_series_from_mujoco`` and
``extract_impact_parameters``; these tests pin (a) agreement with the existing
extractor, (b) the face/path sign conventions the optimiser cost relies on, and
(c) the < 1 s per-rollout budget.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from src.shared.python.impact_parameters import (
    ClubheadSeries,
    TargetFrame,
    extract_impact_parameters,
)
from src.shared.python.impact_parameters.adapters import (
    ClubFaceSpec,
    clubhead_series_from_mujoco,
)
from src.shared.python.impact_parameters.conditions import (
    ImpactConditions,
    evaluate_rollout,
)

pytestmark = pytest.mark.unit

FRAME = TargetFrame()  # ADR-0041: Z-up, target along -Y, right-handed
SPEC = ClubFaceSpec(face_center_body_m=(0.0, 0.0, -0.1))
N = 201
DT = 0.001
OMEGA = 30.0  # rad/s about world z (vertical) -> pure horizontal travel

MJCF = """
<mujoco><compiler angle="radian"/><worldbody>
 <body name="arm" pos="0 0 1.0">
  <joint name="yaw" type="hinge" axis="0 0 1"/>
  <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
  <body name="club" pos="1.0 0 0">
   <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
  </body>
 </body>
</worldbody></mujoco>
"""


def _mujoco():
    mujoco = pytest.importorskip("mujoco")
    if not getattr(mujoco, "__file__", None):
        pytest.skip("mujoco is a test stub")
    return mujoco


def _rollout(omega: float = OMEGA, start: float = -0.3):
    t = np.arange(N) * DT
    q = (start + omega * t)[:, None]
    v = np.full((N, 1), omega)
    return t, q, v


def _series(velocity, normal) -> ClubheadSeries:
    """Straight-line face-centre motion with a constant face normal."""
    t = np.arange(N) * DT
    vel = np.tile(np.asarray(velocity, float), (N, 1))
    pos = vel * t[:, None]
    return ClubheadSeries(
        times_s=t,
        face_center_m=pos,
        velocity_mps=vel,
        face_normal=np.tile(np.asarray(normal, float), (N, 1)),
    )


def test_matches_existing_extractor_on_mujoco_rollout():
    mujoco = _mujoco()
    model = mujoco.MjModel.from_xml_string(MJCF)
    t, q, v = _rollout()
    got = evaluate_rollout(model, t, q, v, "club", FRAME, spec=SPEC, impact_index=150)
    ref = extract_impact_parameters(
        clubhead_series_from_mujoco(model, t, q, v, "club", SPEC),
        FRAME,
        impact_index=150,
        use_tools=False,
    )
    assert got.speed_mph == pytest.approx(ref.clubhead_speed_mph, abs=1e-9)
    assert got.aoa_deg == pytest.approx(ref.attack_angle_deg, abs=1e-9)
    assert got.path_deg == pytest.approx(ref.club_path_deg, abs=1e-9)
    assert got.face_deg == pytest.approx(ref.face_angle_deg, abs=1e-9)
    # analytic: speed = omega * radius at the face centre (|r| = 1.0 m horizontal)
    assert got.speed_mph == pytest.approx(OMEGA * 1.0 * 2.2369362920544, rel=1e-3)
    assert got.aoa_deg == pytest.approx(0.0, abs=1e-6)


def test_evaluation_under_one_second_per_rollout():
    mujoco = _mujoco()
    model = mujoco.MjModel.from_xml_string(MJCF)
    t, q, v = _rollout()
    evaluate_rollout(model, t, q, v, "club", FRAME, spec=SPEC, impact_index=150)
    start = time.perf_counter()
    evaluate_rollout(model, t, q, v, "club", FRAME, spec=SPEC, impact_index=150)
    assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize(
    ("velocity", "expected_path"),
    [
        ((0.0, -40.0, 0.0), 0.0),  # straight down the target line (-Y)
        ((6.0, -40.0, 0.0), -math.degrees(math.atan2(6.0, 40.0))),  # left of target
        ((-6.0, -40.0, 0.0), math.degrees(math.atan2(6.0, 40.0))),  # right: in-to-out
    ],
)
def test_path_sign_positive_is_in_to_out(velocity, expected_path):
    cond = ImpactConditions.from_parameters(
        extract_impact_parameters(
            _series(velocity, (0.0, -1.0, 0.0)),
            FRAME,
            impact_index=100,
            use_tools=False,
        )
    )
    assert cond.path_deg == pytest.approx(expected_path, abs=1e-9)


@pytest.mark.parametrize(("normal_x", "sign"), [(0.0, 0.0), (-0.1, 1.0), (0.1, -1.0)])
def test_face_sign_positive_is_open(normal_x, sign):
    normal = np.array([normal_x, -1.0, 0.0])
    cond = ImpactConditions.from_parameters(
        extract_impact_parameters(
            _series((0.0, -40.0, 0.0), normal), FRAME, impact_index=100, use_tools=False
        )
    )
    assert math.copysign(1.0, cond.face_deg) == math.copysign(1.0, sign) or sign == 0
    assert cond.face_deg == pytest.approx(
        sign * math.degrees(math.atan2(abs(normal_x), 1.0)), abs=1e-9
    )
    # face-to-path = face - path (path is 0 here)


def test_unobservable_face_reports_none_with_reason():
    t = np.arange(N) * DT
    vel = np.tile([0.0, -40.0, 0.0], (N, 1))
    series = ClubheadSeries(
        times_s=t,
        face_center_m=vel * t[:, None],
        velocity_mps=vel,
        face_unobservable_reason="mocap",
    )
    cond = ImpactConditions.from_parameters(
        extract_impact_parameters(series, FRAME, impact_index=100, use_tools=False)
    )
    assert cond.face_deg is None
    assert "face_deg" in cond.unavailable


def test_conditions_are_finite_and_frozen():
    cond = ImpactConditions.from_parameters(
        extract_impact_parameters(
            _series((0.0, -40.0, 0.0), (0.0, -1.0, 0.0)),
            FRAME,
            impact_index=100,
            use_tools=False,
        )
    )
    with pytest.raises(AttributeError):
        cond.path_deg = 1.0  # type: ignore[misc]
    assert all(math.isfinite(x) for x in cond.as_vector())
