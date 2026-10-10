"""GCV-20 (#11767): the same-input integrator applies the ball impulse."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.impact_parameters import ball_impact as bi
from src.shared.python.motion_matching.impact_force import ImpactForce
from src.shared.python.motion_matching.same_input import (
    ROOT_COORDINATES,
    InputBundle,
    StepPolicy,
    integrate,
    open_loop,
    segmented_replay,
)

pytestmark = pytest.mark.unit

ORDER = (*ROOT_COORDINATES, "Joint")
SPEC = json.dumps({"coordinate_order": list(ORDER)}).encode()
DT = 1e-3
STEPS = 20
POLICY = StepPolicy(project=False)


class _Slider:
    """Unit mass on a soft spring (a = -q + tau); the tip moves along +x."""

    coordinate_order = ORDER

    def acceleration(self, q, v, tau):  # noqa: ANN001, ANN201
        return -q + tau

    def closure_pose_residual(self, q):  # noqa: ANN001, ANN201
        return q[:1]  # a weld holding the first root coordinate at zero

    def closure_rate_matrix(self, q):  # noqa: ANN001, ANN201
        return np.eye(1, q.size)

    def kinematic_frames(self, q):  # noqa: ANN001, ANN201
        pose = np.eye(4)
        pose[0, 3] = q[-1]
        return {"tip": pose}


def _plan() -> ImpactForce:
    return ImpactForce(
        frame="tip",
        point_in_frame=np.zeros(3),
        axis_in_frame=np.array([1.0, 0.0, 0.0]),
        t_start_s=0.0103,  # mid-step
        swing_span_s=(0.0, STEPS * DT),
    )


def _start() -> tuple[np.ndarray, np.ndarray]:
    v0 = np.zeros(7)
    v0[-1] = 10.0
    return np.zeros(7), v0


def test_integrate_latches_and_delivers_the_collision() -> None:
    plant, (q0, v0) = _Slider(), _start()
    zero = np.zeros((STEPS, 7))
    free = open_loop(plant, q0, v0, zero, dt_s=DT, policy=POLICY)
    hit = open_loop(plant, q0, v0, zero, dt_s=DT, policy=POLICY, impact=_plan())
    assert free.ball_impact is None and hit.ball_impact is not None
    record = hit.ball_impact["collision"]
    assert record["effective_mass_kg"] == pytest.approx(1.0, rel=1e-6)
    mb, e = bi.BALL_MASS_KG, bi.COR_LIMIT
    expected = (1 + e) * mb * record["approach_speed_mps"] / (1 + mb)
    assert free.v[-1, -1] - hit.v[-1, -1] == pytest.approx(expected, rel=1e-3)
    np.testing.assert_array_equal(free.q[:11], hit.q[:11])  # untouched before


def test_a_recorded_force_replays_identically() -> None:
    plant, (q0, v0) = _Slider(), _start()
    zero = np.zeros((STEPS, 7))
    first = open_loop(plant, q0, v0, zero, dt_s=DT, policy=POLICY, impact=_plan())
    recorded = ImpactForce.from_record(first.ball_impact)
    assert recorded.is_latched
    again = open_loop(plant, q0, v0, zero, dt_s=DT, policy=POLICY, impact=recorded)
    np.testing.assert_allclose(again.v, first.v, rtol=0, atol=1e-12)


def test_segmented_replay_applies_the_impact_on_each_segment_clock() -> None:
    plant, (q0, v0) = _Slider(), _start()
    zero = np.zeros((STEPS, 7))
    reference = integrate(
        plant,
        q0,
        v0,
        lambda *_: np.zeros(7),
        steps=STEPS,
        dt_s=DT,
        policy=StepPolicy(),
        impact=_plan(),
    )
    bundle = InputBundle(
        spec_bytes=SPEC,
        coordinate_order=ORDER,
        dt_s=DT,
        q0=q0,
        v0=v0,
        efforts=zero,
        reference_q=reference.q,
        reference_v=reference.v,
        reference_engine="toy",
        provenance={"ball_impact": reference.ball_impact},
    )
    assert bundle.ball_impact().is_latched
    segments = segmented_replay(plant, bundle, segment_steps=4)
    assert max(s["coordinate_error_rad"] for s in segments) < 1e-12
    without = InputBundle(**{**bundle.__dict__, "provenance": {}})
    worst = max(
        s["coordinate_error_rad"]
        for s in segmented_replay(plant, without, segment_steps=4)
    )
    assert worst > 1e-5
