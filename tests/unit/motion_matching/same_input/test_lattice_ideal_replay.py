"""Recorded off-grid impacts use a stable lattice and observed ideal inputs."""

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input import integrator
from src.shared.python.motion_matching.same_input.integrator import (
    ROOT_COORDINATES,
    StepPolicy,
    integrate,
)
from src.shared.python.motion_matching.same_input.stage_trace import StageRecorder

pytestmark = pytest.mark.unit


class Plant:
    coordinate_order = (*ROOT_COORDINATES, "joint")

    def acceleration(self, q, v, tau):
        return 100 * q + tau

    def acceleration_with_drive_components(self, q, v, drive, external):
        return self.acceleration(q, v, drive + external), drive.copy(), external.copy()

    def acceleration_with_drive(self, q, v, drive):
        return self.acceleration_with_drive_components(
            q, v, drive, np.zeros_like(drive)
        )

    def kinematic_frames(self, q):
        return {}

    def closure_pose_residual(self, q):
        return np.zeros(1)


def test_impact_prefix_is_continuous_at_nominal_lattice_boundary(monkeypatch):
    def fake_impact(impact, t, dt, q, v, step, **kwargs):
        before = step(t, 0.25, q, v, None)[0]
        after = step(t, 0.250000001, q, v, None)[0]
        np.testing.assert_allclose(before, after, atol=1e-6, rtol=0)
        q, v, extra = step(t, dt, q, v, None)
        return q, v, impact, extra

    class FakeImpact:
        def to_record(self):
            return {}

    monkeypatch.setattr(integrator, "step_through_impact", fake_impact)
    integrate(
        Plant(),
        np.ones(7),
        np.zeros(7),
        lambda *_: np.zeros(7),
        steps=1,
        dt_s=1,
        policy=StepPolicy(substeps=4, project=False),
        impact=FakeImpact(),
    )


def test_ideal_drive_recording_does_not_change_no_impact_trajectory():
    q0 = np.zeros(7)

    def source(*_):
        return np.array([0, 0, 0, 0, 0, 0, 2.0])

    policy = StepPolicy(substeps=2, project=False)
    plain = integrate(Plant(), q0, q0, source, steps=2, dt_s=0.01, policy=policy)
    recorder = StageRecorder(capture_ideal_drive=True)
    traced = integrate(
        Plant(), q0, q0, source, steps=2, dt_s=0.01, policy=policy, recorder=recorder
    )
    np.testing.assert_array_equal(plain.q, traced.q)
    np.testing.assert_array_equal(plain.v, traced.v)
    assert recorder.outcome.status == "complete"
    assert all(
        s.applied_drive[-1] == 2 and s.applied_external[-1] == 0
        for step in recorder.steps
        for s in step.stages
    )
