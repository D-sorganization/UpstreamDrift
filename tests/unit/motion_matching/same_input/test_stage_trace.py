"""Transactional RK4 trace must contain only accepted ZOH stages."""

from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input import integrator
from src.shared.python.motion_matching.same_input.integrator import (
    ROOT_COORDINATES,
    StepPolicy,
    integrate,
    open_loop,
)
from src.shared.python.motion_matching.same_input.stage_trace import StageRecorder


class Plant:
    coordinate_order = (*ROOT_COORDINATES, "joint")

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def acceleration(self, q, v, tau):
        self.calls += 1
        if self.calls == self.fail_at:
            raise ValueError("late failure")
        return np.asarray(tau)

    def closure_pose_residual(self, q):
        return np.zeros(1)


@pytest.mark.parametrize(
    "damage", ["segment", "hold", "stage", "duplicate", "reorder", "q", "v", "tau"]
)
def test_finish_rejects_incomplete_or_malformed_trace(damage):
    recorder = StageRecorder()
    q = np.zeros(7)
    integrate(
        Plant(),
        q,
        q,
        lambda *_: np.zeros(7),
        steps=1,
        dt_s=0.01,
        policy=StepPolicy(substeps=2, effort_holds=2, project=False),
        recorder=recorder,
    )
    step = recorder.steps[0]
    if damage in ("segment", "hold"):
        segments = step.segments[1:] if damage == "hold" else ()
        recorder.steps[0] = replace(step, segments=segments)
    elif damage in ("stage", "duplicate", "reorder"):
        stages = list(step.stages)
        if damage == "stage":
            stages.pop()
        elif damage == "duplicate":
            stages[-1] = stages[-2]
        else:
            stages[0], stages[1] = stages[1], stages[0]
        recorder.steps[0] = replace(step, stages=tuple(stages))
    else:
        bad = np.zeros(6) if damage == "v" else np.full(7, np.nan)
        stages = list(step.stages)
        stages[0] = replace(stages[0], **{damage: bad})
        recorder.steps[0] = replace(step, stages=tuple(stages))
    recorder.finish()
    assert recorder.outcome.status == "partial"
    assert recorder.outcome.failure


def test_finish_rejects_duplicate_coordinate_inventory():
    recorder = StageRecorder()
    q = np.zeros(7)
    integrate(
        Plant(),
        q,
        q,
        lambda *_: np.zeros(7),
        steps=1,
        dt_s=0.01,
        policy=StepPolicy(substeps=1, project=False),
        recorder=recorder,
    )
    recorder.plan = replace(recorder.plan, coordinate_order=("same",) * 7)
    recorder.finish()
    assert recorder.outcome.status == "partial"
    assert recorder.outcome.failure


def test_finish_rejects_changing_held_effort():
    recorder = StageRecorder()
    q = np.zeros(7)
    integrate(
        Plant(),
        q,
        q,
        lambda *_: np.zeros(7),
        steps=1,
        dt_s=0.01,
        policy=StepPolicy(substeps=2, project=False),
        recorder=recorder,
    )
    step = recorder.steps[0]
    stages = list(step.stages)
    stages[-1] = replace(stages[-1], tau=np.ones(7))
    recorder.steps[0] = replace(step, stages=tuple(stages))
    recorder.finish()
    assert recorder.outcome.status == "partial"
    assert recorder.outcome.failure


def test_trace_stages_are_committed_without_extra_dynamics():
    q = np.zeros(7)
    p, recorder = Plant(), StageRecorder()
    run = integrate(
        p,
        q,
        q,
        lambda *_: np.ones(7),
        steps=1,
        dt_s=0.01,
        policy=StepPolicy(substeps=1, project=False),
        recorder=recorder,
    )
    assert p.calls == 4
    assert len(recorder.steps) == 1
    stages = recorder.steps[0].stages
    assert [s.stage for s in stages] == [1, 2, 3, 4]
    assert [s.time_s for s in stages] == [0, 0.005, 0.005, 0.01]
    assert all(np.array_equal(s.tau, [0, 0, 0, 0, 0, 0, 1]) for s in stages)
    np.testing.assert_array_equal(recorder.steps[0].post_projection_q, run.q[-1])
    np.testing.assert_array_equal(recorder.steps[0].pre_projection_q, run.q[-1])


def test_open_loop_forwards_recorder():
    q = np.zeros(7)
    recorder = StageRecorder()
    effort = np.zeros((1, 7))
    effort[0, -1] = 1
    run = open_loop(
        Plant(),
        q,
        q,
        effort,
        dt_s=0.01,
        policy=StepPolicy(substeps=1, project=False),
        recorder=recorder,
    )
    assert len(recorder.steps[0].stages) == 4
    assert recorder.outcome.status == "complete"
    np.testing.assert_array_equal(recorder.steps[0].post_projection_q, run.q[-1])


def test_late_failure_discards_entire_step_trace():
    q = np.zeros(7)
    p, recorder = Plant(fail_at=7), StageRecorder()
    run = integrate(
        p,
        q,
        q,
        lambda *_: np.ones(7),
        steps=2,
        dt_s=0.01,
        policy=StepPolicy(substeps=1, project=False, stop_on_failure=True),
        recorder=recorder,
    )
    assert run.failure is not None
    assert len(recorder.steps) == 1
    assert len(recorder.steps[0].stages) == 4
    assert recorder.outcome.status == "partial"
    assert recorder.outcome.completed_steps == 1
    assert recorder.outcome.requested_steps == 2


def test_recording_does_not_change_rollout_or_dynamics_calls():
    q = np.zeros(7)
    policy = StepPolicy(substeps=4, effort_holds=2, project=False)
    plants = (Plant(), Plant())
    recorder = StageRecorder()

    def source(j, *_):
        return np.full(7, j + 1.0)

    plain = integrate(plants[0], q, q, source, steps=2, dt_s=0.01, policy=policy)
    traced = integrate(
        plants[1], q, q, source, steps=2, dt_s=0.01, policy=policy, recorder=recorder
    )
    np.testing.assert_array_equal(plain.q, traced.q)
    np.testing.assert_array_equal(plain.v, traced.v)
    np.testing.assert_array_equal(plain.efforts, traced.efforts)
    assert plants[0].calls == plants[1].calls == 32
    assert len(recorder.steps) == 2
    assert len(recorder.steps[0].stages) == 16
    assert {stage.hold for stage in recorder.steps[0].stages} == {0, 1}
    assert recorder.outcome.status == "complete"
    expected = recorder.expected_stages()
    actual = [
        (s.step, s.hold, s.segment, s.substep, s.stage, s.time_s)
        for step in recorder.steps
        for s in step.stages
    ]
    assert expected == actual
    assert recorder.plan.coordinate_order == plants[1].coordinate_order


def test_impact_probes_do_not_enter_trace(monkeypatch):
    def fake_impact(
        impact, t, dt, q, v, step, *, poses, accel, trial_step, point_velocity
    ):
        trial_step(t, dt / 2, q, v, None)
        q, v, extra = step(t, dt, q, v, None)
        return q, v, impact, extra

    monkeypatch.setattr(integrator, "step_through_impact", fake_impact)
    p, recorder = Plant(), StageRecorder()
    p.kinematic_frames = lambda q: {}
    q = np.zeros(7)

    class FakeImpact:
        def to_record(self):
            return {}

    integrate(
        p,
        q,
        q,
        lambda *_: np.ones(7),
        steps=1,
        dt_s=0.01,
        policy=StepPolicy(substeps=1, project=False),
        impact=FakeImpact(),
        recorder=recorder,
    )
    assert p.calls == 8
    assert len(recorder.steps[0].stages) == 4
    assert len(recorder.steps[0].segments) == 1
    assert len(recorder.expected_stages()) == 4


def test_impact_trial_prefix_is_continuous_and_matches_accepted_trace(monkeypatch):
    class NonlinearPlant(Plant):
        def acceleration(self, q, v, tau):
            return 100.0 * q

    def fake_impact(
        impact, t, dt, q, v, step, *, poses, accel, trial_step, point_velocity
    ):
        # Old ceil-based repartition jumps at 0.25 s, despite smooth dynamics.
        before = trial_step(t, 0.25, q, v, None)[0]
        after = trial_step(t, 0.250000001, q, v, None)[0]
        np.testing.assert_allclose(before, after, atol=1e-6, rtol=0.0)
        trial_q, trial_v, _ = trial_step(t, 0.251, q, v, None)
        accepted_q, accepted_v, extra = step(t, 0.251, q, v, None)
        np.testing.assert_array_equal(trial_q, accepted_q)
        np.testing.assert_array_equal(trial_v, accepted_v)
        end_q, end_v, _ = step(t + 0.251, dt - 0.251, accepted_q, accepted_v, None)
        return end_q, end_v, impact, extra

    class FakeImpact:
        def to_record(self):
            return {}

    monkeypatch.setattr(integrator, "step_through_impact", fake_impact)
    p, recorder = NonlinearPlant(), StageRecorder()
    p.kinematic_frames = lambda q: {}
    q = np.ones(7)
    integrate(
        p,
        q,
        np.zeros(7),
        lambda *_: np.zeros(7),
        steps=1,
        dt_s=1.0,
        policy=StepPolicy(substeps=4, project=False),
        impact=FakeImpact(),
        recorder=recorder,
    )
    assert recorder.outcome.status == "complete"
    assert [segment.substeps for segment in recorder.steps[0].segments] == [2, 3]
    assert recorder.steps[0].stages[4].time_s == pytest.approx(0.25)


def test_armed_surface_search_replays_same_hold_lattice(monkeypatch):
    from src.shared.python.motion_matching.contact_event import find_contact_event

    class NonlinearPlant(Plant):
        def acceleration(self, q, v, tau):
            return 100.0 * q

    triangles = np.array([[[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]])

    def fake_impact(
        impact, t, dt, q, v, step, *, poses, accel, trial_step, point_velocity
    ):
        arm = 0.1
        q_arm, v_arm, _ = trial_step(t, arm, q, v, None)

        def advance_after_arm(h):
            if h == 0.0:
                return q_arm, v_arm
            return trial_step(t + arm, h, q_arm, v_arm, None)[:2]

        result = find_contact_event(
            advance_after_arm,
            lambda x: triangles + np.array([x[0], 0.0, 0.0]),
            np.array([6.4, 0.0, 0.0]),
            0.5,
            duration_s=dt - arm,
            time_tolerance_s=1e-11,
            face_point_velocity_world=lambda qq, vv, point: np.array([vv[0], 0.0, 0.0]),
        )
        event = result.event
        assert event is not None
        assert abs(5.9 - event.q[0]) < 1e-6
        q_a, v_a, extra = step(t, arm, q, v, None)
        q_hit, v_hit, _ = step(t + arm, event.time_s, q_a, v_a, None)
        np.testing.assert_array_equal(q_hit, event.q)
        np.testing.assert_array_equal(v_hit, event.qd)
        q_end, v_end, _ = step(
            t + arm + event.time_s, dt - arm - event.time_s, q_hit, v_hit, None
        )
        return q_end, v_end, impact, extra

    class FakeImpact:
        def to_record(self):
            return {}

    monkeypatch.setattr(integrator, "step_through_impact", fake_impact)
    plant, recorder = NonlinearPlant(), StageRecorder()
    plant.kinematic_frames = lambda q: {}
    integrate(
        plant,
        np.ones(7),
        np.zeros(7),
        lambda *_: np.zeros(7),
        steps=1,
        dt_s=1.0,
        policy=StepPolicy(substeps=4, project=False),
        impact=FakeImpact(),
        recorder=recorder,
    )
    assert recorder.outcome.status == "complete"
    segments = recorder.steps[0].segments
    assert segments[1].start_s == pytest.approx(0.1)
    assert segments[2].start_s < 0.25
    assert segments[2].step_sizes_s[0] == pytest.approx(0.25 - segments[2].start_s)


def test_projection_exception_aborts_step(monkeypatch):
    def fail(*args):
        raise RuntimeError("projection failed")

    monkeypatch.setattr(integrator, "project_to_closure", fail)
    recorder = StageRecorder()
    q = np.zeros(7)
    with pytest.raises(RuntimeError, match="projection failed"):
        integrate(
            Plant(),
            q,
            q,
            lambda *_: np.ones(7),
            steps=1,
            dt_s=0.01,
            policy=StepPolicy(substeps=1),
            recorder=recorder,
        )
    assert recorder.steps == []
    assert recorder._pending == []
    assert recorder.outcome.status == "partial"
    assert recorder.outcome.completed_steps == 0


def test_setup_failure_finalizes_recorder_as_partial():
    recorder = StageRecorder()
    q = np.zeros(7)
    with pytest.raises(TypeError, match="kinematic_frames"):
        integrate(
            Plant(),
            q,
            q,
            lambda *_: np.zeros(7),
            steps=1,
            dt_s=0.01,
            policy=StepPolicy(substeps=1, project=False),
            impact=object(),
            recorder=recorder,
        )
    assert recorder.steps == []
    assert recorder.outcome.status == "partial"
    assert recorder.outcome.completed_steps == 0
    assert recorder.outcome.failure


def test_invalid_initial_state_finalizes_recorder_as_partial():
    recorder = StageRecorder()
    with pytest.raises(ValueError):
        integrate(
            Plant(),
            ["not numeric"],
            np.zeros(7),
            lambda *_: np.zeros(7),
            steps=1,
            dt_s=0.01,
            policy=StepPolicy(substeps=1, project=False),
            recorder=recorder,
        )
    assert recorder.outcome.status == "partial"
    assert recorder.outcome.completed_steps == 0
    assert recorder.outcome.failure
