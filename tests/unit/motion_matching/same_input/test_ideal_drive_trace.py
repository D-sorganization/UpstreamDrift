"""Ideal generalized drive is distinct from external impact and saturation."""

import json
from pathlib import Path

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


class DirectPlant:
    coordinate_order = (*ROOT_COORDINATES, "joint", "passive")

    def acceleration(self, q, v, tau):
        result = np.asarray(tau).copy()
        result[-1] = 0.0
        return result

    def acceleration_with_drive_components(self, q, v, drive, external):
        combined = np.asarray(drive) + np.asarray(external)
        applied_drive, applied_external = (
            np.asarray(drive).copy(),
            np.asarray(external).copy(),
        )
        applied_drive[-1] = applied_external[-1] = 0.0
        return self.acceleration(q, v, combined), applied_drive, applied_external

    def acceleration_with_drive(self, q, v, drive):
        return self.acceleration_with_drive_components(
            q, v, drive, np.zeros_like(drive)
        )

    def closure_pose_residual(self, q):
        return np.zeros(1)

    def kinematic_frames(self, q):
        return {}


def test_actual_accepted_stages_separate_ideal_drive_and_impact(monkeypatch):
    def fake_impact(impact, t, dt, q, v, step, *, poses, accel, trial_step):
        def external(x):
            return np.array([0, 0, 0, 0, 0, 0, 3 + x[6], 9])

        (trial_step or step)(t, dt / 2, q, v, external)
        q, v, extra = step(t, dt, q, v, external)
        return q, v, impact, extra

    class FakeImpact:
        def to_record(self):
            return {}

    monkeypatch.setattr(integrator, "step_through_impact", fake_impact)
    q0 = np.zeros(8)
    policy = StepPolicy(substeps=1, project=False)

    def source(*_):
        return np.array([7, 7, 7, 7, 7, 7, 5, 11])

    plain = integrate(
        DirectPlant(),
        q0,
        q0,
        source,
        steps=1,
        dt_s=0.01,
        policy=policy,
        impact=FakeImpact(),
    )
    recorder = StageRecorder(capture_ideal_drive=True)
    traced = integrate(
        DirectPlant(),
        q0,
        q0,
        source,
        steps=1,
        dt_s=0.01,
        policy=policy,
        impact=FakeImpact(),
        recorder=recorder,
    )
    np.testing.assert_array_equal(traced.q, plain.q)
    np.testing.assert_array_equal(traced.v, plain.v)
    assert recorder.outcome.status == "complete"
    assert len(recorder.steps[0].stages) == 4
    for stage in recorder.steps[0].stages:
        np.testing.assert_array_equal(stage.applied_drive, [0, 0, 0, 0, 0, 0, 5, 0])
        assert stage.applied_external[6] == pytest.approx(3 + stage.q[6])
        assert stage.applied_external[-1] == 0
        assert stage.effort_semantics == "ideal_direct_generalized_no_saturation"


def test_unsupported_plant_cannot_report_ideal_drive():
    class Unsupported(DirectPlant):
        acceleration_with_drive = None

    recorder = StageRecorder(capture_ideal_drive=True)
    with pytest.raises((TypeError, ValueError), match="effort|drive"):
        integrate(
            Unsupported(),
            np.zeros(8),
            np.zeros(8),
            lambda *_: np.zeros(8),
            steps=1,
            dt_s=0.01,
            policy=StepPolicy(substeps=1, project=False),
            recorder=recorder,
        )

    from src.shared.python.motion_matching.same_input.plant import VectorPlant

    vector = VectorPlant.__new__(VectorPlant)
    vector.engine = "pinocchio"
    vector._adapter = object()
    with pytest.raises(TypeError, match="effort"):
        vector.acceleration_with_drive_components(
            np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8)
        )


def test_failed_acceleration_discards_accepted_stage_and_drive_buffers():
    class Failing(DirectPlant):
        calls = 0

        def acceleration_with_drive(self, q, v, drive):
            self.calls += 1
            if self.calls == 3:
                raise ValueError("injected acceleration failure")
            return super().acceleration_with_drive(q, v, drive)

    recorder = StageRecorder(capture_ideal_drive=True)
    run = integrate(
        Failing(),
        np.zeros(8),
        np.zeros(8),
        lambda *_: np.ones(8),
        steps=1,
        dt_s=0.01,
        policy=StepPolicy(substeps=1, project=False, stop_on_failure=True),
        recorder=recorder,
    )
    assert run.failure is not None
    assert recorder.outcome.status == "partial"
    assert recorder.steps == []


def test_native_kkt_observation_matches_actual_masked_input_without_changing_dynamics():
    pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.club_models import DRIVER
    from src.shared.python.motion_matching.club_shaft_chain import (
        apply_shaft_chain,
        build_shaft_chain,
    )
    from src.shared.python.motion_matching.same_input.plant import VectorPlant

    root = Path(__file__).resolve().parents[4]
    spec = json.loads(
        (
            root
            / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
        ).read_text(encoding="utf-8")
    )
    spec = apply_shaft_chain(spec, build_shaft_chain(DRIVER))
    plant = VectorPlant("mujoco", json.dumps(spec).encode(), kkt_regularization=1e-8)
    passive = [
        plant.coordinate_order.index(name)
        for name in plant._adapter.passive_coordinates
    ]
    assert passive
    q = np.zeros(plant.nv)
    v = np.zeros(plant.nv)
    drive = np.linspace(0.0, 1.0, plant.nv)
    external = np.linspace(0.0, 0.5, plant.nv)
    baseline = plant.acceleration(q, v, drive + external)
    result, actual_drive, actual_external = plant.acceleration_with_drive_components(
        q, v, drive, external
    )
    np.testing.assert_array_equal(result, baseline)
    expected_drive, expected_external = drive.copy(), external.copy()
    expected_drive[passive] = expected_external[passive] = 0
    np.testing.assert_array_equal(actual_drive, expected_drive)
    np.testing.assert_array_equal(actual_external, expected_external)

    def named(x):
        return dict(zip(plant.coordinate_order, x, strict=True))

    with pytest.raises(ValueError, match="components"):
        plant._adapter._constrained_solve(
            named(q),
            named(v),
            named(drive),
            effort_parts=(named(drive), named(external)),
            effort_observer=lambda *_: None,
        )
    recorder = StageRecorder(capture_ideal_drive=True)
    policy = StepPolicy(substeps=1, project=False)

    def source(*_):
        return drive

    plain = integrate(plant, q, v, source, steps=1, dt_s=1e-6, policy=policy)
    traced = integrate(
        plant, q, v, source, steps=1, dt_s=1e-6, policy=policy, recorder=recorder
    )
    np.testing.assert_array_equal(traced.q, plain.q)
    np.testing.assert_array_equal(traced.v, plain.v)
    expected_drive[:6] = 0
    assert recorder.outcome.status == "complete"
    assert len(recorder.steps[0].stages) == 4
    for stage in recorder.steps[0].stages:
        np.testing.assert_array_equal(stage.applied_drive, expected_drive)
        np.testing.assert_array_equal(stage.applied_external, np.zeros(plant.nv))
