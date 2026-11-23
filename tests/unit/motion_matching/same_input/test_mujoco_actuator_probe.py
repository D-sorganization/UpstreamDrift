"""Synthetic motor seam tests; limits in these fixtures are not physiological."""

from pathlib import Path
import hashlib
import io

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.shared.python.motion_matching.same_input.mujoco_actuator_probe import (
    ControlledMujocoDrive,
)

pytestmark = pytest.mark.unit

SPEC = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
)


@pytest.fixture(scope="module")
def model():
    return NativeMujocoFullBodyModel(SPEC.read_bytes())


def test_missing_and_stateful_actuators_fail_closed(model):
    with pytest.raises(ValueError, match="actuator"):
        ControlledMujocoDrive(model, "<actuator/>")
    name = model.coordinate_order[6]
    with pytest.raises(ValueError, match="stateful"):
        ControlledMujocoDrive(
            model,
            f'<actuator><general joint="{name}" dyntype="filter" dynprm="1"/></actuator>',
        )


def test_actual_motor_total_drives_existing_kkt_once(model):
    names = model.coordinate_order
    xml = "".join(
        f'<motor joint="{n}" gear="1" ctrllimited="true" ctrlrange="-1 1"/>'
        for n in names[6:]
    )
    system = ControlledMujocoDrive(model, f"<actuator>{xml}</actuator>")
    q = np.zeros(len(names))
    v = np.zeros(len(names))
    command = np.resize(np.array([2.0, -2.0, 0.25]), len(names) - 6)
    acceleration, applied = system.acceleration(q, v, command)
    expected = np.r_[np.zeros(6), np.clip(command, -1, 1)]
    assert np.array_equal(applied, expected)
    plain = model.accelerations(
        dict(zip(names, q, strict=True)),
        dict(zip(names, v, strict=True)),
        dict(zip(names, expected, strict=True)),
    )
    assert np.allclose(acceleration, [plain[n] for n in names], atol=1e-12, rtol=1e-12)
    assert not np.array_equal(applied[6:], command)


def test_control_and_input_validation(model):
    name = model.coordinate_order[6]
    system = ControlledMujocoDrive(
        model, f'<actuator><motor joint="{name}"/></actuator>'
    )
    q = np.zeros(len(model.coordinate_order))
    for ctrl in (np.array([np.nan]), np.array([1.0, 2.0])):
        with pytest.raises(ValueError):
            system.acceleration(q, q, ctrl)
    bad = q.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        system.acceleration(bad, q, np.array([1.0]))


def test_force_limit_and_transmission_are_measured_after_saturation(model):
    name = model.coordinate_order[6]
    system = ControlledMujocoDrive(
        model,
        f'<actuator><motor joint="{name}" gear="2" forcelimited="true" forcerange="-0.5 0.5"/></actuator>',
    )
    q = np.zeros(len(model.coordinate_order))
    _, applied = system.acceleration(q, q, np.array([3.0]))
    assert applied[6] == pytest.approx(1.0)
    assert np.count_nonzero(applied) == 1


def test_repeated_calls_do_not_accumulate_actuation(model):
    name = model.coordinate_order[6]
    system = ControlledMujocoDrive(
        model, f'<actuator><motor joint="{name}"/></actuator>'
    )
    q = np.zeros(len(model.coordinate_order))
    a1, t1 = system.acceleration(q, q, np.array([1.0]))
    system.acceleration(q, q, np.array([-0.5]))
    a2, t2 = system.acceleration(q, q, np.array([1.0]))
    assert np.array_equal(t1, t2)
    assert np.array_equal(a1, a2)


def test_fixed_hold_rollout_records_post_saturation_motor_forces(model):
    from src.shared.python.motion_matching.same_input.integrator import (
        StepPolicy,
        integrate,
    )
    from src.shared.python.motion_matching.same_input.stage_trace import StageRecorder
    from src.shared.python.motion_matching.same_input.mujoco_actuator_probe import (
        MotorRolloutPlant,
    )

    names = model.coordinate_order
    xml = "".join(
        f'<motor joint="{n}" gear="2" ctrllimited="true" ctrlrange="-1 1"/>'
        for n in names[6:]
    )
    drive = ControlledMujocoDrive(model, f"<actuator>{xml}</actuator>")
    plant = MotorRolloutPlant(drive, np.arange(6, len(names)))
    q = np.zeros(len(names))
    command = np.r_[np.zeros(6), np.full(len(names) - 6, 3.0)]
    trace = StageRecorder(capture_motor_drive=True)
    out = integrate(
        plant,
        q,
        q,
        lambda *_: command,
        steps=2,
        dt_s=0.0000001,
        policy=StepPolicy(substeps=2, project=False),
        recorder=trace,
    )
    assert out.failure is None and trace.outcome.status == "complete"
    plain = integrate(
        plant,
        q,
        q,
        lambda *_: command,
        steps=2,
        dt_s=1e-7,
        policy=StepPolicy(substeps=2, project=False),
    )
    np.testing.assert_array_equal(out.q, plain.q)
    np.testing.assert_array_equal(out.v, plain.v)
    stages = [stage for step in trace.steps for stage in step.stages]
    assert len(stages) == 16
    for stage in stages:
        np.testing.assert_array_equal(stage.tau, command)
        np.testing.assert_array_equal(
            stage.applied_drive, np.r_[np.zeros(6), np.full(len(names) - 6, 2.0)]
        )
        np.testing.assert_array_equal(stage.applied_external, np.zeros(len(names)))
        assert stage.effort_semantics == "mujoco_post_saturation_generalized_motor"

    from src.shared.python.motion_matching.same_input.stage_numeric_archive import (
        decode_numeric,
        encode_numeric,
    )

    from types import SimpleNamespace
    from src.shared.python.motion_matching.same_input.evidenced_solution import (
        _check_plan,
    )

    with pytest.raises(ValueError, match="ideal"):
        _check_plan(SimpleNamespace(manifest={"runner": {}}), trace)
    raw = encode_numeric(trace)
    with pytest.raises(ValueError, match="semantics"):
        decode_numeric(raw)
    with np.load(__import__("io").BytesIO(raw), allow_pickle=False) as data:
        fields = {k: data[k] for k in data.files}
    del fields["drive_semantics"]
    stripped = io.BytesIO()
    np.savez_compressed(stripped, **fields)
    with pytest.raises(ValueError, match="semantics"):
        decode_numeric(
            stripped.getvalue(),
            expected_semantics="mujoco_post_saturation_generalized_motor",
        )
    from src.shared.python.motion_matching.same_input.motor_stage_verification import (
        verify_motor_stage_forces,
    )

    spec = SPEC.read_bytes()
    motors = f"<actuator>{xml}</actuator>".encode()
    verification = verify_motor_stage_forces(
        raw,
        spec,
        motors,
        expected_archive_sha256=hashlib.sha256(raw).hexdigest(),
        expected_model_sha256=hashlib.sha256(spec).hexdigest(),
        expected_motor_sha256=hashlib.sha256(motors).hexdigest(),
        atol=0.0,
    )
    assert verification.stages == 16
    assert verification.max_abs_error == 0.0
    assert verification.independent_runtime_verified is False
    assert verification.forward_replay_verified is False
    assert verification.physical_feasibility_verified is False
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        edited = {k: data[k] for k in data.files}
    edited["drive"][0, 6] += 0.25
    corrupt = io.BytesIO()
    np.savez_compressed(corrupt, **edited)
    edited_raw = corrupt.getvalue()
    with pytest.raises(ValueError, match="force differs"):
        verify_motor_stage_forces(
            edited_raw,
            spec,
            motors,
            expected_archive_sha256=hashlib.sha256(edited_raw).hexdigest(),
            expected_model_sha256=hashlib.sha256(spec).hexdigest(),
            expected_motor_sha256=hashlib.sha256(motors).hexdigest(),
            atol=0.0,
        )
    with pytest.raises(ValueError, match="hash"):
        verify_motor_stage_forces(
            raw,
            spec,
            motors,
            expected_archive_sha256=hashlib.sha256(raw).hexdigest(),
            expected_model_sha256="0" * 64,
            expected_motor_sha256=hashlib.sha256(motors).hexdigest(),
            atol=0.0,
        )
    restored = decode_numeric(
        raw, expected_semantics="mujoco_post_saturation_generalized_motor"
    )
    assert restored.drive_semantics == "mujoco_post_saturation_generalized_motor"
    restored_stages = [s for step in restored.steps for s in step.stages]
    for actual, expected in zip(restored_stages, stages, strict=True):
        assert actual.effort_semantics == expected.effort_semantics
        np.testing.assert_array_equal(actual.tau, expected.tau)
        np.testing.assert_array_equal(actual.applied_drive, expected.applied_drive)
        assert not np.array_equal(actual.tau, actual.applied_drive)


def test_motor_rollout_rejects_misleading_recording_and_wrong_mapping(model):
    from src.shared.python.motion_matching.same_input.integrator import (
        StepPolicy,
        integrate,
    )
    from src.shared.python.motion_matching.same_input.stage_trace import StageRecorder
    from src.shared.python.motion_matching.same_input.mujoco_actuator_probe import (
        MotorRolloutPlant,
    )

    names = model.coordinate_order
    xml = f'<actuator><motor joint="{names[6]}"/></actuator>'
    drive = ControlledMujocoDrive(model, xml)
    with pytest.raises(ValueError, match="motor joint order"):
        MotorRolloutPlant(drive, np.array([7]))
    plant = MotorRolloutPlant(drive, np.array([6]))
    q = np.zeros(len(names))
    with pytest.raises(ValueError, match="semantics"):
        integrate(
            plant,
            q,
            q,
            lambda *_: q,
            steps=1,
            dt_s=1e-7,
            policy=StepPolicy(substeps=1, project=False),
            recorder=StageRecorder(capture_ideal_drive=True),
        )
    with pytest.raises(ValueError, match="impact"):
        integrate(
            plant,
            q,
            q,
            lambda *_: q,
            steps=1,
            dt_s=1e-7,
            policy=StepPolicy(substeps=1, project=False),
            impact=object(),
        )
    with pytest.raises(ValueError, match="projection"):
        integrate(plant, q, q, lambda *_: q, steps=1, dt_s=1e-7)
    bad = q.copy()
    bad[7] = 1
    with pytest.raises(ValueError, match="unmapped"):
        plant.acceleration(q, q, bad)
