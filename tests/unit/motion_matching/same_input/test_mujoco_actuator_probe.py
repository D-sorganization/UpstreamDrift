"""Synthetic motor seam tests; limits in these fixtures are not physiological."""

from pathlib import Path
import hashlib
import io
import json

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


def test_forward_replay_recomputes_feedback_at_each_hold(model):
    from src.shared.python.impact_optimization.option_b_prototype import (
        FixedGainController,
        SplineResidual,
    )
    from src.shared.python.motion_matching.same_input.integrator import (
        StepPolicy,
        integrate,
    )
    from src.shared.python.motion_matching.same_input.motor_forward_replay import (
        measured_motor_runtime,
        verify_motor_forward,
    )
    from src.shared.python.motion_matching.same_input.mujoco_actuator_probe import (
        MotorRolloutPlant,
    )
    from src.shared.python.motion_matching.same_input.stage_numeric_archive import (
        encode_numeric,
    )
    from src.shared.python.motion_matching.same_input.stage_trace import StageRecorder

    n = len(model.coordinate_order)
    q0 = np.zeros(n)
    ref = q0.copy()
    ref[6] = 0.1
    gain = q0.copy()
    gain[6] = 4.0
    damping = q0.copy()
    damping[6] = 2.0
    xml = (
        f'<actuator><motor joint="{model.coordinate_order[6]}" '
        'ctrllimited="true" ctrlrange="-0.1 0.1"/></actuator>'
    ).encode()
    plant = MotorRolloutPlant(ControlledMujocoDrive(model, xml.decode()), np.array([6]))
    cfg = {
        "q0": q0.tolist(),
        "v0": q0.tolist(),
        "times_s": [0.0, 2e-7],
        "q_ref": [ref.tolist(), ref.tolist()],
        "qd_ref": [q0.tolist(), q0.tolist()],
        "tau_ff_ref": [q0.tolist(), q0.tolist()],
        "kp": gain.tolist(),
        "kd": damping.tolist(),
        "lower": (-np.ones(n)).tolist(),
        "upper": np.ones(n).tolist(),
        "breakpoints_s": [0.0, 2e-7],
        "coefficients": np.zeros((4, n)).tolist(),
        "scale": np.ones(n).tolist(),
        "trust_radius": 1.0,
        "min_spacing_s": 1e-8,
        "kkt_regularization": 1e-6,
    }
    spline = SplineResidual(
        breakpoints_s=cfg["breakpoints_s"],
        coefficients_nm=cfg["coefficients"],
        scale_nm=cfg["scale"],
        trust_radius=1.0,
        min_spacing_s=1e-8,
    )
    control = FixedGainController(
        times_s=cfg["times_s"],
        q_ref=cfg["q_ref"],
        qd_ref=cfg["qd_ref"],
        tau_ff_ref=cfg["tau_ff_ref"],
        kp=cfg["kp"],
        kd=cfg["kd"],
        lower_nm=cfg["lower"],
        upper_nm=cfg["upper"],
        residual=spline,
    )
    trace = StageRecorder(capture_motor_drive=True)
    result = integrate(
        plant,
        q0,
        q0,
        lambda _i, t, q, v: control.evaluate(t, q, v).commanded_nm,
        steps=2,
        dt_s=1e-7,
        policy=StepPolicy(substeps=2, project=False),
        recorder=trace,
    )
    assert result.failure is None
    assert result.efforts[0, 6] != result.efforts[1, 6]
    raw, spec = encode_numeric(trace), SPEC.read_bytes()
    config = json.dumps(cfg, sort_keys=True).encode()
    checked = verify_motor_forward(
        raw,
        spec,
        xml,
        config,
        expected_archive_sha256=hashlib.sha256(raw).hexdigest(),
        expected_model_sha256=hashlib.sha256(spec).hexdigest(),
        expected_motor_sha256=hashlib.sha256(xml).hexdigest(),
        expected_controller_sha256=hashlib.sha256(config).hexdigest(),
        expected_runtime_sha256=measured_motor_runtime(),
    )
    assert checked.stages == 16 and checked.forward_replay_verified


def test_forward_replay_prescribed_force_separates_motor_input(model):
    from src.shared.python.impact_optimization.option_b_prototype import (
        FixedGainController,
        SplineResidual,
    )
    from src.shared.python.motion_matching.same_input.integrator import (
        StepPolicy,
        integrate,
    )
    from src.shared.python.motion_matching.same_input.motor_forward_replay import (
        measured_motor_runtime,
        verify_motor_forward,
    )
    from src.shared.python.motion_matching.same_input.mujoco_actuator_probe import (
        MotorRolloutPlant,
    )
    from src.shared.python.motion_matching.same_input.stage_numeric_archive import (
        encode_numeric,
    )
    from src.shared.python.motion_matching.same_input.stage_trace import StageRecorder

    n = len(model.coordinate_order)
    q0 = np.zeros(n)
    ref = q0.copy()
    ref[6] = 0.1
    gain = q0.copy()
    gain[6] = 4.0
    damping = q0.copy()
    damping[6] = 2.0
    xml = (
        f'<actuator><motor joint="{model.coordinate_order[6]}" '
        'ctrllimited="true" ctrlrange="-0.1 0.1"/></actuator>'
    ).encode()
    plant = MotorRolloutPlant(ControlledMujocoDrive(model, xml.decode()), np.array([6]))
    cfg = {
        "q0": q0.tolist(),
        "v0": q0.tolist(),
        "times_s": [0.0, 2e-7],
        "q_ref": [ref.tolist(), ref.tolist()],
        "qd_ref": [q0.tolist(), q0.tolist()],
        "tau_ff_ref": [q0.tolist(), q0.tolist()],
        "kp": gain.tolist(),
        "kd": damping.tolist(),
        "lower": (-np.ones(n)).tolist(),
        "upper": np.ones(n).tolist(),
        "breakpoints_s": [0.0, 2e-7],
        "coefficients": np.zeros((4, n)).tolist(),
        "scale": np.ones(n).tolist(),
        "trust_radius": 1.0,
        "min_spacing_s": 1e-8,
        "kkt_regularization": 1e-6,
    }
    spline = SplineResidual(
        breakpoints_s=cfg["breakpoints_s"],
        coefficients_nm=cfg["coefficients"],
        scale_nm=cfg["scale"],
        trust_radius=1.0,
        min_spacing_s=1e-8,
    )
    control = FixedGainController(
        times_s=cfg["times_s"],
        q_ref=cfg["q_ref"],
        qd_ref=cfg["qd_ref"],
        tau_ff_ref=cfg["tau_ff_ref"],
        kp=cfg["kp"],
        kd=cfg["kd"],
        lower_nm=cfg["lower"],
        upper_nm=cfg["upper"],
        residual=spline,
    )
    from src.shared.python.motion_matching.impact_force import ImpactForce

    impact = ImpactForce(
        frame="Clubhead",
        point_in_frame=np.zeros(3),
        axis_in_frame=np.array([1.0, 0.0, 0.0]),
        t_start_s=0.5e-7,
        duration_s=1e-7,
        swing_span_s=(0.0, 2e-7),
        force_world=np.array([-100.0, 0.0, 0.0]),
    )
    cfg["impact"] = {
        "status": "recorded_latched_force",
        "frame": "Clubhead",
        "point_in_frame_m": [0.0, 0.0, 0.0],
        "axis_in_frame": [1.0, 0.0, 0.0],
        "time_s": 0.5e-7,
        "duration_s": 1e-7,
        "force_world_n": [-100.0, 0.0, 0.0],
        "contact_verified": False,
    }
    trace = StageRecorder(capture_motor_drive=True)
    result = integrate(
        plant,
        q0,
        q0,
        lambda _i, t, q, v: control.evaluate(t, q, v).commanded_nm,
        steps=2,
        dt_s=1e-7,
        policy=StepPolicy(substeps=2, project=False),
        recorder=trace,
        impact=impact,
    )
    assert result.failure is None
    plain = integrate(
        plant,
        q0,
        q0,
        lambda _i, t, q, v: control.evaluate(t, q, v).commanded_nm,
        steps=2,
        dt_s=1e-7,
        policy=StepPolicy(substeps=2, project=False),
        impact=impact,
    )
    np.testing.assert_array_equal(result.q, plain.q)
    np.testing.assert_array_equal(result.v, plain.v)
    assert any(
        np.linalg.norm(s.applied_external) > 0
        for step in trace.steps
        for s in step.stages
    )
    assert all(
        abs(s.applied_drive[6]) <= 0.1 for step in trace.steps for s in step.stages
    )
    raw, spec = encode_numeric(trace), SPEC.read_bytes()
    config = json.dumps(cfg, sort_keys=True).encode()
    checked = verify_motor_forward(
        raw,
        spec,
        xml,
        config,
        expected_archive_sha256=hashlib.sha256(raw).hexdigest(),
        expected_model_sha256=hashlib.sha256(spec).hexdigest(),
        expected_motor_sha256=hashlib.sha256(xml).hexdigest(),
        expected_controller_sha256=hashlib.sha256(config).hexdigest(),
        expected_runtime_sha256=measured_motor_runtime(),
    )
    assert checked.forward_replay_verified and not checked.contact_verified
    changed = json.loads(config)
    changed["impact"]["contact_verified"] = True
    forged = json.dumps(changed, sort_keys=True).encode()
    with pytest.raises(ValueError, match="prescribed-force contract"):
        verify_motor_forward(
            raw,
            spec,
            xml,
            forged,
            expected_archive_sha256=hashlib.sha256(raw).hexdigest(),
            expected_model_sha256=hashlib.sha256(spec).hexdigest(),
            expected_motor_sha256=hashlib.sha256(xml).hexdigest(),
            expected_controller_sha256=hashlib.sha256(forged).hexdigest(),
            expected_runtime_sha256=measured_motor_runtime(),
        )


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
    from src.shared.python.motion_matching.same_input.motor_forward_replay import (
        measured_motor_runtime,
        verify_motor_forward,
    )

    config = json.dumps(
        {
            "q0": q.tolist(),
            "v0": q.tolist(),
            "times_s": [0.0, 2e-7],
            "q_ref": [q.tolist(), q.tolist()],
            "qd_ref": [q.tolist(), q.tolist()],
            "tau_ff_ref": [command.tolist(), command.tolist()],
            "kp": q.tolist(),
            "kd": q.tolist(),
            "lower": (-np.ones(len(q)) * 1000).tolist(),
            "upper": (np.ones(len(q)) * 1000).tolist(),
            "breakpoints_s": [0.0, 2e-7],
            "coefficients": np.zeros((4, len(q))).tolist(),
            "scale": np.ones(len(q)).tolist(),
            "trust_radius": 1.0,
            "min_spacing_s": 1e-8,
            "kkt_regularization": 1e-6,
        },
        sort_keys=True,
    ).encode()
    spec = SPEC.read_bytes()
    motors = f"<actuator>{xml}</actuator>".encode()
    hashes = {
        "expected_archive_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_model_sha256": hashlib.sha256(spec).hexdigest(),
        "expected_motor_sha256": hashlib.sha256(motors).hexdigest(),
        "expected_controller_sha256": hashlib.sha256(config).hexdigest(),
        "expected_runtime_sha256": measured_motor_runtime(),
    }
    checked = verify_motor_forward(raw, spec, motors, config, **hashes)
    assert checked.stages == 16 and checked.forward_replay_verified
    assert not checked.independent_runtime_verified
    with pytest.raises(ValueError, match="controller.*hash"):
        verify_motor_forward(raw, spec, motors, config + b" ", **hashes)
    with pytest.raises(ValueError, match="runtime"):
        verify_motor_forward(
            raw, spec, motors, config, **{**hashes, "expected_runtime_sha256": "0" * 64}
        )
    altered = json.loads(config)
    altered["tau_ff_ref"][0][6] = 2.5
    altered["tau_ff_ref"][1][6] = 2.5
    altered_config = json.dumps(altered, sort_keys=True).encode()
    with pytest.raises(ValueError, match="stages differ"):
        verify_motor_forward(
            raw,
            spec,
            motors,
            altered_config,
            **{
                **hashes,
                "expected_controller_sha256": hashlib.sha256(
                    altered_config
                ).hexdigest(),
            },
        )
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
    with pytest.raises(ValueError, match="stages differ"):
        verify_motor_forward(
            edited_raw,
            spec,
            motors,
            config,
            **{
                **hashes,
                "expected_archive_sha256": hashlib.sha256(edited_raw).hexdigest(),
            },
        )
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
    bad = q.copy()
    bad[7] = 1
    with pytest.raises(ValueError, match="unmapped"):
        plant.acceleration(q, q, bad)


@pytest.mark.parametrize("project", [False, True])
@pytest.mark.parametrize("with_impact", [False, True])
def test_motor_bundle_v2_fresh_replay_and_consistency(
    model, tmp_path, project, with_impact
):
    from src.shared.python.impact_optimization.option_b_prototype import (
        FixedGainController,
        SplineResidual,
    )
    from src.shared.python.motion_matching.same_input.integrator import (
        StepPolicy,
        integrate,
    )
    from src.shared.python.motion_matching.same_input.mujoco_actuator_probe import (
        MotorRolloutPlant,
    )
    from src.shared.python.motion_matching.same_input.stage_numeric_archive import (
        encode_numeric,
    )
    from src.shared.python.motion_matching.same_input.stage_trace import StageRecorder
    from src.shared.python.motion_matching.same_input.motor_forward_replay import (
        measured_motor_runtime,
    )
    from src.shared.python.optimization.solution_bundle import (
        SolutionBundle,
        save_solution_bundle,
    )
    from src.shared.python.motion_matching.same_input.motor_solution_bundle import (
        measured_bundle_motor_runtime,
        verify_motor_solution,
    )

    n = len(model.coordinate_order)
    q0 = np.zeros(n)
    if project:
        from scipy.optimize import least_squares

        probe_xml = f'<actuator><motor joint="{model.coordinate_order[6]}"/></actuator>'
        probe = MotorRolloutPlant(
            ControlledMujocoDrive(model, probe_xml), np.array([6])
        )
        closed = least_squares(
            probe.closure_pose_residual, q0, ftol=1e-12, xtol=1e-12, gtol=1e-12
        )
        assert np.max(np.abs(closed.fun)) < 1e-8
        q0 = closed.x
    v0 = np.zeros(n)
    target = q0.copy()
    target[6] = 0.1
    kp = v0.copy()
    kp[6] = 4.0
    kd = v0.copy()
    kd[6] = 2.0
    xml = f'<actuator><motor joint="{model.coordinate_order[6]}" ctrllimited="true" ctrlrange="-0.1 0.1"/></actuator>'.encode()
    cfg = {
        "q0": q0.tolist(),
        "v0": v0.tolist(),
        "times_s": [0.0, 2e-7],
        "q_ref": [target.tolist()] * 2,
        "qd_ref": [v0.tolist()] * 2,
        "tau_ff_ref": [v0.tolist()] * 2,
        "kp": kp.tolist(),
        "kd": kd.tolist(),
        "lower": (-np.ones(n)).tolist(),
        "upper": np.ones(n).tolist(),
        "breakpoints_s": [0.0, 2e-7],
        "coefficients": np.zeros((4, n)).tolist(),
        "scale": np.ones(n).tolist(),
        "trust_radius": 1.0,
        "min_spacing_s": 1e-8,
        "kkt_regularization": 1e-6,
    }
    impact = None
    events = []
    if with_impact:
        from src.shared.python.motion_matching.impact_force import ImpactForce

        impact = ImpactForce(
            frame="Clubhead",
            point_in_frame=np.zeros(3),
            axis_in_frame=np.array([1.0, 0.0, 0.0]),
            t_start_s=0.5e-7,
            duration_s=1e-7,
            swing_span_s=(0.0, 2e-7),
            force_world=np.array([-100.0, 0.0, 0.0]),
        )
        cfg["impact"] = {
            "status": "recorded_latched_force",
            "frame": "Clubhead",
            "point_in_frame_m": [0.0, 0.0, 0.0],
            "axis_in_frame": [1.0, 0.0, 0.0],
            "time_s": 0.5e-7,
            "duration_s": 1e-7,
            "force_world_n": [-100.0, 0.0, 0.0],
            "contact_verified": False,
        }

    def observe(t, q, v):
        events.append((t, q.copy(), v.copy()))

    spline = SplineResidual(
        breakpoints_s=cfg["breakpoints_s"],
        coefficients_nm=cfg["coefficients"],
        scale_nm=cfg["scale"],
        trust_radius=1.0,
        min_spacing_s=1e-8,
    )
    control = FixedGainController(
        times_s=cfg["times_s"],
        q_ref=cfg["q_ref"],
        qd_ref=cfg["qd_ref"],
        tau_ff_ref=cfg["tau_ff_ref"],
        kp=cfg["kp"],
        kd=cfg["kd"],
        lower_nm=cfg["lower"],
        upper_nm=cfg["upper"],
        residual=spline,
    )
    plant = MotorRolloutPlant(ControlledMujocoDrive(model, xml.decode()), np.array([6]))
    trace = StageRecorder(capture_motor_drive=True)
    run = integrate(
        plant,
        q0,
        v0,
        lambda _i, t, q, v: control.evaluate(t, q, v).commanded_nm,
        steps=2,
        dt_s=1e-7,
        policy=StepPolicy(substeps=2, project=project),
        recorder=trace,
        impact=impact,
        impact_event_observer=observe if with_impact else None,
    )
    assert run.failure is None
    raw = encode_numeric(trace)
    spec = SPEC.read_bytes()
    config = json.dumps(cfg, sort_keys=True).encode()

    def digest(x):
        return hashlib.sha256(x).hexdigest()

    drive = np.stack([step.stages[0].applied_drive for step in trace.steps])
    arr = {
        "time": run.time_s,
        "q_ref": np.tile(target, (3, 1)),
        "qd_ref": np.zeros((3, n)),
        "tau_ff_ref": np.zeros((3, n)),
        "q_rollout": run.q,
        "qd_rollout": run.v,
        "tau_cmd": np.vstack((run.efforts, run.efforts[-1])),
        "tau_applied": np.vstack((drive, drive[-1])),
        "knots": np.array([0, 0, 0, 0, 2e-7, 2e-7, 2e-7, 2e-7]),
        "delta_tau_coeffs": np.zeros((4, n)),
        "q0": q0,
        "qd0": v0,
        "controller_state0": np.zeros(0),
        "kp": kp,
        "kd": kd,
        "motor_stage_npz": np.frombuffer(raw, dtype=np.uint8),
        "motor_xml": np.frombuffer(xml, dtype=np.uint8),
        "controller_json": np.frombuffer(config, dtype=np.uint8),
        "model_json": np.frombuffer(spec, dtype=np.uint8),
    }
    bundle = SolutionBundle(
        solution_id="synthetic",
        parent_id=None,
        git_sha="a" * 40,
        model_hash=digest(spec),
        asset_hashes={
            "motor_stage_npz": digest(raw),
            "motor_xml": digest(xml),
            "controller_json": digest(config),
        },
        frame="world",
        joint_order=tuple(model.coordinate_order),
        control={
            "kind": "fixed_pd_bspline",
            "hold": "zoh",
            "saturation": "mujoco_motor",
            "holds_per_step": 1,
        },
        integrator={
            "kind": "rk4",
            "dt_s": 1e-7,
            "substeps": 2,
            "project": project,
            "runtime_sha256": measured_bundle_motor_runtime(),
        },
        environment={},
        optimizer_seed=0,
        impact_event=(
            {
                "t": events[0][0],
                "q": events[0][1].tolist(),
                "qd": events[0][2].tolist(),
                "semantics": "prescribed_latched_force",
                "contact_verified": False,
            }
            if with_impact
            else {"status": "none"}
        ),
        flight_metrics={},
        feasibility_margins={},
        assumptions=("synthetic limits",),
        arrays=arr,
        schema_version=2,
    )
    path = save_solution_bundle(bundle, tmp_path / "b")
    manifest_hash = digest((path / "manifest.json").read_bytes())
    checked = verify_motor_solution(
        path,
        expected_manifest_sha256=manifest_hash,
        expected_model_sha256=digest(spec),
        expected_runtime_sha256=measured_bundle_motor_runtime(),
    )
    assert (
        checked.stages == sum(len(step.stages) for step in trace.steps)
        and checked.forward_replay_verified
        and not checked.independent_runtime_verified
    )
    if with_impact:
        assert checked.event is not None and not checked.contact_verified
        original_event_q = bundle.impact_event["q"][0]
        bundle.impact_event["q"][0] += 0.01
        tampered = save_solution_bundle(bundle, tmp_path / "tampered-event")
        with pytest.raises(ValueError, match="event state mismatch"):
            verify_motor_solution(
                tampered,
                expected_manifest_sha256=digest(
                    (tampered / "manifest.json").read_bytes()
                ),
                expected_model_sha256=digest(spec),
                expected_runtime_sha256=measured_bundle_motor_runtime(),
            )
        bundle.impact_event["q"][0] = original_event_q
    with pytest.raises(ValueError, match="manifest"):
        verify_motor_solution(
            path,
            expected_manifest_sha256="0" * 64,
            expected_model_sha256=digest(spec),
            expected_runtime_sha256=measured_bundle_motor_runtime(),
        )
    with pytest.raises(ValueError, match="runtime"):
        verify_motor_solution(
            path,
            expected_manifest_sha256=manifest_hash,
            expected_model_sha256=digest(spec),
            expected_runtime_sha256="0" * 64,
        )
    for index, (key, col) in enumerate((("tau_applied", 6), ("q_ref", 6))):
        original = bundle.arrays[key][0, col]
        bundle.arrays[key][0, col] += 0.01
        bad = save_solution_bundle(bundle, tmp_path / f"bad-{index}")
        with pytest.raises(ValueError, match="mismatch"):
            verify_motor_solution(
                bad,
                expected_manifest_sha256=digest((bad / "manifest.json").read_bytes()),
                expected_model_sha256=digest(spec),
                expected_runtime_sha256=measured_bundle_motor_runtime(),
            )
        bundle.arrays[key][0, col] = original
    bundle.arrays["q_rollout"][1, 6] += 0.01
    altered = save_solution_bundle(bundle, tmp_path / "altered")
    with pytest.raises(ValueError, match="boundary"):
        verify_motor_solution(
            altered,
            expected_manifest_sha256=digest((altered / "manifest.json").read_bytes()),
            expected_model_sha256=digest(spec),
            expected_runtime_sha256=measured_bundle_motor_runtime(),
        )
