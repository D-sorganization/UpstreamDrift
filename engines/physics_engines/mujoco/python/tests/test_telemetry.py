"""Tests for MuJoCo telemetry capture and reporting."""

import mujoco
import numpy as np
import pytest
from mujoco_humanoid_golf.models import DOUBLE_PENDULUM_XML
from mujoco_humanoid_golf.telemetry import TelemetryRecorder


def test_telemetry_records_forces_and_generates_report() -> None:
    """Test for test telemetry records forces and generates report."""
    model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
    data = mujoco.MjData(model)
    recorder = TelemetryRecorder(model)

    control_input = np.array([5.0, -3.0], dtype=float)
    steps = 5

    for _ in range(steps):
        data.ctrl[:] = control_input
        mujoco.mj_step(model, data)
        recorder.record_step(data)

    report = recorder.generate_report()

    assert report.sample_count == steps
    assert set(report.peak_actuator_torques) == {"shoulder_motor", "wrist_motor"}
    assert all(value >= 0.0 for value in report.peak_actuator_torques.values())
    assert report.duration_seconds > 0.0
    assert report.to_dict()["sample_count"] == steps


def test_non_joint_actuators_are_ignored_in_mapping() -> None:
    """Test for test non joint actuators are ignored in mapping."""
    xml = """
    <mujoco model="mixed-actuators">
        <compiler angle="degree" />
        <worldbody>
            <body name="body" pos="0 0 0">
                <joint name="hinge" type="hinge" axis="0 0 1" />
                <site name="tip" pos="0 0 0.1" />
                <geom type="capsule" size="0.01 0.1" />
            </body>
        </worldbody>
        <actuator>
            <motor name="joint_motor" joint="hinge" gear="1" />
            <general name="site_act" site="tip" gear="1 0 0 0 0 0" />
        </actuator>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    recorder = TelemetryRecorder(model)

    mujoco.mj_step(model, data)
    recorder.record_step(data)

    assert set(recorder._actuator_dof_map.keys()) == {0}
    report = recorder.generate_report()
    assert set(report.peak_actuator_torques.keys()) == {"joint_motor"}


def test_body_actuator_is_ignored_in_mapping() -> None:
    """Test for test body actuator is ignored in mapping."""
    xml = """
    <mujoco model="body-actuator">
        <worldbody>
            <body name="torso" pos="0 0 0">
                <freejoint name="root" />
                <geom type="capsule" size="0.01 0.1" />
            </body>
        </worldbody>
        <actuator>
            <general name="body_wrench" body="torso" gear="1 0 0 0 0 0" />
        </actuator>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    recorder = TelemetryRecorder(model)

    mujoco.mj_step(model, data)
    recorder.record_step(data)

    assert recorder._actuator_dof_map == {}
    report = recorder.generate_report()
    assert report.peak_actuator_torques == {}


def test_tendon_actuators_do_not_break_mapping() -> None:
    """Test for test tendon actuators do not break mapping."""
    xml = """
    <mujoco model="tendon-actuator">
        <worldbody>
            <body name="body" pos="0 0 0">
                <joint name="slide" type="slide" axis="1 0 0" />
                <geom type="capsule" size="0.01 0.1" />
                <site name="anchor" pos="0 0 0" />
                <site name="anchor2" pos="0 0 0.1" />
            </body>
        </worldbody>
        <tendon>
            <spatial name="tendon" width="0.001">
                <site site="anchor" />
                <site site="anchor2" />
            </spatial>
        </tendon>
        <actuator>
            <motor name="joint_motor" joint="slide" gear="1" />
            <general name="tendon_motor" tendon="tendon" gear="1" />
        </actuator>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    recorder = TelemetryRecorder(model)

    mujoco.mj_step(model, data)
    recorder.record_step(data)

    assert set(recorder._actuator_dof_map.keys()) == {0}
    report = recorder.generate_report()
    assert set(report.peak_actuator_torques.keys()) == {"joint_motor"}


def test_jointinparent_actuator_is_mapped() -> None:
    """Test for test jointinparent actuator is mapped."""
    if not hasattr(mujoco.mjtTrn, "mjTRN_JOINTINP"):
        pytest.skip("JOINTINP transmission not available in this MuJoCo version")

    xml = """
    <mujoco model="jointinparent-actuator">
        <compiler angle="degree" />
        <worldbody>
            <body name="body" pos="0 0 0">
                <joint name="hinge" type="hinge" axis="0 0 1" />
                <geom type="capsule" size="0.01 0.1" />
            </body>
        </worldbody>
        <actuator>
            <motor name="joint_motor_parent" joint="hinge" gear="1" />
        </actuator>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    model.actuator_trntype[0] = mujoco.mjtTrn.mjTRN_JOINTINP
    data = mujoco.MjData(model)

    recorder = TelemetryRecorder(model)

    mujoco.mj_step(model, data)
    recorder.record_step(data)

    assert recorder._actuator_dof_map == {0: 0}
    report = recorder.generate_report()
    assert set(report.peak_actuator_torques.keys()) == {"joint_motor_parent"}
    assert report.peak_actuator_torques["joint_motor_parent"] >= 0.0


def test_actuator_with_invalid_joint_index_is_skipped() -> None:
    """Test for test actuator with invalid joint index is skipped."""
    xml = """
    <mujoco model="invalid-joint-target">
        <worldbody>
            <body name="body" pos="0 0 0">
                <joint name="hinge" type="hinge" axis="0 0 1" />
                <geom type="capsule" size="0.01 0.1" />
            </body>
        </worldbody>
        <actuator>
            <motor name="bad_joint_motor" joint="hinge" gear="1" />
        </actuator>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    model.actuator_trnid[0, 0] = model.njnt  # deliberately out of bounds
    data = mujoco.MjData(model)

    recorder = TelemetryRecorder(model)

    mujoco.mj_step(model, data)
    recorder.record_step(data)

    assert recorder._actuator_dof_map == {}
    report = recorder.generate_report()
    assert report.peak_actuator_torques == {}
