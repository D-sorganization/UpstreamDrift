from unittest.mock import MagicMock, mock_open, patch

import mujoco
import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.telemetry import (
    TelemetryRecorder,
    export_telemetry_csv,
    export_telemetry_json,
)


@pytest.fixture
def mock_mujoco_model_data():
    """Create mock MuJoCo model and data."""
    model = MagicMock(spec=mujoco.MjModel)
    model.nbody = 2
    model.njnt = 2
    model.nu = 2
    model.nv = 2
    model.actuator_trnid = np.array(
        [[0, 0], [1, 0]]
    )  # Actuator 0 -> Joint 0, Actuator 1 -> Joint 1
    model.actuator_trntype = np.array(
        [mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINT]
    )
    model.jnt_dofadr = np.array([0, 1])

    data = MagicMock(spec=mujoco.MjData)
    data.time = 0.0
    data.qpos = np.array([0.1, 0.2])
    data.qvel = np.array([1.0, 2.0])
    data.ctrl = np.array([0.5, -0.5])
    data.qfrc_actuator = np.array([10.0, -10.0])
    data.qfrc_constraint = np.array([5.0, -5.0])
    data.cfrc_ext = np.zeros(12)  # 2 bodies * 6
    data.cfrc_ext[6:9] = np.array([1.0, 2.0, 3.0])  # Force on body 1

    return model, data


def test_telemetry_recorder_init(mock_mujoco_model_data):
    model, _ = mock_mujoco_model_data

    with patch("mujoco.mj_id2name", side_effect=lambda m, t, i: f"obj_{i}"):
        recorder = TelemetryRecorder(model)

    assert len(recorder._body_names) == 2
    assert len(recorder._joint_names) == 2
    assert len(recorder._actuator_dof_map) == 2


def test_telemetry_recorder_record_step(mock_mujoco_model_data):
    model, data = mock_mujoco_model_data

    with patch("mujoco.mj_id2name", side_effect=lambda m, t, i: f"obj_{i}"):
        recorder = TelemetryRecorder(model)
        recorder.add_custom_metric("test_metric", 123.45)
        recorder.record_step(data)

    assert len(recorder.samples) == 1
    sample = recorder.samples[0]

    assert sample.time == 0.0
    assert np.allclose(sample.joint_positions, data.qpos)
    assert np.allclose(sample.joint_velocities, data.qvel)
    assert np.allclose(sample.controls, data.ctrl)
    assert sample.actuator_torques["obj_0"] == 10.0
    assert sample.actuator_torques["obj_1"] == -10.0
    assert sample.constraint_torques["obj_0"] == 5.0
    assert sample.body_forces["obj_1"] is not None
    assert sample.custom_metrics["test_metric"] == 123.45


def test_telemetry_report_generation(mock_mujoco_model_data):
    model, data = mock_mujoco_model_data

    with patch("mujoco.mj_id2name", side_effect=lambda m, t, i: f"obj_{i}"):
        recorder = TelemetryRecorder(model)

        # Step 1
        recorder.record_step(data)

        # Step 2
        data.time = 0.1
        data.qfrc_actuator = np.array([20.0, -5.0])
        recorder.record_step(data)

        report = recorder.generate_report()

    assert report.sample_count == 2
    assert report.duration_seconds == 0.1
    assert report.peak_actuator_torques["obj_0"] == 20.0
    assert report.peak_actuator_torques["obj_1"] == 10.0

    report_dict = report.to_dict()
    assert report_dict["sample_count"] == 2


def test_telemetry_reset(mock_mujoco_model_data):
    model, data = mock_mujoco_model_data

    with patch("mujoco.mj_id2name", side_effect=lambda m, t, i: f"obj_{i}"):
        recorder = TelemetryRecorder(model)
        recorder.record_step(data)
        assert len(recorder.samples) == 1

        recorder.reset()
        assert len(recorder.samples) == 0


def test_export_telemetry_json():
    data = {"scalar": 1.0, "array": np.array([1, 2, 3])}

    with patch("builtins.open", mock_open()) as mock_file:
        success = export_telemetry_json("test.json", data)

    assert success
    mock_file.assert_called_with("test.json", "w")
    # Verify json dump called? (Implicit by success returning True)


def test_export_telemetry_csv():
    data = {"time": np.array([0.0, 0.1]), "vec3": np.array([[1, 2, 3], [4, 5, 6]])}

    with patch("builtins.open", mock_open()) as mock_file:
        success = export_telemetry_csv("test.csv", data)

    assert success
    mock_file.assert_called_with("test.csv", "w", newline="")

    # Test with incompatible data length (should handle or fail gracefully depending on
    # impl, code pads with "")
    data_mixed = {"short": np.array([1]), "long": np.array([1, 2])}
    with patch("builtins.open", mock_open()) as mock_file:
        success = export_telemetry_csv("test_mixed.csv", data_mixed)
        assert success


def test_export_fail_handling():
    # Test exceptions
    with patch("builtins.open", side_effect=PermissionError):
        assert not export_telemetry_json("fail.json", {})
        assert not export_telemetry_csv("fail.csv", {"a": [1]})
