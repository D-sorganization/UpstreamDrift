"""Integration tests for connecting Swing, Impact, Flight, and Trajectory Viewers (ORG-14, #10523).

Acceptance criteria:
- RED: a record from each supported family replays in each supported viewer without altering retained samples.
- RED: mismatched frames/units, invalid hashes and unsupported full-body extraction are refused.
- GREEN: manual/MuJoCo supported source -> flight -> both comparison surfaces with method provenance;
  cancellation retains earlier trajectory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.api.routes._ball_flight_trajectory_import import (
    ImportedBallFlightTrajectory,
    TrajectoryImportError as WebTrajectoryImportError,
    import_trajectory_record as import_web_trajectory_record,
)
from src.launchers._shot_tracer_trajectory_import import (
    ImportedTrajectoryCurve,
    TrajectoryImportError as QtTrajectoryImportError,
    import_trajectory_record as import_qt_trajectory_record,
)
from src.shared.python.physics.ball_launch_conditions import (
    EnvironmentalConditions,
    LaunchConditions,
)
from src.shared.python.physics.flight_models import (
    FlightModelRegistry,
    FlightModelType,
    UnifiedLaunchConditions,
)
from src.shared.python.physics.flight_trajectory_export import (
    APP_FRAME_ID,
    BALL_FLIGHT_TRAJECTORY_FORMAT,
    FLIGHT_FRAME_ID,
    UD_FLIGHT_FAMILY,
    VELOCITY_CHANNEL,
    flight_result_to_trajectory_record,
    pipeline_result_to_trajectory_record,
    trajectory_parameter_digest,
    trajectory_record_to_json,
)
from src.shared.python.physics.swing_ball_flight_pipeline import (
    PipelineResult,
    SwingBallFlightPipeline,
    SwingState,
)
from src.shared.python.physics.swing_state_providers import (
    ManualSwingStateProvider,
    MuJoCoSwingStateProvider,
    SwingStateConfig,
)
from src.shared.python.workspace.results_workspace import (
    ActionAvailability,
    ResultArtifactItem,
    ResultCategory,
    ResultsWorkspaceCoordinator,
    WorkspaceActionType,
)
from src.shared.python.workspace.trajectory_handoff import (
    ExtractionAdapterError,
    FrameUnitMismatchError,
    InvalidTrajectoryHashError,
    ShotTrajectoryHandoffCoordinator,
    UnsupportedEngineSourceError,
)

pytestmark = pytest.mark.integration


def _build_ud_family_fixture() -> dict[str, Any]:
    """Generate a valid record from the ud.flight_models family (Waterloo/Penner)."""
    launch = UnifiedLaunchConditions.from_imperial(
        ball_speed_mph=160.0,
        launch_angle_deg=10.5,
        spin_rate_rpm=2400.0,
    )
    model = FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)
    result = model.simulate(launch)
    return flight_result_to_trajectory_record(
        result,
        source_id="ud:test-waterloo-01",
        model_type=FlightModelType.WATERLOO_PENNER,
    )


def _build_tools_family_fixture() -> dict[str, Any]:
    """Generate a valid wire record representing the Tools swing_sim.flight family."""
    parameters = {"cd": 0.22, "cl": 0.16, "spin_decay": 0.075}
    digest = trajectory_parameter_digest(parameters)
    t = np.linspace(0.0, 5.0, 51)
    samples: list[dict[str, Any]] = []
    for time_val in t:
        x = float(45.0 * time_val)
        y = float(0.5 * math.sin(time_val))
        z = float(max(0.0, 25.0 * time_val - 4.9 * (time_val**2)))
        vx = 45.0
        vy = float(0.5 * math.cos(time_val))
        vz = float(25.0 - 9.8 * time_val)
        samples.append(
            {
                "time_s": float(time_val),
                "position_m": [x, y, z],
                VELOCITY_CHANNEL: [vx, vy, vz],
            }
        )
    return {
        "format": BALL_FLIGHT_TRAJECTORY_FORMAT,
        "source_id": "tools:swing_sim-flight-01",
        "frame_id": FLIGHT_FRAME_ID,
        "channels": [VELOCITY_CHANNEL],
        "provenance": {
            "model_family": "swing_sim.flight",
            "model_name": "CapabilityEvaluator",
            "parameter_digest": digest,
        },
        "samples": samples,
    }


def test_red_cross_family_records_replay_without_altering_retained_samples(
    tmp_path: Path,
) -> None:
    """Acceptance RED 1: records from each supported family replay in each viewer without altering retained samples."""
    ud_record = _build_ud_family_fixture()
    tools_record = _build_tools_family_fixture()

    coordinator = ShotTrajectoryHandoffCoordinator(repo_root=tmp_path)

    for record, family_id in [
        (ud_record, UD_FLIGHT_FAMILY),
        (tools_record, "swing_sim.flight"),
    ]:
        file_path = tmp_path / f"traj_{family_id.replace('.', '_')}.json"
        raw_json = trajectory_record_to_json(record)
        file_path.write_text(raw_json, encoding="utf-8")

        original_samples_copy = copy.deepcopy(record["samples"])
        original_bytes = file_path.read_bytes()

        # 1. Replay / import in Shot Tracer (Qt)
        qt_curve = coordinator.load_into_shot_tracer(file_path)
        assert isinstance(qt_curve, ImportedTrajectoryCurve)
        assert qt_curve.model_family == family_id
        assert len(qt_curve.positions) == len(original_samples_copy)
        for i, sample in enumerate(original_samples_copy):
            np.testing.assert_allclose(qt_curve.positions[i], sample["position_m"])

        # 2. Replay / import in BallFlight (Web route model)
        web_traj = coordinator.load_into_ball_flight_web(file_path)
        assert isinstance(web_traj, ImportedBallFlightTrajectory)
        assert web_traj.model_family == family_id
        assert len(web_traj.samples) == len(original_samples_copy)
        for i, sample in enumerate(original_samples_copy):
            assert web_traj.samples[i].time_s == sample["time_s"]
            assert list(web_traj.samples[i].position_m) == sample["position_m"]

        # 3. Replay in Impact Explorer 3D playback format
        impact_playback = coordinator.load_into_impact_explorer(file_path)
        assert impact_playback["model_family"] == family_id
        assert impact_playback["frame_id"] == FLIGHT_FRAME_ID
        assert len(impact_playback["samples"]) == len(original_samples_copy)
        for i, sample in enumerate(original_samples_copy):
            assert impact_playback["samples"][i]["time_s"] == sample["time_s"]
            assert impact_playback["samples"][i]["position_m"] == sample["position_m"]

        # CRITICAL: Retained samples and disk bytes must remain completely unaltered
        assert file_path.read_bytes() == original_bytes
        assert record["samples"] == original_samples_copy


def test_red_refusal_of_mismatched_frames_units_invalid_hashes_and_unsupported_extraction(
    tmp_path: Path,
) -> None:
    """Acceptance RED 2: mismatched frames/units, invalid hashes and unsupported full-body extraction are refused."""
    coordinator = ShotTrajectoryHandoffCoordinator(repo_root=tmp_path)

    # 1. Mismatched / unsupported frame refusal
    bad_frame_record = _build_tools_family_fixture()
    bad_frame_record["frame_id"] = "unsupported_frame_xyz"
    bad_frame_path = tmp_path / "bad_frame.json"
    bad_frame_path.write_text(json.dumps(bad_frame_record), encoding="utf-8")

    with pytest.raises(FrameUnitMismatchError) as exc_info:
        coordinator.validate_and_register_trajectory(bad_frame_path)
    assert "unsupported frame" in str(exc_info.value).lower()

    # 2. Invalid cryptographic hash refusal
    good_record = _build_ud_family_fixture()
    good_path = tmp_path / "good.json"
    good_path.write_text(trajectory_record_to_json(good_record), encoding="utf-8")
    item = coordinator.validate_and_register_trajectory(good_path)

    # Corrupt the file after registration
    good_path.write_text(
        json.dumps({"corrupted": True, "format": "tampered"}), encoding="utf-8"
    )
    with pytest.raises(InvalidTrajectoryHashError) as exc_hash:
        coordinator.verify_artifact_integrity(item)
    assert "cryptographic hash mismatch" in str(exc_hash.value).lower()

    # 3. Unsupported full-body engine extraction refusal (Drake / Pinocchio)
    with pytest.raises(UnsupportedEngineSourceError) as exc_drake:
        coordinator.extract_swing_state_from_run(
            engine="drake",
            run_data={"run_id": "drake_run_01", "q": [0.1, 0.2]},
        )
    assert "drake" in str(exc_drake.value).lower()
    assert (
        "not supported" in str(exc_drake.value).lower()
        or "not implemented" in str(exc_drake.value).lower()
    )

    with pytest.raises(UnsupportedEngineSourceError) as exc_pin:
        coordinator.extract_swing_state_from_run(
            engine="pinocchio",
            run_data={"run_id": "pinocchio_run_01", "tau": [1.0, 2.0]},
        )
    assert "pinocchio" in str(exc_pin.value).lower()

    # 4. Arbitrary unvalidated full-body run extraction refusal (cannot guess clubhead state)
    with pytest.raises(ExtractionAdapterError) as exc_guess:
        coordinator.extract_swing_state_from_run(
            engine="mujoco",
            run_data={
                "unvalidated_raw_dump": True
            },  # Missing validated kinematics/adapter
        )
    assert (
        "validated extraction adapter" in str(exc_guess.value).lower()
        or "missing required" in str(exc_guess.value).lower()
    )


def test_green_supported_source_to_flight_to_comparison_with_provenance_and_cancellation_rollback(
    tmp_path: Path,
) -> None:
    """Acceptance GREEN: manual/MuJoCo supported source -> flight -> both comparison surfaces with method provenance;
    cancellation retains earlier trajectory.
    """
    coordinator = ShotTrajectoryHandoffCoordinator(repo_root=tmp_path)
    ws_coord = ResultsWorkspaceCoordinator(repo_root=tmp_path)

    # Step 1: Execute manual swing source -> impact -> flight pipeline
    manual_config = SwingStateConfig(
        clubhead_speed_ms=46.0,
        loft_deg=10.5,
        clubhead_mass_kg=0.200,
    )
    manual_swing = coordinator.get_swing_state("manual", manual_config)
    assert manual_swing.engine_name == "manual"
    assert math.isclose(float(manual_swing.clubhead_velocity[0]), 46.0)

    env = EnvironmentalConditions.from_altitude(
        altitude_m=0.0,
        temperature_c=25.0,
        pressure_pa=101325.0,
        wind_velocity=np.array(
            [
                3.0 * math.cos(math.radians(45.0)),
                3.0 * math.sin(math.radians(45.0)),
                0.0,
            ]
        ),
    )
    coordinator.set_environment_conditions(env)

    # Run pipeline through coordinator
    pipe_result1, traj_path1 = coordinator.simulate_and_export_trajectory(
        swing_state=manual_swing,
        trajectory_name="shot_01_manual",
    )
    assert isinstance(pipe_result1, PipelineResult)
    assert traj_path1.exists()
    assert pipe_result1.carry_m > 50.0

    # Verify action availability in ResultsWorkspaceCoordinator
    traj_item1 = ResultArtifactItem(
        run_id="shot_01",
        category=ResultCategory.FLIGHT_TRAJECTORY,
        path=str(traj_path1.relative_to(tmp_path)),
        engine="manual",
    )
    avail_compare = ws_coord.get_action_availability(
        WorkspaceActionType.COMPARE_FLIGHT_MODELS, traj_item1
    )
    assert avail_compare.enabled is True
    assert avail_compare.action == WorkspaceActionType.COMPARE_FLIGHT_MODELS

    avail_impact = ws_coord.get_action_availability(
        WorkspaceActionType.OPEN_IN_IMPACT_EXPLORER, traj_item1
    )
    assert avail_impact.enabled is True
    assert avail_impact.action == WorkspaceActionType.OPEN_IN_IMPACT_EXPLORER

    # Step 2: Set as active session trajectory
    coordinator.set_active_trajectory(traj_path1)
    assert coordinator.get_active_trajectory() == traj_path1

    # Step 3: Now attempt a second run that gets canceled or encounters an extraction failure
    # Verify rollback / retention of earlier trajectory
    with pytest.raises(UnsupportedEngineSourceError):
        coordinator.simulate_and_export_trajectory_from_run(
            engine="drake",
            run_data={"invalid": True},
            trajectory_name="shot_02_failed",
        )

    # Prior active trajectory MUST be preserved intact
    assert coordinator.get_active_trajectory() == traj_path1
    assert traj_path1.exists()

    # Step 4: MuJoCo source execution (if available or via validated provider mock)
    mujoco_provider = MuJoCoSwingStateProvider()
    if mujoco_provider.is_available():
        mujoco_swing = coordinator.get_swing_state("mujoco", manual_config)
        assert mujoco_swing.engine_name == "mujoco"
        pipe_result2, traj_path2 = coordinator.simulate_and_export_trajectory(
            swing_state=mujoco_swing,
            trajectory_name="shot_02_mujoco",
        )
        assert traj_path2.exists()
        assert pipe_result2.swing_state.engine_name == "mujoco"
        coordinator.set_active_trajectory(traj_path2)
        assert coordinator.get_active_trajectory() == traj_path2

    # Step 5: Test explicit user cancellation retains previous trajectory
    previous_traj = coordinator.get_active_trajectory()
    coordinator.begin_session_transaction()
    coordinator.stage_tentative_trajectory(tmp_path / "tentative_cancelled.json")
    coordinator.cancel_session_transaction()
    assert coordinator.get_active_trajectory() == previous_traj
