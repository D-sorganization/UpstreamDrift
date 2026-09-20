"""Integration tests for Shot & Course Lab workspace composition (ORG-15, #10524).

Authoritative Acceptance Cases:
- RED: incompatible ground/flight record rejected;
       terrain edit invalidates dependent run rather than silently changing history.
- RED: scene-only view cannot report computed shot;
       unsupported simulator destination is disabled.
- GREEN: putting fixture save/reopen,
         bunker fidelity export round-trip and
         simulator failure/cancel flows use existing services.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.golf_simulator.contracts import (
    AimContext,
    CapabilityDescriptor,
    CapabilityState,
    ContactStatus,
    NumericalStatus,
    ScientificStatus,
    ShotEnvelope,
    ShotQualification,
    SimulatorCapabilities,
    SourceKind,
    SubmissionState,
)
from src.shared.python.workspace.shot_course_workspace import (
    BunkerFidelityTier,
    BunkerRunRecord,
    IncompatibleGroundFlightRecordError,
    PuttingFixture,
    SceneNonPhysicsError,
    ShotCourseMode,
    ShotCourseRun,
    ShotCourseWorkspaceCoordinator,
    SimulatorDeliveryRequest,
    TerrainConfig,
    TerrainMutationInvalidationError,
    UnsupportedSimulatorDestinationError,
)


@pytest.fixture
def workspace_coordinator(tmp_path: Path) -> ShotCourseWorkspaceCoordinator:
    """Create a test coordinator backed by a temporary workspace directory."""
    return ShotCourseWorkspaceCoordinator(workspace_dir=tmp_path)


# ---------------------------------------------------------------------------
# RED Case 1: Incompatible ground/flight record rejected
# ---------------------------------------------------------------------------


def test_incompatible_ground_flight_record_rejected(
    workspace_coordinator: ShotCourseWorkspaceCoordinator,
) -> None:
    """Refuse unsupported flight-to-ground transition without valid kinetic/spatial state."""
    # Attempting to convert or inject a flight record missing impact angles/velocity
    # or lacking valid coordinates into the putting/ground rolling engine must fail closed.
    invalid_flight_record = {
        "flight_id": "fl_test_001",
        "landing_position_m": [150.0, 0.0],  # Missing z elevation coordinate
        "spin_decay_rate": None,
    }

    with pytest.raises(IncompatibleGroundFlightRecordError) as exc_info:
        workspace_coordinator.transition_flight_to_ground(invalid_flight_record)

    assert (
        "Missing required 3D landing coordinates" in str(exc_info.value)
        or "invalid" in str(exc_info.value).lower()
    )


# ---------------------------------------------------------------------------
# RED Case 2: Terrain edit invalidates dependent run rather than silently changing history
# ---------------------------------------------------------------------------


def test_terrain_edit_invalidates_dependent_run(
    workspace_coordinator: ShotCourseWorkspaceCoordinator,
) -> None:
    """Environment mutation creates an explicit new config revision and invalidates dependent runs."""
    # 1. Establish initial terrain config (revision 1)
    t1 = TerrainConfig(
        terrain_id="green_01",
        revision=1,
        elevation_grid=[[0.0, 0.05], [0.02, 0.08]],
        stimp_rating=10.5,
        firmness_norm=0.75,
    )
    workspace_coordinator.set_terrain(t1)

    # 2. Execute a putting simulation run dependent on terrain revision 1
    run1 = workspace_coordinator.execute_putting_run(
        initial_position=(0.0, 0.0, 0.0),
        initial_velocity=(2.5, 0.2, 0.0),
        run_label="putt_baseline",
    )
    assert run1.revision == 1
    assert run1.is_valid is True
    assert run1.terrain_revision == 1

    # 3. Mutate the terrain (new revision 2)
    t2 = TerrainConfig(
        terrain_id="green_01",
        revision=2,
        elevation_grid=[[0.0, 0.15], [0.05, 0.18]],  # Slope modified
        stimp_rating=11.2,
        firmness_norm=0.80,
    )
    workspace_coordinator.set_terrain(t2)

    # 4. Verify run1 is marked invalidated rather than silently recomputed or claimed current
    assert workspace_coordinator.get_run(run1.run_id).is_valid is False
    assert (
        workspace_coordinator.get_run(run1.run_id).invalidation_reason
        == "Terrain modified (revision advanced from 1 to 2)"
    )

    # Re-evaluating run1 against current terrain must raise an error requiring a new run revision
    with pytest.raises(TerrainMutationInvalidationError):
        workspace_coordinator.verify_run_validity(run1.run_id)


# ---------------------------------------------------------------------------
# RED Case 3: Scene-only view cannot report computed shot
# ---------------------------------------------------------------------------


def test_scene_only_view_cannot_report_computed_shot(
    workspace_coordinator: ShotCourseWorkspaceCoordinator,
) -> None:
    """Scene view is explicitly for visual inspection only and cannot report computed shots."""
    workspace_coordinator.set_mode(ShotCourseMode.SCENE)

    with pytest.raises(SceneNonPhysicsError) as exc_info:
        workspace_coordinator.report_computed_shot()

    assert (
        "Scene view is visual inspection only and cannot report computed shots"
        in str(exc_info.value)
    )


# ---------------------------------------------------------------------------
# RED Case 4: Unsupported simulator destination is disabled
# ---------------------------------------------------------------------------


def test_unsupported_simulator_destination_is_disabled(
    workspace_coordinator: ShotCourseWorkspaceCoordinator,
) -> None:
    """Simulator destination lacking supported shot_input capability is disabled."""
    # Destination declaring unsupported shot input
    unsupported_caps = SimulatorCapabilities(
        shot_input=CapabilityDescriptor(
            state=CapabilityState.UNSUPPORTED,
            evidence="Destination hardware does not support remote shot injection v1",
        ),
        club_data=CapabilityDescriptor(
            state=CapabilityState.UNSUPPORTED,
            evidence="No club data support",
        ),
        native_avatar_animation=CapabilityDescriptor(
            state=CapabilityState.UNSUPPORTED,
            evidence="No avatar support",
        ),
        course_state_feedback=CapabilityDescriptor(
            state=CapabilityState.UNSUPPORTED,
            evidence="No feedback support",
        ),
    )

    workspace_coordinator.register_simulator_destination(
        destination_id="legacy_box_dest",
        capabilities=unsupported_caps,
    )

    availability = workspace_coordinator.check_delivery_availability("legacy_box_dest")
    assert availability.enabled is False
    assert "unsupported" in availability.reason.lower()

    # Attempting to deliver must raise UnsupportedSimulatorDestinationError
    qual = ShotQualification(
        contact=ContactStatus.QUALIFIED,
        numerical=NumericalStatus.CONVERGED,
        scientific=ScientificStatus.BENCHMARKED,
    )
    aim = AimContext(
        source_to_target_rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        provenance="test",
    )
    envelope = ShotEnvelope(
        schema_version=1,
        shot_id="shot_test_01",
        session_id="session_test_01",
        source_kind=SourceKind.MODEL_CONTACT,
        qualification=qual,
        ball_velocity_m_s=(45.0, 2.0, 15.0),
        ball_angular_velocity_rad_s=(0.0, 300.0, 0.0),
        aim_context=aim,
        created_at_utc="2026-09-20T08:00:00Z",
        model_run_id="run_m01",
        trace_digest="digest_abc",
        impact_id="imp_01",
        impact_time_s=0.25,
    )

    req = SimulatorDeliveryRequest(
        destination_id="legacy_box_dest",
        payload=envelope,
    )
    with pytest.raises(UnsupportedSimulatorDestinationError):
        workspace_coordinator.deliver_to_simulator(req)


# ---------------------------------------------------------------------------
# GREEN Case 1: Putting fixture save / reopen round-trip
# ---------------------------------------------------------------------------


def test_putting_fixture_save_and_reopen(
    workspace_coordinator: ShotCourseWorkspaceCoordinator,
    tmp_path: Path,
) -> None:
    """Putting scenario preserves terrain, ball parameters, and roll model across save/reopen."""
    workspace_coordinator.set_mode(ShotCourseMode.PUTTING)

    fixture = PuttingFixture(
        fixture_id="putt_fic_4ft_straight",
        stimp=10.5,
        green_slope_deg=1.2,
        initial_speed_m_s=1.85,
        launch_direction_deg=0.5,
        roll_model="usga_stimp",
        metadata={"author": "test_agent", "elevation_profile": "gentle_rise"},
    )

    fixture_path = tmp_path / "fixtures" / "putt_fic_4ft_straight.json"
    workspace_coordinator.save_putting_fixture(fixture, fixture_path)

    assert fixture_path.exists()

    reopened = workspace_coordinator.load_putting_fixture(fixture_path)
    assert reopened.fixture_id == fixture.fixture_id
    assert reopened.stimp == pytest.approx(10.5)
    assert reopened.green_slope_deg == pytest.approx(1.2)
    assert reopened.initial_speed_m_s == pytest.approx(1.85)
    assert reopened.roll_model == "usga_stimp"
    assert reopened.metadata["author"] == "test_agent"


# ---------------------------------------------------------------------------
# GREEN Case 2: Bunker fidelity export round-trip
# ---------------------------------------------------------------------------


def test_bunker_fidelity_export_round_trip(
    workspace_coordinator: ShotCourseWorkspaceCoordinator,
    tmp_path: Path,
) -> None:
    """Bunker shot record preserves fidelity tier and physical domain properties on export/import."""
    workspace_coordinator.set_mode(ShotCourseMode.BUNKER)

    bunker_run = BunkerRunRecord(
        run_id="bunker_f1_sand_001",
        fidelity_tier=BunkerFidelityTier.F1_RESISTANCE_FORCE,
        wedge_loft_deg=56.0,
        wedge_bounce_deg=12.0,
        sand_density_kg_m3=1600.0,
        grain_friction_angle_deg=34.0,
        exit_ball_speed_m_s=22.4,
        exit_launch_angle_deg=28.5,
        exit_spin_rpm=4500.0,
        provenance={"solver": "bunkershot3d.f1", "calibrated": True},
    )

    export_path = tmp_path / "exports" / "bunker_run.json"
    workspace_coordinator.export_bunker_run(bunker_run, export_path)

    assert export_path.exists()

    imported = workspace_coordinator.import_bunker_run(export_path)
    assert imported.run_id == bunker_run.run_id
    assert imported.fidelity_tier == BunkerFidelityTier.F1_RESISTANCE_FORCE
    assert imported.wedge_loft_deg == pytest.approx(56.0)
    assert imported.exit_spin_rpm == pytest.approx(4500.0)
    assert imported.provenance["solver"] == "bunkershot3d.f1"


# ---------------------------------------------------------------------------
# GREEN Case 3: Simulator failure / cancel flows
# ---------------------------------------------------------------------------


def test_simulator_failure_and_cancel_flows(
    workspace_coordinator: ShotCourseWorkspaceCoordinator,
) -> None:
    """Simulator delivery gracefully handles network/hardware failure and user cancellation."""
    # 1. Register a destination with supported shot input
    supported_caps = SimulatorCapabilities(
        shot_input=CapabilityDescriptor(
            state=CapabilityState.SUPPORTED,
            evidence="Open connect v1 supported",
        ),
        club_data=CapabilityDescriptor(
            state=CapabilityState.UNSUPPORTED,
            evidence="No club data",
        ),
        native_avatar_animation=CapabilityDescriptor(
            state=CapabilityState.UNSUPPORTED,
            evidence="No avatar",
        ),
        course_state_feedback=CapabilityDescriptor(
            state=CapabilityState.UNSUPPORTED,
            evidence="No feedback",
        ),
    )
    workspace_coordinator.register_simulator_destination(
        destination_id="gsp_valid_dest",
        capabilities=supported_caps,
    )

    qual = ShotQualification(
        contact=ContactStatus.QUALIFIED,
        numerical=NumericalStatus.CONVERGED,
        scientific=ScientificStatus.BENCHMARKED,
    )
    aim = AimContext(
        source_to_target_rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        provenance="test",
    )
    envelope = ShotEnvelope(
        schema_version=1,
        shot_id="shot_test_fail",
        session_id="session_test_fail",
        source_kind=SourceKind.MODEL_CONTACT,
        qualification=qual,
        ball_velocity_m_s=(42.0, 0.0, 18.0),
        ball_angular_velocity_rad_s=(0.0, 280.0, 0.0),
        aim_context=aim,
        created_at_utc="2026-09-20T08:15:00Z",
        model_run_id="run_m02",
        trace_digest="digest_fail",
        impact_id="imp_02",
        impact_time_s=0.30,
    )

    req = SimulatorDeliveryRequest(
        destination_id="gsp_valid_dest",
        payload=envelope,
        simulate_network_failure=True,
    )

    # Delivery under network failure produces an explicit failure receipt with evidence
    receipt = workspace_coordinator.deliver_to_simulator(req)
    assert receipt.state == SubmissionState.FAILED_BEFORE_SEND
    assert "Connection refused or timed out" in receipt.detail
    assert receipt.destination_id == "gsp_valid_dest"

    # Delivery cancellation flow
    req_cancel = SimulatorDeliveryRequest(
        destination_id="gsp_valid_dest",
        payload=envelope,
        canceled_by_user=True,
    )
    receipt_cancel = workspace_coordinator.deliver_to_simulator(req_cancel)
    assert receipt_cancel.state == SubmissionState.FAILED_BEFORE_SEND
    assert "Delivery canceled by user" in receipt_cancel.detail
