"""F09 consumes real native positions through the frozen observation contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.engines.feedback_comparison import (
    ComparisonEvidence,
    ComparisonLevel,
    DriveMode,
    EvidenceMode,
    FeedbackComparisonRegistry,
    InputKind,
    ReplayPolicy,
)
from src.engines.model_inventory import EngineModelInventory, TARGET_ENGINES
from src.engines.feedback_observation_qualification import (
    NativeObservationCase,
    build_feedback_observation_report,
    feedback_replay_identity_sha256,
)
from src.shared.python.motion_matching.acceptance import Horizon
from src.shared.python.motion_matching.replay_metrics import (
    NativeMarkerPositionOutput,
    ObservedMarkerPositions,
    PositionInterpolation,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sidekick.lab.mocap import ExperimentReplayBundle

pytestmark = pytest.mark.unit


@pytest.fixture()
def registry() -> FeedbackComparisonRegistry:
    root = Path(__file__).resolve().parents[3]
    return FeedbackComparisonRegistry(EngineModelInventory.load(repo_root=root))


def _evidence_and_bundle(
    registry: FeedbackComparisonRegistry,
    *,
    q_initial: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    input_values: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.1, 0.1),
    ),
) -> tuple[ComparisonEvidence, ExperimentReplayBundle]:
    from src.engines.native_replay_contracts import native_replay_contract_types

    contracts = native_replay_contract_types()
    ActuationInputKind = contracts.ActuationInputKind
    CapabilityAvailability = contracts.CapabilityAvailability
    CapabilityDeclaration = contracts.CapabilityDeclaration
    CapabilitySupport = contracts.CapabilitySupport
    InitialStateSchema = contracts.InitialStateSchema
    InputChannel = contracts.InputChannel
    InputInterpolation = contracts.InputInterpolation
    ModelIdentity = contracts.ModelIdentity
    ReplayExecutionPolicy = contracts.ReplayExecutionPolicy
    ReplayMode = contracts.ReplayMode
    StateComponentRole = contracts.StateComponentRole
    StateComponentSpec = contracts.StateComponentSpec
    build_experiment_replay_bundle = contracts.build_experiment_replay_bundle

    row = registry.get("mujoco/driver", "default", DriveMode.TORQUE)
    channels = (
        InputChannel("hip", "actuator:hip", "N*m"),
        InputChannel("knee", "actuator:knee", "N*m"),
    )
    model = ModelIdentity(
        engine_id=row.engine,
        model_id="synthetic-native-model",
        variant_id=row.variant_id,
        model_version="1.0.0",
        source_model_sha256=row.source_model_sha256,
        provider_id=row.provider_id,
        provider_version="1.0.0",
        provider_sha256=row.provider_sha256,
        state_schema=InitialStateSchema(
            schema_id="synthetic-state",
            version="1.0.0",
            components=(
                StateComponentSpec(
                    "q",
                    StateComponentRole.POSITION,
                    4,
                    "1",
                    "unit_quaternion_xyzw",
                ),
                StateComponentSpec(
                    "v", StateComponentRole.VELOCITY, 3, "rad/s", "angular_velocity"
                ),
            ),
        ),
        ordered_input_channel_ids=("hip", "knee"),
        loaded_native_model_sha256="6" * 64,
    )
    policy = ReplayExecutionPolicy(
        replay_mode=ReplayMode.SHARED_RIGID_BODY_EMULATION,
        solver_id="synthetic-solver",
        solver_version="1.0.0",
        integration_method="fixed-step-test",
        step_policy="fixed",
        step_size_seconds=0.01,
        initialization_policy_id="zero-state",
        initialization_policy_version="1.0.0",
        input_player_id="time-only-player",
        input_player_version="1.0.0",
        observation_access=False,
        state_feedback_access=False,
        state_reset_allowed=False,
        contact_policy_id="synthetic-contact",
        contact_policy_version="1.0.0",
        contact_policy_sha256="3" * 64,
    )
    bundle = build_experiment_replay_bundle(
        "synthetic-experiment",
        model,
        (
            CapabilityDeclaration(
                "native_replay",
                required=True,
                support=CapabilitySupport.SUPPORTED,
                availability=CapabilityAvailability.AVAILABLE,
            ),
        ),
        (("q", q_initial), ("v", (0.0, 0.0, 0.0))),
        channels,
        ActuationInputKind.GENERALIZED_EFFORT,
        InputInterpolation.ZERO_ORDER_HOLD,
        (0.0, 0.6, 1.2, 1.8),
        input_values,
        policy,
    )
    evidence = ComparisonEvidence(
        package_id=row.package_id,
        variant_id=row.variant_id,
        drive_mode=row.drive_mode,
        source_model_sha256=row.source_model_sha256,
        provider_id=row.provider_id,
        provider_sha256=row.provider_sha256,
        state_schema_sha256=bundle.state_schema_sha256,
        policy_sha256=bundle.policy_sha256,
        applied_input_sha256=bundle.applied_input_sha256,
        input_kind=InputKind.GENERALIZED_EFFORT,
        input_interpolation="zero_order_hold",
        timebase_id="simulation_relative",
        replay_policy=ReplayPolicy.INDEPENDENT_TIME_ONLY,
        evidence_mode=EvidenceMode.SHARED_RIGID_BODY_EMULATION,
        horizon_s=bundle.input_history.time_seconds[-1],
        observation_sha256="e" * 64,
        channel_ids=("hip", "knee"),
        full_state=True,
        full_horizon=True,
        state_resets=0,
        nq=4,
        nv=3,
        physical_model_sha256="1" * 64,
        loaded_native_model_sha256="6" * 64,
        time_grid_sha256=bundle.time_grid_sha256,
        physics_sha256="2" * 64,
        contact_sha256="3" * 64,
        integrator_sha256="4" * 64,
        input_channel_schema_sha256=bundle.input_channel_schema_sha256,
        initial_state_sha256=bundle.integrity.initial_state_sha256,
        comparison_contract_version="feedback-comparison/1.1.0",
    )
    return evidence, bundle


def _case(registry: FeedbackComparisonRegistry) -> NativeObservationCase:
    evidence, bundle = _evidence_and_bundle(registry)
    native_times = np.array([0.0, 0.4, 0.8, 1.2, 1.8])
    observation_times = np.array([0.1, 0.5, 0.9, 1.5, 1.8])
    labels = ("WaistLeft", "WaistRight", "Marker_2", "Marker_3")
    native = np.zeros((len(native_times), len(labels), 3), dtype=np.float64)
    native[:, 0, 0] = 0.2
    native[:, 1, 0] = 0.5
    native[:, 1, 1] = 0.1
    native[:, 2, 0] = native_times
    native[:, 3, 1] = native_times * 0.5
    expected_at_observations = np.zeros(
        (len(observation_times), len(labels), 3), dtype=np.float64
    )
    expected_at_observations[:, 0, 0] = 0.2
    expected_at_observations[:, 1, 0] = 0.5
    expected_at_observations[:, 1, 1] = 0.1
    expected_at_observations[:, 2, 0] = observation_times
    expected_at_observations[:, 3, 1] = observation_times * 0.5
    observed = expected_at_observations.copy()
    observed[:, :, 2] += 0.01
    return NativeObservationCase(
        evidence=evidence,
        replay_bundle=bundle,
        source_replay_identity_sha256=feedback_replay_identity_sha256(evidence, bundle),
        native_output=NativeMarkerPositionOutput(
            time_s=native_times,
            positions_m=native,
            marker_labels=labels,
            frame_id="world-z-up-v1",
            timebase_id="simulation_relative",
        ),
        observations=ObservedMarkerPositions(
            time_s=observation_times,
            positions_m=observed,
            valid=np.ones((len(observation_times), len(labels)), dtype=bool),
            marker_labels=labels,
            frame_id="world-z-up-v1",
            timebase_id="simulation_relative",
        ),
        native_acceptance_receipt={
            "duration_s": 1.8,
            "engine": "mujoco",
            "capture": "driver",
        },
        horizon=Horizon.G3,
        interpolation=PositionInterpolation.LINEAR_POSITION,
    )


def test_campaign_keeps_all_six_engine_rows_when_no_native_output_exists(
    registry: FeedbackComparisonRegistry,
) -> None:
    report = build_feedback_observation_report(registry, ())

    assert set(report.required_engine_ids) == set(TARGET_ENGINES)
    assert set(report.missing_engine_ids) == set(TARGET_ENGINES)
    assert report.required_row_count == len(registry.rows)
    assert report.scored_row_count == 0
    assert report.is_qualified is False
    assert all(row.status == "missing_evidence" for row in report.rows)
    assert all(row.qualification == "unqualified" for row in report.rows)


def test_campaign_scores_actual_positions_but_does_not_promote_registry_row(
    registry: FeedbackComparisonRegistry,
) -> None:
    report = build_feedback_observation_report(registry, (_case(registry),))
    row = next(row for row in report.rows if row.package_id == "mujoco/driver")

    assert report.scored_row_count == 1
    assert "mujoco" in report.observed_engine_ids
    assert row.score is not None
    assert row.score.observation_times_s == (0.1, 0.5, 0.9, 1.5, 1.8)
    assert row.score.metrics.whole_rms_m == pytest.approx(0.01)
    assert row.score.alignment_identity_sha256
    assert row.score.criteria_sha256
    assert (
        row.score.initial_state_sha256
        == _case(registry).replay_bundle.integrity.initial_state_sha256
    )
    assert row.score.capture == "driver"
    assert row.score.native_acceptance_receipt_sha256
    assert row.score.acceptance_verdict.is_physically_accepted is False
    assert row.qualification == "unqualified"
    assert report.is_qualified is False


def test_stale_source_replay_identity_is_rejected(
    registry: FeedbackComparisonRegistry,
) -> None:
    case = _case(registry)
    with pytest.raises(ValueError, match="source replay identity"):
        build_feedback_observation_report(
            registry,
            (replace(case, source_replay_identity_sha256="0" * 64),),
        )


def test_replay_identity_binds_full_initial_state_payload(
    registry: FeedbackComparisonRegistry,
) -> None:
    evidence, original = _evidence_and_bundle(registry)
    _, changed_state = _evidence_and_bundle(registry, q_initial=(0.0, 0.0, 0.1, 0.995))

    assert original.state_schema_sha256 == changed_state.state_schema_sha256
    assert original.integrity.initial_state_sha256 != (
        changed_state.integrity.initial_state_sha256
    )
    assert feedback_replay_identity_sha256(evidence, original) != (
        feedback_replay_identity_sha256(evidence, changed_state)
    )


def test_stale_initial_state_evidence_is_rejected(
    registry: FeedbackComparisonRegistry,
) -> None:
    case = _case(registry)
    evidence = replace(case.evidence, initial_state_sha256="0" * 64)
    case = replace(
        case,
        evidence=evidence,
        source_replay_identity_sha256=feedback_replay_identity_sha256(
            evidence, case.replay_bundle
        ),
    )
    with pytest.raises(ValueError, match="initial_state_payload"):
        build_feedback_observation_report(registry, (case,))


def test_unsupported_replay_comparison_contract_is_rejected(
    registry: FeedbackComparisonRegistry,
) -> None:
    case = _case(registry)
    evidence = replace(
        case.evidence, comparison_contract_version="feedback-comparison/1.0.0"
    )
    case = replace(
        case,
        evidence=evidence,
        source_replay_identity_sha256=feedback_replay_identity_sha256(
            evidence, case.replay_bundle
        ),
    )
    with pytest.raises(ValueError, match="comparison_contract_version"):
        build_feedback_observation_report(registry, (case,))


def test_changed_applied_input_cannot_be_rehashed_around_f01_evidence(
    registry: FeedbackComparisonRegistry,
) -> None:
    case = _case(registry)
    _, changed_bundle = _evidence_and_bundle(
        registry,
        input_values=((0.0, 0.0), (0.8, 0.2), (0.2, 0.3), (0.1, 0.1)),
    )
    changed_case = replace(
        case,
        replay_bundle=changed_bundle,
        source_replay_identity_sha256=feedback_replay_identity_sha256(
            case.evidence, changed_bundle
        ),
    )

    with pytest.raises(ValueError, match="applied_input"):
        build_feedback_observation_report(registry, (changed_case,))


def test_manifold_position_and_velocity_dimensions_remain_distinct(
    registry: FeedbackComparisonRegistry,
) -> None:
    case = _case(registry)
    evidence = replace(case.evidence, nq=3)
    mismatched_case = replace(
        case,
        evidence=evidence,
        source_replay_identity_sha256=feedback_replay_identity_sha256(
            evidence, case.replay_bundle
        ),
    )

    with pytest.raises(ValueError, match="nq"):
        build_feedback_observation_report(registry, (mismatched_case,))


def test_g3_requires_an_explicit_capture_for_frozen_acceptance_thresholds(
    registry: FeedbackComparisonRegistry,
) -> None:
    case = _case(registry)
    case = replace(
        case,
        native_acceptance_receipt={"duration_s": 1.8, "engine": "mujoco"},
    )
    with pytest.raises(ValueError, match="explicit driver or iron capture"):
        build_feedback_observation_report(registry, (case,))


def test_native_acceptance_engine_must_match_registered_model(
    registry: FeedbackComparisonRegistry,
) -> None:
    case = _case(registry)
    case = replace(
        case,
        native_acceptance_receipt={
            "duration_s": 1.8,
            "engine": "opensim",
            "capture": "driver",
        },
    )

    with pytest.raises(ValueError, match="engine differs from registry row"):
        build_feedback_observation_report(registry, (case,))


def test_duplicate_or_unregistered_native_rows_are_rejected(
    registry: FeedbackComparisonRegistry,
) -> None:
    case = _case(registry)
    with pytest.raises(ValueError, match="duplicate"):
        build_feedback_observation_report(registry, (case, case))
    bad_evidence = replace(case.evidence, package_id="unregistered/model")
    bad_case = replace(
        case,
        evidence=bad_evidence,
        source_replay_identity_sha256=feedback_replay_identity_sha256(
            bad_evidence, case.replay_bundle
        ),
    )
    with pytest.raises(ValueError, match="unknown comparison row"):
        build_feedback_observation_report(registry, (bad_case,))


def test_unavailable_required_row_remains_in_denominator(
    registry: FeedbackComparisonRegistry,
) -> None:
    from dataclasses import replace as dataclass_replace

    from src.engines.model_inventory import PackageStatus

    inventory = dataclass_replace(
        registry.inventory,
        packages=tuple(
            dataclass_replace(package, status=PackageStatus.RETIRED)
            if package.id == "simscape/driver"
            else package
            for package in registry.inventory.packages
        ),
    )
    unavailable = FeedbackComparisonRegistry(inventory)
    report = build_feedback_observation_report(unavailable, ())
    row = next(row for row in report.rows if row.package_id == "simscape/driver")

    assert row.required is True
    assert row.availability == "unavailable"
    assert row.status == "missing_evidence"
    assert row.qualification == "unqualified"
    assert report.required_row_count == len(unavailable.rows)
