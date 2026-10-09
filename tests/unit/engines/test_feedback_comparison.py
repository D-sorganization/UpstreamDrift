"""F01 executable admission tests for controlled-swing comparisons."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.engines.feedback_comparison import (
    BaselineRun,
    ComparisonEvidence,
    ComparisonLevel,
    ComparisonRequest,
    DriveMode,
    EvidenceMode,
    FeedbackComparisonRegistry,
    InputKind,
    ReplayPolicy,
)
from src.engines.model_inventory import EngineModelInventory, PackageStatus

pytestmark = pytest.mark.unit


@pytest.fixture()
def registry() -> FeedbackComparisonRegistry:
    root = Path(__file__).resolve().parents[3]
    return FeedbackComparisonRegistry(EngineModelInventory.load(repo_root=root))


def _valid_evidence(registry: FeedbackComparisonRegistry) -> ComparisonEvidence:
    row = registry.get("mujoco/driver", "default", DriveMode.TORQUE)
    return ComparisonEvidence(
        package_id=row.package_id,
        variant_id=row.variant_id,
        drive_mode=row.drive_mode,
        source_model_sha256=row.source_model_sha256,
        provider_id=row.provider_id,
        provider_sha256=row.provider_sha256,
        state_schema_sha256="b" * 64,
        policy_sha256="c" * 64,
        applied_input_sha256="d" * 64,
        input_kind=InputKind.GENERALIZED_EFFORT,
        input_interpolation="zero_order_hold",
        timebase_id="simulation_relative",
        replay_policy=ReplayPolicy.INDEPENDENT_TIME_ONLY,
        evidence_mode=EvidenceMode.SHARED_RIGID_BODY_EMULATION,
        horizon_s=1.0,
        observation_sha256="e" * 64,
        observation_time_grid_sha256="f" * 64,
        channel_ids=("hip", "knee"),
        full_state=True,
        full_horizon=True,
        state_resets=0,
        nq=44,
        nv=44,
        physical_model_sha256="1" * 64,
        loaded_native_model_sha256="6" * 64,
        time_grid_sha256="7" * 64,
        physics_sha256="2" * 64,
        contact_sha256="3" * 64,
        integrator_sha256="4" * 64,
        input_channel_schema_sha256="5" * 64,
    )


def test_inventory_keeps_every_package_in_required_denominator(
    registry: FeedbackComparisonRegistry,
) -> None:
    rows = registry.rows
    assert {row.package_id for row in rows} == {
        package.id for package in registry.inventory.packages
    }
    assert all(row.required for row in rows)
    assert all("flexible_club" in row.required_capabilities for row in rows)
    assert all(row.capability_support["flexible_club"] == "unknown" for row in rows)
    assert (
        registry.get(
            "opensim/native-golf-humanoid",
            "golf_humanoid_muscle_variant",
            DriveMode.MUSCLE_EXCITATION,
        ).qualification
        == "unqualified"
    )
    assert (
        registry.get(
            "myosuite/driver", "default", DriveMode.MUSCLE_EXCITATION
        ).qualification
        == "unqualified"
    )


def test_unavailable_required_package_stays_in_denominator(
    registry: FeedbackComparisonRegistry,
) -> None:
    packages = tuple(
        replace(package, status=PackageStatus.RETIRED)
        if package.id == "simscape/driver"
        else package
        for package in registry.inventory.packages
    )
    altered = FeedbackComparisonRegistry(replace(registry.inventory, packages=packages))
    row = altered.get("simscape/driver", "default", DriveMode.TORQUE)
    assert row.required
    assert row.availability == "unavailable"
    assert row.qualification == "unqualified"


def test_unknown_variant_and_stale_model_identity_fail(
    registry: FeedbackComparisonRegistry,
) -> None:
    evidence = _valid_evidence(registry)
    with pytest.raises(ValueError, match="unknown comparison row"):
        registry.get("missing/driver", "default", DriveMode.TORQUE)
    with pytest.raises(ValueError, match="model identity"):
        registry.admit(
            replace(evidence, source_model_sha256="f" * 64),
            ComparisonLevel.WITHIN_ENGINE_REPLAY,
        )
    with pytest.raises(ValueError, match="provider source hash"):
        registry.admit(
            replace(evidence, provider_sha256="f" * 64),
            ComparisonLevel.WITHIN_ENGINE_REPLAY,
        )


def test_unqualified_muscle_cannot_be_reported_as_qualified(
    registry: FeedbackComparisonRegistry,
) -> None:
    evidence = _valid_evidence(registry)
    muscle = registry.get(
        "opensim/native-golf-humanoid",
        "golf_humanoid_muscle_variant",
        DriveMode.MUSCLE_EXCITATION,
    )
    with pytest.raises(ValueError, match="unqualified"):
        registry.admit(
            replace(
                evidence,
                package_id=muscle.package_id,
                variant_id=muscle.variant_id,
                drive_mode=muscle.drive_mode,
                source_model_sha256=muscle.source_model_sha256,
                provider_id=muscle.provider_id,
                provider_sha256=muscle.provider_sha256,
                input_kind=InputKind.MUSCLE_EXCITATION,
                evidence_mode=EvidenceMode.NATIVE_OWN_CONTACT,
            ),
            ComparisonLevel.WITHIN_ENGINE_REPLAY,
            claim_qualified=True,
        )


def test_open_sim_native_contact_claim_remains_unavailable_without_capability(
    registry: FeedbackComparisonRegistry,
) -> None:
    row = registry.get(
        "opensim/native-golf-humanoid",
        "golf_humanoid_muscle_variant",
        DriveMode.MUSCLE_EXCITATION,
    )
    evidence = replace(
        _valid_evidence(registry),
        package_id=row.package_id,
        variant_id=row.variant_id,
        drive_mode=row.drive_mode,
        source_model_sha256=row.source_model_sha256,
        provider_id=row.provider_id,
        provider_sha256=row.provider_sha256,
        input_kind=InputKind.MUSCLE_EXCITATION,
        evidence_mode=EvidenceMode.NATIVE_OWN_CONTACT,
        contact_evidence_sha256="a" * 64,
        force_evidence_sha256="b" * 64,
        muscle_state_evidence_sha256="c" * 64,
    )
    assert row.capability_support["native_own_contact"] == "unknown"
    with pytest.raises(ValueError, match="native own-contact capability"):
        registry.admit(
            evidence,
            ComparisonLevel.BIOMECHANICAL_EQUIVALENCE,
            claim_native_muscle=True,
        )


def test_independent_replay_requires_bound_inputs_and_no_feedback(
    registry: FeedbackComparisonRegistry,
) -> None:
    evidence = _valid_evidence(registry)
    for bad in (
        replace(evidence, applied_input_sha256=""),
        replace(evidence, policy_sha256=""),
        replace(evidence, replay_policy=ReplayPolicy.STATE_FEEDBACK),
        replace(evidence, state_resets=1),
        replace(evidence, full_state=False),
    ):
        with pytest.raises(ValueError):
            registry.admit(bad, ComparisonLevel.WITHIN_ENGINE_REPLAY)


def test_transcription_and_observation_receipts_are_distinct(
    registry: FeedbackComparisonRegistry,
) -> None:
    evidence = _valid_evidence(registry)
    with pytest.raises(ValueError, match="transcription_receipt"):
        registry.admit(evidence, ComparisonLevel.TRANSCRIPTION_FEASIBILITY)
    with pytest.raises(ValueError, match="observation_score_receipt"):
        registry.admit(evidence, ComparisonLevel.OBSERVATION_ACCURACY)
    registry.admit(
        replace(evidence, transcription_receipt_sha256="a" * 64),
        ComparisonLevel.TRANSCRIPTION_FEASIBILITY,
    )
    registry.admit(
        replace(evidence, observation_score_receipt_sha256="a" * 64),
        ComparisonLevel.OBSERVATION_ACCURACY,
    )


def test_euclidean_v1_cannot_qualify_manifold_or_muscle_replay(
    registry: FeedbackComparisonRegistry,
) -> None:
    evidence = _valid_evidence(registry)
    with pytest.raises(ValueError, match="Euclidean"):
        registry.admit(
            replace(evidence, nq=45, bundle_schema="same-input-bundle/v1"),
            ComparisonLevel.WITHIN_ENGINE_REPLAY,
        )
    with pytest.raises(ValueError, match="v1"):
        registry.admit(
            replace(
                evidence,
                input_kind=InputKind.MUSCLE_EXCITATION,
                bundle_schema="same-input-bundle/v1",
            ),
            ComparisonLevel.WITHIN_ENGINE_REPLAY,
        )


def test_same_input_rejects_mixed_torque_and_excitation(
    registry: FeedbackComparisonRegistry,
) -> None:
    left = _valid_evidence(registry)
    right = replace(
        left,
        package_id="myosuite/driver",
        source_model_sha256=registry.get(
            "myosuite/driver", "default", DriveMode.MUSCLE_EXCITATION
        ).source_model_sha256,
        drive_mode=DriveMode.MUSCLE_EXCITATION,
        input_kind=InputKind.MUSCLE_EXCITATION,
    )
    with pytest.raises(ValueError, match="input kind"):
        registry.compare(ComparisonRequest(ComparisonLevel.SAME_INPUT, left, right))


def test_same_input_requires_common_physics_and_time_grid(
    registry: FeedbackComparisonRegistry,
) -> None:
    left = _valid_evidence(registry)
    for bad in (
        replace(left, time_grid_sha256="8" * 64),
        replace(left, physics_sha256="9" * 64),
        replace(left, channel_ids=("hip",)),
    ):
        with pytest.raises(ValueError, match="differs"):
            registry.compare(ComparisonRequest(ComparisonLevel.SAME_INPUT, left, bad))


def test_direct_effort_must_be_held_for_each_step(
    registry: FeedbackComparisonRegistry,
) -> None:
    with pytest.raises(ValueError, match="zero-order hold"):
        registry.admit(
            replace(_valid_evidence(registry), input_interpolation="linear"),
            ComparisonLevel.WITHIN_ENGINE_REPLAY,
        )
    with pytest.raises(ValueError, match="timebase"):
        registry.admit(
            replace(_valid_evidence(registry), timebase_id="capture_source"),
            ComparisonLevel.WITHIN_ENGINE_REPLAY,
        )


@pytest.mark.parametrize(
    "level",
    [ComparisonLevel.WITHIN_ENGINE_REPLAY, ComparisonLevel.SAME_INPUT],
)
def test_truncated_replay_cannot_pass_full_horizon_levels(
    registry: FeedbackComparisonRegistry, level: ComparisonLevel
) -> None:
    with pytest.raises(ValueError, match="full horizon"):
        registry.admit(replace(_valid_evidence(registry), full_horizon=False), level)


def test_observation_score_requires_separate_output_clock_digest(
    registry: FeedbackComparisonRegistry,
) -> None:
    evidence = _valid_evidence(registry)
    with pytest.raises(ValueError, match="observation_time_grid_sha256"):
        registry.admit(
            replace(evidence, observation_time_grid_sha256=""),
            ComparisonLevel.OBSERVATION_ACCURACY,
        )


def test_native_muscle_claim_rejects_short_shared_or_restarted_evidence(
    registry: FeedbackComparisonRegistry,
) -> None:
    evidence = _valid_evidence(registry)
    for bad in (
        replace(evidence, evidence_mode=EvidenceMode.SHARED_RIGID_BODY_EMULATION),
        replace(evidence, full_horizon=False),
        replace(evidence, state_resets=1),
    ):
        with pytest.raises(ValueError):
            registry.admit(
                bad, ComparisonLevel.BIOMECHANICAL_EQUIVALENCE, claim_native_muscle=True
            )


def test_externally_forced_path_cannot_qualify_biomechanics(
    registry: FeedbackComparisonRegistry,
) -> None:
    evidence = replace(
        _valid_evidence(registry),
        evidence_mode=EvidenceMode.EXTERNALLY_FORCED,
        contact_evidence_sha256="a" * 64,
        force_evidence_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="externally forced"):
        registry.admit(evidence, ComparisonLevel.BIOMECHANICAL_EQUIVALENCE)


def test_baselines_must_share_observations_horizon_and_budget(
    registry: FeedbackComparisonRegistry,
) -> None:
    runs = tuple(
        BaselineRun(name, "a" * 64, 1.0, 100.0) for name in registry.required_baselines
    )
    registry.validate_baselines(runs)
    with pytest.raises(ValueError, match="budget"):
        registry.validate_baselines(
            runs[:-1] + (replace(runs[-1], compute_budget_s=101.0),)
        )
    with pytest.raises(ValueError, match="missing baseline"):
        registry.validate_baselines(runs[:-1])
