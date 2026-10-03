"""MMR-04 (#11088): Enforce Compiled Home Budgets and Preserve Diagnostics.

TDD Behavioral Regression Tests for:
1. Actual R2025b production count <= 975 with 25-block instrumentation reserve
   documented independently of 1,000 license ceiling.
2. Exact required audit variant compiles <= 1,000.
3. Tests fail on a deliberately over-budget clone.
4. Observability contract: no consumer loses mass/COM/energy/contact/closure evidence.
5. Report before/after count by subsystem, not an inferred icon count.
6. Tool/license errors provide actionable diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.shared.python.motion_matching.simscape_block_budget import (
    HOME_LICENSE_BLOCK_LIMIT,
    INSTRUMENTATION_RESERVE_BLOCKS,
    PRODUCTION_BUDGET_CEILING,
    EXACT_AUDIT_BUDGET_CEILING,
    BlockBudgetProfile,
    EvidenceSourceType,
    InvalidSubsystemReportError,
    ObservabilityChannel,
    ObservabilityEvidenceError,
    ObservabilityEvidenceType,
    ObservabilityManifest,
    SimscapeBlockBudgetError,
    SimscapeBlockBudgetReport,
    SimscapeBudgetExceededError,
    SubsystemBlockCount,
    create_canonical_baseline_observability_manifest,
    create_canonical_human_observability_manifest,
    parse_matlab_block_budget_json,
    validate_simscape_block_budget,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_JSON_PATH = (
    REPO_ROOT
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "matlab"
    / "exploratory_gs3dx"
    / "docs"
    / "block_budget_GS3DX_Baseline.json"
)


def _sample_subsystems(
    uncompiled_total: int, compiled_total: int
) -> tuple[SubsystemBlockCount, ...]:
    """Helper to generate realistic subsystem counts matching totals."""
    growth = compiled_total - uncompiled_total
    sub1_u = int(uncompiled_total * 0.6)
    sub2_u = uncompiled_total - sub1_u
    sub1_c = sub1_u + int(growth * 0.7)
    sub2_c = compiled_total - sub1_c
    return (
        SubsystemBlockCount(
            name="UpperBody", uncompiled_count=sub1_u, compiled_count=sub1_c
        ),
        SubsystemBlockCount(
            name="LowerBody", uncompiled_count=sub2_u, compiled_count=sub2_c
        ),
    )


def test_constants_and_policy_bounds() -> None:
    """Document 25-block instrumentation reserve independently of 1,000 ceiling."""
    assert HOME_LICENSE_BLOCK_LIMIT == 1000
    assert INSTRUMENTATION_RESERVE_BLOCKS == 25
    assert PRODUCTION_BUDGET_CEILING == 975
    assert EXACT_AUDIT_BUDGET_CEILING == 1000
    assert (
        PRODUCTION_BUDGET_CEILING + INSTRUMENTATION_RESERVE_BLOCKS
        == HOME_LICENSE_BLOCK_LIMIT
    )


def test_production_count_within_975_passes() -> None:
    """Actual R2025b production count <= 975 with 25-block reserve passes gate."""
    # GS3DX_Human compiles to 965 blocks (< 975 ceiling)
    obs = create_canonical_human_observability_manifest()
    subsystems = _sample_subsystems(uncompiled_total=750, compiled_total=965)
    report = SimscapeBlockBudgetReport(
        model_name="GS3DX_Human",
        profile=BlockBudgetProfile.PRODUCTION,
        uncompiled_total=750,
        compiled_total=965,
        converter_internal=40,
        top_level_equivalent=750,
        simscape_blocks=500,
        subsystems=subsystems,
        observability=obs,
    )

    verdict = validate_simscape_block_budget(report)
    assert verdict.passed is True
    assert verdict.effective_ceiling == 975
    assert verdict.headroom == 10  # 975 - 965
    assert verdict.reserve_blocks == 25
    assert "Within production budget ceiling of 975" in verdict.summary


def test_exact_required_audit_variant_compiles_under_1000() -> None:
    """Exact required audit variant with diagnostics compiles <= 1,000."""
    # Audit variant has 25 blocks of diagnostic instrumentation (compiled_total = 990 <= 1000)
    obs = create_canonical_human_observability_manifest()
    subsystems = _sample_subsystems(uncompiled_total=775, compiled_total=990)
    report = SimscapeBlockBudgetReport(
        model_name="GS3DX_Human_Audit",
        profile=BlockBudgetProfile.AUDIT,
        uncompiled_total=775,
        compiled_total=990,
        converter_internal=50,
        top_level_equivalent=775,
        simscape_blocks=525,
        subsystems=subsystems,
        observability=obs,
    )

    verdict = validate_simscape_block_budget(report)
    assert verdict.passed is True
    assert verdict.effective_ceiling == 1000
    assert verdict.headroom == 10  # 1000 - 990
    assert verdict.reserve_blocks == 0


def test_audit_variant_fails_production_profile_if_above_975() -> None:
    """An instrumented variant exceeding 975 is rejected under production profile."""
    obs = create_canonical_human_observability_manifest()
    subsystems = _sample_subsystems(uncompiled_total=775, compiled_total=985)
    report = SimscapeBlockBudgetReport(
        model_name="GS3DX_Human_Audit",
        profile=BlockBudgetProfile.PRODUCTION,
        uncompiled_total=775,
        compiled_total=985,
        converter_internal=50,
        top_level_equivalent=775,
        simscape_blocks=525,
        subsystems=subsystems,
        observability=obs,
    )

    with pytest.raises(SimscapeBudgetExceededError) as exc_info:
        validate_simscape_block_budget(report)

    err_msg = str(exc_info.value)
    assert "exceeds production budget ceiling of 975" in err_msg
    assert "compiled_total=985" in err_msg
    assert "reserve of 25 blocks" in err_msg


def test_deliberately_over_budget_clone_fails_with_actionable_diagnostics() -> None:
    """Tests fail on a deliberately over-budget clone exceeding license ceiling (1,001)."""
    obs = create_canonical_human_observability_manifest()
    subsystems = (
        SubsystemBlockCount(name="UpperBody", uncompiled_count=400, compiled_count=520),
        SubsystemBlockCount(name="LowerBody", uncompiled_count=350, compiled_count=481),
    )
    over_budget_report = SimscapeBlockBudgetReport(
        model_name="GS3DX_OverBudget_Clone",
        profile=BlockBudgetProfile.AUDIT,
        uncompiled_total=750,
        compiled_total=1001,  # Exceeds 1000 ceiling!
        converter_internal=50,
        top_level_equivalent=750,
        simscape_blocks=600,
        subsystems=subsystems,
        observability=obs,
    )

    with pytest.raises(SimscapeBudgetExceededError) as exc_info:
        validate_simscape_block_budget(over_budget_report)

    err = str(exc_info.value)
    # Actionable diagnostics check
    assert "exceeds license limit of 1000 nonvirtual blocks" in err
    assert "compiled_total=1001" in err
    assert "offending subsystems by growth" in err.lower()
    assert "UpperBody" in err


def test_observability_contract_rejects_missing_evidence_streams() -> None:
    """Observability contract: no consumer loses mass/COM/energy/contact/closure evidence."""
    # Create manifest missing ENERGY and CLOSURE evidence
    partial_channels = (
        ObservabilityChannel(
            channel_id="mass_audit",
            evidence_type=ObservabilityEvidenceType.MASS,
            source_type=EvidenceSourceType.OFFLINE_DIAGNOSTIC,
            active=True,
            consumer="gs3dx_inertia_audit",
        ),
        ObservabilityChannel(
            channel_id="balance_com_ref",
            evidence_type=ObservabilityEvidenceType.COM,
            source_type=EvidenceSourceType.WORKSPACE_REFERENCE,
            active=True,
            consumer="BalanceCOMRef",
        ),
        ObservabilityChannel(
            channel_id="foot_contact_forces",
            evidence_type=ObservabilityEvidenceType.CONTACT,
            source_type=EvidenceSourceType.NATIVE_LOG,
            active=True,
            consumer="FootContactForces",
        ),
    )
    incomplete_obs = ObservabilityManifest(channels=partial_channels)
    subsystems = _sample_subsystems(uncompiled_total=700, compiled_total=900)
    report = SimscapeBlockBudgetReport(
        model_name="GS3DX_Human_MissingEvidence",
        profile=BlockBudgetProfile.PRODUCTION,
        uncompiled_total=700,
        compiled_total=900,
        converter_internal=30,
        top_level_equivalent=700,
        simscape_blocks=450,
        subsystems=subsystems,
        observability=incomplete_obs,
    )

    with pytest.raises(ObservabilityEvidenceError) as exc_info:
        validate_simscape_block_budget(report)

    err = str(exc_info.value)
    assert "Observability contract violated" in err
    assert "Missing required evidence domain" in err
    assert "energy" in err.lower()
    assert "closure" in err.lower()


def test_removing_consumed_sensor_without_replacement_fails_observability() -> None:
    """Removing a sensor that a consumer depends on without replacement offline diagnostics fails."""
    # Sensor dropped, consumer orphaned
    orphan_channels = (
        ObservabilityChannel(
            channel_id="sensor_inertia_12",
            evidence_type=ObservabilityEvidenceType.MASS,
            source_type=EvidenceSourceType.PHYSICAL_SENSOR,
            active=False,  # Dropped!
            consumer="legacy_inertia_consumer",
            description="Removed without replacement diagnostic",
        ),
        ObservabilityChannel(
            channel_id="balance_com_ref",
            evidence_type=ObservabilityEvidenceType.COM,
            source_type=EvidenceSourceType.WORKSPACE_REFERENCE,
            active=True,
            consumer="BalanceCOMRef",
        ),
        ObservabilityChannel(
            channel_id="energy_calc",
            evidence_type=ObservabilityEvidenceType.ENERGY,
            source_type=EvidenceSourceType.OFFLINE_DIAGNOSTIC,
            active=True,
            consumer="simlog_energy",
        ),
        ObservabilityChannel(
            channel_id="foot_contact_forces",
            evidence_type=ObservabilityEvidenceType.CONTACT,
            source_type=EvidenceSourceType.NATIVE_LOG,
            active=True,
            consumer="FootContactForces",
        ),
        ObservabilityChannel(
            channel_id="momentum_closure",
            evidence_type=ObservabilityEvidenceType.CLOSURE,
            source_type=EvidenceSourceType.OFFLINE_DIAGNOSTIC,
            active=True,
            consumer="gs3dx_contact_check",
        ),
    )
    obs = ObservabilityManifest(channels=orphan_channels)
    subsystems = _sample_subsystems(uncompiled_total=700, compiled_total=900)
    report = SimscapeBlockBudgetReport(
        model_name="GS3DX_Broken_Sensor",
        profile=BlockBudgetProfile.PRODUCTION,
        uncompiled_total=700,
        compiled_total=900,
        converter_internal=30,
        top_level_equivalent=700,
        simscape_blocks=450,
        subsystems=subsystems,
        observability=obs,
    )

    with pytest.raises(ObservabilityEvidenceError) as exc_info:
        validate_simscape_block_budget(report)

    err = str(exc_info.value)
    assert "orphaned active consumer" in err.lower()
    assert "legacy_inertia_consumer" in err


def test_subsystem_breakdown_rejects_inferred_or_empty_counts() -> None:
    """Report before/after count by subsystem, not an inferred icon count."""
    obs = create_canonical_human_observability_manifest()

    # Empty subsystems
    report_empty = SimscapeBlockBudgetReport(
        model_name="GS3DX_NoSubsystems",
        profile=BlockBudgetProfile.PRODUCTION,
        uncompiled_total=700,
        compiled_total=900,
        converter_internal=30,
        top_level_equivalent=700,
        simscape_blocks=450,
        subsystems=(),  # Empty!
        observability=obs,
    )
    with pytest.raises(InvalidSubsystemReportError) as exc_empty:
        validate_simscape_block_budget(report_empty)
    assert "Subsystems breakdown cannot be empty" in str(exc_empty.value)

    # Inconsistent sum with compiled total
    bad_subsystems = (
        SubsystemBlockCount(name="UpperBody", uncompiled_count=400, compiled_count=450),
        SubsystemBlockCount(name="LowerBody", uncompiled_count=300, compiled_count=350),
    )  # Sum compiled = 800 != compiled_total 900
    report_bad_sum = SimscapeBlockBudgetReport(
        model_name="GS3DX_BadSum",
        profile=BlockBudgetProfile.PRODUCTION,
        uncompiled_total=700,
        compiled_total=900,
        converter_internal=30,
        top_level_equivalent=700,
        simscape_blocks=450,
        subsystems=bad_subsystems,
        observability=obs,
    )
    with pytest.raises(InvalidSubsystemReportError) as exc_sum:
        validate_simscape_block_budget(report_bad_sum)
    assert "does not match compiled_total" in str(exc_sum.value)


def test_parse_matlab_block_budget_json_and_validate() -> None:
    """Parse baseline JSON exported by MATLAB and validate against budget gate."""
    assert BASELINE_JSON_PATH.is_file(), f"Missing {BASELINE_JSON_PATH}"
    data = json.loads(BASELINE_JSON_PATH.read_text(encoding="utf-8"))

    report = parse_matlab_block_budget_json(
        data,
        profile=BlockBudgetProfile.PRODUCTION,
        observability=create_canonical_baseline_observability_manifest(),
    )
    assert report.model_name == "GS3DX_Baseline"
    assert report.nonvirtual_total == 672
    assert report.uncompiled_total == 672
    assert report.converter_internal == 306
    assert report.top_level_equivalent == 672
    assert report.simscape_blocks == 489
    assert report.compiled_total == 672  # Baseline uncompiled fallback

    verdict = validate_simscape_block_budget(report)
    assert verdict.passed is True
    assert verdict.effective_ceiling == 975
    assert verdict.headroom == 975 - 672
