"""Unit tests for GS3DX Variants Promotion and R2025b Evidence Contracts (MMR-03, #11087).

Enforces:
1. Complete inventory and receipts for Baseline, Slim, Quat, FullBody, Contact, Golfer, Fit, Shape, Neck, Human.
2. Clean-host explicit R2025b build/save/reopen validation (fails closed on non-R2025b, build errors, dirty reopen).
3. Hand-built original models' hashes protected (protection contract).
4. Stable-drive equivalence strictly distinguished from C3D fit.
5. Motion-prescribed neck and servo tracking strictly distinguished from autonomous balance.
6. Cold-replay commands, provider/model/capture provenance, and candidate/image/metric integrity.
7. Main ledger consumes only the designated reviewed variant.
8. Reported MATLAB suite rerun at promoted SHA with skipped tests disclosed.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.shared.python.motion_matching.gs3dx_variants import (
    ActuationClassificationError,
    BalanceClassificationError,
    BalanceMode,
    CandidateIntegrityError,
    CandidateIntegrityReceipt,
    CleanHostBuildReceipt,
    CleanHostBuildValidationError,
    ColdReplaySpec,
    DriveClassification,
    DriveClassificationMismatchError,
    GS3DXDriveMode,
    GS3DXPromotionError,
    GS3DXVariant,
    GS3DXVariantReceipt,
    LedgerPromotionError,
    MatlabSuiteResult,
    MatlabSuiteValidationError,
    NeckActuationMode,
    OriginalModelProtectedError,
    OriginalProtectionSummary,
    ReviewStatus,
    VariantBlockBudget,
    VariantProvenance,
    build_gs3dx_variant_receipt,
    filter_promotable_ledger_variants,
    generate_canonical_variant_receipt,
    get_canonical_gs3dx_variant_definitions,
    load_variant_receipt,
    save_variant_receipt,
    validate_candidate_package_integrity,
    validate_gs3dx_variant_receipt,
    validate_matlab_suite_run,
    verify_original_models_protection,
)


@pytest.fixture
def valid_clean_host_build() -> CleanHostBuildReceipt:
    return CleanHostBuildReceipt(
        matlab_release="2025b",
        matlab_version="24.2.0.2741519 (R2025b)",
        host="test-builder-box",
        built_clean=True,
        saved_clean=True,
        reopened_clean=True,
    )


@pytest.fixture
def valid_provenance() -> VariantProvenance:
    return VariantProvenance(
        provider="MATLAB R2025b / Simscape Multibody",
        host="test-builder-box",
        matlab_release="2025b",
        matlab_version="24.2.0.2741519 (R2025b)",
        model_name="GS3DX_Human",
        model_sha256="919974719a4e24ee7d04ff004818c01dc3c31212383fe6a84b6e6de84abc919f",
        capture_file="data/C3D_TA_Driver.c3d",
        capture_sha256="cbcb4d84aca0a1f558073f0c2328c3f501c4726aaf699e382cf20c5861a0971d",
    )


@pytest.fixture
def valid_human_receipt(
    valid_clean_host_build: CleanHostBuildReceipt,
    valid_provenance: VariantProvenance,
) -> GS3DXVariantReceipt:
    return GS3DXVariantReceipt(
        schema_version="gs3dx-variant-receipt/1",
        variant=GS3DXVariant.HUMAN,
        model_name="GS3DX_Human",
        builder="gs3dx_build_human",
        parent_variant=GS3DXVariant.NECK,
        model_file="src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/models/GS3DX_Human.slx",
        model_sha256="919974719a4e24ee7d04ff004818c01dc3c31212383fe6a84b6e6de84abc919f",
        block_budget=VariantBlockBudget(
            uncompiled_count=773,
            compiled_count=965,
            license_ceiling=1000,
            reserve_blocks=25,
        ),
        drive_mode=GS3DXDriveMode.HUMAN_SPRUNG_FEET_BALANCE,
        drive_classification=DriveClassification.C3D_FIT,
        neck_actuation=NeckActuationMode.MOTION_PRESCRIBED,
        balance_mode=BalanceMode.JACOBIAN_BALANCE_LOOP,
        c3d_tracking=True,
        clean_host_build=valid_clean_host_build,
        original_protection=OriginalProtectionSummary(
            originals_verified=True,
            files_checked=("GolfSwing3D_Kinetic.slx",),
            hashes_match=True,
            details={
                "GolfSwing3D_Kinetic.slx": "daca9a90ad0ab819c7d61641ed594b8f230f8658f2a5f4ce34026db44b52ddc9"
            },
        ),
        cold_replay=ColdReplaySpec(
            command="matlab -batch \"addpath('tools'); gs3dx_setup; gs3dx_simulate('GS3DX_Human')\"",
            cwd="src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx",
            environment={"MATLABPATH": "tools;models"},
            input_files=("models/GS3DX_Human.slx", "data/C3D_TA_Driver.c3d"),
        ),
        provenance=valid_provenance,
        review_status=ReviewStatus.REVIEWED_PROMOTED,
    )


@pytest.mark.unit
def test_ten_promoted_variants_inventory() -> None:
    """All 10 required variants must be registered with exact builder, lineage, and block budgets."""
    definitions = get_canonical_gs3dx_variant_definitions()
    expected_variants = {
        GS3DXVariant.BASELINE,
        GS3DXVariant.SLIM,
        GS3DXVariant.QUAT,
        GS3DXVariant.FULL_BODY,
        GS3DXVariant.CONTACT,
        GS3DXVariant.GOLFER,
        GS3DXVariant.FIT,
        GS3DXVariant.SHAPE,
        GS3DXVariant.NECK,
        GS3DXVariant.HUMAN,
    }
    assert set(definitions.keys()) == expected_variants
    assert len(definitions) == 10

    # Lineage verification
    assert definitions[GS3DXVariant.BASELINE].parent_variant is None
    assert definitions[GS3DXVariant.SLIM].parent_variant == GS3DXVariant.BASELINE
    assert definitions[GS3DXVariant.QUAT].parent_variant == GS3DXVariant.SLIM
    assert definitions[GS3DXVariant.FULL_BODY].parent_variant == GS3DXVariant.QUAT
    assert definitions[GS3DXVariant.CONTACT].parent_variant == GS3DXVariant.FULL_BODY
    assert definitions[GS3DXVariant.GOLFER].parent_variant == GS3DXVariant.CONTACT
    assert definitions[GS3DXVariant.FIT].parent_variant == GS3DXVariant.GOLFER
    assert definitions[GS3DXVariant.SHAPE].parent_variant == GS3DXVariant.FIT
    assert definitions[GS3DXVariant.NECK].parent_variant == GS3DXVariant.SHAPE
    assert definitions[GS3DXVariant.HUMAN].parent_variant == GS3DXVariant.NECK

    # All compiled models must observe the 1,000 block ceiling and 25-block reserve (<= 975)
    for variant, defn in definitions.items():
        assert defn.compiled_blocks <= 975, (
            f"{variant} compiled blocks {defn.compiled_blocks} > 975 reserve cap"
        )
        assert defn.model_file.endswith(".slx")
        assert defn.builder.startswith("gs3dx_")


@pytest.mark.unit
def test_machine_readable_variant_receipt_lifecycle(
    tmp_path: Path, valid_human_receipt: GS3DXVariantReceipt
) -> None:
    """Per-variant receipt must validate, serialize to JSON, and deserialize faithfully."""
    validate_gs3dx_variant_receipt(valid_human_receipt)
    out_file = tmp_path / "gs3dx_human_receipt.json"
    save_variant_receipt(valid_human_receipt, out_file)
    assert out_file.is_file()

    loaded = load_variant_receipt(out_file)
    assert loaded.variant == GS3DXVariant.HUMAN
    assert loaded.model_sha256 == valid_human_receipt.model_sha256
    assert loaded.clean_host_build.matlab_release == "2025b"
    assert loaded.review_status == ReviewStatus.REVIEWED_PROMOTED

    receipt_dict = loaded.as_dict()
    assert receipt_dict["schema_version"] == "gs3dx-variant-receipt/1"
    assert receipt_dict["variant"] == "Human"
    assert receipt_dict["drive_classification"] == "c3d_fit"
    assert receipt_dict["neck_actuation"] == "motion_prescribed"


@pytest.mark.unit
def test_clean_host_r2025b_validation_fails_closed(
    valid_human_receipt: GS3DXVariantReceipt,
) -> None:
    """Must reject non-R2025b releases, dirty builds, or missing host."""
    # 1. R2026a substitution prohibited
    bad_release = CleanHostBuildReceipt(
        matlab_release="2026a",
        matlab_version="25.1.0",
        host="host-1",
        built_clean=True,
        saved_clean=True,
        reopened_clean=True,
    )
    receipt_r2026a = build_gs3dx_variant_receipt(
        valid_human_receipt, clean_host_build=bad_release
    )
    with pytest.raises(
        CleanHostBuildValidationError, match="requires MATLAB R2025b; got '2026a'"
    ):
        validate_gs3dx_variant_receipt(receipt_r2026a)

    # 2. Build not clean
    bad_build = CleanHostBuildReceipt(
        matlab_release="2025b",
        matlab_version="24.2.0",
        host="host-1",
        built_clean=False,
        saved_clean=True,
        reopened_clean=True,
    )
    with pytest.raises(CleanHostBuildValidationError, match="clean-host build failed"):
        validate_gs3dx_variant_receipt(
            build_gs3dx_variant_receipt(valid_human_receipt, clean_host_build=bad_build)
        )

    # 3. Save not clean
    bad_save = CleanHostBuildReceipt(
        matlab_release="2025b",
        matlab_version="24.2.0",
        host="host-1",
        built_clean=True,
        saved_clean=False,
        reopened_clean=True,
    )
    with pytest.raises(CleanHostBuildValidationError, match="clean-host save failed"):
        validate_gs3dx_variant_receipt(
            build_gs3dx_variant_receipt(valid_human_receipt, clean_host_build=bad_save)
        )

    # 4. Reopen not clean
    bad_reopen = CleanHostBuildReceipt(
        matlab_release="2025b",
        matlab_version="24.2.0",
        host="host-1",
        built_clean=True,
        saved_clean=True,
        reopened_clean=False,
    )
    with pytest.raises(CleanHostBuildValidationError, match="clean-host reopen failed"):
        validate_gs3dx_variant_receipt(
            build_gs3dx_variant_receipt(
                valid_human_receipt, clean_host_build=bad_reopen
            )
        )

    # 5. Empty host
    empty_host = CleanHostBuildReceipt(
        matlab_release="2025b",
        matlab_version="24.2.0",
        host="   ",
        built_clean=True,
        saved_clean=True,
        reopened_clean=True,
    )
    with pytest.raises(
        CleanHostBuildValidationError,
        match="host must name the licensed execution machine",
    ):
        validate_gs3dx_variant_receipt(
            build_gs3dx_variant_receipt(
                valid_human_receipt, clean_host_build=empty_host
            )
        )


@pytest.mark.unit
def test_hand_built_originals_protection_contract() -> None:
    """Hand-built originals must match committed digests; changes fail closed."""
    # Live verification against actual checkout repository
    summary = verify_original_models_protection()
    assert summary.originals_verified is True
    assert summary.hashes_match is True
    assert len(summary.files_checked) == 4

    # Simulated corruption / modification of an original model
    tampered_hashes = {
        "GolfSwing3D_Kinetic.slx": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    }
    with pytest.raises(
        OriginalModelProtectedError, match="Original hand-built model hash mismatch"
    ):
        verify_original_models_protection(override_hashes=tampered_hashes)


@pytest.mark.unit
def test_stable_drive_equivalence_distinguished_from_c3d_fit(
    valid_human_receipt: GS3DXVariantReceipt,
) -> None:
    """Baseline/Slim/Quat must NOT claim C3D fit, and C3D fit requires capture provenance."""
    # Baseline cannot claim C3D_FIT
    baseline_bad = build_gs3dx_variant_receipt(
        valid_human_receipt,
        variant=GS3DXVariant.BASELINE,
        model_name="GS3DX_Baseline",
        drive_classification=DriveClassification.C3D_FIT,
        c3d_tracking=True,
    )
    with pytest.raises(
        DriveClassificationMismatchError,
        match="Baseline only satisfies stable-drive equivalence",
    ):
        validate_gs3dx_variant_receipt(baseline_bad)

    # Slim cannot claim C3D_FIT
    slim_bad = build_gs3dx_variant_receipt(
        valid_human_receipt,
        variant=GS3DXVariant.SLIM,
        model_name="GS3DX_Slim",
        drive_classification=DriveClassification.C3D_FIT,
        c3d_tracking=True,
    )
    with pytest.raises(
        DriveClassificationMismatchError,
        match="Slim only satisfies stable-drive equivalence",
    ):
        validate_gs3dx_variant_receipt(slim_bad)

    # Quat cannot claim C3D_FIT
    quat_bad = build_gs3dx_variant_receipt(
        valid_human_receipt,
        variant=GS3DXVariant.QUAT,
        model_name="GS3DX_Quat",
        drive_classification=DriveClassification.C3D_FIT,
        c3d_tracking=True,
    )
    with pytest.raises(
        DriveClassificationMismatchError,
        match="Quat only satisfies stable-drive equivalence",
    ):
        validate_gs3dx_variant_receipt(quat_bad)

    # Stable-drive equivalence receipt requires valid drive reference disclosure
    baseline_good = build_gs3dx_variant_receipt(
        valid_human_receipt,
        variant=GS3DXVariant.BASELINE,
        model_name="GS3DX_Baseline",
        drive_classification=DriveClassification.STABLE_DRIVE_EQUIVALENCE,
        c3d_tracking=False,
        drive_reference="baselines/original_GolfSwing3D_Kinetic_impact_0p3S.mat",
    )
    validate_gs3dx_variant_receipt(baseline_good)


@pytest.mark.unit
def test_motion_prescribed_neck_and_servo_tracking_distinguished_from_autonomous(
    valid_human_receipt: GS3DXVariantReceipt,
) -> None:
    """Neck and Human must disclose motion prescription and servo balance; cannot claim autonomous."""
    # 1. Reject autonomous neck claim
    bad_neck = build_gs3dx_variant_receipt(
        valid_human_receipt,
        neck_actuation=NeckActuationMode.AUTONOMOUS,
    )
    with pytest.raises(
        ActuationClassificationError,
        match="Neck/Human actuation uses motion prescription",
    ):
        validate_gs3dx_variant_receipt(bad_neck)

    # 2. Reject autonomous balance claim
    bad_balance = build_gs3dx_variant_receipt(
        valid_human_receipt,
        balance_mode=BalanceMode.AUTONOMOUS_BALANCE,
    )
    with pytest.raises(
        BalanceClassificationError,
        match="Leg stabilization uses servo tracking and Jacobian compensation",
    ):
        validate_gs3dx_variant_receipt(bad_balance)


@pytest.mark.unit
def test_candidate_image_and_metric_package_integrity() -> None:
    """Candidate, image, and metric packages must carry matching hashes."""
    model_sha = "919974719a4e24ee7d04ff004818c01dc3c31212383fe6a84b6e6de84abc919f"
    candidate_sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    image_sha = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    metric_sha = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"

    # Valid package
    pkg = validate_candidate_package_integrity(
        model_sha256=model_sha,
        candidate_sha256=candidate_sha,
        image_package_sha256=image_sha,
        metric_package_sha256=metric_sha,
        declared_candidate_in_images=candidate_sha,
        declared_candidate_in_metrics=candidate_sha,
    )
    assert isinstance(pkg, CandidateIntegrityReceipt)
    assert pkg.candidate_sha256 == candidate_sha

    # Mismatch in images package candidate reference
    with pytest.raises(
        CandidateIntegrityError, match="Image package candidate SHA mismatch"
    ):
        validate_candidate_package_integrity(
            model_sha256=model_sha,
            candidate_sha256=candidate_sha,
            image_package_sha256=image_sha,
            metric_package_sha256=metric_sha,
            declared_candidate_in_images="deadbeef" * 8,
            declared_candidate_in_metrics=candidate_sha,
        )

    # Mismatch in metrics package candidate reference
    with pytest.raises(
        CandidateIntegrityError, match="Metric package candidate SHA mismatch"
    ):
        validate_candidate_package_integrity(
            model_sha256=model_sha,
            candidate_sha256=candidate_sha,
            image_package_sha256=image_sha,
            metric_package_sha256=metric_sha,
            declared_candidate_in_images=candidate_sha,
            declared_candidate_in_metrics="deadbeef" * 8,
        )


@pytest.mark.unit
def test_main_ledger_consumes_only_reviewed_variant(
    valid_human_receipt: GS3DXVariantReceipt,
) -> None:
    """Main ledger consumer gate must reject unreviewed or intermediate variants."""
    reviewed_human = valid_human_receipt
    intermediate_neck = build_gs3dx_variant_receipt(
        valid_human_receipt,
        variant=GS3DXVariant.NECK,
        model_name="GS3DX_Neck",
        review_status=ReviewStatus.EXPLORATORY_PROMOTED,
    )
    unreviewed_variant = build_gs3dx_variant_receipt(
        valid_human_receipt,
        variant=GS3DXVariant.FIT,
        model_name="GS3DX_Fit",
        review_status=ReviewStatus.UNREVIEWED,
    )

    promotable = filter_promotable_ledger_variants(
        [reviewed_human, intermediate_neck, unreviewed_variant]
    )
    assert len(promotable) == 1
    assert promotable[0].variant == GS3DXVariant.HUMAN

    # If ledger consumer receives unreviewed variant directly, fail closed
    with pytest.raises(
        LedgerPromotionError,
        match="Main ledger only consumes reviewed promoted variants",
    ):
        filter_promotable_ledger_variants([intermediate_neck], fail_on_unreviewed=True)


@pytest.mark.unit
def test_matlab_suite_rerun_at_promoted_sha_with_skipped_tests_disclosed() -> None:
    """MATLAB test suite rerun validation must match SHA, require R2025b, and disclose skips."""
    promoted_sha = "919974719a4e24ee7d04ff004818c01dc3c31212383fe6a84b6e6de84abc919f"

    # Valid suite execution
    valid_suite = MatlabSuiteResult(
        promoted_sha=promoted_sha,
        matlab_release="2025b",
        total_tests=26,
        passed_tests=24,
        failed_tests=0,
        skipped_tests=("test_gs3dx_license_limit_probe", "test_gs3dx_contact_trial"),
        skip_reasons={
            "test_gs3dx_license_limit_probe": "License limit probe is a manual stress benchmark",
            "test_gs3dx_contact_trial": "Physical trial requires interactive viewer session",
        },
    )
    validate_matlab_suite_run(valid_suite, expected_sha=promoted_sha)

    # Mismatched SHA
    with pytest.raises(
        MatlabSuiteValidationError, match="Promoted SHA mismatch in suite run"
    ):
        validate_matlab_suite_run(
            valid_suite,
            expected_sha="0000000000000000000000000000000000000000000000000000000000000000",
        )

    # Non-R2025b release
    bad_release_suite = MatlabSuiteResult(
        promoted_sha=promoted_sha,
        matlab_release="2026a",
        total_tests=26,
        passed_tests=26,
        failed_tests=0,
        skipped_tests=(),
        skip_reasons={},
    )
    with pytest.raises(
        MatlabSuiteValidationError,
        match="MATLAB suite rerun requires R2025b; got '2026a'",
    ):
        validate_matlab_suite_run(bad_release_suite, expected_sha=promoted_sha)

    # Suite failures
    failing_suite = MatlabSuiteResult(
        promoted_sha=promoted_sha,
        matlab_release="2025b",
        total_tests=26,
        passed_tests=25,
        failed_tests=1,
        skipped_tests=(),
        skip_reasons={},
    )
    with pytest.raises(
        MatlabSuiteValidationError, match="Suite run reported 1 failures"
    ):
        validate_matlab_suite_run(failing_suite, expected_sha=promoted_sha)

    # Empty suite (0 tests run)
    empty_suite = MatlabSuiteResult(
        promoted_sha=promoted_sha,
        matlab_release="2025b",
        total_tests=0,
        passed_tests=0,
        failed_tests=0,
        skipped_tests=(),
        skip_reasons={},
    )
    with pytest.raises(
        MatlabSuiteValidationError, match="No tests were run; unqualified"
    ):
        validate_matlab_suite_run(empty_suite, expected_sha=promoted_sha)


@pytest.mark.unit
def test_generate_canonical_receipts_for_all_ten_variants(tmp_path: Path) -> None:
    """Generate, validate, and serialize canonical receipts for all 10 promoted variants."""
    all_variants = [
        GS3DXVariant.BASELINE,
        GS3DXVariant.SLIM,
        GS3DXVariant.QUAT,
        GS3DXVariant.FULL_BODY,
        GS3DXVariant.CONTACT,
        GS3DXVariant.GOLFER,
        GS3DXVariant.FIT,
        GS3DXVariant.SHAPE,
        GS3DXVariant.NECK,
        GS3DXVariant.HUMAN,
    ]
    for variant in all_variants:
        receipt = generate_canonical_variant_receipt(variant, host="ci-builder-r2025b")
        assert receipt.variant == variant
        assert receipt.model_sha256 is not None
        assert len(receipt.model_sha256) == 64
        assert receipt.block_budget.within_production_ceiling is True

        # Save and reload
        out_path = tmp_path / f"{receipt.model_name}_receipt.json"
        save_variant_receipt(receipt, out_path)
        loaded = load_variant_receipt(out_path)
        assert loaded.model_name == receipt.model_name
        assert loaded.model_sha256 == receipt.model_sha256
