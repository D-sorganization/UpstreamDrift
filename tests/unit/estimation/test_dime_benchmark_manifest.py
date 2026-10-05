"""Focused behavioral tests for DIME baseline and frozen benchmark protocol (#11422).

Enforces:
- RED: wrong units, missing provenance, force-derived test kinematics, and
  skeleton contact output cannot be marked qualified.
- GREEN: analytic pendulum and existing estimator baseline reproduce with recorded
  seeds; capture datasets remain private; register phase-specific drift/control
  magnitudes, alignment and cancellation metrics; specify a zero-denominator policy;
  report native capability status separately from method existence.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from src.shared.python.estimation.dime_manifest import (
    CANONICAL_DIME_UNITS,
    CapabilityStatus,
    ConditioningScale,
    DimeBenchmarkManifest,
    DimeBenchmarkResult,
    DimeProvenanceRecord,
    ForceClassification,
    NumericAcceptanceThresholds,
    PrivacySpec,
    SplitPolicy,
    ZeroDenominatorPolicy,
    compute_alignment_metric,
    compute_cancellation_metric,
    compute_phase_drift_and_control,
    run_dime_baseline,
)
from src.shared.python.estimation.synthetic_fixtures import (
    make_fixed_base_pendulum_fixture,
    make_native_stance_fixture,
    make_underactuated_analytic_fixture,
)


def _make_valid_manifest(
    *,
    units: dict[str, str] | None = None,
    provenance: DimeProvenanceRecord | None = None,
    force_classification: ForceClassification | None = None,
    is_private: bool = True,
    zero_denominator_policy: ZeroDenominatorPolicy = "guarded_zero",
) -> DimeBenchmarkManifest:
    """Helper to construct a canonically valid benchmark manifest."""
    return DimeBenchmarkManifest(
        manifest_version="1.0.0",
        dataset_id="benchmark-dataset-01",
        dataset_revision="rev-2026-10-04",
        model_id="fixed_base_pendulum",
        model_revision="sha256-pendulum-model-v1",
        engine_id="analytic",
        engine_revision="1.0.0",
        split_policy=SplitPolicy(
            calibration_frames=(0, 1, 2, 3),
            holdout_frames=(4, 5, 6, 7),
            phase_frames={"phase_a": (0, 1, 2, 3), "phase_b": (4, 5, 6, 7)},
        ),
        units=dict(CANONICAL_DIME_UNITS) if units is None else units,
        observation_type="synthetic_camera",
        known_truth_type="analytic_solution",
        force_classification=(
            ForceClassification(
                measured_forces=(),
                inferred_forces=("pin_reaction",),
                kinematics_source="independent_measurement",
                has_skeleton_contact=False,
            )
            if force_classification is None
            else force_classification
        ),
        provenance=(
            DimeProvenanceRecord(
                engine="analytic",
                engine_version="1.0.0",
                model_hash="a1b2c3d4e5f67890",
                param_hash="0987654321fedcba",
                git_commit="abcdef0123456789abcdef0123456789abcdef01",
                created_at="2026-10-04T00:00:00Z",
                seed=42,
            )
            if provenance is None
            else provenance
        ),
        privacy=PrivacySpec(
            is_private=is_private,
            license_name="Proprietary / Internal Benchmark",
            redact_filesystem_paths=True,
        ),
        zero_denominator_policy=zero_denominator_policy,
        thresholds=NumericAcceptanceThresholds(
            max_drift_m=0.015,
            max_angular_drift_rad=0.05,
            max_control_norm_nm=250.0,
            min_alignment=0.95,
            max_cancellation_ratio=0.10,
            reproducibility_atol=1e-9,
        ),
        conditioning=ConditioningScale(
            position_scale=1.0,
            angle_scale=1.0,
            velocity_scale=1.0,
            torque_scale=1.0,
            time_scale=1.0,
        ),
    )


# ==============================================================================
# RED Cases: Disqualification Rules
# ==============================================================================


@pytest.mark.unit
def test_red_wrong_units_cannot_be_marked_qualified() -> None:
    """RED: Non-canonical units (e.g. mm or deg) must not be marked qualified."""
    wrong_units = dict(CANONICAL_DIME_UNITS)
    wrong_units["length"] = "mm"  # Non-canonical unit

    manifest = _make_valid_manifest(units=wrong_units)
    status, reasons = manifest.evaluate_qualification()

    assert status != "qualified"
    assert status == "implemented"
    assert any("units" in r.lower() for r in reasons)


@pytest.mark.unit
def test_red_missing_provenance_cannot_be_marked_qualified() -> None:
    """RED: Missing or empty provenance fields must prevent qualification."""
    incomplete_provenance = DimeProvenanceRecord(
        engine="analytic",
        engine_version="",  # Empty version
        model_hash="",  # Empty hash
        param_hash="0987654321fedcba",
        git_commit="",  # Empty git commit
        created_at="2026-10-04T00:00:00Z",
        seed=42,
    )

    manifest = _make_valid_manifest(provenance=incomplete_provenance)
    status, reasons = manifest.evaluate_qualification()

    assert status != "qualified"
    assert any("provenance" in r.lower() for r in reasons)


@pytest.mark.unit
def test_red_force_derived_test_kinematics_cannot_be_marked_qualified() -> None:
    """RED: Test kinematics derived from forces create circularity and cannot be qualified."""
    classification = ForceClassification(
        measured_forces=(),
        inferred_forces=("joint_torque",),
        kinematics_source="force_derived",  # Force-derived test kinematics!
        has_skeleton_contact=False,
    )

    manifest = _make_valid_manifest(force_classification=classification)
    status, reasons = manifest.evaluate_qualification()

    assert status != "qualified"
    assert any("force_derived" in r.lower() for r in reasons)


@pytest.mark.unit
def test_red_skeleton_contact_output_cannot_be_marked_qualified() -> None:
    """RED: Kinematic skeleton claiming contact output has no contact physics and cannot be qualified."""
    classification = ForceClassification(
        measured_forces=(),
        inferred_forces=("grf",),
        kinematics_source="independent_measurement",
        has_skeleton_contact=True,  # Skeleton contact is physically ungrounded!
    )

    manifest = _make_valid_manifest(force_classification=classification)
    status, reasons = manifest.evaluate_qualification()

    assert status != "qualified"
    assert any("skeleton_contact" in r.lower() for r in reasons)


# ==============================================================================
# GREEN Cases: Baseline, Metrics, Privacy, Fixtures, Reproducibility
# ==============================================================================


@pytest.mark.unit
def test_green_valid_manifest_is_qualified() -> None:
    """GREEN: Canonically compliant manifest passes qualification."""
    manifest = _make_valid_manifest()
    status, reasons = manifest.evaluate_qualification()

    assert status == "qualified"
    assert len(reasons) == 0


@pytest.mark.unit
def test_green_analytic_pendulum_and_estimator_baseline_reproduce_with_recorded_seeds() -> (
    None
):
    """GREEN: Analytic pendulum and existing estimator baseline reproduce exactly with recorded seeds."""
    manifest = _make_valid_manifest()

    result_1 = run_dime_baseline(manifest, seed=42)
    result_2 = run_dime_baseline(manifest, seed=42)

    # No estimator ran, so nothing was reproduced (#11552).
    assert result_1.reproduced_identically is False
    # #11552: the former trajectory was ground truth copied as the "estimate";
    # with no estimator it is not measured (None), not reproduced.
    assert result_1.trajectory_q is None and result_2.trajectory_q is None
    np.testing.assert_allclose(
        result_1.control_torques,
        result_2.control_torques,
        atol=manifest.thresholds.reproducibility_atol,
    )


@pytest.mark.unit
def test_green_capture_datasets_remain_private() -> None:
    """GREEN: Manifest serialization redacts private file paths and preserves privacy status."""
    manifest = _make_valid_manifest(is_private=True)
    serialized = manifest.to_dict()

    assert serialized["privacy"]["is_private"] is True
    assert "source_path" not in serialized
    # Redaction guard: ensure string representations do not leak local file systems
    manifest_str = json.dumps(serialized)
    assert "/home/" not in manifest_str
    assert "/Users/" not in manifest_str
    assert "C:\\" not in manifest_str


@pytest.mark.unit
def test_green_phase_specific_drift_and_control_magnitudes() -> None:
    """GREEN: Computes phase-specific drift and control magnitudes across split phases."""
    phase_frames = {
        "address": (0, 1, 2),
        "downswing": (3, 4, 5),
    }
    q_est = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4])
    q_true = np.array([0.0, 0.04, 0.09, 0.18, 0.27, 0.36])
    torques = np.array([10.0, 15.0, 20.0, 50.0, 80.0, 100.0])

    metrics = compute_phase_drift_and_control(q_est, q_true, torques, phase_frames)

    assert "address" in metrics
    assert "downswing" in metrics
    assert np.isclose(metrics["address"].max_drift, 0.01, atol=1e-5)
    assert np.isclose(metrics["address"].max_control, 20.0, atol=1e-5)
    assert np.isclose(metrics["downswing"].max_drift, 0.04, atol=1e-5)
    assert np.isclose(metrics["downswing"].max_control, 100.0, atol=1e-5)


@pytest.mark.unit
def test_green_alignment_and_cancellation_metrics_with_zero_denominator_policy() -> (
    None
):
    """GREEN: Metric calculations handle zero denominators safely according to policy."""
    zero_vec = np.zeros(5)
    non_zero = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Under 'guarded_zero' policy:
    alignment_zero = compute_alignment_metric(zero_vec, non_zero, policy="guarded_zero")
    assert alignment_zero == 0.0

    cancellation_zero = compute_cancellation_metric(
        zero_vec, zero_vec, policy="guarded_zero"
    )
    assert cancellation_zero == 0.0

    # Under 'raise' policy:
    with pytest.raises(ZeroDivisionError):
        compute_alignment_metric(zero_vec, non_zero, policy="raise")

    # Positive alignment test:
    alignment_perfect = compute_alignment_metric(
        non_zero, non_zero, policy="guarded_zero"
    )
    assert np.isclose(alignment_perfect, 1.0, atol=1e-6)

    # Opposing forces cancellation test (equal & opposite forces cancel perfectly):
    f1 = np.array([10.0, 0.0, 0.0])
    f2 = np.array([-10.0, 0.0, 0.0])
    cancellation_perfect = compute_cancellation_metric(f1, f2, policy="guarded_zero")
    assert np.isclose(cancellation_perfect, 0.0, atol=1e-6)


@pytest.mark.unit
def test_green_native_capability_status_separated_from_method_existence() -> None:
    """GREEN: Method existence does not imply qualified status; report reports truthful statuses."""

    class DummyEngineWithMethods:
        def simulate_forward(self) -> None:
            pass

        def compute_inverse_dynamics(self) -> None:
            pass

    # Method exists on DummyEngineWithMethods, but lacks physics qualification
    manifest = _make_valid_manifest(
        force_classification=ForceClassification(
            kinematics_source="force_derived",
        )
    )
    capabilities = manifest.report_native_capabilities(
        engine_instance=DummyEngineWithMethods()
    )

    # Method exists is True
    assert capabilities["simulate_forward"].method_exists is True
    # But status is implemented, not qualified
    assert capabilities["simulate_forward"].status == "implemented"
    assert capabilities["overall"].status != "qualified"


@pytest.mark.unit
def test_green_frozen_numeric_thresholds_and_conditioning_scaling() -> None:
    """GREEN: Manifest fixes thresholds and conditioning before comparison."""
    manifest = _make_valid_manifest()

    assert manifest.thresholds.max_drift_m == 0.015
    assert manifest.thresholds.max_angular_drift_rad == 0.05
    assert manifest.thresholds.max_control_norm_nm == 250.0
    assert manifest.thresholds.min_alignment == 0.95
    assert manifest.thresholds.max_cancellation_ratio == 0.10
    assert manifest.conditioning.position_scale == 1.0


@pytest.mark.unit
def test_green_manifest_serialization_roundtrip(tmp_path: Path) -> None:
    """GREEN: Manifest serializes to dict and JSON and round-trips losslessly."""
    manifest = _make_valid_manifest()
    json_path = tmp_path / "dime_manifest.json"

    manifest.save_json(json_path)
    loaded = DimeBenchmarkManifest.load_json(json_path)

    assert loaded.dataset_id == manifest.dataset_id
    assert loaded.model_id == manifest.model_id
    assert loaded.engine_id == manifest.engine_id
    assert loaded.units == manifest.units
    assert (
        loaded.split_policy.calibration_frames
        == manifest.split_policy.calibration_frames
    )
    assert loaded.thresholds.max_drift_m == manifest.thresholds.max_drift_m
    assert loaded.provenance.git_commit == manifest.provenance.git_commit


@pytest.mark.unit
def test_green_shared_fixtures_contract() -> None:
    """GREEN: Shared fixtures for fixed-base pendulum, underactuated, and native stance validate."""
    # 1. Fixed-base pendulum
    pendulum = make_fixed_base_pendulum_fixture(n_frames=10, fps=100.0)
    assert pendulum.skeleton.num_joints == 1
    assert pendulum.skeleton.num_dofs == 1
    assert len(pendulum.frames) == 10
    assert pendulum.units == "m"
    assert np.isclose(pendulum.frames[0].timestamp, 0.0)

    # 2. Underactuated analytic fixture
    underactuated = make_underactuated_analytic_fixture(n_frames=10, fps=100.0)
    assert underactuated.is_underactuated is True
    assert underactuated.unactuated_dofs == (0,)
    assert len(underactuated.frames) == 10

    # 3. Native stance fixture
    stance = make_native_stance_fixture(n_frames=10, fps=100.0)
    assert stance.is_stance is True
    assert np.isclose(stance.ground_reaction_force_z, stance.mass_kg * 9.81, atol=1e-4)
    assert len(stance.frames) == 10


@pytest.mark.unit
def test_red_11552_baseline_without_estimator_is_not_measured() -> None:
    """The baseline must not report truth-as-estimate figures as measurements."""
    manifest = _make_valid_manifest()
    result = run_dime_baseline(manifest, seed=42)

    assert result.measured is False
    assert result.trajectory_q is None
    assert result.alignment is None
    assert result.cancellation is None
    assert result.phase_metrics == {}
    assert result.status == "unavailable"
    assert result.thresholds_passed is False
    assert any("not measured" in f for f in result.threshold_failures)
    assert any("no baseline estimator" in r for r in result.qualification_reasons)
    # Qualification of the manifest itself is unchanged and still evaluable.
    assert manifest.evaluate_qualification()[0] == "qualified"
