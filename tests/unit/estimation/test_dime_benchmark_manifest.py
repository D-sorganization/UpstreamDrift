"""Tests for DIME-01: Baseline and Frozen Benchmark Protocol (#11422).

TDD test-first suite verifying:
1. RED: wrong units, missing provenance, force-derived test kinematics, and
   skeleton contact output cannot be marked qualified.
2. GREEN: analytic pendulum and existing estimator baseline reproduce with
   recorded seeds; capture datasets remain private.
3. Register phase-specific drift/control magnitudes, alignment and cancellation
   metrics with zero-denominator safety.
4. Shared deterministic fixed-base pendulum, underactuated analytic and native
   stance fixtures with exact truth derivation.
5. Frozen numeric thresholds and conditioning/scaling before solver comparison.
6. Reviewer can run baseline from saved manifest and see truthful
   implemented/qualified/unavailable status.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.shared.python.data_io.provenance import ProvenanceInfo
from src.shared.python.estimation.benchmark_manifest import (
    BenchmarkExecutionResult,
    BenchmarkManifest,
    BenchmarkManifestInput,
    BenchmarkSplitPolicy,
    ForceDerivedKinematicsError,
    ForceMeasurementType,
    MissingProvenanceError,
    NativeCapabilityStatus,
    NumericAcceptanceThreshold,
    ObservationType,
    PhaseMetrics,
    SkeletonContactQualificationError,
    UnitMismatchError,
    compute_phase_metrics,
    create_baseline_manifest,
    run_baseline_from_manifest,
    validate_benchmark_manifest,
)
from src.shared.python.estimation.benchmark_fixtures import (
    BenchmarkFixture,
    make_deterministic_pendulum_fixture,
    make_native_stance_fixture,
    make_underactuated_analytic_fixture,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _valid_provenance() -> ProvenanceInfo:
    return ProvenanceInfo(
        timestamp_utc="2026-10-04T00:00:00Z",
        timestamp_local="2026-10-04T00:00:00Z",
        software_name="UpstreamDrift-DIME",
        software_version="1.0.0",
        git_commit_sha="2deef170502deef170502deef170502deef17050",
        engine_name="synthetic-pendulum",
        parameters={"seed": 42},
    )


def _valid_manifest_input() -> BenchmarkManifestInput:
    return BenchmarkManifestInput(
        dataset_id="benchmark-synthetic-pendulum-v1",
        model_revision="rev-001",
        engine_type="analytic-euler-lagrange",
        split_policy=BenchmarkSplitPolicy.ALL_FRAMES,
        seed=42,
    )


# ---------------------------------------------------------------------------
# RED Cases: validation must fail closed
# ---------------------------------------------------------------------------


def test_manifest_rejects_wrong_units() -> None:
    """RED: Non-SI units (e.g. degrees, millimetres, inches) must be rejected."""
    manifest = BenchmarkManifest(
        manifest_id="test-manifest-wrong-units",
        input=_valid_manifest_input(),
        provenance=_valid_provenance(),
        units_and_frames={
            "position": "mm",  # Invalid! Must be SI 'm'
            "angle": "deg",  # Invalid! Must be SI 'rad'
            "torque": "N*m",
            "time": "s",
            "coordinate_frame": "world_z_up",
        },
        observation_type=ObservationType.MARKER_3D,
        known_truth={"type": "analytic_pendulum"},
        forces_type=ForceMeasurementType.MEASURED,
        license_and_privacy={"license": "Apache-2.0", "privacy": "synthetic_public"},
        native_capability={"synthetic_rig": NativeCapabilityStatus.IMPLEMENTED},
        metrics_and_thresholds=(
            NumericAcceptanceThreshold("drift_m", 0.05, "<=", "m"),
        ),
        phase_metrics=(),
    )
    with pytest.raises(UnitMismatchError, match="wrong units"):
        validate_benchmark_manifest(manifest)


def test_manifest_rejects_missing_provenance() -> None:
    """RED: Missing or unverified provenance block must fail closed."""
    manifest = BenchmarkManifest(
        manifest_id="test-manifest-no-prov",
        input=_valid_manifest_input(),
        provenance=None,  # Missing!
        units_and_frames={
            "position": "m",
            "angle": "rad",
            "torque": "N*m",
            "time": "s",
            "coordinate_frame": "world_z_up",
        },
        observation_type=ObservationType.MARKER_3D,
        known_truth={"type": "analytic_pendulum"},
        forces_type=ForceMeasurementType.MEASURED,
        license_and_privacy={"license": "Apache-2.0", "privacy": "synthetic_public"},
        native_capability={"synthetic_rig": NativeCapabilityStatus.IMPLEMENTED},
        metrics_and_thresholds=(),
        phase_metrics=(),
    )
    with pytest.raises(MissingProvenanceError, match="provenance"):
        validate_benchmark_manifest(manifest)


def test_manifest_rejects_force_derived_test_kinematics_as_qualified() -> None:
    """RED: Kinematics derived from unconstrained/inferred forces cannot be qualified."""
    manifest = BenchmarkManifest(
        manifest_id="test-manifest-force-derived",
        input=_valid_manifest_input(),
        provenance=_valid_provenance(),
        units_and_frames={
            "position": "m",
            "angle": "rad",
            "torque": "N*m",
            "time": "s",
            "coordinate_frame": "world_z_up",
        },
        observation_type=ObservationType.MARKER_3D,
        known_truth={"type": "synthetic_observations"},
        forces_type=ForceMeasurementType.INFERRED,  # Inferred, not measured!
        license_and_privacy={"license": "Apache-2.0", "privacy": "synthetic_public"},
        native_capability={
            "force_solver": NativeCapabilityStatus.QUALIFIED
        },  # Invalid!
        metrics_and_thresholds=(),
        phase_metrics=(),
    )
    with pytest.raises(
        ForceDerivedKinematicsError, match="force-derived test kinematics"
    ):
        validate_benchmark_manifest(manifest)


def test_manifest_rejects_skeleton_contact_output_as_qualified() -> None:
    """RED: Skeleton contact output without physics cannot be marked qualified."""
    manifest = BenchmarkManifest(
        manifest_id="test-manifest-skeleton-contact",
        input=_valid_manifest_input(),
        provenance=_valid_provenance(),
        units_and_frames={
            "position": "m",
            "angle": "rad",
            "torque": "N*m",
            "time": "s",
            "coordinate_frame": "world_z_up",
        },
        observation_type=ObservationType.KEYPOINT_2D,
        known_truth={"type": "skeleton_rig_fk"},  # Pure kinematic skeleton
        forces_type=ForceMeasurementType.UNCONSTRAINED,
        license_and_privacy={"license": "Apache-2.0", "privacy": "synthetic_public"},
        native_capability={
            "skeleton_contact": NativeCapabilityStatus.QUALIFIED
        },  # Invalid!
        metrics_and_thresholds=(),
        phase_metrics=(),
    )
    with pytest.raises(
        SkeletonContactQualificationError, match="skeleton contact output"
    ):
        validate_benchmark_manifest(manifest)


def test_manifest_distinguishes_method_existence_from_native_capability() -> None:
    """RED/GREEN: Method existence on disk != native capability qualification."""
    manifest = create_baseline_manifest()
    # Check that external/native engines not verified in the local environment
    # are truthfully labeled UNAVAILABLE or IMPLEMENTED, NEVER falsely QUALIFIED.
    assert (
        manifest.native_capability.get("simscape") != NativeCapabilityStatus.QUALIFIED
    )
    assert (
        manifest.native_capability.get("pinocchio") != NativeCapabilityStatus.QUALIFIED
    )
    # Synthetic rig is verified and implemented
    assert (
        manifest.native_capability.get("synthetic_rig")
        == NativeCapabilityStatus.IMPLEMENTED
    )


# ---------------------------------------------------------------------------
# GREEN Cases: deterministic fixtures and reproducible baseline
# ---------------------------------------------------------------------------


def test_analytic_pendulum_baseline_reproduces_with_recorded_seeds() -> None:
    """GREEN: Fixed-base pendulum fixture reproduces exactly across multiple runs."""
    fix1 = make_deterministic_pendulum_fixture(n_frames=60, dt=1 / 60.0, seed=123)
    fix2 = make_deterministic_pendulum_fixture(n_frames=60, dt=1 / 60.0, seed=123)

    assert fix1.trajectory.q.shape == (60, 1)
    assert fix1.trajectory.qdot.shape == (60, 1)
    np.testing.assert_allclose(fix1.trajectory.q, fix2.trajectory.q, atol=1e-12)
    np.testing.assert_allclose(fix1.trajectory.qdot, fix2.trajectory.qdot, atol=1e-12)

    # Energy conservation check (conservative pendulum, zero damping)
    energies = fix1.total_energy()
    assert len(energies) == 60
    np.testing.assert_allclose(energies, energies[0], atol=1e-5)


def test_underactuated_analytic_fixture_validates_constraints() -> None:
    """GREEN: Underactuated fixture enforces zero control on passive DOF."""
    fixture = make_underactuated_analytic_fixture(n_frames=60, dt=1 / 60.0, seed=42)
    assert fixture.num_dofs == 2
    assert fixture.actuated_mask == (True, False)  # 2nd DOF is passive
    # Passive joint control torque must be identically zero
    assert np.all(fixture.controls[:, 1] == 0.0)
    assert fixture.units["torque"] == "N*m"
    assert fixture.units["angle"] == "rad"


def test_native_stance_fixture_validates_contact_constraints() -> None:
    """GREEN: Native stance fixture enforces non-negative normal forces and friction cone."""
    fixture = make_native_stance_fixture(n_frames=30, dt=1 / 60.0, seed=42)
    assert fixture.forces_type == ForceMeasurementType.MEASURED
    # Normal force F_z must be non-negative
    f_z = fixture.contact_forces[:, 2]
    assert np.all(f_z >= 0.0)
    # Friction cone: sqrt(Fx^2 + Fy^2) <= mu * Fz
    mu = fixture.friction_coefficient
    tangential = np.sqrt(
        fixture.contact_forces[:, 0] ** 2 + fixture.contact_forces[:, 1] ** 2
    )
    assert np.all(tangential <= mu * f_z + 1e-6)


def test_capture_dataset_privacy_invariants() -> None:
    """GREEN: Private capture datasets do not leak absolute paths or personal IDs."""
    manifest = create_baseline_manifest(privacy="private_held")
    payload = manifest.to_dict()
    payload_str = json.dumps(payload)

    # Must NOT contain raw personal paths or sensitive patterns
    assert "C:\\Users\\" not in payload_str
    assert "/home/" not in payload_str
    assert "owner_" not in payload_str
    assert payload["license_and_privacy"]["privacy"] == "private_held"
    assert payload["license_and_privacy"]["personal_data_retained"] is False


def test_phase_specific_metrics_and_zero_denominator_policy() -> None:
    """GREEN: Phase-specific metrics compute safely under zero denominators."""
    t_ref = np.zeros((30, 3))
    t_est = np.zeros((30, 3))
    controls = np.zeros((30, 1))

    # All zeros test zero-denominator policy
    metrics = compute_phase_metrics(
        reference_trajectory=t_ref,
        estimated_trajectory=t_est,
        controls=controls,
        phases={"backswing": (0, 10), "downswing": (10, 20), "impact": (20, 30)},
        zero_denom_eps=1e-9,
    )
    assert len(metrics) == 3
    for m in metrics:
        assert np.isfinite(m.drift_magnitude_m)
        assert np.isfinite(m.control_magnitude_nm)
        assert np.isfinite(m.alignment_score)
        assert np.isfinite(m.cancellation_ratio)
        assert m.drift_magnitude_m == 0.0
        assert m.cancellation_ratio == 0.0  # Zero-denominator policy returns 0.0


def test_frozen_numeric_thresholds_and_scaling() -> None:
    """GREEN: Manifest defines frozen thresholds that can be evaluated against metrics."""
    manifest = create_baseline_manifest()
    assert len(manifest.metrics_and_thresholds) >= 4

    # Check key frozen metrics
    names = {t.metric_name: t for t in manifest.metrics_and_thresholds}
    assert "max_drift_m" in names
    assert "trajectory_alignment" in names
    assert "control_cancellation_ratio" in names
    assert "mean_marker_rms_m" in names

    assert names["max_drift_m"].threshold <= 0.05
    assert names["trajectory_alignment"].threshold >= 0.95
    assert names["control_cancellation_ratio"].threshold >= 0.80


def test_reviewer_can_run_baseline_from_manifest_truthfully() -> None:
    """GREEN: Reviewer executes baseline from manifest and sees truthful status."""
    manifest = create_baseline_manifest()
    result = run_baseline_from_manifest(manifest)

    assert isinstance(result, BenchmarkExecutionResult)
    assert result.manifest_id == manifest.manifest_id
    assert result.status_summary["implemented"] >= 1
    assert result.status_summary["unavailable"] >= 1
    assert result.improvement_claim_made is False  # Explicit truthfulness invariant
    assert result.estimator_rewritten is False


def test_manifest_serialization_round_trip(tmp_path: Path) -> None:
    """GREEN: Complete manifest JSON serialization round-trips bitwise."""
    manifest = create_baseline_manifest()
    json_path = tmp_path / "benchmark_manifest.json"
    manifest.save(json_path)

    loaded = BenchmarkManifest.load(json_path)
    assert loaded.manifest_id == manifest.manifest_id
    assert loaded.schema_version == manifest.schema_version
    assert loaded.observation_type == manifest.observation_type
    assert loaded.forces_type == manifest.forces_type
    assert len(loaded.metrics_and_thresholds) == len(manifest.metrics_and_thresholds)
    assert loaded.native_capability == manifest.native_capability
