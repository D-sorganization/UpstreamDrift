"""Tests for versioned baseline package manifest, export, import, and verification (TB-02 #10587)."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.shared.python.tour_baselines.metrics import (
    DynamicFeasibilityStatus,
    KinematicAccuracyStatus,
    ProductPromotionStatus,
    ScientificQualificationStatus,
    SolverStatus,
    TourFitMetrics,
)
from src.shared.python.tour_baselines.models import ModelClass
from src.shared.python.tour_baselines.packages import (
    BaselinePackage,
    BaselinePackageManifest,
    CoordinatePlane,
    FitMode,
    NativeReplayEvidence,
    export_baseline_package,
    import_baseline_package,
    load_legacy_receipt_as_unverified_package,
)
from src.shared.python.tour_baselines.qualification import (
    DoublePendulumPlanarProfile,
    evaluate_qualification,
)

pytestmark = pytest.mark.unit


def _sample_package(tmp_path: Path) -> BaselinePackage:
    profile = DoublePendulumPlanarProfile()
    metrics = TourFitMetrics(
        observed_valid_denominator=500,
        excluded_sample_count=10,
        coverage_fraction=0.98,
        whole_marker_rmse_m=0.035,
        p95_marker_error_m=0.045,
        max_marker_error_m=0.060,
        per_marker_rmse_m={"ClubHead": 0.040, "Grip": 0.025},
        per_phase_rmse_m={"downswing": 0.035, "impact": 0.040},
        impact_marker_error_m=0.040,
        endpoint_clubhead_rmse_m=0.040,
        optimizer_weighted_loss=0.15,
        original_frame_rmse_m=0.035,
        in_plane_rmse_m=0.030,
        out_of_plane_residual_m=0.012,
        landmarks_hash="b" * 64,
    )
    replay = NativeReplayEvidence(
        engine="mujoco",
        replay_command="python -m src.engines.physics_engines.mujoco.replay",
        replay_sha256="c" * 64,
        verified_reproduced=True,
    )
    verdict = evaluate_qualification(profile, metrics, replay)

    manifest = BaselinePackageManifest(
        schema_version="tour-baseline-package/1.0.0",
        package_id="967eac5b-2e78-4207-a99f-d57437296d70",
        capture="driver",
        capture_sha256="d" * 64,
        target_hash="e" * 64,
        model_id="double_pendulum_planar",
        model_class=ModelClass.DOUBLE_PENDULUM_PLANAR,
        backend="mujoco",
        backend_version="3.6.0",
        fit_mode=FitMode.TORQUE_DRIVEN,
        horizon="G3",
        coordinate_plane=CoordinatePlane.TWO_DIMENSIONAL_SWING_PLANE,
        measurement_map_version="tour-measurement-map/1.0.0",
        fixed_geometry={"arm_length_m": 0.60, "club_length_m": 1.15},
        q0=[0.0, 0.0],
        v0=[0.0, 0.0],
        controls_parameterization={"type": "spline_torques", "n_knots": 10},
        solver_config={"optimizer": "scipy_slsqp", "ftol": 1e-6},
        budgets={"max_iterations": 500, "wall_clock_limit_s": 60.0},
        candidate_ancestry=["f" * 64],
        runtime_hashes={"python": "3.12.0"},
        statuses={
            "solver": SolverStatus.CONVERGED.value,
            "kinematic": KinematicAccuracyStatus.ACCURATE.value,
            "dynamic": DynamicFeasibilityStatus.FEASIBLE.value,
            "scientific": verdict.scientific_status.value,
            "product": verdict.product_status.value,
        },
        metrics=metrics,
        qualification_gate_version=profile.gate_version,
        qualification_verdict=verdict.as_dict(),
        replay=replay,
        is_synthetic_test_data=False,
    )

    trajectories = {
        "time_s": np.linspace(0.0, 1.8, 100),
        "q": np.zeros((100, 2)),
        "marker_pred_m": np.zeros((100, 5, 3)),
    }

    return BaselinePackage(manifest=manifest, trajectories=trajectories)


def test_baseline_package_export_and_import_roundtrip(tmp_path: Path) -> None:
    """Baseline package can be exported to disk and imported with byte/hash verification."""
    package = _sample_package(tmp_path)
    export_dir = tmp_path / "exported_package"
    exported_path = export_baseline_package(package, export_dir)

    assert exported_path.exists()
    assert (export_dir / "manifest.json").exists()
    assert (export_dir / "trajectories.npz").exists()

    imported = import_baseline_package(export_dir)
    assert imported.manifest.package_id == package.manifest.package_id
    assert imported.manifest.capture == "driver"
    assert imported.manifest.model_id == "double_pendulum_planar"
    assert imported.manifest.fit_mode == FitMode.TORQUE_DRIVEN
    assert np.allclose(imported.trajectories["time_s"], package.trajectories["time_s"])


def test_tampered_trajectory_fails_import(tmp_path: Path) -> None:
    """Tampering with trajectory data invalidates package checksum and fails import."""
    package = _sample_package(tmp_path)
    export_dir = tmp_path / "tampered_pkg"
    export_baseline_package(package, export_dir)

    # Tamper with the npz file
    npz_path = export_dir / "trajectories.npz"
    data = npz_path.read_bytes()
    # Mutate a byte in the array data
    tampered_data = bytearray(data)
    tampered_data[-10] ^= 0xFF
    npz_path.write_bytes(tampered_data)

    with pytest.raises(ValueError, match="Hash mismatch|Checksum verification failed"):
        import_baseline_package(export_dir)


def test_legacy_receipt_backward_compatibility(tmp_path: Path) -> None:
    """Legacy receipts remain readable as unverified historical packages."""
    legacy_json = {
        "backend": "mujoco",
        "engine": "mujoco",
        "base_spec_sha256": "1" * 64,
        "spec_sha256": "2" * 64,
        "spec_file": "model.xml",
        "hipcal_spec_file": "hipcal.xml",
        "recalibrate_upper": False,
        "candidate_sha256": "3" * 64,
        "capture_sha256": "4" * 64,
        "capture": "driver",
    }
    legacy_path = tmp_path / "legacy_receipt.json"
    legacy_path.write_text(json.dumps(legacy_json), encoding="utf-8")

    pkg = load_legacy_receipt_as_unverified_package(legacy_path)
    assert pkg.manifest.capture == "driver"
    assert (
        pkg.manifest.statuses["scientific"]
        == ScientificQualificationStatus.HISTORICAL.value
    )
    assert pkg.manifest.statuses["product"] == ProductPromotionStatus.UNPROMOTED.value
    assert pkg.manifest.replay.verified_reproduced is False
