"""Versioned baseline package manifest, serialization, and clean-machine verification (TB-02 #10587).

Links capture, model class/topology, backend/provider pin, fit mode, horizon, frame/plane,
marker map, fixed geometry/inertia, initial state, controls, solver config, seed, budgets,
candidate ancestry, runtime and file hashes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.tour_baselines.metrics import (
    DynamicFeasibilityStatus,
    KinematicAccuracyStatus,
    ProductPromotionStatus,
    ScientificQualificationStatus,
    SolverStatus,
    TourFitMetrics,
)
from src.shared.python.tour_baselines.models import FitMode, ModelClass


class CoordinatePlane(str, Enum):
    """Geometric reference frame/plane."""

    THREE_DIMENSIONAL = "three_dimensional"
    TWO_DIMENSIONAL_SWING_PLANE = "two_dimensional_swing_plane"
    TWO_DIMENSIONAL_SAGITTAL = "two_dimensional_sagittal"


@dataclass(frozen=True)
class NativeReplayEvidence:
    """Evidence verifying native engine reproduction under identical inputs."""

    engine: str
    replay_command: str
    replay_sha256: str
    verified_reproduced: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "replay_command": self.replay_command,
            "replay_sha256": self.replay_sha256,
            "verified_reproduced": self.verified_reproduced,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NativeReplayEvidence:
        return cls(
            engine=str(data.get("engine", "")),
            replay_command=str(data.get("replay_command", "")),
            replay_sha256=str(data.get("replay_sha256", "")),
            verified_reproduced=bool(data.get("verified_reproduced", False)),
        )


@dataclass(frozen=True)
class BaselinePackageManifest:
    """Authoritative manifest for a versioned tour baseline package."""

    schema_version: str
    package_id: str
    capture: str
    capture_sha256: str
    target_hash: str
    model_id: str
    model_class: ModelClass
    backend: str
    backend_version: str
    fit_mode: FitMode
    horizon: str
    coordinate_plane: CoordinatePlane
    measurement_map_version: str
    fixed_geometry: dict[str, Any]
    q0: list[float]
    v0: list[float]
    controls_parameterization: dict[str, Any]
    solver_config: dict[str, Any]
    budgets: dict[str, Any]
    candidate_ancestry: list[str]
    runtime_hashes: dict[str, str]
    statuses: dict[str, str]
    metrics: TourFitMetrics
    qualification_gate_version: str
    qualification_verdict: dict[str, Any]
    replay: NativeReplayEvidence
    file_hashes: dict[str, str] = field(default_factory=dict)
    seed: int | None = None
    is_synthetic_test_data: bool = False

    def as_dict(self) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "package_id": self.package_id,
            "capture": self.capture,
            "capture_sha256": self.capture_sha256,
            "target_hash": self.target_hash,
            "model_id": self.model_id,
            "model_class": self.model_class.value,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "fit_mode": self.fit_mode.value,
            "horizon": self.horizon,
            "coordinate_plane": self.coordinate_plane.value,
            "measurement_map_version": self.measurement_map_version,
            "fixed_geometry": dict(self.fixed_geometry),
            "q0": list(self.q0),
            "v0": list(self.v0),
            "controls_parameterization": dict(self.controls_parameterization),
            "solver_config": dict(self.solver_config),
            "budgets": dict(self.budgets),
            "candidate_ancestry": list(self.candidate_ancestry),
            "runtime_hashes": dict(self.runtime_hashes),
            "statuses": dict(self.statuses),
            "metrics": self.metrics.as_dict(),
            "qualification_gate_version": self.qualification_gate_version,
            "qualification_verdict": dict(self.qualification_verdict),
            "replay": self.replay.as_dict(),
            "file_hashes": dict(self.file_hashes),
            "seed": self.seed,
            "is_synthetic_test_data": self.is_synthetic_test_data,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselinePackageManifest:
        metrics_dict = data["metrics"]
        metrics = TourFitMetrics(
            observed_valid_denominator=int(metrics_dict["observed_valid_denominator"]),
            excluded_sample_count=int(metrics_dict["excluded_sample_count"]),
            coverage_fraction=float(metrics_dict["coverage_fraction"]),
            whole_marker_rmse_m=float(metrics_dict["whole_marker_rmse_m"]),
            p95_marker_error_m=float(metrics_dict["p95_marker_error_m"]),
            max_marker_error_m=float(metrics_dict["max_marker_error_m"]),
            per_marker_rmse_m=dict(metrics_dict["per_marker_rmse_m"]),
            per_phase_rmse_m=dict(metrics_dict["per_phase_rmse_m"]),
            impact_marker_error_m=float(metrics_dict["impact_marker_error_m"]),
            endpoint_clubhead_rmse_m=float(metrics_dict["endpoint_clubhead_rmse_m"]),
            optimizer_weighted_loss=float(metrics_dict["optimizer_weighted_loss"]),
            original_frame_rmse_m=float(metrics_dict["original_frame_rmse_m"]),
            in_plane_rmse_m=float(metrics_dict["in_plane_rmse_m"]),
            out_of_plane_residual_m=float(metrics_dict["out_of_plane_residual_m"]),
            landmarks_hash=str(metrics_dict["landmarks_hash"]),
            clubhead_speed_error_m_s=metrics_dict.get("clubhead_speed_error_m_s"),
        )

        return cls(
            schema_version=str(data["schema_version"]),
            package_id=str(data["package_id"]),
            capture=str(data["capture"]),
            capture_sha256=str(data["capture_sha256"]),
            target_hash=str(data["target_hash"]),
            model_id=str(data["model_id"]),
            model_class=ModelClass(data["model_class"]),
            backend=str(data["backend"]),
            backend_version=str(data["backend_version"]),
            fit_mode=FitMode(data["fit_mode"]),
            horizon=str(data["horizon"]),
            coordinate_plane=CoordinatePlane(data["coordinate_plane"]),
            measurement_map_version=str(data["measurement_map_version"]),
            fixed_geometry=dict(data["fixed_geometry"]),
            q0=list(data["q0"]),
            v0=list(data["v0"]),
            controls_parameterization=dict(data["controls_parameterization"]),
            solver_config=dict(data["solver_config"]),
            budgets=dict(data["budgets"]),
            candidate_ancestry=list(data["candidate_ancestry"]),
            runtime_hashes=dict(data["runtime_hashes"]),
            statuses=dict(data["statuses"]),
            metrics=metrics,
            qualification_gate_version=str(data["qualification_gate_version"]),
            qualification_verdict=dict(data["qualification_verdict"]),
            replay=NativeReplayEvidence.from_dict(data["replay"]),
            file_hashes=dict(data.get("file_hashes", {})),
            seed=data.get("seed"),
            is_synthetic_test_data=bool(data.get("is_synthetic_test_data", False)),
        )


@dataclass(frozen=True)
class BaselinePackage:
    """Self-contained, hash-verified baseline package."""

    manifest: BaselinePackageManifest
    trajectories: dict[str, NDArray[np.float64]]


def export_baseline_package(package: BaselinePackage, target_dir: Path) -> Path:
    """Export baseline package manifest and arrays to target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    npz_path = target_dir / "trajectories.npz"
    np.savez_compressed(npz_path, allow_pickle=True, **package.trajectories)

    hasher = hashlib.sha256()
    hasher.update(npz_path.read_bytes())
    npz_hash = hasher.hexdigest()

    manifest_dict = package.manifest.as_dict()
    manifest_dict["file_hashes"]["trajectories.npz"] = npz_hash

    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")
    return target_dir


def import_baseline_package(package_dir: Path) -> BaselinePackage:
    """Import and hash-verify a baseline package from directory."""
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest in {package_dir}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = BaselinePackageManifest.from_dict(manifest_data)

    npz_path = package_dir / "trajectories.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing trajectories.npz in {package_dir}")

    npz_bytes = npz_path.read_bytes()
    observed_hash = hashlib.sha256(npz_bytes).hexdigest()
    expected_hash = manifest.file_hashes.get("trajectories.npz", "")
    if observed_hash != expected_hash:
        raise ValueError(
            f"Checksum verification failed for trajectories.npz: "
            f"expected {expected_hash}, got {observed_hash}"
        )

    with np.load(npz_path) as loaded:
        trajectories = {key: loaded[key] for key in loaded.files}

    return BaselinePackage(manifest=manifest, trajectories=trajectories)


def load_legacy_receipt_as_unverified_package(
    legacy_path: Path,
) -> BaselinePackage:
    """Convert historical execution receipt into an unverified baseline package."""
    content = json.loads(legacy_path.read_text(encoding="utf-8"))
    capture = str(content.get("capture", "driver"))
    capture_hash = str(content.get("capture_sha256", "0" * 64))

    metrics = TourFitMetrics(
        observed_valid_denominator=1,
        excluded_sample_count=0,
        coverage_fraction=1.0,
        whole_marker_rmse_m=0.0,
        p95_marker_error_m=0.0,
        max_marker_error_m=0.0,
        per_marker_rmse_m={},
        per_phase_rmse_m={},
        impact_marker_error_m=0.0,
        endpoint_clubhead_rmse_m=0.0,
        optimizer_weighted_loss=0.0,
        original_frame_rmse_m=0.0,
        in_plane_rmse_m=0.0,
        out_of_plane_residual_m=0.0,
        landmarks_hash="0" * 64,
    )

    manifest = BaselinePackageManifest(
        schema_version="tour-baseline-package/1.0.0",
        package_id=str(uuid.uuid4()),
        capture=capture,
        capture_sha256=capture_hash,
        target_hash="0" * 64,
        model_id=str(content.get("model_id", "historical_legacy")),
        model_class=ModelClass.FULL_BODY_MECH,
        backend=str(content.get("backend", "unknown")),
        backend_version="historical",
        fit_mode=FitMode.KINEMATIC,
        horizon=str(content.get("horizon", "historical")),
        coordinate_plane=CoordinatePlane.THREE_DIMENSIONAL,
        measurement_map_version="tour-measurement-map/1.0.0",
        fixed_geometry={},
        q0=[],
        v0=[],
        controls_parameterization={},
        solver_config={},
        budgets={},
        candidate_ancestry=[],
        runtime_hashes={},
        statuses={
            "solver": SolverStatus.CONVERGED.value,
            "kinematic": KinematicAccuracyStatus.UNTESTED.value,
            "dynamic": DynamicFeasibilityStatus.NOT_APPLICABLE.value,
            "scientific": ScientificQualificationStatus.HISTORICAL.value,
            "product": ProductPromotionStatus.UNPROMOTED.value,
        },
        metrics=metrics,
        qualification_gate_version="legacy",
        qualification_verdict={
            "is_qualified": False,
            "status": "historical",
            "reason": "Legacy unverified receipt",
        },
        replay=NativeReplayEvidence(
            engine=str(content.get("engine", "unknown")),
            replay_command="",
            replay_sha256="",
            verified_reproduced=False,
        ),
        is_synthetic_test_data=False,
    )
    return BaselinePackage(manifest=manifest, trajectories={})
