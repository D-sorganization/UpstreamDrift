"""Baseline packages, identities, status bundles, and clean-machine persistence (TB-02 #10587).

Extends the existing result and portable package contracts (#10379, #10334):
1. Separates solver convergence, kinematic accuracy, dynamic feasibility, scientific qualification,
   and product promotion into orthogonal statuses.
2. Complete manifest with missing native replay cannot qualify scientifically.
3. BaselineIdentity links capture, model class/topology, backend pin, fit mode, horizon,
   coordinate frame, measurement map, fixed geometry/inertia, controls, solver config, seed, budgets, and hashes.
4. Hash-verified export and clean-machine import with checksum validation (allow_pickle=False).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.tour_baselines.fit_metrics import PhysicalFitMetrics
from src.shared.python.tour_baselines.models import (
    BackendType,
    FitMode,
    ModelTopology,
)

logger = logging.getLogger(__name__)

BASELINE_SCHEMA_VERSION = "tour-baseline-package/1.0.0"
MANIFEST_KEY = "manifest_json"


class SolverConvergenceStatus(str, Enum):
    """Numerical optimization outcome."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    INFEASIBLE = "infeasible"
    DIVERGED = "diverged"
    UNSOLVED = "unsolved"


class KinematicAccuracyStatus(str, Enum):
    """Marker tracking accuracy against declared threshold."""

    WITHIN_TOLERANCE = "within_tolerance"
    EXCEEDS_THRESHOLD = "exceeds_threshold"
    UNEVALUATED = "unevaluated"


class DynamicFeasibilityStatus(str, Enum):
    """Physical consistency (forces, penetrations, closure, actuator bounds)."""

    PHYSICALLY_FEASIBLE = "physically_feasible"
    INFEASIBLE = "infeasible"
    KINEMATIC_ONLY = "kinematic_only"
    UNEVALUATED = "unevaluated"


class ScientificQualificationStatus(str, Enum):
    """Authoritative scientific gate decision."""

    QUALIFIED = "qualified"
    DISQUALIFIED = "disqualified"
    UNVERIFIED = "unverified"
    PENDING_REVIEW = "pending_review"


class ProductPromotionStatus(str, Enum):
    """Release readiness in customer-facing application shells."""

    PROMOTED = "promoted"
    EXPLORATORY = "exploratory"
    REJECTED = "rejected"
    DEMOTED = "demoted"


@dataclass(frozen=True)
class StatusBundle:
    """Explicit separation of five orthogonal statuses."""

    solver_convergence: SolverConvergenceStatus
    kinematic_accuracy: KinematicAccuracyStatus
    dynamic_feasibility: DynamicFeasibilityStatus
    scientific_qualification: ScientificQualificationStatus
    product_promotion: ProductPromotionStatus
    has_native_replay: bool = False

    def __post_init__(self) -> None:
        if (
            not self.has_native_replay
            and self.scientific_qualification == ScientificQualificationStatus.QUALIFIED
        ):
            raise ValueError(
                "Baseline candidate cannot be QUALIFIED without native replay evidence."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver_convergence": self.solver_convergence.value,
            "kinematic_accuracy": self.kinematic_accuracy.value,
            "dynamic_feasibility": self.dynamic_feasibility.value,
            "scientific_qualification": self.scientific_qualification.value,
            "product_promotion": self.product_promotion.value,
            "has_native_replay": self.has_native_replay,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StatusBundle:
        return cls(
            solver_convergence=SolverConvergenceStatus(data["solver_convergence"]),
            kinematic_accuracy=KinematicAccuracyStatus(data["kinematic_accuracy"]),
            dynamic_feasibility=DynamicFeasibilityStatus(data["dynamic_feasibility"]),
            scientific_qualification=ScientificQualificationStatus(
                data["scientific_qualification"]
            ),
            product_promotion=ProductPromotionStatus(data["product_promotion"]),
            has_native_replay=bool(data.get("has_native_replay", False)),
        )


@dataclass(frozen=True)
class BaselineIdentity:
    """Immutable, typed identity and manifest parameters of a baseline fit."""

    model_id: str
    topology: ModelTopology
    backend: BackendType
    provider_pin: str
    fit_mode: FitMode
    capture: str
    capture_sha256: str
    horizon: str
    frame_convention: str = "z_up_y_forward"
    plane_convention: str = "transverse_sagittal_frontal"
    measurement_map_version: str = "tour-measurement-map/1.0.0"
    fixed_geometry_hash: str = ""
    fixed_inertia_hash: str = ""
    q0_hash: str | None = None
    v0_hash: str | None = None
    controls_hash: str | None = None
    solver_name: str = ""
    solver_config: dict[str, Any] = field(default_factory=dict)
    integrator: str = ""
    seed: int | None = None
    wall_clock_budget_s: float | None = None
    max_evaluations_budget: int | None = None
    candidate_ancestry: tuple[str, ...] = ()
    runtime_hashes: dict[str, str] = field(default_factory=dict)
    file_hashes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if self.capture not in ("driver", "iron"):
            raise ValueError(
                f"capture must be 'driver' or 'iron', got {self.capture!r}"
            )
        if self.horizon not in ("G1", "G2", "G3"):
            raise ValueError(f"horizon must be G1, G2, or G3, got {self.horizon!r}")

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 fingerprint for this baseline identity."""
        payload = {
            "model_id": self.model_id,
            "topology": self.topology.value,
            "backend": self.backend.value,
            "provider_pin": self.provider_pin,
            "fit_mode": self.fit_mode.value,
            "capture": self.capture,
            "capture_sha256": self.capture_sha256,
            "horizon": self.horizon,
            "frame_convention": self.frame_convention,
            "plane_convention": self.plane_convention,
            "measurement_map_version": self.measurement_map_version,
            "fixed_geometry_hash": self.fixed_geometry_hash,
            "fixed_inertia_hash": self.fixed_inertia_hash,
            "q0_hash": self.q0_hash,
            "v0_hash": self.v0_hash,
            "controls_hash": self.controls_hash,
            "solver_name": self.solver_name,
            "solver_config": self.solver_config,
            "integrator": self.integrator,
            "seed": self.seed,
            "candidate_ancestry": list(self.candidate_ancestry),
        }
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "topology": self.topology.value,
            "backend": self.backend.value,
            "provider_pin": self.provider_pin,
            "fit_mode": self.fit_mode.value,
            "capture": self.capture,
            "capture_sha256": self.capture_sha256,
            "horizon": self.horizon,
            "frame_convention": self.frame_convention,
            "plane_convention": self.plane_convention,
            "measurement_map_version": self.measurement_map_version,
            "fixed_geometry_hash": self.fixed_geometry_hash,
            "fixed_inertia_hash": self.fixed_inertia_hash,
            "q0_hash": self.q0_hash,
            "v0_hash": self.v0_hash,
            "controls_hash": self.controls_hash,
            "solver_name": self.solver_name,
            "solver_config": dict(self.solver_config),
            "integrator": self.integrator,
            "seed": self.seed,
            "wall_clock_budget_s": self.wall_clock_budget_s,
            "max_evaluations_budget": self.max_evaluations_budget,
            "candidate_ancestry": list(self.candidate_ancestry),
            "runtime_hashes": dict(self.runtime_hashes),
            "file_hashes": dict(self.file_hashes),
            "identity_hash": self.compute_hash(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BaselineIdentity:
        d = dict(data)
        d.pop("identity_hash", None)
        return cls(
            model_id=d["model_id"],
            topology=ModelTopology(d["topology"]),
            backend=BackendType(d["backend"]),
            provider_pin=d["provider_pin"],
            fit_mode=FitMode(d["fit_mode"]),
            capture=d["capture"],
            capture_sha256=d["capture_sha256"],
            horizon=d["horizon"],
            frame_convention=d.get("frame_convention", "z_up_y_forward"),
            plane_convention=d.get("plane_convention", "transverse_sagittal_frontal"),
            measurement_map_version=d.get(
                "measurement_map_version", "tour-measurement-map/1.0.0"
            ),
            fixed_geometry_hash=d.get("fixed_geometry_hash", ""),
            fixed_inertia_hash=d.get("fixed_inertia_hash", ""),
            q0_hash=d.get("q0_hash"),
            v0_hash=d.get("v0_hash"),
            controls_hash=d.get("controls_hash"),
            solver_name=d.get("solver_name", ""),
            solver_config=d.get("solver_config", {}),
            integrator=d.get("integrator", ""),
            seed=d.get("seed"),
            wall_clock_budget_s=d.get("wall_clock_budget_s"),
            max_evaluations_budget=d.get("max_evaluations_budget"),
            candidate_ancestry=tuple(d.get("candidate_ancestry", ())),
            runtime_hashes=d.get("runtime_hashes", {}),
            file_hashes=d.get("file_hashes", {}),
        )


@dataclass(frozen=True)
class BaselinePackage:
    """Complete, self-contained portable baseline package."""

    identity: BaselineIdentity
    statuses: StatusBundle
    metrics: PhysicalFitMetrics
    replay_command: str
    trajectories: dict[str, np.ndarray] = field(default_factory=dict)
    coefficients: np.ndarray | None = None
    reports: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    is_synthetic: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        st = self.statuses
        if (
            self.is_synthetic
            and st.product_promotion == ProductPromotionStatus.PROMOTED
        ):
            raise ValueError(
                "Synthetic test package cannot have product_promotion set to PROMOTED."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize package manifest to a dictionary without embedded binary arrays."""
        return {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "identity": self.identity.to_dict(),
            "statuses": self.statuses.to_dict(),
            "metrics": self.metrics.to_dict(),
            "replay_command": self.replay_command,
            "reports": dict(self.reports),
            "artifacts": dict(self.artifacts),
            "is_synthetic": self.is_synthetic,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize package manifest deterministically to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"


def _compute_array_hash(arr: np.ndarray) -> str:
    """Compute SHA-256 of array's contiguous byte representation."""
    contiguous = np.ascontiguousarray(arr)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def export_baseline_package(package: BaselinePackage, output_path: Path | str) -> Path:
    """Export a BaselinePackage to a self-contained .npz archive."""
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    ident_dict = package.identity.to_dict()
    status_dict = package.statuses.to_dict()
    metrics_dict = package.metrics.to_dict()

    array_checksums: dict[str, str] = {}
    arrays_to_save: dict[str, np.ndarray] = {}

    for name, arr in package.trajectories.items():
        arrays_to_save[f"traj_{name}"] = arr
        array_checksums[f"traj_{name}"] = _compute_array_hash(arr)

    if package.coefficients is not None:
        arrays_to_save["coefficients"] = package.coefficients
        array_checksums["coefficients"] = _compute_array_hash(package.coefficients)

    manifest = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "identity": ident_dict,
        "statuses": status_dict,
        "metrics": metrics_dict,
        "replay_command": package.replay_command,
        "reports": package.reports,
        "artifacts": package.artifacts,
        "is_synthetic": package.is_synthetic,
        "notes": package.notes,
        "array_checksums": array_checksums,
    }

    manifest_json = json.dumps(manifest, indent=2, sort_keys=True)
    arrays_to_save[MANIFEST_KEY] = np.array(manifest_json)

    np.savez(out, **arrays_to_save)  # type: ignore[arg-type]
    logger.debug("Exported BaselinePackage -> %s", out)
    return out


def import_baseline_package(input_path: Path | str) -> BaselinePackage:
    """Import and validate a BaselinePackage from an archive file."""
    src = Path(input_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Baseline package archive does not exist: {src}")

    with np.load(src, allow_pickle=False) as data:
        if MANIFEST_KEY not in data:
            raise ValueError(f"Archive {src} is missing required {MANIFEST_KEY} header")

        manifest_str = str(data[MANIFEST_KEY])
        manifest = json.loads(manifest_str)

        schema = manifest.get("schema_version")
        if schema != BASELINE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported baseline schema version: {schema!r} (expected {BASELINE_SCHEMA_VERSION!r})"
            )

        array_checksums: dict[str, str] = manifest.get("array_checksums", {})

        trajectories: dict[str, np.ndarray] = {}
        coefficients: np.ndarray | None = None

        for key in data.files:
            if key == MANIFEST_KEY:
                continue
            if key not in array_checksums:
                raise ValueError(
                    f"Unverified array '{key}' present in archive without checksum"
                )
            arr = data[key]
            expected_hash = array_checksums[key]
            actual_hash = _compute_array_hash(arr)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Checksum mismatch for array '{key}': expected {expected_hash}, got {actual_hash}"
                )

            if key.startswith("traj_"):
                traj_name = key[len("traj_") :]
                trajectories[traj_name] = arr
            elif key == "coefficients":
                coefficients = arr

        ident = BaselineIdentity.from_dict(manifest["identity"])
        statuses = StatusBundle.from_dict(manifest["statuses"])
        metrics = PhysicalFitMetrics.from_dict(manifest["metrics"])

        return BaselinePackage(
            identity=ident,
            statuses=statuses,
            metrics=metrics,
            replay_command=manifest.get("replay_command", ""),
            trajectories=trajectories,
            coefficients=coefficients,
            reports=manifest.get("reports", {}),
            artifacts=manifest.get("artifacts", {}),
            is_synthetic=bool(manifest.get("is_synthetic", False)),
            notes=manifest.get("notes", ""),
        )
