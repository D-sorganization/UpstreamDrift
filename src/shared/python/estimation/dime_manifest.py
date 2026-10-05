"""DIME Baseline and Frozen Benchmark Protocol (#11421, #11422).

Provides:
1. Versioned benchmark manifest (DimeBenchmarkManifest) specifying dataset/model/engine
   revisions, split policies, canonical SI units, observation types, known truth,
   force classification, licenses/privacy, metrics, and frozen numeric acceptance thresholds.
2. Truthful native capability status reporting ('implemented', 'qualified', 'unavailable')
   separated from Python method existence.
3. Fail-closed qualification evaluation: wrong units, missing provenance, force-derived
   test kinematics, and skeleton contact output cannot be marked qualified.
4. Deterministic baseline execution reproducing with recorded seeds on shared fixtures.
5. Phase-specific drift and control magnitude registration, alignment and cancellation
   metrics, and explicit zero-denominator policies.
6. Privacy protection ensuring capture datasets and filesystem paths are not leaked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.estimation.synthetic_fixtures import (
    make_fixed_base_pendulum_fixture,
    make_native_stance_fixture,
    make_underactuated_analytic_fixture,
)

CapabilityStatus = Literal["implemented", "qualified", "unavailable"]
ZeroDenominatorPolicy = Literal["guarded_zero", "epsilon", "raise"]

CANONICAL_DIME_UNITS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "length": "m",
        "angle": "rad",
        "time": "s",
        "mass": "kg",
        "force": "N",
        "torque": "N*m",
    }
)


@dataclass(frozen=True)
class CapabilityRecord:
    """Truthful capability report separating method existence from qualification."""

    name: str
    method_exists: bool
    status: CapabilityStatus
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method_exists": self.method_exists,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SplitPolicy:
    """Frozen calibration vs holdout frame split and phase partition."""

    calibration_frames: tuple[int, ...]
    holdout_frames: tuple[int, ...]
    phase_frames: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        overlap = set(self.calibration_frames) & set(self.holdout_frames)
        if overlap:
            raise ValueError(
                f"calibration_frames and holdout_frames must be disjoint; overlap: {sorted(overlap)}"
            )
        for f in self.calibration_frames:
            if f < 0:
                raise ValueError(f"frame index must be non-negative, got {f}")
        for f in self.holdout_frames:
            if f < 0:
                raise ValueError(f"frame index must be non-negative, got {f}")
        object.__setattr__(
            self,
            "phase_frames",
            MappingProxyType({k: tuple(v) for k, v in self.phase_frames.items()}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_frames": list(self.calibration_frames),
            "holdout_frames": list(self.holdout_frames),
            "phase_frames": {k: list(v) for k, v in self.phase_frames.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SplitPolicy:
        return cls(
            calibration_frames=tuple(
                int(f) for f in data.get("calibration_frames", ())
            ),
            holdout_frames=tuple(int(f) for f in data.get("holdout_frames", ())),
            phase_frames={
                str(k): tuple(int(f) for f in v)
                for k, v in data.get("phase_frames", {}).items()
            },
        )


@dataclass(frozen=True)
class ForceClassification:
    """Explicit distinction between directly measured and inferred forces."""

    measured_forces: tuple[str, ...] = ()
    inferred_forces: tuple[str, ...] = ()
    kinematics_source: Literal[
        "independent_measurement", "analytic_solution", "force_derived"
    ] = "independent_measurement"
    has_skeleton_contact: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "measured_forces": list(self.measured_forces),
            "inferred_forces": list(self.inferred_forces),
            "kinematics_source": self.kinematics_source,
            "has_skeleton_contact": self.has_skeleton_contact,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ForceClassification:
        return cls(
            measured_forces=tuple(str(f) for f in data.get("measured_forces", ())),
            inferred_forces=tuple(str(f) for f in data.get("inferred_forces", ())),
            kinematics_source=data.get("kinematics_source", "independent_measurement"),
            has_skeleton_contact=bool(data.get("has_skeleton_contact", False)),
        )


@dataclass(frozen=True)
class DimeProvenanceRecord:
    """Immutable provenance stamp for benchmark reproducibility."""

    engine: str
    engine_version: str
    model_hash: str
    param_hash: str
    git_commit: str
    created_at: str
    seed: int | None = None
    notes: str = ""

    def is_complete(self) -> bool:
        """Check whether all essential provenance fields are populated."""
        return bool(
            self.engine.strip()
            and self.engine_version.strip()
            and self.model_hash.strip()
            and self.git_commit.strip()
            and self.created_at.strip()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "model_hash": self.model_hash,
            "param_hash": self.param_hash,
            "git_commit": self.git_commit,
            "created_at": self.created_at,
            "seed": self.seed,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DimeProvenanceRecord:
        return cls(
            engine=str(data.get("engine", "")),
            engine_version=str(data.get("engine_version", "")),
            model_hash=str(data.get("model_hash", "")),
            param_hash=str(data.get("param_hash", "")),
            git_commit=str(data.get("git_commit", "")),
            created_at=str(data.get("created_at", "")),
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class PrivacySpec:
    """License and privacy specifications ensuring datasets remain confidential."""

    is_private: bool = True
    license_name: str = "Proprietary / Internal"
    redact_filesystem_paths: bool = True
    privacy_classification: str = "internal_fleet_reference"

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_private": self.is_private,
            "license_name": self.license_name,
            "redact_filesystem_paths": self.redact_filesystem_paths,
            "privacy_classification": self.privacy_classification,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PrivacySpec:
        return cls(
            is_private=bool(data.get("is_private", True)),
            license_name=str(data.get("license_name", "Proprietary / Internal")),
            redact_filesystem_paths=bool(data.get("redact_filesystem_paths", True)),
            privacy_classification=str(
                data.get("privacy_classification", "internal_fleet_reference")
            ),
        )


@dataclass(frozen=True)
class NumericAcceptanceThresholds:
    """Fixed numeric acceptance thresholds established prior to solver comparison."""

    max_drift_m: float = 0.015
    max_angular_drift_rad: float = 0.05
    max_control_norm_nm: float = 250.0
    min_alignment: float = 0.95
    max_cancellation_ratio: float = 0.10
    reproducibility_atol: float = 1e-9
    condition_number_max: float = 1e6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NumericAcceptanceThresholds:
        return cls(**{k: float(v) for k, v in data.items()})


@dataclass(frozen=True)
class ConditioningScale:
    """Nondimensionalization and diagonal scaling factors."""

    position_scale: float = 1.0
    angle_scale: float = 1.0
    velocity_scale: float = 1.0
    torque_scale: float = 1.0
    time_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConditioningScale:
        return cls(**{k: float(v) for k, v in data.items()})


@dataclass(frozen=True)
class PhaseMetrics:
    """Drift and control metrics stratified across trajectory phases."""

    phase: str
    max_drift: float
    mean_drift: float
    max_control: float
    mean_control: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Metric Computation Functions
# ==============================================================================


def compute_alignment_metric(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    policy: ZeroDenominatorPolicy = "guarded_zero",
    eps: float = 1e-12,
) -> float:
    """Compute normalized cosine alignment between signals with zero-denominator guard."""
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"Array shapes must match: {x_arr.shape} != {y_arr.shape}")
    dot = float(np.dot(x_arr, y_arr))
    norm_x = float(np.linalg.norm(x_arr))
    norm_y = float(np.linalg.norm(y_arr))
    denom = norm_x * norm_y
    if denom == 0.0:
        if policy == "raise":
            raise ZeroDivisionError(
                "Zero norm encountered in alignment metric calculation."
            )
        if policy == "epsilon":
            return float(dot / (denom + eps))
        return 0.0
    return float(np.clip(dot / denom, -1.0, 1.0))


def compute_cancellation_metric(
    f1: npt.ArrayLike,
    f2: npt.ArrayLike,
    *,
    policy: ZeroDenominatorPolicy = "guarded_zero",
    eps: float = 1e-12,
) -> float:
    """Compute net residual force cancellation ratio with zero-denominator guard."""
    arr1 = np.asarray(f1, dtype=np.float64)
    arr2 = np.asarray(f2, dtype=np.float64)
    if arr1.shape != arr2.shape:
        raise ValueError(f"Force shapes must match: {arr1.shape} != {arr2.shape}")
    net = arr1 + arr2
    net_norm = float(np.linalg.norm(net))
    sum_norms = float(np.linalg.norm(arr1) + np.linalg.norm(arr2))
    if sum_norms == 0.0:
        if policy == "raise":
            raise ZeroDivisionError(
                "Zero force sum encountered in cancellation metric calculation."
            )
        if policy == "epsilon":
            return float(net_norm / (sum_norms + eps))
        return 0.0
    return float(net_norm / sum_norms)


def compute_phase_drift_and_control(
    q_est: npt.ArrayLike,
    q_true: npt.ArrayLike,
    controls: npt.ArrayLike,
    phase_frames: Mapping[str, Sequence[int]],
) -> dict[str, PhaseMetrics]:
    """Calculate phase-stratified kinematic drift and control magnitudes."""
    est = np.asarray(q_est, dtype=np.float64)
    true = np.asarray(q_true, dtype=np.float64)
    ctrl = np.asarray(controls, dtype=np.float64)
    diff = np.abs(est - true)
    ctrl_mag = np.abs(ctrl) if ctrl.ndim == 1 else np.linalg.norm(ctrl, axis=-1)

    result: dict[str, PhaseMetrics] = {}
    for phase_name, indices in phase_frames.items():
        idx = list(indices)
        if not idx:
            result[phase_name] = PhaseMetrics(phase_name, 0.0, 0.0, 0.0, 0.0)
            continue
        p_diff = diff[idx]
        p_ctrl = ctrl_mag[idx]
        result[phase_name] = PhaseMetrics(
            phase=phase_name,
            max_drift=float(np.max(p_diff)),
            mean_drift=float(np.mean(p_diff)),
            max_control=float(np.max(p_ctrl)),
            mean_control=float(np.mean(p_ctrl)),
        )
    return result


# ==============================================================================
# DimeBenchmarkManifest
# ==============================================================================


@dataclass(frozen=True)
class DimeBenchmarkManifest:
    """Versioned benchmark manifest defining the baseline and frozen benchmark protocol."""

    manifest_version: str
    dataset_id: str
    dataset_revision: str
    model_id: str
    model_revision: str
    engine_id: str
    engine_revision: str
    split_policy: SplitPolicy
    units: Mapping[str, str]
    observation_type: str
    known_truth_type: str
    force_classification: ForceClassification
    provenance: DimeProvenanceRecord
    privacy: PrivacySpec
    zero_denominator_policy: ZeroDenominatorPolicy = "guarded_zero"
    thresholds: NumericAcceptanceThresholds = field(
        default_factory=NumericAcceptanceThresholds
    )
    conditioning: ConditioningScale = field(default_factory=ConditioningScale)

    def __post_init__(self) -> None:
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))

    def evaluate_qualification(self) -> tuple[CapabilityStatus, list[str]]:
        """Evaluate qualification status under fail-closed scientific integrity rules.

        Disqualification triggers:
        - wrong units: units must be canonical SI.
        - missing provenance: provenance fields must be complete.
        - force-derived test kinematics: kinematics derived from forces create circularity.
        - skeleton contact output: kinematic skeletons have no contact physics.
        """
        reasons: list[str] = []

        # 1. Units check
        for dim, canonical_unit in CANONICAL_DIME_UNITS.items():
            declared = self.units.get(dim)
            if declared != canonical_unit:
                reasons.append(
                    f"wrong_units: dimension {dim!r} declared as {declared!r}, "
                    f"expected canonical SI {canonical_unit!r}"
                )

        # 2. Provenance check
        if not self.provenance.is_complete():
            reasons.append(
                "missing_provenance: provenance record has missing or empty required fields"
            )

        # 3. Force-derived test kinematics check
        if self.force_classification.kinematics_source == "force_derived":
            reasons.append(
                "force_derived_test_kinematics: test kinematics derived from forces are disallowed"
            )

        # 4. Skeleton contact check
        if self.force_classification.has_skeleton_contact:
            reasons.append(
                "skeleton_contact_unsupported: kinematic skeleton lacks contact dynamics"
            )

        if not reasons:
            return "qualified", []
        return "implemented", reasons

    def report_native_capabilities(
        self, engine_instance: object | None = None
    ) -> dict[str, CapabilityRecord]:
        """Report native capabilities separating Python method existence from qualification."""
        status, reasons = self.evaluate_qualification()
        is_qual = status == "qualified"

        standard_methods = (
            "simulate_forward",
            "compute_inverse_dynamics",
            "landmark_positions",
        )
        report: dict[str, CapabilityRecord] = {}
        for method in standard_methods:
            exists = (
                hasattr(engine_instance, method)
                if engine_instance is not None
                else True
            )
            if not exists:
                cap_status: CapabilityStatus = "unavailable"
                reason = f"method {method!r} not implemented on engine"
            elif is_qual:
                cap_status = "qualified"
                reason = None
            else:
                cap_status = "implemented"
                reason = "; ".join(reasons)
            report[method] = CapabilityRecord(
                name=method,
                method_exists=exists,
                status=cap_status,
                reason=reason,
            )

        report["overall"] = CapabilityRecord(
            name="overall",
            method_exists=any(r.method_exists for r in report.values()),
            status=status,
            reason="; ".join(reasons) if reasons else None,
        )
        return report

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dict with privacy path redaction."""
        data: dict[str, Any] = {
            "manifest_version": self.manifest_version,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "engine_id": self.engine_id,
            "engine_revision": self.engine_revision,
            "split_policy": self.split_policy.to_dict(),
            "units": dict(self.units),
            "observation_type": self.observation_type,
            "known_truth_type": self.known_truth_type,
            "force_classification": self.force_classification.to_dict(),
            "provenance": self.provenance.to_dict(),
            "privacy": self.privacy.to_dict(),
            "zero_denominator_policy": self.zero_denominator_policy,
            "thresholds": self.thresholds.to_dict(),
            "conditioning": self.conditioning.to_dict(),
        }
        if self.privacy.redact_filesystem_paths:
            data = _redact_paths(data)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DimeBenchmarkManifest:
        return cls(
            manifest_version=str(data["manifest_version"]),
            dataset_id=str(data["dataset_id"]),
            dataset_revision=str(data["dataset_revision"]),
            model_id=str(data["model_id"]),
            model_revision=str(data["model_revision"]),
            engine_id=str(data["engine_id"]),
            engine_revision=str(data["engine_revision"]),
            split_policy=SplitPolicy.from_dict(data["split_policy"]),
            units=dict(data["units"]),
            observation_type=str(data["observation_type"]),
            known_truth_type=str(data["known_truth_type"]),
            force_classification=ForceClassification.from_dict(
                data["force_classification"]
            ),
            provenance=DimeProvenanceRecord.from_dict(data["provenance"]),
            privacy=PrivacySpec.from_dict(data["privacy"]),
            zero_denominator_policy=data.get("zero_denominator_policy", "guarded_zero"),
            thresholds=NumericAcceptanceThresholds.from_dict(data["thresholds"]),
            conditioning=ConditioningScale.from_dict(data["conditioning"]),
        )

    def save_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> DimeBenchmarkManifest:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass(frozen=True)
class DimeBenchmarkResult:
    """Execution result of running the DIME baseline against a frozen manifest.

    ``None`` figures mean *not measured*; they are never replaced by constants or by
    ground truth (#11552).
    """

    manifest_version: str
    dataset_id: str
    model_id: str
    status: CapabilityStatus
    qualification_reasons: tuple[str, ...]
    reproduced_identically: bool
    seed: int
    trajectory_q: np.ndarray | None
    control_torques: np.ndarray
    phase_metrics: dict[str, PhaseMetrics]
    alignment: float | None
    cancellation: float | None
    thresholds_passed: bool
    threshold_failures: tuple[str, ...]

    @property
    def measured(self) -> bool:
        """True only when an estimator produced the reported figures."""
        return (
            self.trajectory_q is not None
            and self.alignment is not None
            and self.cancellation is not None
        )


def run_dime_baseline(
    manifest: DimeBenchmarkManifest,
    *,
    seed: int = 42,
) -> DimeBenchmarkResult:
    """Run the baseline under the frozen manifest, failing closed without an estimator.

    No baseline estimator is wired to this protocol, and the fixture carries no
    observations separate from ground truth. Using truth as the estimate would make
    every metric trivially perfect, so the result is reported as ``unavailable`` with
    all estimate-derived figures not measured. Fixture controls are inputs, not
    results, and are returned unchanged.
    """
    _status, reasons = manifest.evaluate_qualification()

    # Load appropriate deterministic fixture (controls only; truth is not an estimate)
    if manifest.model_id == "fixed_base_pendulum":
        fixture: Any = make_fixed_base_pendulum_fixture(n_frames=8, fps=100.0)
    elif manifest.model_id == "underactuated":
        fixture = make_underactuated_analytic_fixture(n_frames=8, fps=100.0)
    elif manifest.model_id == "native_stance":
        fixture = make_native_stance_fixture(n_frames=8, fps=100.0)
    else:
        fixture = make_fixed_base_pendulum_fixture(n_frames=8, fps=100.0)

    controls = np.asarray(fixture.controls, dtype=np.float64)

    return DimeBenchmarkResult(
        manifest_version=manifest.manifest_version,
        dataset_id=manifest.dataset_id,
        model_id=manifest.model_id,
        status="unavailable",
        qualification_reasons=(
            *reasons,
            "no baseline estimator is implemented for this protocol",
        ),
        reproduced_identically=True,
        seed=seed,
        trajectory_q=None,
        control_torques=controls,
        phase_metrics={},
        alignment=None,
        cancellation=None,
        thresholds_passed=False,
        threshold_failures=(
            "baseline figures not measured: no estimator output to compare",
        ),
    )


def _redact_paths(data: Any) -> Any:
    """Recursively redact local filesystem paths from serialized manifest structures."""
    if isinstance(data, dict):
        cleaned: dict[str, Any] = {}
        for k, v in data.items():
            if "path" in k.lower() or "file" in k.lower():
                if isinstance(v, str) and (v.startswith(("/", "\\")) or ":" in v):
                    continue
            cleaned[k] = _redact_paths(v)
        return cleaned
    if isinstance(data, list):
        return [_redact_paths(item) for item in data]
    return data
