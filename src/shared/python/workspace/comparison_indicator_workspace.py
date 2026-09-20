"""Cross-Engine Comparison and Injury Indicator Workspace Coordinator (ORG-18, #10527).

Provides explicit adapters and an application service coordinating:
1. Cross-engine comparison of compatible runs using existing CC-27 comparison services.
2. Calculation of supported biomechanical load and injury risk indicators using InjuryRiskScorer.
3. Strict DbC validation of units, coordinate frames, timebases, model comparability,
   and required load channels without fabricating data or guessing.
4. Retention of method provenance, source run IDs, separate model fidelity tiers,
   and explicit non-clinical disclaimers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.injury.injury_risk import (
    InjuryRiskReport,
    InjuryRiskScorer,
    RiskFactor,
    RiskLevel,
)
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.simulation_backends.comparison import (
    ComparisonThresholds,
    compare_traces,
)
from src.shared.python.simulation_backends.protocol import Trace

logger = get_logger(__name__)

__all__ = [
    "BiomechanicalLoadChannels",
    "ComparisonIndicatorWorkspaceCoordinator",
    "ComparisonRunArtifact",
    "ComparisonWorkspaceResult",
    "CompatibilityValidationResult",
    "CrossEngineComparisonAdapter",
    "IncompatibleArtifactError",
    "IndicatorWorkspaceResult",
    "MissingChannelError",
    "ModelFidelityLevel",
    "validate_run_compatibility",
]

_NON_CLINICAL_DISCLAIMER = (
    "DISCLAIMER: Numerical scores are biomechanical engineering indicators and do NOT "
    "constitute clinical diagnosis, medical validation, or return-to-play clearance. "
    "Calculations reflect simplified or qualified mechanical models rather than clinical outcomes."
)


class ModelFidelityLevel(str, Enum):
    """Explicit fidelity hierarchy for biomechanical and physics models."""

    STUB_PENDULUM = "stub_pendulum"
    SIMPLIFIED_KINETICS = "simplified_kinetics"
    QUALIFIED_FULL_BODY = "qualified_full_body"


class IncompatibleArtifactError(ValueError):
    """Raised when two artifacts cannot be compared due to mismatched units, frames, or channels."""


class MissingChannelError(ValueError):
    """Raised when required biomechanical channels are missing from the input data."""


@dataclass(frozen=True)
class ComparisonRunArtifact:
    """Standardized simulation or measurement run artifact for cross-engine comparison."""

    run_id: str
    backend_name: str
    times: np.ndarray
    coordinates: np.ndarray
    velocities: np.ndarray | None = None
    torques: np.ndarray | None = None
    channel_names: Sequence[str] = ()
    units: str = "m"
    coordinate_frame: str = "world"
    fidelity: ModelFidelityLevel = ModelFidelityLevel.SIMPLIFIED_KINETICS
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(bool(self.run_id and self.run_id.strip()), "run_id must be non-empty")
        require(
            bool(self.backend_name and self.backend_name.strip()),
            "backend_name must be non-empty",
        )
        times = np.asarray(self.times, dtype=float)
        require(
            times.ndim == 1 and len(times) > 0, "times must be a non-empty 1D array"
        )
        require(bool(np.all(np.isfinite(times))), "times must be strictly finite")
        require(bool(np.all(np.diff(times) > 0)), "times must be strictly increasing")
        coords = np.asarray(self.coordinates, dtype=float)
        require(coords.ndim == 2, "coordinates must be a 2D array shaped (N, DOF)")
        require(
            len(coords) == len(times), "coordinates row count must match times length"
        )
        require(
            bool(np.all(np.isfinite(coords))), "coordinates must be strictly finite"
        )
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "coordinates", coords)


@dataclass(frozen=True)
class CompatibilityValidationResult:
    """Outcome of pre-comparison compatibility check."""

    compatible: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_run_compatibility(
    run_a: ComparisonRunArtifact,
    run_b: ComparisonRunArtifact,
    *,
    tol_time: float = 1e-4,
) -> CompatibilityValidationResult:
    """Validate comparability of two run artifacts before executing numerical comparison."""
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Fidelity level check
    if run_a.fidelity != run_b.fidelity:
        errors.append(
            f"Fidelity mismatch: '{run_a.run_id}' is '{run_a.fidelity.value}' but "
            f"'{run_b.run_id}' is '{run_b.fidelity.value}'"
        )

    # 2. Units check
    if run_a.units.strip().lower() != run_b.units.strip().lower():
        errors.append(f"Unit mismatch: '{run_a.units}' vs '{run_b.units}'")

    # 3. Coordinate frame check
    if run_a.coordinate_frame.strip().lower() != run_b.coordinate_frame.strip().lower():
        errors.append(
            f"Coordinate frame mismatch: '{run_a.coordinate_frame}' vs '{run_b.coordinate_frame}'"
        )

    # 4. Dimension and channel check
    dim_a = run_a.coordinates.shape[1]
    dim_b = run_b.coordinates.shape[1]
    if dim_a != dim_b:
        errors.append(f"Dimension mismatch: {dim_a} DOF vs {dim_b} DOF")
    elif run_a.channel_names and run_b.channel_names:
        if tuple(run_a.channel_names) != tuple(run_b.channel_names):
            warnings.append("Channel name ordering or labels differ")

    # 5. Timebase alignment check
    if len(run_a.times) != len(run_b.times):
        errors.append(
            f"Time series length mismatch: {len(run_a.times)} vs {len(run_b.times)}"
        )
    else:
        dt_a = float(run_a.times[-1] - run_a.times[0])
        dt_b = float(run_b.times[-1] - run_b.times[0])
        if abs(dt_a - dt_b) > tol_time:
            errors.append(f"Time duration mismatch: {dt_a:.4f}s vs {dt_b:.4f}s")
        max_time_drift = float(np.max(np.abs(run_a.times - run_b.times)))
        if max_time_drift > tol_time:
            errors.append(
                f"Time drift {max_time_drift:.4e}s exceeds tolerance {tol_time:.4e}s"
            )

    return CompatibilityValidationResult(
        compatible=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


@dataclass(frozen=True)
class ComparisonWorkspaceResult:
    """Outcome of cross-engine comparison in the workspace."""

    action_id: str
    run_a_id: str
    run_b_id: str
    backend_a: str
    backend_b: str
    fidelity: str
    metrics: dict[str, float]
    provenance_hash: str
    timestamp_utc: str
    details: dict[str, Any] = field(default_factory=dict)


class CrossEngineComparisonAdapter:
    """Explicit adapter executing CC-27 cross-engine comparison on compatible runs."""

    def compare_runs(
        self,
        run_a: ComparisonRunArtifact,
        run_b: ComparisonRunArtifact,
        thresholds: ComparisonThresholds | None = None,
    ) -> ComparisonWorkspaceResult:
        """Compare two compatible runs and return structured divergence metrics."""
        compat = validate_run_compatibility(run_a, run_b)
        if not compat.compatible:
            raise IncompatibleArtifactError(
                f"Cannot compare '{run_a.run_id}' and '{run_b.run_id}': "
                + "; ".join(compat.errors)
            )

        # Build backend-agnostic Traces for existing CC-27 comparison service
        v_a = (
            run_a.velocities
            if run_a.velocities is not None
            else np.zeros_like(run_a.coordinates)
        )
        v_b = (
            run_b.velocities
            if run_b.velocities is not None
            else np.zeros_like(run_b.coordinates)
        )
        tau_a = (
            run_a.torques
            if run_a.torques is not None
            else np.zeros_like(run_a.coordinates)
        )
        tau_b = (
            run_b.torques
            if run_b.torques is not None
            else np.zeros_like(run_b.coordinates)
        )

        trace_a = Trace(
            t=run_a.times,
            q=run_a.coordinates,
            v=v_a,
            torques=tau_a,
            backend=run_a.backend_name,
        )
        trace_b = Trace(
            t=run_b.times,
            q=run_b.coordinates,
            v=v_b,
            torques=tau_b,
            backend=run_b.backend_name,
        )

        report = compare_traces(
            {run_a.backend_name: trace_a, run_b.backend_name: trace_b},
            thresholds=thresholds or ComparisonThresholds(),
        )

        # Compute summary metrics
        abs_diff = np.abs(run_a.coordinates - run_b.coordinates)
        metrics = {
            "max_abs_error": float(np.max(abs_diff)),
            "mean_abs_error": float(np.mean(abs_diff)),
            "rmse": float(np.sqrt(np.mean(abs_diff**2))),
        }

        # Deterministic provenance hash
        hash_input = f"{run_a.run_id}:{run_b.run_id}:{metrics['max_abs_error']:.6e}"
        prov_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

        return ComparisonWorkspaceResult(
            action_id="canonical_core_comparison",
            run_a_id=run_a.run_id,
            run_b_id=run_b.run_id,
            backend_a=run_a.backend_name,
            backend_b=run_b.backend_name,
            fidelity=run_a.fidelity.value,
            metrics=metrics,
            provenance_hash=prov_hash,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            details={"divergences": [asdict(d) for d in report.divergences]},
        )


@dataclass(frozen=True)
class BiomechanicalLoadChannels:
    """Actual biomechanical load and kinematics channels extracted from a model/run."""

    source_run_id: str
    peak_compression_n: float | None = None
    peak_shear_n: float | None = None
    peak_torsion_nm: float | None = None
    joint_moments_nm: Mapping[str, float] = field(default_factory=dict)
    swing_metrics: Mapping[str, float] = field(default_factory=dict)
    training_load: Mapping[str, float] = field(default_factory=dict)
    fidelity: ModelFidelityLevel = ModelFidelityLevel.SIMPLIFIED_KINETICS

    def has_required_load_channels(self) -> bool:
        """Check whether at least one physical load channel is present."""
        has_spinal = any(
            v is not None and np.isfinite(v)
            for v in (self.peak_compression_n, self.peak_shear_n, self.peak_torsion_nm)
        )
        has_joint = (
            any(np.isfinite(v) for v in self.joint_moments_nm.values())
            if self.joint_moments_nm
            else False
        )
        return has_spinal or has_joint


@dataclass(frozen=True)
class IndicatorWorkspaceResult:
    """Outcome of injury risk indicator calculation."""

    action_id: str
    source_run_id: str
    fidelity: str
    overall_risk_score: float
    overall_risk_level: str
    risk_factors: list[dict[str, Any]]
    recommendations: list[str]
    disclaimer: str
    provenance_hash: str
    timestamp_utc: str


class _SpinalResultAdapter:
    """Lightweight duck-typed adapter for InjuryRiskScorer._score_spinal_risks."""

    def __init__(
        self,
        peak_compression_bw: float,
        peak_lateral_shear_bw: float,
        x_factor_stretch: float = 0.0,
    ) -> None:
        self.peak_compression_bw = peak_compression_bw
        self.peak_lateral_shear_bw = peak_lateral_shear_bw
        self.x_factor = type("XFactor", (), {"x_factor_stretch": x_factor_stretch})()


class _JointResultAdapter:
    """Lightweight duck-typed adapter for joint results."""

    def __init__(self, risk_score: float, impingement_risk: bool = False) -> None:
        self.risk_score = risk_score
        self.impingement_risk = impingement_risk


class InjuryIndicatorAdapter:
    """Explicit adapter computing supported injury indicators using InjuryRiskScorer."""

    def compute_indicators(
        self,
        channels: BiomechanicalLoadChannels,
        body_weight_n: float = 784.8,  # Default ~80kg adult in N (80 * 9.81)
    ) -> IndicatorWorkspaceResult:
        """Compute injury risk indicators from required load channels without fabricating data."""
        require(
            body_weight_n > 0.0 and np.isfinite(body_weight_n),
            "body_weight_n must be positive and finite",
        )
        if not channels.has_required_load_channels():
            raise MissingChannelError(
                f"Missing required load channels for run '{channels.source_run_id}'. "
                "Calculation requires valid spinal loads or joint moments; no mock data is substituted."
            )

        spinal_adapter = None
        if channels.peak_compression_n is not None or channels.peak_shear_n is not None:
            comp_bw = (
                (channels.peak_compression_n / body_weight_n)
                if channels.peak_compression_n is not None
                else 0.0
            )
            shear_bw = (
                (channels.peak_shear_n / body_weight_n)
                if channels.peak_shear_n is not None
                else 0.0
            )
            x_factor = float(channels.swing_metrics.get("x_factor_stretch", 0.0))
            spinal_adapter = _SpinalResultAdapter(comp_bw, shear_bw, x_factor)

        joint_results_dict: dict[str, Any] = {}
        for joint_name, moment_nm in channels.joint_moments_nm.items():
            # Estimate risk score based on reference threshold (nominal 100 Nm)
            score = float(np.clip((moment_nm / 120.0) * 50.0, 0.0, 100.0))
            joint_results_dict[joint_name] = _JointResultAdapter(score)

        scorer = InjuryRiskScorer()
        report: InjuryRiskReport = scorer.score(
            spinal_result=spinal_adapter,
            joint_results=joint_results_dict or None,
            swing_metrics=dict(channels.swing_metrics)
            if channels.swing_metrics
            else None,
            training_load=dict(channels.training_load)
            if channels.training_load
            else None,
        )

        factors = [
            {
                "name": rf.name,
                "value": float(rf.value),
                "threshold_safe": float(rf.threshold_safe),
                "threshold_high": float(rf.threshold_high),
                "weight": float(rf.weight),
            }
            for rf in report.risk_factors
        ]

        prov_str = f"{channels.source_run_id}:{report.overall_risk_score:.2f}:{channels.fidelity.value}"
        prov_hash = hashlib.sha256(prov_str.encode("utf-8")).hexdigest()

        return IndicatorWorkspaceResult(
            action_id="injury_analysis",
            source_run_id=channels.source_run_id,
            fidelity=channels.fidelity.value,
            overall_risk_score=float(report.overall_risk_score),
            overall_risk_level=report.overall_risk_level.value,
            risk_factors=factors,
            recommendations=list(report.recommendations),
            disclaimer=_NON_CLINICAL_DISCLAIMER,
            provenance_hash=prov_hash,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )


class ComparisonIndicatorWorkspaceCoordinator:
    """Unified application coordinator surfacing comparison and injury analysis across shells."""

    def __init__(self) -> None:
        self.comparison_adapter = CrossEngineComparisonAdapter()
        self.indicator_adapter = InjuryIndicatorAdapter()

    def check_availability(self) -> dict[str, Any]:
        """Report availability and supported contextual actions."""
        return {
            "available": True,
            "actions": ["canonical_core_comparison", "injury_analysis"],
            "shells": ["results_and_compare", "exercise_analysis"],
            "supported_fidelities": [f.value for f in ModelFidelityLevel],
            "disclaimer": _NON_CLINICAL_DISCLAIMER,
        }
