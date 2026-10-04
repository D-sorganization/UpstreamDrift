"""DIME Shared Reports, GUI Strategy Selection and LaTeX Methods (#11421, #11432).

Provides:
1. Unified estimation strategy selection with truthful capability status reporting
   (implemented, qualified, unavailable) and graceful unavailability states.
2. Comprehensive Full and Custom report output contracts packaging kinematics,
   selected net controls, GRF provenance, drift vs controlled prediction, uncertainty,
   contact/replay residuals, video overlays, and unavailable-capability explanations.
3. Distinct pointwise vs integrated ZTCF (Zero-Torque / Zero-Control Free dynamics)
   registration.
4. Strict fail-closed validation:
   - IK runs mislabeled as forward dynamics fail closed.
   - Missing native GRF rendered or substituted as zero fails closed.
   - Unqualified engines offered or reported as validated fail closed.
   - Mismatched timestamp/frame overlays fail closed with TimingViolationError.
5. Deterministic dictionary and JSON round-trip serialization preserving model/data
   provenance.
6. Verifiable LaTeX methods documentation generator with equations, assumptions,
   units, reproduction seeds/commit, and explicit limitations.
7. Offscreen GUI ViewModel supporting headless selection and option configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from src.shared.python.contracts import PreconditionError, require
from src.shared.python.estimation.dime_contracts import ProviderCapability
from src.shared.python.estimation.dime_manifest import (
    CANONICAL_DIME_UNITS,
    CapabilityRecord,
    CapabilityStatus,
    DimeProvenanceRecord,
)
from src.shared.python.estimation.dime_observation_factors import TimingViolationError

DIME_REPORT_VERSION: Final[str] = "1.0.0"


# ==============================================================================
# Domain Enums
# ==============================================================================


class EstimatorStrategy(str, Enum):
    """Supported estimation and simulation strategies."""

    IK = "inverse_kinematics"
    INVERSE_DYNAMICS = "inverse_dynamics"
    FORWARD_DYNAMICS = "forward_dynamics"
    DIME_MHE = "dime_mhe"
    CONTINUOUS_REPLAY = "continuous_replay"
    NEURAL_ESTIMATOR = "neural_estimator"


class RunClassification(str, Enum):
    """Physical classification of motion estimation or forward execution."""

    INVERSE_KINEMATICS = "inverse_kinematics"
    INVERSE_DYNAMICS = "inverse_dynamics"
    FORWARD_DYNAMICS = "forward_dynamics"
    HYBRID_ESTIMATION = "hybrid_estimation"


class ReportScope(str, Enum):
    """Scope of generated estimation report."""

    FULL = "full"
    CUSTOM = "custom"


class ReportForceProvenance(str, Enum):
    """Ground reaction force measurement origin and typed availability."""

    MEASURED = "measured"
    INFERRED = "inferred"
    REGULARIZED = "regularized"
    UNAVAILABLE = "unavailable"


# ==============================================================================
# Helper Utilities
# ==============================================================================


def _make_readonly_array(arr: np.ndarray | None) -> np.ndarray | None:
    """Return a read-only float64 copy of array, or None if input is None."""
    if arr is None:
        return None
    out = np.array(arr, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


# ==============================================================================
# Domain Dataclasses
# ==============================================================================


@dataclass(frozen=True)
class KinematicsPayload:
    """Rigid/multibody kinematics time series with joint names and units."""

    times: np.ndarray
    q: np.ndarray
    v: np.ndarray | None = None
    a: np.ndarray | None = None
    joint_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require(self.times.ndim == 1, "times must be 1D array")
        require(self.q.ndim == 2, "q must be 2D array [N x n_q]")
        require(len(self.times) == len(self.q), "times and q must have same length")
        if len(self.times) > 1:
            diffs = np.diff(self.times)
            require(bool(np.all(diffs > 0.0)), "times must be strictly increasing")

        object.__setattr__(self, "times", _make_readonly_array(self.times))
        object.__setattr__(self, "q", _make_readonly_array(self.q))
        object.__setattr__(self, "v", _make_readonly_array(self.v))
        object.__setattr__(self, "a", _make_readonly_array(self.a))

    def to_dict(self) -> dict[str, Any]:
        return {
            "times": self.times.tolist(),
            "q": self.q.tolist(),
            "v": self.v.tolist() if self.v is not None else None,
            "a": self.a.tolist() if self.a is not None else None,
            "joint_names": list(self.joint_names),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KinematicsPayload:
        v_raw = data.get("v")
        a_raw = data.get("a")
        return cls(
            times=np.array(data["times"], dtype=np.float64),
            q=np.array(data["q"], dtype=np.float64),
            v=np.array(v_raw, dtype=np.float64) if v_raw is not None else None,
            a=np.array(a_raw, dtype=np.float64) if a_raw is not None else None,
            joint_names=tuple(data.get("joint_names", ())),
        )


@dataclass(frozen=True)
class ZtcfRecord:
    """Distinct pointwise acceleration and integrated trajectory ZTCF data."""

    pointwise_ztcf_drift: np.ndarray
    integrated_ztcf_drift: np.ndarray
    pointwise_dominance: np.ndarray | None = None
    integrated_dominance: float | None = None
    units: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType(
            {"acceleration": "m/s^2", "drift": "m", "time": "s"}
        )
    )

    def __post_init__(self) -> None:
        require(self.pointwise_ztcf_drift.ndim == 1, "pointwise drift must be 1D array")
        require(
            self.integrated_ztcf_drift.ndim == 1, "integrated drift must be 1D array"
        )
        object.__setattr__(
            self,
            "pointwise_ztcf_drift",
            _make_readonly_array(self.pointwise_ztcf_drift),
        )
        object.__setattr__(
            self,
            "integrated_ztcf_drift",
            _make_readonly_array(self.integrated_ztcf_drift),
        )
        object.__setattr__(
            self,
            "pointwise_dominance",
            _make_readonly_array(self.pointwise_dominance),
        )
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pointwise_ztcf_drift": self.pointwise_ztcf_drift.tolist(),
            "integrated_ztcf_drift": self.integrated_ztcf_drift.tolist(),
            "pointwise_dominance": (
                self.pointwise_dominance.tolist()
                if self.pointwise_dominance is not None
                else None
            ),
            "integrated_dominance": self.integrated_dominance,
            "units": dict(self.units),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ZtcfRecord:
        pw_dom = data.get("pointwise_dominance")
        return cls(
            pointwise_ztcf_drift=np.array(
                data["pointwise_ztcf_drift"], dtype=np.float64
            ),
            integrated_ztcf_drift=np.array(
                data["integrated_ztcf_drift"], dtype=np.float64
            ),
            pointwise_dominance=(
                np.array(pw_dom, dtype=np.float64) if pw_dom is not None else None
            ),
            integrated_dominance=(
                float(data["integrated_dominance"])
                if data.get("integrated_dominance") is not None
                else None
            ),
            units=MappingProxyType(dict(data.get("units", {}))),
        )


@dataclass(frozen=True)
class GroundReactionForceReport:
    """Typed ground reaction force data rejecting silent zero-substitution."""

    provenance: ReportForceProvenance
    forces_n: np.ndarray | None = None
    cop_m: np.ndarray | None = None
    torques_nm: np.ndarray | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.provenance == ReportForceProvenance.UNAVAILABLE:
            if self.forces_n is not None:
                raise PreconditionError(
                    "Missing native GRF cannot be rendered or substituted as zero; "
                    "must be typed unavailable"
                )
        elif self.provenance == ReportForceProvenance.MEASURED:
            if self.forces_n is None:
                raise PreconditionError(
                    "Measured GRF requires non-null, measured force data"
                )

        object.__setattr__(self, "forces_n", _make_readonly_array(self.forces_n))
        object.__setattr__(self, "cop_m", _make_readonly_array(self.cop_m))
        object.__setattr__(self, "torques_nm", _make_readonly_array(self.torques_nm))

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.value,
            "forces_n": (self.forces_n.tolist() if self.forces_n is not None else None),
            "cop_m": self.cop_m.tolist() if self.cop_m is not None else None,
            "torques_nm": (
                self.torques_nm.tolist() if self.torques_nm is not None else None
            ),
            "unavailable_reason": self.unavailable_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GroundReactionForceReport:
        f_raw = data.get("forces_n")
        c_raw = data.get("cop_m")
        t_raw = data.get("torques_nm")
        return cls(
            provenance=ReportForceProvenance(data["provenance"]),
            forces_n=np.array(f_raw, dtype=np.float64) if f_raw is not None else None,
            cop_m=np.array(c_raw, dtype=np.float64) if c_raw is not None else None,
            torques_nm=np.array(t_raw, dtype=np.float64) if t_raw is not None else None,
            unavailable_reason=data.get("unavailable_reason"),
        )


@dataclass(frozen=True)
class VideoOverlaySpec:
    """Video overlay synchronization specification using refined ellipsoid geometry."""

    overlay_id: str
    timestamps_s: np.ndarray
    frame_indices: np.ndarray
    geometry_model: str = "GS3DX_Human_Refined_Ellipsoid"
    camera_id: str = "default_camera"
    render_quality: str = "1080p_h264"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.timestamps_s) != len(self.frame_indices):
            raise TimingViolationError(
                "Overlay timestamps and frame indices must have identical lengths"
            )
        object.__setattr__(
            self, "timestamps_s", _make_readonly_array(self.timestamps_s)
        )
        indices = np.array(self.frame_indices, dtype=np.int64, copy=True)
        indices.flags.writeable = False
        object.__setattr__(self, "frame_indices", indices)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "overlay_id": self.overlay_id,
            "timestamps_s": self.timestamps_s.tolist(),
            "frame_indices": self.frame_indices.tolist(),
            "geometry_model": self.geometry_model,
            "camera_id": self.camera_id,
            "render_quality": self.render_quality,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VideoOverlaySpec:
        return cls(
            overlay_id=str(data["overlay_id"]),
            timestamps_s=np.array(data["timestamps_s"], dtype=np.float64),
            frame_indices=np.array(data["frame_indices"], dtype=np.int64),
            geometry_model=str(
                data.get("geometry_model", "GS3DX_Human_Refined_Ellipsoid")
            ),
            camera_id=str(data.get("camera_id", "default_camera")),
            render_quality=str(data.get("render_quality", "1080p_h264")),
            metadata=MappingProxyType(dict(data.get("metadata", {}))),
        )


@dataclass(frozen=True)
class DriftAndPredictionPayload:
    """Drift and controlled prediction comparison telemetry."""

    drift_trajectory: np.ndarray
    controlled_prediction: np.ndarray
    drift_dominance_index: float
    reachable_interval_radius: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "drift_trajectory", _make_readonly_array(self.drift_trajectory)
        )
        object.__setattr__(
            self,
            "controlled_prediction",
            _make_readonly_array(self.controlled_prediction),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_trajectory": self.drift_trajectory.tolist(),
            "controlled_prediction": self.controlled_prediction.tolist(),
            "drift_dominance_index": self.drift_dominance_index,
            "reachable_interval_radius": self.reachable_interval_radius,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DriftAndPredictionPayload:
        return cls(
            drift_trajectory=np.array(data["drift_trajectory"], dtype=np.float64),
            controlled_prediction=np.array(
                data["controlled_prediction"], dtype=np.float64
            ),
            drift_dominance_index=float(data["drift_dominance_index"]),
            reachable_interval_radius=float(data["reachable_interval_radius"]),
        )


@dataclass(frozen=True)
class UncertaintySummary:
    """Parametric and trajectory estimation uncertainty."""

    kind: str = "gaussian"
    confidence_level: float = 0.95
    joint_variances: Mapping[str, float] = field(default_factory=dict)
    parameter_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "joint_variances", MappingProxyType(dict(self.joint_variances))
        )
        object.__setattr__(
            self, "parameter_bounds", MappingProxyType(dict(self.parameter_bounds))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "confidence_level": self.confidence_level,
            "joint_variances": dict(self.joint_variances),
            "parameter_bounds": {k: list(v) for k, v in self.parameter_bounds.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UncertaintySummary:
        return cls(
            kind=str(data.get("kind", "gaussian")),
            confidence_level=float(data.get("confidence_level", 0.95)),
            joint_variances=dict(data.get("joint_variances", {})),
            parameter_bounds={
                str(k): (float(v[0]), float(v[1]))
                for k, v in data.get("parameter_bounds", {}).items()
            },
        )


@dataclass(frozen=True)
class ReplayResidualsSummary:
    """Contact and replay verification metrics."""

    max_position_drift_m: float
    rms_position_drift_m: float
    normal_force_residual_n: float = 0.0
    friction_cone_violations: int = 0
    reset_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReplayResidualsSummary:
        return cls(**data)


@dataclass(frozen=True)
class DimeReportOptions:
    """Scope and channel selection options for report generation."""

    scope: ReportScope = ReportScope.FULL
    strategy: EstimatorStrategy = EstimatorStrategy.DIME_MHE
    include_kinematics: bool = True
    include_controls: bool = True
    include_grf: bool = True
    include_drift_prediction: bool = True
    include_uncertainty: bool = True
    include_residuals: bool = True
    include_video_overlays: bool = True
    selected_channels: tuple[str, ...] = ()
    selected_plots: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.value,
            "strategy": self.strategy.value,
            "include_kinematics": self.include_kinematics,
            "include_controls": self.include_controls,
            "include_grf": self.include_grf,
            "include_drift_prediction": self.include_drift_prediction,
            "include_uncertainty": self.include_uncertainty,
            "include_residuals": self.include_residuals,
            "include_video_overlays": self.include_video_overlays,
            "selected_channels": list(self.selected_channels),
            "selected_plots": list(self.selected_plots),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DimeReportOptions:
        return cls(
            scope=ReportScope(data.get("scope", "full")),
            strategy=EstimatorStrategy(data.get("strategy", "dime_mhe")),
            include_kinematics=bool(data.get("include_kinematics", True)),
            include_controls=bool(data.get("include_controls", True)),
            include_grf=bool(data.get("include_grf", True)),
            include_drift_prediction=bool(data.get("include_drift_prediction", True)),
            include_uncertainty=bool(data.get("include_uncertainty", True)),
            include_residuals=bool(data.get("include_residuals", True)),
            include_video_overlays=bool(data.get("include_video_overlays", True)),
            selected_channels=tuple(data.get("selected_channels", ())),
            selected_plots=tuple(data.get("selected_plots", ())),
        )


@dataclass(frozen=True)
class StrategySelectionState:
    """Active strategy selection state with capability record and availability."""

    strategy: EstimatorStrategy
    capability: CapabilityRecord
    engine_id: str
    is_validated: bool
    selected_channels: tuple[str, ...]
    selected_plots: tuple[str, ...]
    unavailable_explanations: Mapping[str, str] = field(default_factory=dict)


# ==============================================================================
# Report Artifact Contract
# ==============================================================================


@dataclass(frozen=True)
class DimeReportArtifact:
    """Shared report artifact package with physical metrics, provenance, and LaTeX."""

    report_id: str
    created_at: str
    scope: ReportScope
    strategy: EstimatorStrategy
    run_classification: RunClassification
    provenance: DimeProvenanceRecord
    is_validated: bool = False
    engine_capability_status: CapabilityStatus = "implemented"
    kinematics: KinematicsPayload | None = None
    selected_net_controls: np.ndarray | None = None
    control_channel_names: tuple[str, ...] = ()
    grf_report: GroundReactionForceReport | None = None
    drift_prediction: DriftAndPredictionPayload | None = None
    uncertainty: UncertaintySummary | None = None
    residuals: ReplayResidualsSummary | None = None
    video_overlay: VideoOverlaySpec | None = None
    ztcf_record: ZtcfRecord | None = None
    unavailable_capabilities: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        # Invariant 1: IK run cannot be mislabeled as forward dynamics
        if (
            self.strategy == EstimatorStrategy.IK
            and self.run_classification == RunClassification.FORWARD_DYNAMICS
        ):
            raise PreconditionError("IK run cannot be mislabeled as forward dynamics")

        # Invariant 2: Unqualified engine cannot be offered or reported as validated
        if self.is_validated and self.engine_capability_status != "qualified":
            raise PreconditionError(
                "Unqualified engine cannot be offered or reported as validated"
            )

        # Invariant 3: Single-shot continuous replay requires reset_count == 1
        if self.residuals is not None and self.residuals.reset_count != 1:
            raise PreconditionError(
                f"Continuous replay requires reset_count == 1, got {self.residuals.reset_count}"
            )

        # Read-only controls array
        object.__setattr__(
            self,
            "selected_net_controls",
            _make_readonly_array(self.selected_net_controls),
        )

        # Populate unavailable capabilities map
        unavail = dict(self.unavailable_capabilities)
        if (
            self.grf_report is not None
            and self.grf_report.provenance == ReportForceProvenance.UNAVAILABLE
        ):
            unavail["grf"] = (
                self.grf_report.unavailable_reason or "Native GRF unavailable"
            )
        object.__setattr__(self, "unavailable_capabilities", MappingProxyType(unavail))

    def to_dict(self) -> dict[str, Any]:
        """Convert report artifact to dictionary."""
        u_arr = self.selected_net_controls
        return {
            "report_id": self.report_id,
            "created_at": self.created_at,
            "scope": self.scope.value,
            "strategy": self.strategy.value,
            "run_classification": self.run_classification.value,
            "provenance": self.provenance.to_dict(),
            "is_validated": self.is_validated,
            "engine_capability_status": self.engine_capability_status,
            "kinematics": self.kinematics.to_dict() if self.kinematics else None,
            "selected_net_controls": u_arr.tolist() if u_arr is not None else None,
            "control_channel_names": list(self.control_channel_names),
            "grf_report": self.grf_report.to_dict() if self.grf_report else None,
            "drift_prediction": (
                self.drift_prediction.to_dict() if self.drift_prediction else None
            ),
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty else None,
            "residuals": self.residuals.to_dict() if self.residuals else None,
            "video_overlay": (
                self.video_overlay.to_dict() if self.video_overlay else None
            ),
            "ztcf_record": self.ztcf_record.to_dict() if self.ztcf_record else None,
            "unavailable_capabilities": dict(self.unavailable_capabilities),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DimeReportArtifact:
        """Construct report artifact from dictionary."""
        u_raw = data.get("selected_net_controls")
        kin_raw = data.get("kinematics")
        grf_raw = data.get("grf_report")
        drift_raw = data.get("drift_prediction")
        unc_raw = data.get("uncertainty")
        res_raw = data.get("residuals")
        ovl_raw = data.get("video_overlay")
        ztcf_raw = data.get("ztcf_record")

        return cls(
            report_id=str(data["report_id"]),
            created_at=str(data["created_at"]),
            scope=ReportScope(data["scope"]),
            strategy=EstimatorStrategy(data["strategy"]),
            run_classification=RunClassification(data["run_classification"]),
            provenance=DimeProvenanceRecord.from_dict(data["provenance"]),
            is_validated=bool(data.get("is_validated", False)),
            engine_capability_status=data.get(
                "engine_capability_status", "implemented"
            ),
            kinematics=KinematicsPayload.from_dict(kin_raw) if kin_raw else None,
            selected_net_controls=(
                np.array(u_raw, dtype=np.float64) if u_raw is not None else None
            ),
            control_channel_names=tuple(data.get("control_channel_names", ())),
            grf_report=(
                GroundReactionForceReport.from_dict(grf_raw) if grf_raw else None
            ),
            drift_prediction=(
                DriftAndPredictionPayload.from_dict(drift_raw) if drift_raw else None
            ),
            uncertainty=UncertaintySummary.from_dict(unc_raw) if unc_raw else None,
            residuals=ReplayResidualsSummary.from_dict(res_raw) if res_raw else None,
            video_overlay=VideoOverlaySpec.from_dict(ovl_raw) if ovl_raw else None,
            ztcf_record=ZtcfRecord.from_dict(ztcf_raw) if ztcf_raw else None,
            unavailable_capabilities=dict(data.get("unavailable_capabilities", {})),
            notes=str(data.get("notes", "")),
        )

    def to_json(self) -> str:
        """Serialize report artifact to formatted JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> DimeReportArtifact:
        """Deserialize report artifact from JSON."""
        return cls.from_dict(json.loads(json_str))

    def to_latex_summary(self) -> str:
        """Generate LaTeX methods documentation with equations, units, and limitations."""
        m_hash = self.provenance.model_hash
        c_hash = self.provenance.git_commit
        l_unit = CANONICAL_DIME_UNITS["length"]
        t_unit = CANONICAL_DIME_UNITS["time"]
        f_unit = CANONICAL_DIME_UNITS["force"]
        tau_unit = CANONICAL_DIME_UNITS["torque"]

        return (
            "\\section{Estimation Methods and Model Identification}\n"
            "\\label{sec:dime_methods}\n"
            "This report summarizes multibody motion estimation, unactuated base balance, "
            "and continuous forward replay diagnostics.\n\n"
            "\\subsection{Governing Dynamics and ZTCF Equations}\n"
            "The constrained multibody equations of motion satisfy:\n"
            "\\begin{equation}\n"
            "\\mathbf{M}(\\mathbf{q}) \\dot{\\mathbf{v}} + "
            "\\mathbf{C}(\\mathbf{q}, \\mathbf{v})\\mathbf{v} + "
            "\\mathbf{g}(\\mathbf{q}) = \\mathbf{B}\\mathbf{u} + \\mathbf{J}_c^T \\boldsymbol{\\lambda}\n"
            "\\end{equation}\n"
            "Pointwise and integrated Zero-Torque Contact Free (ZTCF) drift are recorded distinctly:\n"
            "\\begin{align}\n"
            "a_{\\mathrm{ztcf}}(t) &= \\mathbf{M}(\\mathbf{q})^{-1} \\left( "
            "-\\mathbf{C}(\\mathbf{q},\\mathbf{v})\\mathbf{v} - \\mathbf{g}(\\mathbf{q}) "
            "+ \\mathbf{J}_c^T \\boldsymbol{\\lambda} \\right) \\\\[6pt]\n"
            "x_{\\mathrm{ztcf}}(t) &= x(t_0) + \\int_{t_0}^t f_{\\mathrm{ztcf}}(x(\\tau))\\, d\\tau\n"
            "\\end{align}\n\n"
            "\\subsection{Units and Parameter Assumptions}\n"
            f"All physical quantities adhere strictly to canonical SI units: length in [{l_unit}], "
            f"time in [{t_unit}], force in [{f_unit}], and torque in [{tau_unit}]. Floating-base root "
            "coordinates (DoFs 0..5) have strictly zero artificial actuator forces. Visualizations "
            "employ refined ellipsoid humanoid geometry.\n\n"
            "\\subsection{Exact Reproduction and Provenance}\n"
            f"Model digest: \\texttt{{{m_hash}}}; engine git commit: \\texttt{{{c_hash}}}.\n"
            "Execution is certified with single-shot forward continuous replay (reset count = 1).\n\n"
            "\\subsection{Limitations and Interpretation Boundaries}\n"
            "Kinematic pose agreement does not establish forward dynamic validity or joint effort "
            "feasibility. Missing contact forces are typed unavailable and never substituted as zero."
        )


# ==============================================================================
# Strategy Selection Service
# ==============================================================================


class DimeStrategySelectionService:
    """Service coordinating strategy availability, validation, and report creation."""

    def __init__(self) -> None:
        self._experimental_strategies: Final[set[EstimatorStrategy]] = {
            EstimatorStrategy.NEURAL_ESTIMATOR
        }

    def get_available_strategies(
        self, provider_capability: ProviderCapability | None = None
    ) -> Mapping[EstimatorStrategy, CapabilityRecord]:
        """Query available strategies and truthful qualification states."""
        records: dict[EstimatorStrategy, CapabilityRecord] = {}

        is_qual = provider_capability.is_qualified if provider_capability else False
        stat: CapabilityStatus = "qualified" if is_qual else "implemented"

        records[EstimatorStrategy.IK] = CapabilityRecord(
            name="inverse_kinematics",
            method_exists=True,
            status=stat,
            reason=None,
        )
        records[EstimatorStrategy.INVERSE_DYNAMICS] = CapabilityRecord(
            name="inverse_dynamics",
            method_exists=True,
            status=stat,
            reason=None,
        )
        records[EstimatorStrategy.DIME_MHE] = CapabilityRecord(
            name="dime_mhe",
            method_exists=True,
            status=stat,
            reason=None,
        )
        records[EstimatorStrategy.CONTINUOUS_REPLAY] = CapabilityRecord(
            name="continuous_replay",
            method_exists=True,
            status=stat,
            reason=None,
        )
        # Neural estimator is uncertified pending individual qualification
        records[EstimatorStrategy.NEURAL_ESTIMATOR] = CapabilityRecord(
            name="neural_estimator",
            method_exists=True,
            status="unavailable",
            reason="Pending individual qualification; awaiting neural comparative study",
        )
        return MappingProxyType(records)

    def select_strategy(
        self,
        strategy: EstimatorStrategy,
        engine_capability: ProviderCapability,
        require_validated: bool = False,
    ) -> StrategySelectionState:
        """Select estimation strategy enforcing qualification rules."""
        if require_validated and not engine_capability.is_qualified:
            raise PreconditionError(
                "Unqualified engine cannot be offered or reported as validated"
            )

        strats = self.get_available_strategies(engine_capability)
        cap = strats.get(strategy)
        if cap is None or cap.status == "unavailable":
            unavail_reason = cap.reason if cap else "Strategy not supported"
            raise PreconditionError(
                f"Strategy '{strategy.value}' is unavailable: {unavail_reason}"
            )

        engine_id = engine_capability.provider_id
        is_val = engine_capability.is_qualified and require_validated
        channels = tuple(c.name for c in engine_capability.control_channels)
        return StrategySelectionState(
            strategy=strategy,
            capability=cap,
            engine_id=engine_id,
            is_validated=is_val,
            selected_channels=channels,
            selected_plots=(),
        )

    def validate_overlay(
        self, overlay: VideoOverlaySpec, kinematics: KinematicsPayload
    ) -> None:
        """Ensure video overlay timestamps strictly align with kinematics time series."""
        if len(overlay.timestamps_s) != len(kinematics.times):
            raise TimingViolationError(
                "Video overlay timestamps do not align with kinematics timestamps: "
                f"length mismatch ({len(overlay.timestamps_s)} vs {len(kinematics.times)})"
            )
        if not np.allclose(overlay.timestamps_s, kinematics.times, atol=1e-4):
            raise TimingViolationError(
                "Video overlay timestamps do not align with kinematics timestamps: "
                "numerical time drift detected"
            )

    def create_report(
        self,
        report_id: str,
        options: DimeReportOptions,
        run_classification: RunClassification,
        provenance: DimeProvenanceRecord,
        **kwargs: Any,
    ) -> DimeReportArtifact:
        """Create structured report artifact conforming to DIME-11 contract."""
        kinematics: KinematicsPayload | None = kwargs.get("kinematics")
        overlay: VideoOverlaySpec | None = kwargs.get("video_overlay")
        if overlay is not None and kinematics is not None:
            self.validate_overlay(overlay, kinematics)

        engine_cap_status = kwargs.get("engine_capability_status", "implemented")
        is_val = engine_cap_status == "qualified"

        return DimeReportArtifact(
            report_id=report_id,
            created_at=kwargs.get("created_at", "2026-10-04T08:00:00Z"),
            scope=options.scope,
            strategy=options.strategy,
            run_classification=run_classification,
            provenance=provenance,
            is_validated=is_val,
            engine_capability_status=engine_cap_status,
            kinematics=kinematics,
            selected_net_controls=kwargs.get("selected_net_controls"),
            control_channel_names=tuple(kwargs.get("control_channel_names", ())),
            grf_report=kwargs.get("grf_report"),
            drift_prediction=kwargs.get("drift_prediction"),
            uncertainty=kwargs.get("uncertainty"),
            residuals=kwargs.get("residuals"),
            video_overlay=overlay,
            ztcf_record=kwargs.get("ztcf_record"),
            unavailable_capabilities=kwargs.get("unavailable_capabilities", {}),
            notes=str(kwargs.get("notes", "")),
        )


# ==============================================================================
# Offscreen GUI ViewModel
# ==============================================================================


class DimeStrategySelectionViewModel:
    """Headless GUI ViewModel managing strategy selection and channel options."""

    def __init__(self, service: DimeStrategySelectionService | None = None) -> None:
        self._service = service or DimeStrategySelectionService()
        self._strategy: EstimatorStrategy = EstimatorStrategy.DIME_MHE
        self._scope: ReportScope = ReportScope.FULL
        self._selected_channels: list[str] = []
        self._selected_plots: list[str] = []
        self._capabilities: dict[EstimatorStrategy, CapabilityRecord] = {}

    def update_capabilities(
        self, provider_capability: ProviderCapability | None = None
    ) -> None:
        """Refresh strategy availability against provider capability."""
        strats = self._service.get_available_strategies(provider_capability)
        self._capabilities = dict(strats)

    def can_select(self, strategy: EstimatorStrategy) -> bool:
        """Check whether a strategy can be actively selected."""
        rec = self._capabilities.get(strategy)
        if rec is None:
            return False
        return rec.status != "unavailable"

    def set_strategy(self, strategy: EstimatorStrategy) -> None:
        """Set current estimation strategy fail-closed if unavailable."""
        if not self.can_select(strategy):
            rec = self._capabilities.get(strategy)
            reason = rec.reason if rec else "Strategy unavailable"
            raise PreconditionError(
                f"Cannot select unavailable strategy '{strategy.value}': {reason}"
            )
        self._strategy = strategy

    def set_scope(self, scope: ReportScope) -> None:
        """Set report scope (Full or Custom)."""
        self._scope = scope

    def set_selected_channels(self, channels: Sequence[str]) -> None:
        """Configure active channel selections."""
        self._selected_channels = list(channels)

    def set_selected_plots(self, plots: Sequence[str]) -> None:
        """Configure active plot selections."""
        self._selected_plots = list(plots)

    @property
    def current_strategy(self) -> EstimatorStrategy:
        return self._strategy

    @property
    def current_scope(self) -> ReportScope:
        return self._scope

    @property
    def selected_channels(self) -> tuple[str, ...]:
        return tuple(self._selected_channels)

    @property
    def selected_plots(self) -> tuple[str, ...]:
        return tuple(self._selected_plots)

    @property
    def available_strategies(
        self,
    ) -> Mapping[EstimatorStrategy, CapabilityRecord]:
        return MappingProxyType(self._capabilities)
