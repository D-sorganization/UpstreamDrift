"""Versioned matching strategy contract and stage-separated qualification schema (PF-10, #10440).

Parent Epic: #10430, Program: #10363.

Provides:
1. QualificationStage:
   - MODEL_AVAILABLE: Engine SDK and model specification files are available.
   - KINEMATIC_FIT: Generalized coordinates match capture markers within RMS tolerance.
   - FORCE_FEASIBLE: Dynamic equilibrium M*a + b = S^T*tau + J^T*f satisfied with bounded residuals.
   - REPLAY_ACCEPTED: Forward simulation rollout matches candidate without divergence.
   - RUNTIME_BUDGET_MET: Optimization/simulation execution satisfies latency budget.
   - MUSCLE_QUALIFIED: Biological actuator forces/activations within physiological limits.

2. StageState: PASSED, FAILED, SKIPPED, NOT_EVALUATED.
3. StageReport: Detailed report per stage (state, metrics, thresholds, reason, details).
4. StageQualificationMatrix: Independent evaluation of all six qualification stages.
   - Invariant: Never marks a candidate 'accepted' merely because it animates or kinematic fit passed!
5. StrategyPreset: Canonical optimization and torque allocation presets.
6. ControllerSpecification: Low-level controller metadata and channel definitions.
7. ContactReactionHistory: Foot contact ground reactions and bilateral weld closure wrenches.
8. MatchingStrategyContract: Immutable strategy specification bound to model and capture.
9. CandidateStrategyPackage: Bundled candidate trajectory, strategy contract, and dynamics.
10. StrategyComparisonService: Cross-engine comparison, torque profiles, and capability auditing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.candidate import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateAuxiliary,
    CandidateMarkers,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)

logger = logging.getLogger(__name__)

STRATEGY_SCHEMA_VERSION = "matched-strategy-v1"

ALL_ENGINES: tuple[str, ...] = (
    "mujoco",
    "pinocchio",
    "drake",
    "opensim",
    "simscape",
    "myosuite",
)


class QualificationStage(str, Enum):
    """The six stage-separated qualification milestones in the matched swing program."""

    MODEL_AVAILABLE = "model_available"
    KINEMATIC_FIT = "kinematic_fit"
    FORCE_FEASIBLE = "force_feasible"
    REPLAY_ACCEPTED = "replay_accepted"
    RUNTIME_BUDGET_MET = "runtime_budget_met"
    MUSCLE_QUALIFIED = "muscle_qualified"


class StageState(str, Enum):
    """Execution status of a qualification stage."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    NOT_EVALUATED = "not_evaluated"


class StrategyPreset(str, Enum):
    """Standardized optimization objectives and solver presets."""

    MINIMUM_EFFORT = "minimum_effort"
    MINIMUM_TRAIL_ARM = "minimum_trail_arm"
    BARRIER_REDUCED = "barrier_reduced"
    CROCODDYL_DDP = "crocoddyl_ddp"
    DRAKE_QP = "drake_qp"
    MOCO_MUSCLE_TRACK = "moco_muscle_track"
    SIMSCAPE_FEEDFORWARD = "simscape_feedforward"


@dataclass(frozen=True)
class StageReport:
    """Detailed evaluation report for an individual qualification stage."""

    stage: QualificationStage
    state: StageState = StageState.NOT_EVALUATED
    metrics: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "state": self.state.value,
            "metrics": dict(self.metrics),
            "thresholds": dict(self.thresholds),
            "reason": self.reason,
            "details": dict(self.details),
            "evaluated_at": self.evaluated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StageReport:
        d = dict(data)
        d["stage"] = QualificationStage(d["stage"])
        d["state"] = StageState(d["state"])
        return cls(**d)


class StageQualificationMatrix:
    """Independent evaluation matrix across all six qualification stages."""

    def __init__(
        self, stages: Mapping[QualificationStage, StageReport] | None = None
    ) -> None:
        self._stages: dict[QualificationStage, StageReport] = {}
        if stages:
            for s, r in stages.items():
                self._stages[QualificationStage(s)] = r
        else:
            for s in QualificationStage:
                self._stages[s] = StageReport(stage=s, state=StageState.NOT_EVALUATED)

    @property
    def stages(self) -> dict[QualificationStage, StageReport]:
        return dict(self._stages)

    def get_report(self, stage: QualificationStage | str) -> StageReport:
        stg = QualificationStage(stage)
        return self._stages[stg]

    def record_stage(
        self,
        stage: QualificationStage | str,
        state: StageState | str,
        metrics: Mapping[str, float] | None = None,
        thresholds: Mapping[str, float] | None = None,
        reason: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        stg = QualificationStage(stage)
        st = StageState(state)
        report = StageReport(
            stage=stg,
            state=st,
            metrics=dict(metrics or {}),
            thresholds=dict(thresholds or {}),
            reason=reason,
            details=dict(details or {}),
        )
        self._stages[stg] = report

    def is_stage_passed(self, stage: QualificationStage | str) -> bool:
        stg = QualificationStage(stage)
        return self._stages[stg].state == StageState.PASSED

    def is_qualified_through(self, stage: QualificationStage | str) -> bool:
        """True if every stage in sequence up to and including `stage` has PASSED."""
        stg = QualificationStage(stage)
        ordered_stages = list(QualificationStage)
        target_idx = ordered_stages.index(stg)
        for i in range(target_idx + 1):
            s = ordered_stages[i]
            if self._stages[s].state != StageState.PASSED:
                return False
        return True

    def failing_stage(self) -> QualificationStage | None:
        """Return the first stage that encountered a FAILED status, if any."""
        for s in QualificationStage:
            if self._stages[s].state == StageState.FAILED:
                return s
        return None

    def overall_verdict(self) -> str:
        """Compute the high-level verdict.

        Rule: Replay acceptance is mandatory for 'accepted'.
        A purely kinematic fit or animating model is NEVER marked 'accepted'.
        """
        if self.failing_stage() is not None:
            return "rejected"

        if self.is_stage_passed(QualificationStage.REPLAY_ACCEPTED):
            # Check if all prior stages also passed
            if self.is_qualified_through(QualificationStage.REPLAY_ACCEPTED):
                return "accepted"
            return "rejected"

        # If any prior stage passed without failure, it is provisional
        for s in (
            QualificationStage.MODEL_AVAILABLE,
            QualificationStage.KINEMATIC_FIT,
            QualificationStage.FORCE_FEASIBLE,
        ):
            if self.is_stage_passed(s):
                return "provisional"

        return "not_evaluated"

    def to_dict(self) -> dict[str, Any]:
        failing = self.failing_stage()
        return {
            "stages": {s.value: r.to_dict() for s, r in self._stages.items()},
            "overall_verdict": self.overall_verdict(),
            "failing_stage": (failing.value if failing is not None else None),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StageQualificationMatrix:
        stages_dict = data.get("stages", {})
        stages = {
            QualificationStage(k): StageReport.from_dict(v)
            for k, v in stages_dict.items()
        }
        return cls(stages)


@dataclass(frozen=True)
class ControllerSpecification:
    """Specification of the control policy and actuation channels."""

    controller_type: str
    control_channels: tuple[str, ...] = ()
    gains: dict[str, float] = field(default_factory=dict)
    limits: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller_type": self.controller_type,
            "control_channels": list(self.control_channels),
            "gains": dict(self.gains),
            "limits": dict(self.limits),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ControllerSpecification:
        d = dict(data)
        if "control_channels" in d:
            d["control_channels"] = tuple(d["control_channels"])
        return cls(**d)


@dataclass(frozen=True)
class ContactReactionHistory:
    """Historical ground contact and bilateral grip closure reactions."""

    contact_names: tuple[str, ...] = ()
    reactions: NDArray[np.float64] | None = None
    closure_wrench: NDArray[np.float64] | None = None
    root_wrench_residual: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        for arr in (self.reactions, self.closure_wrench, self.root_wrench_residual):
            if arr is not None:
                arr.flags.writeable = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "contact_names": list(self.contact_names),
            "has_reactions": self.reactions is not None,
            "has_closure_wrench": self.closure_wrench is not None,
            "has_root_residual": self.root_wrench_residual is not None,
        }


@dataclass(frozen=True)
class MatchingStrategyContract:
    """Immutable contract specifying matching strategy, solver settings, and qualifications."""

    strategy_id: str
    engine: str
    model_id: str
    model_sha256: str
    capture_id: str
    strategy_preset: StrategyPreset | str = StrategyPreset.MINIMUM_EFFORT
    schema_version: str = STRATEGY_SCHEMA_VERSION
    frame_convention: str = "z_up_y_forward"
    coordinate_mapping: dict[str, int] = field(default_factory=dict)
    actuator_mapping: dict[str, int] = field(default_factory=dict)
    qualification: StageQualificationMatrix = field(
        default_factory=StageQualificationMatrix
    )
    controller: ControllerSpecification | None = None
    runtime_budget_ms: float | None = None
    measured_runtime_ms: float | None = None
    solver_hyperparameters: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != STRATEGY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported strategy schema version: {self.schema_version!r} (expected {STRATEGY_SCHEMA_VERSION!r})"
            )
        if isinstance(self.strategy_preset, str):
            try:
                object.__setattr__(
                    self, "strategy_preset", StrategyPreset(self.strategy_preset)
                )
            except ValueError:
                pass  # Allow custom preset strings

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategy_id": self.strategy_id,
            "engine": self.engine,
            "model_id": self.model_id,
            "model_sha256": self.model_sha256,
            "capture_id": self.capture_id,
            "strategy_preset": (
                self.strategy_preset.value
                if isinstance(self.strategy_preset, StrategyPreset)
                else str(self.strategy_preset)
            ),
            "frame_convention": self.frame_convention,
            "coordinate_mapping": dict(self.coordinate_mapping),
            "actuator_mapping": dict(self.actuator_mapping),
            "qualification": self.qualification.to_dict(),
            "controller": self.controller.to_dict() if self.controller else None,
            "runtime_budget_ms": self.runtime_budget_ms,
            "measured_runtime_ms": self.measured_runtime_ms,
            "solver_hyperparameters": dict(self.solver_hyperparameters),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MatchingStrategyContract:
        d = dict(data)
        if "qualification" in d and isinstance(d["qualification"], Mapping):
            d["qualification"] = StageQualificationMatrix.from_dict(d["qualification"])
        if "controller" in d and isinstance(d["controller"], Mapping):
            d["controller"] = ControllerSpecification.from_dict(d["controller"])
        if "strategy_preset" in d:
            try:
                d["strategy_preset"] = StrategyPreset(d["strategy_preset"])
            except ValueError:
                pass
        return cls(**d)


class CandidateStrategyPackage:
    """Unified container binding a MatchedSwingCandidate with its MatchingStrategyContract."""

    def __init__(
        self,
        candidate: MatchedSwingCandidate,
        strategy: MatchingStrategyContract,
        a: NDArray[np.float64] | None = None,
        reactions: ContactReactionHistory | None = None,
    ) -> None:
        self._candidate = candidate
        self._strategy = strategy

        if a is not None:
            a_arr = np.asarray(a, dtype=np.float64)
            if a_arr.shape[0] != candidate.n_frames:
                raise ValueError(
                    f"Acceleration frame count {a_arr.shape[0]} != candidate frames {candidate.n_frames}"
                )
            a_arr.flags.writeable = False
            self._a: NDArray[np.float64] | None = a_arr
        else:
            self._a = None

        self._reactions = reactions

    @property
    def candidate(self) -> MatchedSwingCandidate:
        return self._candidate

    @property
    def strategy(self) -> MatchingStrategyContract:
        return self._strategy

    @property
    def a(self) -> NDArray[np.float64] | None:
        return self._a

    @property
    def reactions(self) -> ContactReactionHistory | None:
        return self._reactions

    def remap_coordinates(
        self, target_coordinate_order: Sequence[str]
    ) -> CandidateStrategyPackage:
        """Remap generalized coordinate columns according to target_coordinate_order.

        Fails closed with KeyError if any coordinate in target_coordinate_order
        is missing from the source candidate.
        """
        source_names = list(self._candidate.metadata.coordinate_names)
        source_index = {name: i for i, name in enumerate(source_names)}

        indices: list[int] = []
        for name in target_coordinate_order:
            if name not in source_index:
                raise KeyError(
                    f"Missing coordinate {name!r} during remapping. "
                    f"Available coordinates: {source_names}"
                )
            indices.append(source_index[name])

        idx_arr = np.array(indices, dtype=int)
        remapped_q = self._candidate.q[:, idx_arr]
        remapped_v = (
            self._candidate.v[:, idx_arr] if self._candidate.v is not None else None
        )
        remapped_a = self._a[:, idx_arr] if self._a is not None else None

        meta_dict = self._candidate.metadata.to_dict()
        meta_dict["coordinate_names"] = list(target_coordinate_order)
        if meta_dict.get("velocity_names"):
            meta_dict["velocity_names"] = list(target_coordinate_order)
        new_meta = CandidateMetadata.from_dict(meta_dict)

        new_cand = MatchedSwingCandidate(
            metadata=new_meta,
            time_s=self._candidate.time_s,
            q=remapped_q,
            v=remapped_v,
            tau=self._candidate.tau,
            markers=self._candidate.markers,
            auxiliary=self._candidate.auxiliary,
        )

        strat_dict = self._strategy.to_dict()
        strat_dict["coordinate_mapping"] = {
            name: i for i, name in enumerate(target_coordinate_order)
        }
        new_strat = MatchingStrategyContract.from_dict(strat_dict)

        return CandidateStrategyPackage(
            candidate=new_cand,
            strategy=new_strat,
            a=remapped_a,
            reactions=self._reactions,
        )

    def save_package(self, path: Path | str) -> None:
        """Serialize CandidateStrategyPackage to an .npz archive without pickle."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, Any] = {
            "strategy_manifest_json": np.array(
                json.dumps(self._strategy.to_dict(), indent=2)
            ),
            "candidate_manifest_json": np.array(
                json.dumps(self._candidate.metadata.to_dict(), indent=2)
            ),
            "time_s": np.ascontiguousarray(self._candidate.time_s),
            "q": np.ascontiguousarray(self._candidate.q),
        }
        if self._candidate.v is not None:
            arrays["v"] = np.ascontiguousarray(self._candidate.v)
        if self._candidate.tau is not None:
            arrays["tau"] = np.ascontiguousarray(self._candidate.tau)
        if self._a is not None:
            arrays["a"] = np.ascontiguousarray(self._a)
        if self._reactions is not None and self._reactions.reactions is not None:
            arrays["reactions"] = np.ascontiguousarray(self._reactions.reactions)
            arrays["contact_names_json"] = np.array(
                json.dumps(list(self._reactions.contact_names))
            )
            if self._reactions.closure_wrench is not None:
                arrays["closure_wrench"] = np.ascontiguousarray(
                    self._reactions.closure_wrench
                )
            if self._reactions.root_wrench_residual is not None:
                arrays["root_wrench_residual"] = np.ascontiguousarray(
                    self._reactions.root_wrench_residual
                )

        np.savez(target_path, **arrays)

    @classmethod
    def load_package(cls, path: Path | str) -> CandidateStrategyPackage:
        """Load a CandidateStrategyPackage from an .npz archive without pickle."""
        p = Path(path)
        with np.load(p, allow_pickle=False) as data:
            if (
                "strategy_manifest_json" not in data
                or "candidate_manifest_json" not in data
            ):
                raise ValueError(f"Archive {p} is missing required manifest headers.")

            strat_raw = str(data["strategy_manifest_json"])
            cand_raw = str(data["candidate_manifest_json"])
            strat_dict = json.loads(strat_raw)
            cand_dict = json.loads(cand_raw)

            strategy = MatchingStrategyContract.from_dict(strat_dict)
            cand_meta = CandidateMetadata.from_dict(cand_dict)

            time_s = np.asarray(data["time_s"], dtype=np.float64)
            q = np.asarray(data["q"], dtype=np.float64)
            v = np.asarray(data["v"], dtype=np.float64) if "v" in data else None
            tau = np.asarray(data["tau"], dtype=np.float64) if "tau" in data else None
            a = np.asarray(data["a"], dtype=np.float64) if "a" in data else None

            cand = MatchedSwingCandidate(
                metadata=cand_meta,
                time_s=time_s,
                q=q,
                v=v,
                tau=tau,
            )

            reactions: ContactReactionHistory | None = None
            if "reactions" in data:
                names = (
                    tuple(json.loads(str(data["contact_names_json"])))
                    if "contact_names_json" in data
                    else ()
                )
                closure = (
                    np.asarray(data["closure_wrench"], dtype=np.float64)
                    if "closure_wrench" in data
                    else None
                )
                root_res = (
                    np.asarray(data["root_wrench_residual"], dtype=np.float64)
                    if "root_wrench_residual" in data
                    else None
                )
                reactions = ContactReactionHistory(
                    contact_names=names,
                    reactions=np.asarray(data["reactions"], dtype=np.float64),
                    closure_wrench=closure,
                    root_wrench_residual=root_res,
                )

            return cls(candidate=cand, strategy=strategy, a=a, reactions=reactions)


class StrategyComparisonService:
    """Service to evaluate and compare strategies and capabilities across engines."""

    def compare_strategies(
        self, packages: Sequence[CandidateStrategyPackage]
    ) -> dict[str, Any]:
        """Generate a comparison matrix across candidate strategy packages."""
        rows: list[dict[str, Any]] = []
        for pkg in packages:
            cand = pkg.candidate
            strat = pkg.strategy
            q_matrix = strat.qualification

            peak_tau = float(np.max(np.abs(cand.tau))) if cand.tau is not None else 0.0
            rms_tau = (
                float(np.sqrt(np.mean(cand.tau**2))) if cand.tau is not None else 0.0
            )

            kin_report = q_matrix.get_report(QualificationStage.KINEMATIC_FIT)
            marker_rmse = kin_report.metrics.get("marker_rmse_mm", 0.0)

            failing = q_matrix.failing_stage()
            row: dict[str, Any] = {
                "strategy_id": strat.strategy_id,
                "engine": strat.engine,
                "capture_id": strat.capture_id,
                "preset": (
                    strat.strategy_preset.value
                    if isinstance(strat.strategy_preset, StrategyPreset)
                    else str(strat.strategy_preset)
                ),
                "verdict": q_matrix.overall_verdict(),
                "failing_stage": failing.value if failing is not None else None,
                "whole_marker_rmse_mm": marker_rmse,
                "peak_effort_nm": peak_tau,
                "rms_effort_nm": rms_tau,
                "stages": {
                    s.value: q_matrix.get_report(s).state.value
                    for s in QualificationStage
                },
            }
            rows.append(row)

        return {"comparison_table": rows, "count": len(rows)}

    def extract_torque_profiles(
        self, packages: Sequence[CandidateStrategyPackage]
    ) -> dict[str, dict[str, Any]]:
        """Extract torque metrics per strategy package."""
        profiles: dict[str, dict[str, Any]] = {}
        for pkg in packages:
            key = pkg.strategy.engine
            cand = pkg.candidate
            if cand.tau is not None:
                peak = float(np.max(np.abs(cand.tau)))
                rms = float(np.sqrt(np.mean(cand.tau**2)))
                channel_peaks = [
                    float(np.max(np.abs(cand.tau[:, col])))
                    for col in range(cand.tau.shape[1])
                ]
            else:
                peak, rms, channel_peaks = 0.0, 0.0, []

            profiles[key] = {
                "strategy_id": pkg.strategy.strategy_id,
                "peak_effort_nm": peak,
                "mean_rms_effort_nm": rms,
                "channel_peaks": channel_peaks,
            }
        return profiles

    def extract_tracking_and_closure_errors(
        self, packages: Sequence[CandidateStrategyPackage]
    ) -> dict[str, dict[str, Any]]:
        """Extract kinematics and loop closure tracking error profiles."""
        errors: dict[str, dict[str, Any]] = {}
        for pkg in packages:
            key = pkg.strategy.engine
            kin_report = pkg.strategy.qualification.get_report(
                QualificationStage.KINEMATIC_FIT
            )
            force_report = pkg.strategy.qualification.get_report(
                QualificationStage.FORCE_FEASIBLE
            )

            closure_res = 0.0
            if pkg.reactions and pkg.reactions.closure_wrench is not None:
                closure_res = float(np.max(np.abs(pkg.reactions.closure_wrench)))

            errors[key] = {
                "strategy_id": pkg.strategy.strategy_id,
                "whole_marker_rmse_mm": kin_report.metrics.get("marker_rmse_mm", 0.0),
                "max_equilibrium_residual": force_report.metrics.get(
                    "max_residual", 0.0
                ),
                "closure_residual_norm": closure_res,
            }
        return errors

    def evaluate_engine_capabilities(
        self, engine: str, override_available: bool | None = None
    ) -> dict[str, Any]:
        """Evaluate if an engine backend has SDK support on the host.

        Invalidates supported state if engine runtime/SDK is missing.
        """
        is_available: bool
        if override_available is not None:
            is_available = override_available
        else:
            try:
                from src.shared.python.feature_registry import get_registry

                reg = get_registry()
                is_available = reg.is_available(engine)
            except Exception:
                # Fallback to direct import checks
                if engine == "mujoco":
                    try:
                        import mujoco  # noqa: F401

                        is_available = True
                    except ImportError:
                        is_available = False
                elif engine == "pinocchio":
                    try:
                        import pinocchio  # noqa: F401

                        is_available = True
                    except ImportError:
                        is_available = False
                else:
                    is_available = False

        status = StageState.PASSED.value if is_available else StageState.FAILED.value
        reason = (
            f"Engine SDK {engine} verified"
            if is_available
            else f"Engine SDK unavailable for {engine} on host system"
        )
        return {
            "engine": engine,
            "supported": is_available,
            "qualification_stage_status": status,
            "reason": reason,
        }


def create_sample_strategy_package(
    engine: str,
    capture: str = "driver",
    preset: StrategyPreset | str = StrategyPreset.MINIMUM_EFFORT,
    stage_up_to: QualificationStage | None = None,
    n_frames: int = 10,
    nv: int = 41,
) -> CandidateStrategyPackage:
    """Create a concrete sample strategy package for test fixtures or handoff (MV-03 #10479)."""
    coord_names = tuple(f"coord_{i}" for i in range(nv))
    time_s = np.linspace(0.0, 0.3, n_frames)
    q = np.zeros((n_frames, nv), dtype=np.float64)
    q[:, 2] = 0.85
    v = np.zeros((n_frames, nv), dtype=np.float64)
    a = np.zeros((n_frames, nv), dtype=np.float64)
    tau = np.ones((n_frames, nv - 6), dtype=np.float64) * 15.0

    meta = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.DYNAMIC,
        engine=engine,
        model_name=f"anthro_{capture}_{engine}",
        model_sha256=hashlib.sha256(f"model_{engine}".encode()).hexdigest(),
        coordinate_names=coord_names,
        velocity_names=coord_names,
        actuator_names=tuple(f"act_{i}" for i in range(nv - 6)),
    )
    cand = MatchedSwingCandidate(
        metadata=meta,
        time_s=time_s,
        q=q,
        v=v,
        tau=tau,
    )

    q_matrix = StageQualificationMatrix()
    if stage_up_to is not None:
        ordered = list(QualificationStage)
        limit_idx = ordered.index(stage_up_to)
        for i in range(limit_idx + 1):
            q_matrix.record_stage(
                ordered[i],
                StageState.PASSED,
                metrics={"marker_rmse_mm": 15.0, "max_residual": 5e-5},
                reason="Simulated benchmark verification",
            )

    strat = MatchingStrategyContract(
        strategy_id=f"sample-{engine}-{capture}",
        engine=engine,
        model_id=f"anthro_{capture}",
        model_sha256=meta.model_sha256,
        capture_id=capture,
        strategy_preset=preset,
        coordinate_mapping={name: i for i, name in enumerate(coord_names)},
        actuator_mapping={f"act_{i}": i for i in range(nv - 6)},
        qualification=q_matrix,
        runtime_budget_ms=2000.0,
        measured_runtime_ms=1450.0,
    )

    reactions = ContactReactionHistory(
        contact_names=("heel_r", "forefoot_r", "heel_l", "forefoot_l"),
        reactions=np.ones((n_frames, 12), dtype=np.float64) * 80.0,
        closure_wrench=np.zeros((n_frames, 6), dtype=np.float64),
        root_wrench_residual=np.zeros((n_frames, 6), dtype=np.float64),
    )

    return CandidateStrategyPackage(
        candidate=cand,
        strategy=strat,
        a=a,
        reactions=reactions,
    )
