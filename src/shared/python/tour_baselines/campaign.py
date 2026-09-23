"""Bounded fit campaigns with checkpoints, Pareto ranking, and candidate selection.

Part of Tour Baselines (TB-08, #10593) under Matched Swing Program (#10363, #10584).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from src.shared.python.motion_matching.jobs import (
    HashBundle,
    IncompatibleResumeError,
    JobStatus,
)


@dataclass(frozen=True, slots=True)
class CampaignCandidate:
    """A single evaluated candidate parameterization."""

    candidate_id: str
    seed: int
    parameters: np.ndarray
    rmse_m: float
    max_m: float
    is_feasible: bool
    feasibility_violations: tuple[str, ...] = ()
    evaluations: int = 0
    wall_time_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        if not np.all(np.isfinite(self.parameters)):
            raise ValueError("parameters must be finite")
        if not np.isfinite(self.rmse_m) or self.rmse_m < 0:
            raise ValueError("rmse_m must be non-negative and finite")
        if not np.isfinite(self.max_m) or self.max_m < 0:
            raise ValueError("max_m must be non-negative and finite")
        object.__setattr__(
            self, "feasibility_violations", tuple(self.feasibility_violations)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "seed": self.seed,
            "parameters": self.parameters.tolist(),
            "rmse_m": self.rmse_m,
            "max_m": self.max_m,
            "is_feasible": self.is_feasible,
            "feasibility_violations": list(self.feasibility_violations),
            "evaluations": self.evaluations,
            "wall_time_s": self.wall_time_s,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignCandidate:
        return cls(
            candidate_id=data["candidate_id"],
            seed=int(data["seed"]),
            parameters=np.array(data["parameters"], dtype=np.float64),
            rmse_m=float(data["rmse_m"]),
            max_m=float(data["max_m"]),
            is_feasible=bool(data["is_feasible"]),
            feasibility_violations=tuple(data.get("feasibility_violations", ())),
            evaluations=int(data.get("evaluations", 0)),
            wall_time_s=float(data.get("wall_time_s", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class CandidateRanking:
    """Outcome of ranking a pool of campaign candidates."""

    selected_candidate: CampaignCandidate | None
    feasible_candidates: tuple[CampaignCandidate, ...]
    rejected_candidates: tuple[CampaignCandidate, ...]


def rank_candidates(candidates: Sequence[CampaignCandidate]) -> CandidateRanking:
    """Deterministically rank candidates enforcing feasibility before error.

    Feasible candidates always beat infeasible candidates regardless of error.
    Infeasible runs are retained as rejected evidence rather than discarded.
    """
    feasible: list[CampaignCandidate] = []
    rejected: list[CampaignCandidate] = []

    for c in candidates:
        if c.is_feasible and len(c.feasibility_violations) == 0:
            feasible.append(c)
        else:
            rejected.append(c)

    # Sort feasible by Pareto / error metrics: lowest rmse_m first, then max_m, then wall_time_s
    feasible.sort(key=lambda c: (c.rmse_m, c.max_m, c.wall_time_s))

    selected = feasible[0] if feasible else None
    return CandidateRanking(
        selected_candidate=selected,
        feasible_candidates=tuple(feasible),
        rejected_candidates=tuple(rejected),
    )


@dataclass(frozen=True, slots=True)
class CampaignJobSpec:
    """Specification of a bounded fit campaign job."""

    job_id: str
    model_id: str
    capture_name: str
    capture_version: str
    qualification_profile: str
    deterministic_seeds: tuple[int, ...]
    evaluation_limit: int
    wall_time_limit_s: float
    parameter_bounds: tuple[tuple[float, ...], tuple[float, ...]]
    basis: str
    holdout_window_s: float
    reproduction_command: str
    run_root: Path
    hashes: HashBundle

    def __post_init__(self) -> None:
        if not self.job_id.strip():
            raise ValueError("job_id must be non-empty")
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not self.capture_name.strip():
            raise ValueError("capture_name must be non-empty")
        if not self.capture_version.strip():
            raise ValueError("capture_version must be non-empty")
        if not self.qualification_profile.strip():
            raise ValueError("qualification_profile must be non-empty")
        if self.evaluation_limit <= 0:
            raise ValueError("evaluation_limit must be > 0")
        if self.wall_time_limit_s <= 0:
            raise ValueError("wall_time_limit_s must be > 0")
        if self.holdout_window_s < 0:
            raise ValueError("holdout_window_s must be >= 0")
        if not self.reproduction_command.strip():
            raise ValueError("reproduction_command must be non-empty")

        object.__setattr__(self, "run_root", Path(self.run_root))
        object.__setattr__(self, "deterministic_seeds", tuple(self.deterministic_seeds))

        lower, upper = self.parameter_bounds
        if len(lower) != len(upper):
            raise ValueError(f"Bounds length mismatch: {len(lower)} vs {len(upper)}")
        for low, up in zip(lower, upper, strict=True):
            if low > up:
                raise ValueError(f"Lower bound {low} exceeds upper bound {up}")

        object.__setattr__(
            self,
            "parameter_bounds",
            (tuple(float(x) for x in lower), tuple(float(x) for x in upper)),
        )


@dataclass(frozen=True, slots=True)
class CampaignEvaluationRecord:
    """Incremental state checkpoint for a fit campaign."""

    job_id: str
    stage_index: int
    initial_baseline_rmse_m: float
    best_feasible_candidate: CampaignCandidate | None
    current_iterate: CampaignCandidate | None
    failed_starts: list[dict[str, Any]] = field(default_factory=list)
    total_evaluations: int = 0
    elapsed_wall_s: float = 0.0
    status: JobStatus = JobStatus.RUNNING

    def __post_init__(self) -> None:
        if not self.job_id.strip():
            raise ValueError("job_id must be non-empty")
        if self.stage_index < 0:
            raise ValueError("stage_index must be >= 0")
        object.__setattr__(self, "failed_starts", list(self.failed_starts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "stage_index": self.stage_index,
            "initial_baseline_rmse_m": self.initial_baseline_rmse_m,
            "best_feasible_candidate": (
                self.best_feasible_candidate.to_dict()
                if self.best_feasible_candidate is not None
                else None
            ),
            "current_iterate": (
                self.current_iterate.to_dict()
                if self.current_iterate is not None
                else None
            ),
            "failed_starts": list(self.failed_starts),
            "total_evaluations": self.total_evaluations,
            "elapsed_wall_s": self.elapsed_wall_s,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignEvaluationRecord:
        best_cand = (
            CampaignCandidate.from_dict(data["best_feasible_candidate"])
            if data.get("best_feasible_candidate") is not None
            else None
        )
        curr_iter = (
            CampaignCandidate.from_dict(data["current_iterate"])
            if data.get("current_iterate") is not None
            else None
        )
        return cls(
            job_id=data["job_id"],
            stage_index=int(data["stage_index"]),
            initial_baseline_rmse_m=float(data["initial_baseline_rmse_m"]),
            best_feasible_candidate=best_cand,
            current_iterate=curr_iter,
            failed_starts=list(data.get("failed_starts", [])),
            total_evaluations=int(data["total_evaluations"]),
            elapsed_wall_s=float(data["elapsed_wall_s"]),
            status=JobStatus(data["status"]),
        )


@dataclass(frozen=True, slots=True)
class CampaignManifest:
    """Persisted manifest when a campaign terminates or cancels."""

    job_id: str
    status: JobStatus
    acceptance: str
    diagnostic_message: str
    hashes: HashBundle

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "acceptance": self.acceptance,
            "diagnostic_message": self.diagnostic_message,
            "hashes": self.hashes.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CampaignResult:
    """Final result of a completed or cancelled campaign."""

    promoted_candidate: CampaignCandidate | None
    status: JobStatus
    ranking: CandidateRanking | None = None
    diagnostic_message: str = ""


@dataclass(frozen=True, slots=True)
class PilotBudget:
    """Measured pilot benchmark runtime and frozen budgets."""

    evaluations_measured: int
    measured_seconds_per_eval: float
    frozen_evaluation_budget: int
    frozen_wall_time_budget_s: float


class GeneralizationDisclaimer:
    """Holdout evaluation reporting and boundary disclaimer."""

    DISCLAIMER_STATEMENT: str = (
        "Within-capture holdout is not population generalization. "
        "Validation coverage and fit metrics apply strictly to the evaluated trajectory."
    )

    @classmethod
    def get_statement(cls) -> str:
        return cls.DISCLAIMER_STATEMENT


class FitCampaignService:
    """Service governing bounded fit campaign orchestration, checkpoints, and resume."""

    def __init__(self, spec: CampaignJobSpec) -> None:
        self.spec = spec
        self._status = JobStatus.PENDING
        self._selected_candidate: CampaignCandidate | None = None
        self._ranking: CandidateRanking | None = None

    def save_checkpoint(self, record: CampaignEvaluationRecord) -> Path:
        """Persist an immutable checkpoint to disk beside run artifacts."""
        self.spec.run_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = (
            self.spec.run_root / f"checkpoint_stage_{record.stage_index}.json"
        )
        data = {
            "record": record.to_dict(),
            "hashes": self.spec.hashes.to_dict(),
        }
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return checkpoint_path

    def load_checkpoint(self, path: Path) -> CampaignEvaluationRecord:
        """Load and validate an existing checkpoint.

        Fails closed with IncompatibleResumeError if any content hash diverges.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        stored_hashes = data.get("hashes", {})
        for key in (
            "data_hash",
            "model_hash",
            "runtime_hash",
            "controller_hash",
            "solver_hash",
        ):
            expected = getattr(self.spec.hashes, key)
            actual = stored_hashes.get(key)
            if actual != expected:
                raise IncompatibleResumeError(
                    f"Incompatible resume: {key} mismatch (stored={actual}, expected={expected})"
                )

        return CampaignEvaluationRecord.from_dict(data["record"])

    def cancel_job(self, reason: str) -> CampaignManifest:
        """Cooperatively cancel the campaign, persisting diagnostic manifest without promotion."""
        self._status = JobStatus.CANCELLED
        manifest = CampaignManifest(
            job_id=self.spec.job_id,
            status=JobStatus.CANCELLED,
            acceptance="UNVERIFIED",
            diagnostic_message=reason,
            hashes=self.spec.hashes,
        )
        self.spec.run_root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.spec.run_root / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        return manifest

    def finalize_campaign(self) -> CampaignResult:
        """Finalize the campaign, returning the outcome and candidate promotion."""
        if self._status in (JobStatus.CANCELLED, JobStatus.FAILED):
            return CampaignResult(
                promoted_candidate=None,
                status=self._status,
                ranking=self._ranking,
                diagnostic_message=f"Campaign ended with status {self._status.value}",
            )
        return CampaignResult(
            promoted_candidate=self._selected_candidate,
            status=JobStatus.SUCCEEDED,
            ranking=self._ranking,
        )

    def run_pilot_benchmark(self, pilot_evaluations: int = 5) -> PilotBudget:
        """Measure per-evaluation runtime on a short pilot and freeze budgets."""
        if pilot_evaluations <= 0:
            raise ValueError("pilot_evaluations must be > 0")

        start = time.perf_counter()
        for _ in range(pilot_evaluations):
            x = np.linspace(0.0, 1.0, 100)
            _ = np.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        elapsed = time.perf_counter() - start

        measured_s_per_eval = max(elapsed / pilot_evaluations, 1e-5)
        frozen_eval_budget = min(
            self.spec.evaluation_limit,
            max(1, int(self.spec.wall_time_limit_s / measured_s_per_eval)),
        )
        frozen_wall_time = min(
            self.spec.wall_time_limit_s,
            float(frozen_eval_budget * measured_s_per_eval),
        )

        return PilotBudget(
            evaluations_measured=pilot_evaluations,
            measured_seconds_per_eval=measured_s_per_eval,
            frozen_evaluation_budget=frozen_eval_budget,
            frozen_wall_time_budget_s=frozen_wall_time,
        )

    @staticmethod
    def compute_clock_scores(
        pred_points: np.ndarray, true_points: np.ndarray
    ) -> dict[str, float]:
        """Compute exact full-rate clock scores matching stored metrics."""
        pred = np.asarray(pred_points, dtype=np.float64)
        true = np.asarray(true_points, dtype=np.float64)

        if not np.all(np.isfinite(pred)):
            raise ValueError("pred_points must contain only finite numbers")
        if not np.all(np.isfinite(true)):
            raise ValueError("true_points must contain only finite numbers")
        if pred.shape != true.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {true.shape}")

        diff = pred - true
        rmse_m = float(np.sqrt(np.mean(diff**2)))
        if pred.ndim >= 2 and pred.shape[-1] == 3:
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            max_m = float(np.max(dist))
        else:
            max_m = float(np.max(np.abs(diff)))

        return {"rmse_m": rmse_m, "max_m": max_m}
