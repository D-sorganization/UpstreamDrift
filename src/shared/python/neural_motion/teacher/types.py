"""Typed contracts for NM-04 teacher generation (#10619)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.shared.python.dataset_tools.canonical import N_COEFFS

__all__ = [
    "TEACHER_CAMPAIGN_SCHEMA",
    "AcquisitionMode",
    "TeacherAnchor",
    "TeacherGenerationSpec",
    "TeacherRolloutRequest",
    "TeacherRolloutResult",
]

TEACHER_CAMPAIGN_SCHEMA = "neural-teacher-campaign/1.0.0"


class AcquisitionMode(str, Enum):
    """Active-learning acquisition policies (test labels are never consumed)."""

    UNCERTAINTY = "uncertainty"
    DISAGREEMENT = "disagreement"
    RANDOM_CONTROL = "random_control"


@dataclass(frozen=True)
class TeacherAnchor:
    """Reviewed starting point near a valid solution (CO-03 / baseline swing)."""

    anchor_id: str
    trial_id: str
    model_id: str
    source: str
    q0: np.ndarray
    coefficients: np.ndarray
    geometry_stratum: str
    contact_stratum: str
    club_stratum: str

    def __post_init__(self) -> None:
        for name, value in (
            ("anchor_id", self.anchor_id),
            ("trial_id", self.trial_id),
            ("model_id", self.model_id),
            ("source", self.source),
            ("geometry_stratum", self.geometry_stratum),
            ("contact_stratum", self.contact_stratum),
            ("club_stratum", self.club_stratum),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        q0 = np.asarray(self.q0, dtype=np.float64)
        if q0.ndim != 1 or q0.size < 1 or not np.all(np.isfinite(q0)):
            raise ValueError("q0 must be a finite 1-D array")
        coeffs = np.asarray(self.coefficients, dtype=np.float64)
        if coeffs.shape != (N_COEFFS,):
            raise ValueError(f"coefficients must have shape ({N_COEFFS},)")
        object.__setattr__(self, "q0", q0.copy())
        object.__setattr__(self, "coefficients", coeffs.copy())

    def content_digest(self) -> str:
        payload = {
            "anchor_id": self.anchor_id,
            "trial_id": self.trial_id,
            "model_id": self.model_id,
            "source": self.source,
            "q0": hashlib.sha256(self.q0.tobytes()).hexdigest(),
            "coefficients": hashlib.sha256(self.coefficients.tobytes()).hexdigest(),
            "geometry_stratum": self.geometry_stratum,
            "contact_stratum": self.contact_stratum,
            "club_stratum": self.club_stratum,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TeacherGenerationSpec:
    """Frozen campaign parameters aligned with NM-01 nested stages."""

    model_id: str
    campaign_id: str
    master_seed: int
    nested_stages: tuple[int, ...]
    max_episodes_per_stage: int
    store_root: Path
    rejected_root: Path
    ledger_path: Path
    acquisition_log_path: Path
    state_path: Path

    def __post_init__(self) -> None:
        if not self.model_id or not self.campaign_id:
            raise ValueError("model_id and campaign_id required")
        if self.master_seed < 0:
            raise ValueError("master_seed must be non-negative")
        if not self.nested_stages:
            raise ValueError("nested_stages must be non-empty")
        if any(stage <= 0 for stage in self.nested_stages):
            raise ValueError("nested_stages must be positive")
        if self.max_episodes_per_stage <= 0:
            raise ValueError("max_episodes_per_stage must be positive")
        object.__setattr__(self, "store_root", Path(self.store_root))
        object.__setattr__(self, "rejected_root", Path(self.rejected_root))
        object.__setattr__(self, "ledger_path", Path(self.ledger_path))
        object.__setattr__(
            self, "acquisition_log_path", Path(self.acquisition_log_path)
        )
        object.__setattr__(self, "state_path", Path(self.state_path))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": TEACHER_CAMPAIGN_SCHEMA,
            "model_id": self.model_id,
            "campaign_id": self.campaign_id,
            "master_seed": self.master_seed,
            "nested_stages": list(self.nested_stages),
            "max_episodes_per_stage": self.max_episodes_per_stage,
            "store_root": str(self.store_root),
            "rejected_root": str(self.rejected_root),
        }


@dataclass(frozen=True)
class TeacherRolloutRequest:
    """One bounded rollout attempt from an anchor plus perturbation."""

    anchor: TeacherAnchor
    attempt_index: int
    stage_index: int
    master_seed: int
    perturbation: np.ndarray
    duration_s: float = 0.5
    n_samples: int = 6

    def __post_init__(self) -> None:
        pert = np.asarray(self.perturbation, dtype=np.float64)
        if pert.ndim != 1 or pert.size < 1 or not np.all(np.isfinite(pert)):
            raise ValueError("perturbation must be a finite 1-D array")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if self.n_samples < 2:
            raise ValueError("n_samples must be >= 2")
        object.__setattr__(self, "perturbation", pert.copy())

    def attempt_key(self) -> str:
        pert_digest = hashlib.sha256(self.perturbation.tobytes()).hexdigest()[:12]
        return (
            f"{self.master_seed}:{self.stage_index}:{self.attempt_index}:{pert_digest}"
        )


@dataclass(frozen=True)
class TeacherRolloutResult:
    """Outcome of one teacher solve + optional independent replay digest."""

    feasible: bool
    teacher_objective: float
    convergence_iterations: int
    independent_replay_digest: str
    simulation_cost_units: float
    rejection_reason: str = ""
    channel_availability: Mapping[str, str] | None = None
    sample_times_s: np.ndarray | None = None
    q: np.ndarray | None = None
    v: np.ndarray | None = None
    u: np.ndarray | None = None
    a_native: np.ndarray | None = None
    q_next: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.teacher_objective) or self.teacher_objective < 0.0:
            raise ValueError("teacher_objective must be finite and >= 0")
        if self.convergence_iterations < 0:
            raise ValueError("convergence_iterations must be >= 0")
        if not self.independent_replay_digest:
            raise ValueError("independent_replay_digest required")
        if len(self.independent_replay_digest) != 64:
            raise ValueError("independent_replay_digest must be 64 hex chars")
        if (
            not np.isfinite(self.simulation_cost_units)
            or self.simulation_cost_units < 0
        ):
            raise ValueError("simulation_cost_units must be finite and >= 0")
        if self.feasible and self.q is None:
            raise ValueError("feasible rollouts must include q trajectory")
