"""Club-only pendulum match contracts (CO-04 #10608).

Pure request/result types and seed mapping live here. Fit orchestration that
needs pendulum engine adapters lives in
``src.engines.physics_engines.pendulum.python.motion_matching.club_pendulum_match``
so shared code does not import engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.data_io.path_utils import get_repo_root
from src.shared.python.motion_matching.club_only.fit_outcomes import load_fit_outcomes
from src.shared.python.motion_matching.club_only.hub_accounting import HubMode
from src.shared.python.motion_matching.club_only.observation import ClubObservation
from src.shared.python.motion_matching.club_only.seeds import CandidateSeed

__all__ = [
    "PendulumMatchRequest",
    "PendulumMatchResult",
    "load_pendulum_match_outcomes",
    "map_seed_to_pendulum_q0",
    "reject_reconstruction_as_club_evidence",
]

_DEFAULT_PENDULUM_EVIDENCE = (
    Path("docs")
    / "plans"
    / "club_only_matching"
    / "evidence"
    / "club_pendulum_match.json"
)
_RECONSTRUCTION_PREFIX = "reconstruction_"
_DRIVEN_DOUBLE = "driven_double_pendulum"
_DRIVEN_TRIPLE = "driven_triple_pendulum"
_DOF = {_DRIVEN_DOUBLE: 2, _DRIVEN_TRIPLE: 3}


def load_pendulum_match_outcomes(
    path: Path | str | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load recorded fit outcomes keyed by ``(model_id, trial_id)`` (#10602)."""
    return load_fit_outcomes(
        get_repo_root() / _DEFAULT_PENDULUM_EVIDENCE if path is None else path
    )


@dataclass(frozen=True)
class PendulumMatchRequest:
    """Inputs for one model/trial club-only pendulum match."""

    observation: ClubObservation
    model_id: str
    hub_mode: HubMode
    max_nfev: int = 25
    seeds: tuple[CandidateSeed, ...] = ()
    geometry_hash: str | None = None
    profile_hash: str | None = None

    def __post_init__(self) -> None:
        reject_reconstruction_as_club_evidence(self.model_id)
        if self.model_id not in _DOF:
            raise ValueError(f"unsupported pendulum model_id={self.model_id!r}")
        if self.max_nfev < 1:
            raise ValueError("max_nfev must be >= 1")
        if self.model_id == _DRIVEN_DOUBLE and self.hub_mode is not HubMode.FIXED_PIVOT:
            raise ValueError("driven_double_pendulum requires HubMode.FIXED_PIVOT")
        mid = np.asarray(self.observation.mid_hands_xyz, dtype=np.float64)
        face = np.asarray(self.observation.face_xyz, dtype=np.float64)
        if not (np.all(np.isfinite(mid)) and np.all(np.isfinite(face))):
            raise ValueError("observation mid_hands/face positions must be finite")


@dataclass(frozen=True)
class PendulumMatchResult:
    """Best feasible club-only pendulum match for one model/trial cell."""

    model_id: str
    trial_id: str
    hub_variant_id: str
    hub_mode: HubMode
    theta: np.ndarray
    q0: np.ndarray
    v0: np.ndarray
    times_s: np.ndarray
    l_arm_m: float
    l_club_m: float
    l_hub_m: float | None
    in_plane_rmse_m: float
    original_3d_rmse_m: float
    first_frame_rmse_m: float
    t0_evaluated_before_step: bool
    coverage_fraction: float
    cold_start_rmse_m: float | None
    retrieval_start_rmse_m: float | None
    selected_start: str
    external_work_joules: float | None
    native_g1_pass: bool
    qualification_blockers: tuple[str, ...]
    claims_body_reconstruction_evidence: bool = False

    def __post_init__(self) -> None:
        if self.claims_body_reconstruction_evidence:
            raise ValueError("club match cannot claim body reconstruction evidence")
        if self.native_g1_pass:
            raise ValueError("CO-04 must not invent native G1 pass")
        if not self.qualification_blockers:
            raise ValueError("unmet native gates require named blockers")
        if self.selected_start not in {"cold", "retrieval"}:
            raise ValueError("selected_start must be cold or retrieval")
        object.__setattr__(
            self, "theta", np.asarray(self.theta, dtype=np.float64).copy()
        )
        object.__setattr__(self, "q0", np.asarray(self.q0, dtype=np.float64).copy())
        object.__setattr__(self, "v0", np.asarray(self.v0, dtype=np.float64).copy())
        object.__setattr__(
            self, "times_s", np.asarray(self.times_s, dtype=np.float64).copy()
        )
        object.__setattr__(
            self, "qualification_blockers", tuple(self.qualification_blockers)
        )


def reject_reconstruction_as_club_evidence(model_id: str) -> None:
    """Fail closed when body-only reconstruction IDs are offered as club evidence."""
    if not model_id:
        raise ValueError("model_id required")
    if model_id.startswith(_RECONSTRUCTION_PREFIX) or "reconstruction" in model_id:
        raise ValueError(
            f"cannot reuse body-only reconstruction scores as club match evidence "
            f"(model_id={model_id!r})"
        )


def map_seed_to_pendulum_q0(
    seed: CandidateSeed,
    *,
    model_id: str,
    geometry_hash: str,
    profile_hash: str,
) -> np.ndarray | None:
    """Map a CO-03 seed to pendulum q0 when DOF and content hashes are valid."""
    reject_reconstruction_as_club_evidence(model_id)
    expected = _DOF.get(model_id)
    if expected is None:
        return None
    if seed.model_id != model_id:
        return None
    if seed.geometry_hash != geometry_hash or seed.profile_hash != profile_hash:
        return None
    q = np.asarray(seed.q, dtype=np.float64)
    if q.ndim != 1 or q.size != expected or not np.all(np.isfinite(q)):
        return None
    return q.copy()
