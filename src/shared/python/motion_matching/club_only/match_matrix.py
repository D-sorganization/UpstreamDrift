"""Eight-cell double/triple × four-trial club-only pendulum match matrix (CO-04)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.shared.python.motion_matching.club_only.hub_accounting import HubMode
from src.shared.python.motion_matching.club_only.observation import (
    build_calibrated_observation_fixture,
)
from src.shared.python.motion_matching.club_only.pendulum_match import (
    PendulumMatchRequest,
    PendulumMatchResult,
    match_club_pendulum,
)
from src.shared.python.motion_matching.club_only.replay_package import (
    build_replay_package,
)
from src.shared.python.motion_matching.club_only.workbook_identity import (
    CANONICAL_TRIAL_SHEETS,
)

MATCH_SCHEMA = "club-pendulum-match/1.0.0"
PENDULUM_MATCH_MODELS: tuple[str, ...] = (
    "driven_double_pendulum",
    "driven_triple_pendulum",
)
_GOVERNING_ISSUE = 10608

__all__ = [
    "MATCH_SCHEMA",
    "PENDULUM_MATCH_MODELS",
    "MatchMatrixOutcome",
    "PendulumMatchMatrix",
    "build_pendulum_match_matrix",
    "evidence_payload",
]


@dataclass(frozen=True)
class MatchMatrixOutcome:
    """One model/trial software-contract outcome with replay inputs and blockers."""

    model_id: str
    trial_id: str
    hub_variant_id: str
    in_plane_rmse_m: float
    original_3d_rmse_m: float
    first_frame_rmse_m: float
    t0_evaluated_before_step: bool
    coverage_fraction: float
    selected_start: str
    external_work_joules: float | None
    native_g1_pass: bool
    qualification_blockers: tuple[str, ...]
    replay_inputs: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "trial_id": self.trial_id,
            "hub_variant_id": self.hub_variant_id,
            "in_plane_rmse_m": self.in_plane_rmse_m,
            "original_3d_rmse_m": self.original_3d_rmse_m,
            "first_frame_rmse_m": self.first_frame_rmse_m,
            "t0_evaluated_before_step": self.t0_evaluated_before_step,
            "coverage_fraction": self.coverage_fraction,
            "selected_start": self.selected_start,
            "external_work_joules": self.external_work_joules,
            "native_g1_pass": self.native_g1_pass,
            "qualification_blockers": list(self.qualification_blockers),
            "replay_inputs": dict(self.replay_inputs),
        }


@dataclass(frozen=True)
class PendulumMatchMatrix:
    """Complete 2×4 club-only pendulum match matrix."""

    schema: str
    governing_issue: int
    outcomes: tuple[MatchMatrixOutcome, ...]

    def __post_init__(self) -> None:
        if self.schema != MATCH_SCHEMA:
            raise ValueError(f"schema must be {MATCH_SCHEMA!r}")
        if len(self.outcomes) != 8:
            raise ValueError("matrix must contain exactly eight model/trial outcomes")


def _hub_mode_for(model_id: str) -> HubMode:
    if model_id == "driven_double_pendulum":
        return HubMode.FIXED_PIVOT
    if model_id == "driven_triple_pendulum":
        return HubMode.PRESCRIBED_MOVING_HUB
    raise ValueError(f"unsupported model_id={model_id!r}")


def _outcome_from_result(result: PendulumMatchResult) -> MatchMatrixOutcome:
    package = build_replay_package(result)
    return MatchMatrixOutcome(
        model_id=result.model_id,
        trial_id=result.trial_id,
        hub_variant_id=result.hub_variant_id,
        in_plane_rmse_m=result.in_plane_rmse_m,
        original_3d_rmse_m=result.original_3d_rmse_m,
        first_frame_rmse_m=result.first_frame_rmse_m,
        t0_evaluated_before_step=result.t0_evaluated_before_step,
        coverage_fraction=result.coverage_fraction,
        selected_start=result.selected_start,
        external_work_joules=result.external_work_joules,
        native_g1_pass=package.native_g1_pass,
        qualification_blockers=package.qualification_blockers,
        replay_inputs=dict(package.replay_inputs),
    )


def build_pendulum_match_matrix(*, max_nfev: int = 8) -> PendulumMatchMatrix:
    """Fit every driven double/triple × canonical trial cell (software contracts)."""
    outcomes: list[MatchMatrixOutcome] = []
    for model_id in PENDULUM_MATCH_MODELS:
        hub_mode = _hub_mode_for(model_id)
        for trial_id in CANONICAL_TRIAL_SHEETS:
            obs = build_calibrated_observation_fixture(trial_id)
            result = match_club_pendulum(
                PendulumMatchRequest(
                    observation=obs,
                    model_id=model_id,
                    hub_mode=hub_mode,
                    max_nfev=max_nfev,
                    seeds=(),
                )
            )
            outcomes.append(_outcome_from_result(result))
    return PendulumMatchMatrix(
        schema=MATCH_SCHEMA,
        governing_issue=_GOVERNING_ISSUE,
        outcomes=tuple(outcomes),
    )


def evidence_payload(matrix: PendulumMatchMatrix | None = None) -> dict[str, Any]:
    """Serialize the CO-04 evidence receipt."""
    mat = matrix if matrix is not None else build_pendulum_match_matrix()
    return {
        "schema": mat.schema,
        "governing_issue": mat.governing_issue,
        "models": list(PENDULUM_MATCH_MODELS),
        "trials": list(CANONICAL_TRIAL_SHEETS),
        "outcomes": [o.as_dict() for o in mat.outcomes],
        "notes": [
            "Software-contract matrix only; native G1 remains blocked with named gates.",
            "In-plane and original 3D errors are reported separately per outcome.",
            "Fixed-pivot double and prescribed moving-hub triple use distinct hub IDs.",
            "Body-only reconstruction scores are rejected as club match evidence.",
        ],
    }
