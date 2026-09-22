"""Plausible upper-body and full-body candidate generation (CO-05 #10609).

Consumes club-only observations, CO-02 priors/profiles, and CO-03 seeds.
Produces a bounded, explainable candidate set with separated observation-fit,
plausibility, contact/effort, and runtime lanes. Qualification remains with
CO-06/08 and native owners — this module never claims G1/G3 acceptance.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.club_only.nullspace_proposals import (
    analyze_grip_jacobian_nullspace,
    propose_nullspace_offsets,
    reproject_onto_closure,
    synthetic_grip_jacobian,
)
from src.shared.python.motion_matching.club_only.observation import (
    ClubObservation,
    build_calibrated_observation_fixture,
    require_strictly_increasing_timestamps,
)
from src.shared.python.motion_matching.club_only.priors import GolfPlausibilityPriors
from src.shared.python.motion_matching.club_only.profiles import (
    ClubOnlyProfile,
    build_roster_profiles,
    get_club_only_profile,
)
from src.shared.python.motion_matching.club_only.seeds import (
    CandidateSeed,
    geometry_content_hash,
    profile_content_hash,
)
from src.shared.python.motion_matching.club_only.topology_mapping import (
    TopologyMapping,
    get_topology_mapping,
    map_reduced_seed_to_body,
)
from src.shared.python.motion_matching.club_only.workbook_identity import (
    CANONICAL_TRIAL_SHEETS,
)
from src.shared.python.tour_baselines.models import ModelTopology
from src.shared.python.tour_baselines.registry import (
    get_golf_model,
    init_default_registry,
    list_golf_models,
)

CANDIDATE_SCHEMA = "club-body-candidates/1.0.0"
_GOVERNING_ISSUE = 10609
_DEFAULT_SOURCE_MODEL = "driven_triple_pendulum"

# Models whose native runtime is commonly absent on agent hosts; cells stay
# unqualified with an explicit blocker rather than a surrogate animation.
_DEFAULT_MISSING_RUNTIME: frozenset[str] = frozenset(
    {
        "full_body_opensim",
        "full_body_simscape",
        "full_body_myosuite",
        "opensim_golfer",
        "myosuite_body",
    }
)

__all__ = [
    "CANDIDATE_SCHEMA",
    "BodyCandidate",
    "BodyCandidateOptions",
    "BodyCandidateReport",
    "BodyCandidateResult",
    "ModelTrialCell",
    "RejectionReason",
    "body_candidate_evidence_payload",
    "build_body_candidate_report",
    "generate_plausible_body_candidates",
]


class RejectionReason(str, Enum):
    """Fail-closed rejection modes for body candidates."""

    INCOMPATIBLE_GRIP = "incompatible_grip"
    CONTACT_INFEASIBLE = "contact_infeasible"
    SELF_COLLISION = "self_collision"
    SINGULAR_NULLSPACE = "singular_nullspace"
    MISSING_RUNTIME = "missing_runtime"
    NONFINITE = "nonfinite"
    WRONG_DIMS = "wrong_dims"
    INCOMPATIBLE_CLOCK = "incompatible_clock"


@dataclass(frozen=True)
class BodyCandidate:
    """One plausible body candidate with separated score lanes."""

    candidate_id: str
    trial_id: str
    model_id: str
    topology: str
    seed_id: str
    q: NDArray[np.float64]
    v: NDArray[np.float64]
    a: NDArray[np.float64]
    timestamps_s: NDArray[np.float64]
    observation_fit_m: float
    plausibility_score: float
    contact_effort: float
    runtime_s: float | None
    prior_strength: float
    body_configuration_hash: str
    closure_q_m: float
    closure_v_m_s: float
    closure_a_m_s2: float
    is_kinematic_preview: bool = True
    claims_native_acceptance: bool = False
    claims_surrogate_as_native: bool = False
    accepted: bool = True
    rejection_reasons: tuple[str, ...] = ()
    prior_sensitivity: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.trial_id or not self.model_id:
            raise ValueError("candidate_id, trial_id, and model_id required")
        if not self.seed_id:
            raise ValueError("seed_id required")
        q = np.asarray(self.q, dtype=np.float64)
        v = np.asarray(self.v, dtype=np.float64)
        a = np.asarray(self.a, dtype=np.float64)
        times = np.asarray(self.timestamps_s, dtype=np.float64)
        if q.ndim != 1 or v.ndim != 1 or a.ndim != 1:
            raise ValueError("q, v, a must be 1-D")
        if q.shape != v.shape or q.shape != a.shape:
            raise ValueError("q, v, a must share dimensions")
        if q.size < 1:
            raise ValueError("q, v, a must be non-empty")
        if not (
            np.all(np.isfinite(q)) and np.all(np.isfinite(v)) and np.all(np.isfinite(a))
        ):
            raise ValueError("q, v, a must be finite")
        times = require_strictly_increasing_timestamps(times)
        for name, value in (
            ("observation_fit_m", self.observation_fit_m),
            ("plausibility_score", self.plausibility_score),
            ("contact_effort", self.contact_effort),
            ("prior_strength", self.prior_strength),
            ("closure_q_m", self.closure_q_m),
            ("closure_v_m_s", self.closure_v_m_s),
            ("closure_a_m_s2", self.closure_a_m_s2),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0.0 and name != "plausibility_score":
                raise ValueError(f"{name} must be >= 0")
        if not 0.0 <= self.plausibility_score <= 1.0:
            raise ValueError("plausibility_score must be in [0, 1]")
        if self.runtime_s is not None and (
            not np.isfinite(self.runtime_s) or self.runtime_s < 0.0
        ):
            raise ValueError("runtime_s must be finite and >= 0 when set")
        if not self.body_configuration_hash:
            raise ValueError("body_configuration_hash required")
        if not self.is_kinematic_preview:
            raise ValueError("CO-05 candidates remain kinematic previews (CO-06)")
        if self.claims_native_acceptance:
            raise ValueError("CO-05 cannot claim native acceptance")
        if self.claims_surrogate_as_native:
            raise ValueError("surrogate/animation cannot stand in for native model")
        object.__setattr__(self, "q", q.copy())
        object.__setattr__(self, "v", v.copy())
        object.__setattr__(self, "a", a.copy())
        object.__setattr__(self, "timestamps_s", times.copy())
        object.__setattr__(self, "rejection_reasons", tuple(self.rejection_reasons))
        if self.prior_sensitivity is not None:
            object.__setattr__(self, "prior_sensitivity", dict(self.prior_sensitivity))

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "trial_id": self.trial_id,
            "model_id": self.model_id,
            "topology": self.topology,
            "seed_id": self.seed_id,
            "observation_fit_m": self.observation_fit_m,
            "plausibility_score": self.plausibility_score,
            "contact_effort": self.contact_effort,
            "runtime_s": self.runtime_s,
            "prior_strength": self.prior_strength,
            "body_configuration_hash": self.body_configuration_hash,
            "closure_q_m": self.closure_q_m,
            "closure_v_m_s": self.closure_v_m_s,
            "closure_a_m_s2": self.closure_a_m_s2,
            "is_kinematic_preview": self.is_kinematic_preview,
            "claims_native_acceptance": self.claims_native_acceptance,
            "claims_surrogate_as_native": self.claims_surrogate_as_native,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "prior_sensitivity": (
                dict(self.prior_sensitivity)
                if self.prior_sensitivity is not None
                else None
            ),
            "q_dim": int(self.q.size),
            "n_timestamps": int(self.timestamps_s.size),
        }


@dataclass(frozen=True)
class ModelTrialCell:
    """One model × trial roster cell with status and optional blocker."""

    model_id: str
    trial_id: str
    status: str
    blocker: str | None
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.model_id or not self.trial_id:
            raise ValueError("model_id and trial_id required")
        if self.status not in {
            "generated",
            "rejected",
            "missing_runtime",
            "unsupported",
        }:
            raise ValueError(f"unsupported cell status: {self.status!r}")
        if self.status != "generated" and not self.blocker:
            raise ValueError("non-generated cells require a precise blocker")
        object.__setattr__(self, "candidate_ids", tuple(self.candidate_ids))

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "trial_id": self.trial_id,
            "status": self.status,
            "blocker": self.blocker,
            "candidate_ids": list(self.candidate_ids),
        }


@dataclass(frozen=True)
class BodyCandidateResult:
    """Result of generating candidates for one observation/profile pair."""

    candidates: tuple[BodyCandidate, ...]
    cells: tuple[ModelTrialCell, ...]
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.as_dict() for c in self.candidates],
            "cells": [c.as_dict() for c in self.cells],
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class BodyCandidateReport:
    """Roster-wide body-candidate evidence report for CO-05."""

    schema: str
    governing_issue: int
    prior_strength: float
    candidates: tuple[BodyCandidate, ...]
    cells: tuple[ModelTrialCell, ...]
    limitations: tuple[str, ...]
    topology_mappings: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if self.schema != CANDIDATE_SCHEMA:
            raise ValueError(f"schema must be {CANDIDATE_SCHEMA!r}")
        if self.governing_issue != _GOVERNING_ISSUE:
            raise ValueError(f"governing_issue must be {_GOVERNING_ISSUE}")
        if not np.isfinite(self.prior_strength) or self.prior_strength <= 0.0:
            raise ValueError("prior_strength must be finite and > 0")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "governing_issue": self.governing_issue,
            "prior_strength": self.prior_strength,
            "candidates": [c.as_dict() for c in self.candidates],
            "cells": [c.as_dict() for c in self.cells],
            "limitations": list(self.limitations),
            "topology_mappings": list(self.topology_mappings),
        }


def _body_hash(q: NDArray[np.floating]) -> str:
    return hashlib.sha256(np.asarray(q, dtype=np.float64).tobytes()).hexdigest()


def _assert_compatible_clock(observation: ClubObservation, seed: CandidateSeed) -> None:
    obs_t = np.asarray(observation.native_time_s, dtype=np.float64)
    seed_t = np.asarray(seed.timestamps_s, dtype=np.float64)
    if obs_t.shape != seed_t.shape or not np.allclose(
        obs_t, seed_t, rtol=0.0, atol=0.0
    ):
        raise ValueError(
            "incompatible clocks: seed timestamps must match observation "
            "native_time_s exactly (no hidden time warp)"
        )


def _assert_finite_seed_q(seed: CandidateSeed) -> None:
    q = np.asarray(seed.q, dtype=np.float64)
    if q.ndim != 1 or q.size < 1:
        raise ValueError("wrong dims: seed.q must be a non-empty 1-D array")
    if not np.all(np.isfinite(q)):
        raise ValueError("seed.q must be finite")


def _reduced_source_for(target_model_id: str) -> str:
    """Choose a reduced source whose DOF is mappable into the target."""
    init_default_registry()
    target = get_golf_model(target_model_id)
    if target.topology is ModelTopology.CONSTRAINED_UPPER_BODY:
        return "driven_triple_pendulum"
    if target.topology is ModelTopology.FULL_BODY_MULTIBODY:
        return "driven_double_pendulum"
    return _DEFAULT_SOURCE_MODEL


def _default_mapping(target_model_id: str) -> TopologyMapping:
    return get_topology_mapping(
        source_model_id=_reduced_source_for(target_model_id),
        target_model_id=target_model_id,
    )


def _closure_triplet(
    q: NDArray[np.floating],
    *,
    grip_tol_m: float,
) -> tuple[float, float, float]:
    """Synthetic q/v/a closure residuals from configuration magnitude."""
    mag = float(np.linalg.norm(q))
    closure_q = min(grip_tol_m * 0.5, 1.0e-4 + 1.0e-3 * mag)
    closure_v = 1.0e-3 * mag
    closure_a = 1.0e-2 * mag
    return closure_q, closure_v, closure_a


def _plausibility(
    q: NDArray[np.floating],
    *,
    priors: GolfPlausibilityPriors,
    prior_strength: float,
    base_prior: float,
) -> tuple[float, dict[str, float]]:
    """Prior-dependent plausibility in [0, 1] with sensitivity envelope."""
    rate_penalty = float(np.mean(np.abs(q))) / max(priors.max_joint_rate_rad_s, 1.0)
    posture_term = priors.posture_regularization_weight * rate_penalty
    effort_term = priors.effort_regularization_weight * rate_penalty
    # Stronger prior pulls toward the seed prior_score; weaker leaves room.
    blended = base_prior - prior_strength * (posture_term + effort_term) * 0.1
    score = float(max(0.0, min(1.0, blended)))
    # Sensitivity: finite difference around the requested strength.
    alt = base_prior - (prior_strength * 1.1) * (posture_term + effort_term) * 0.1
    alt_score = float(max(0.0, min(1.0, alt)))
    return score, {
        "plausibility_delta": float(score - alt_score),
        "posture_term": float(posture_term),
        "effort_term": float(effort_term),
    }


def _contact_effort(q: NDArray[np.floating], *, prior_strength: float) -> float:
    return float(np.linalg.norm(q) * (0.5 + 0.25 * prior_strength))


def _rejected_candidate(
    *,
    candidate_id: str,
    seed: CandidateSeed,
    profile: ClubOnlyProfile,
    q: NDArray[np.float64],
    prior_strength: float,
    reason: RejectionReason,
    runtime_s: float,
) -> BodyCandidate:
    times = np.asarray(seed.timestamps_s, dtype=np.float64)
    zeros = np.zeros_like(q)
    return BodyCandidate(
        candidate_id=candidate_id,
        trial_id=seed.trial_id,
        model_id=profile.model_id,
        topology=profile.topology,
        seed_id=seed.seed_id,
        q=q,
        v=zeros,
        a=zeros,
        timestamps_s=times,
        observation_fit_m=float(seed.observed_residual_m),
        plausibility_score=0.0,
        contact_effort=0.0,
        runtime_s=runtime_s,
        prior_strength=prior_strength,
        body_configuration_hash=_body_hash(q),
        closure_q_m=float(profile.max_closure_residual_m() * 2.0),
        closure_v_m_s=0.0,
        closure_a_m_s2=0.0,
        accepted=False,
        rejection_reasons=(reason.value,),
    )


@dataclass(frozen=True)
class BodyCandidateOptions:
    """Optional knobs for CO-05 candidate generation (keeps the public arity low)."""

    n_nullspace_proposals: int = 3
    runtime_available: bool = True
    runtime_blocker: str | None = None
    force_reject: RejectionReason | None = None


def _co05_limitations(profile: ClubOnlyProfile) -> list[str]:
    limitations = list(profile.limitations)
    limitations.append(
        "CO-05 candidates are kinematic previews; torque replay and scientific "
        "qualification remain with CO-06/08 and native owners."
    )
    return limitations


def _missing_runtime_result(
    *,
    observation: ClubObservation,
    profile: ClubOnlyProfile,
    options: BodyCandidateOptions,
) -> BodyCandidateResult:
    missing_blocker = options.runtime_blocker or (
        f"native runtime unavailable for model={profile.model_id!r}; "
        "cell remains unqualified"
    )
    cell = ModelTrialCell(
        model_id=profile.model_id,
        trial_id=observation.trial_id,
        status="missing_runtime",
        blocker=missing_blocker,
        candidate_ids=(),
    )
    return BodyCandidateResult(
        candidates=(),
        cells=(cell,),
        limitations=tuple(_co05_limitations(profile)),
    )


@dataclass(frozen=True)
class _SeedExpansionCtx:
    observation: ClubObservation
    profile: ClubOnlyProfile
    priors: GolfPlausibilityPriors
    prior_strength: float
    options: BodyCandidateOptions
    started: float


def _accepted_from_seed(
    *,
    ctx: _SeedExpansionCtx,
    seed: CandidateSeed,
    q0: NDArray[np.float64],
    analysis: Any,
) -> list[BodyCandidate]:
    """Expand one seed into accepted null-space proposals (or a singular reject)."""
    if analysis.is_singular:
        return [
            _rejected_candidate(
                candidate_id=f"reject:{seed.seed_id}:singular",
                seed=seed,
                profile=ctx.profile,
                q=q0,
                prior_strength=ctx.prior_strength,
                reason=RejectionReason.SINGULAR_NULLSPACE,
                runtime_s=float(time.perf_counter() - ctx.started),
            )
        ]

    n_prop = int(ctx.options.n_nullspace_proposals)
    amplitudes = tuple(0.015 * (i + 1) for i in range(max(1, n_prop - 1)))
    offsets = propose_nullspace_offsets(analysis, amplitudes=amplitudes)[:n_prop]
    return [
        _build_accepted_proposal(ctx=ctx, seed=seed, q_prop=q0 + offset, index=index)
        for index, offset in enumerate(offsets)
    ]


def _build_accepted_proposal(
    *,
    ctx: _SeedExpansionCtx,
    seed: CandidateSeed,
    q_prop: NDArray[np.float64],
    index: int,
) -> BodyCandidate:
    closure_q, _, _ = _closure_triplet(q_prop, grip_tol_m=ctx.priors.grip_closure_tol_m)
    q_proj = reproject_onto_closure(
        q_prop,
        closure_residual_m=closure_q,
        tol_m=ctx.profile.max_closure_residual_m(),
    )
    closure_q, closure_v, closure_a = _closure_triplet(
        q_proj, grip_tol_m=ctx.priors.grip_closure_tol_m
    )
    plaus, sensitivity = _plausibility(
        q_proj,
        priors=ctx.priors,
        prior_strength=ctx.prior_strength,
        base_prior=float(seed.prior_score),
    )
    return BodyCandidate(
        candidate_id=f"body:{ctx.profile.model_id}:{seed.seed_id}:{index}",
        trial_id=ctx.observation.trial_id,
        model_id=ctx.profile.model_id,
        topology=ctx.profile.topology,
        seed_id=seed.seed_id,
        q=q_proj,
        v=np.zeros_like(q_proj),
        a=np.zeros_like(q_proj),
        timestamps_s=np.asarray(ctx.observation.native_time_s, dtype=np.float64),
        observation_fit_m=float(seed.observed_residual_m),
        plausibility_score=plaus,
        contact_effort=_contact_effort(q_proj, prior_strength=ctx.prior_strength),
        runtime_s=float(time.perf_counter() - ctx.started),
        prior_strength=ctx.prior_strength,
        body_configuration_hash=_body_hash(q_proj),
        closure_q_m=closure_q,
        closure_v_m_s=closure_v,
        closure_a_m_s2=closure_a,
        accepted=True,
        prior_sensitivity=sensitivity,
    )


def _candidates_for_seeds(
    *,
    observation: ClubObservation,
    profile: ClubOnlyProfile,
    seeds: Sequence[CandidateSeed],
    priors: GolfPlausibilityPriors,
    prior_strength: float,
    options: BodyCandidateOptions,
) -> list[BodyCandidate]:
    mapping = _default_mapping(profile.model_id)
    target_nq = int(mapping.target_nq)
    jacobian = synthetic_grip_jacobian(target_nq, n_constraints=min(3, target_nq))
    analysis = analyze_grip_jacobian_nullspace(jacobian)
    ctx = _SeedExpansionCtx(
        observation=observation,
        profile=profile,
        priors=priors,
        prior_strength=prior_strength,
        options=options,
        started=time.perf_counter(),
    )
    candidates: list[BodyCandidate] = []
    for seed in seeds:
        q0 = map_reduced_seed_to_body(
            q_reduced=np.asarray(seed.q, dtype=np.float64),
            mapping=mapping,
            target_nq=target_nq,
        )
        if options.force_reject is not None:
            reason = options.force_reject
            candidates.append(
                _rejected_candidate(
                    candidate_id=f"reject:{seed.seed_id}:{reason.value}",
                    seed=seed,
                    profile=profile,
                    q=q0,
                    prior_strength=prior_strength,
                    reason=reason,
                    runtime_s=float(time.perf_counter() - ctx.started),
                )
            )
            continue
        candidates.extend(
            _accepted_from_seed(ctx=ctx, seed=seed, q0=q0, analysis=analysis)
        )
    return candidates


@precondition(
    lambda observation, profile, seeds, priors, prior_strength, options=None: (
        isinstance(observation, ClubObservation)
        and isinstance(profile, ClubOnlyProfile)
        and isinstance(priors, GolfPlausibilityPriors)
    ),
    "observation, profile, priors required",
)
@postcondition(
    lambda result: isinstance(result, BodyCandidateResult),
    "must return BodyCandidateResult",
)
def generate_plausible_body_candidates(
    *,
    observation: ClubObservation,
    profile: ClubOnlyProfile,
    seeds: Sequence[CandidateSeed],
    priors: GolfPlausibilityPriors,
    prior_strength: float,
    options: BodyCandidateOptions | None = None,
) -> BodyCandidateResult:
    """Generate a bounded set of upper/full-body candidates from club seeds.

    Fail-closed on nonfinite/wrong dims/incompatible clocks. Missing runtime
    yields an unqualified cell without fabricating native success.
    """
    opts = options or BodyCandidateOptions()
    if not np.isfinite(prior_strength) or prior_strength <= 0.0:
        raise ValueError("prior_strength must be finite and > 0")
    if int(opts.n_nullspace_proposals) < 1:
        raise ValueError("n_nullspace_proposals must be >= 1")
    if not opts.runtime_available:
        return _missing_runtime_result(
            observation=observation, profile=profile, options=opts
        )
    if not seeds:
        raise ValueError("seeds must be non-empty when runtime is available")

    for seed in seeds:
        _assert_finite_seed_q(seed)
        _assert_compatible_clock(observation, seed)
        if seed.trial_id != observation.trial_id:
            raise ValueError("seed trial_id incompatible with observation")

    candidates = _candidates_for_seeds(
        observation=observation,
        profile=profile,
        seeds=seeds,
        priors=priors,
        prior_strength=prior_strength,
        options=opts,
    )
    accepted_ids = tuple(c.candidate_id for c in candidates if c.accepted)
    if accepted_ids:
        cell_status, cell_blocker = "generated", None
    else:
        cell_status = "rejected"
        cell_blocker = "all proposals rejected (grip/contact/collision/singular)"
    cell = ModelTrialCell(
        model_id=profile.model_id,
        trial_id=observation.trial_id,
        status=cell_status,
        blocker=cell_blocker,
        candidate_ids=accepted_ids,
    )
    return BodyCandidateResult(
        candidates=tuple(candidates),
        cells=(cell,),
        limitations=tuple(_co05_limitations(profile)),
    )


def _synthetic_seed_for(
    *,
    observation: ClubObservation,
    profile: ClubOnlyProfile,
) -> CandidateSeed:
    g_hash = geometry_content_hash(
        club_type=observation.club_type,
        catalog_length_m=observation.catalog_length_m,
        tool_to_model_residual_m=0.0,
    )
    p_hash = profile_content_hash(profile)
    source = _reduced_source_for(profile.model_id)
    source_nq = int(get_golf_model(source).dof)
    q = np.zeros(source_nq, dtype=np.float64)
    grip = np.asarray(observation.mid_hands_xyz, dtype=np.float64)
    mean_z = float(np.nanmean(grip[:, 2])) if grip.size else 0.0
    if np.isfinite(mean_z) and source_nq >= 1:
        q[0] = 0.05 * mean_z
    if source_nq >= 2:
        q[1] = -0.02 * mean_z
    return CandidateSeed(
        seed_id=f"co05-seed:{observation.trial_id}:{profile.model_id}",
        trial_id=observation.trial_id,
        model_id=profile.model_id,
        source="constrained_ik",
        q=q,
        body_configuration_hash=_body_hash(q),
        observed_residual_m=0.01,
        prior_score=0.65,
        feasibility_reasons=("synthetic_co05_seed", "kinematic_preview"),
        timestamps_s=np.asarray(observation.native_time_s, dtype=np.float64),
        geometry_hash=g_hash,
        profile_hash=p_hash,
        body_is_prior=True,
    )


def _cell_blocker_for(model_id: str, topology: str) -> tuple[str, str] | None:
    """Return (status, blocker) when the model cannot generate body candidates."""
    if model_id in _DEFAULT_MISSING_RUNTIME:
        return (
            "missing_runtime",
            f"native runtime for {model_id} not available on agent host; "
            "unqualified pending CO-06/native owner",
        )
    if topology == ModelTopology.PLANAR_DRIVEN_PENDULUM.value:
        return (
            "unsupported",
            "planar pendulum owned by CO-04; upper/full-body candidates out of scope",
        )
    if topology == ModelTopology.KINEMATIC_RECONSTRUCTION.value:
        return (
            "unsupported",
            "kinematic reconstruction is not an upper/full-body native model",
        )
    if topology == ModelTopology.REFERENCE_CATALOG_URDF.value:
        return (
            "unsupported",
            "reference catalog URDF/MJCF requires a native adapter receipt",
        )
    return None


def build_body_candidate_report(
    *,
    model_ids: Sequence[str] | None = None,
    trial_ids: Sequence[str] | None = None,
    prior_strength: float = 1.0,
    n_nullspace_proposals: int = 2,
) -> BodyCandidateReport:
    """Build the roster × trial body-candidate matrix for evidence."""
    if not np.isfinite(prior_strength) or prior_strength <= 0.0:
        raise ValueError("prior_strength must be finite and > 0")
    init_default_registry()
    roster = build_roster_profiles()
    models = (
        list(model_ids)
        if model_ids is not None
        else [m.model_id for m in list_golf_models()]
    )
    trials = list(trial_ids) if trial_ids is not None else list(CANONICAL_TRIAL_SHEETS)

    all_candidates: list[BodyCandidate] = []
    all_cells: list[ModelTrialCell] = []
    limitations: list[str] = [
        "Synthetic fixtures validate software contracts only; not native G1 evidence.",
        "Pendulum fit paths remain owned by CO-04 (#10608).",
    ]
    mappings: list[dict[str, Any]] = []
    seen_maps: set[tuple[str, str]] = set()

    for model_id in models:
        profile = (
            roster[model_id] if model_id in roster else get_club_only_profile(model_id)
        )
        blocked = _cell_blocker_for(model_id, profile.topology)
        for trial_id in trials:
            if blocked is not None:
                status, blocker = blocked
                all_cells.append(
                    ModelTrialCell(
                        model_id=model_id,
                        trial_id=trial_id,
                        status=status,
                        blocker=blocker,
                        candidate_ids=(),
                    )
                )
                continue

            obs = build_calibrated_observation_fixture(trial_id)
            seed = _synthetic_seed_for(observation=obs, profile=profile)
            map_key = (_reduced_source_for(model_id), model_id)
            if map_key not in seen_maps:
                mapping = get_topology_mapping(
                    source_model_id=map_key[0],
                    target_model_id=map_key[1],
                )
                mappings.append(mapping.as_dict())
                seen_maps.add(map_key)

            result = generate_plausible_body_candidates(
                observation=obs,
                profile=profile,
                seeds=(seed,),
                priors=profile.golf_priors(),
                prior_strength=prior_strength,
                options=BodyCandidateOptions(
                    n_nullspace_proposals=n_nullspace_proposals,
                    runtime_available=True,
                ),
            )
            all_candidates.extend(result.candidates)
            all_cells.extend(result.cells)
            for note in result.limitations:
                if note not in limitations:
                    limitations.append(note)

    return BodyCandidateReport(
        schema=CANDIDATE_SCHEMA,
        governing_issue=_GOVERNING_ISSUE,
        prior_strength=prior_strength,
        candidates=tuple(all_candidates),
        cells=tuple(all_cells),
        limitations=tuple(limitations),
        topology_mappings=tuple(mappings),
    )


def body_candidate_evidence_payload(
    report: BodyCandidateReport | None = None,
) -> dict[str, Any]:
    """Serialize the CO-05 evidence receipt."""
    built = report if report is not None else build_body_candidate_report()
    models: dict[str, Any] = {}
    trials: dict[str, Any] = {}
    for cell in built.cells:
        models.setdefault(cell.model_id, {"cells": []})
        models[cell.model_id]["cells"].append(cell.as_dict())
        trials.setdefault(cell.trial_id, {"cells": []})
        trials[cell.trial_id]["cells"].append(cell.as_dict())

    return {
        "schema": built.schema,
        "governing_issue": built.governing_issue,
        "prior_strength": built.prior_strength,
        "models": {
            model_id: {
                "model_id": model_id,
                "cell_count": len(payload["cells"]),
                "statuses": sorted({c["status"] for c in payload["cells"]}),
            }
            for model_id, payload in models.items()
        },
        "trials": {
            trial_id: {
                "trial_id": trial_id,
                "cell_count": len(payload["cells"]),
            }
            for trial_id, payload in trials.items()
        },
        "candidate_count": len(built.candidates),
        "accepted_candidate_count": sum(1 for c in built.candidates if c.accepted),
        "cell_count": len(built.cells),
        "topology_mappings": list(built.topology_mappings),
        "limitations": list(built.limitations),
        "notes": [
            "Upper/full-body providers consume club-only observations via this contract.",
            "Observation fit, plausibility, contact/effort, and runtime stay separate.",
            "Native qualification and continuous replay remain with CO-06/08.",
        ],
    }
