"""Fast club-only matching presets, caching, pruning, and Pareto diversity (CO-07 #10611).

Orchestrates CO-03 seeds, CO-05-style branch generation, and CO-06 verification
budgets without inventing native Fit/G1 success. Software-contract scoring only.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.data_io.path_utils import get_repo_root
from src.shared.python.motion_matching.club_only.constrained_ik import (
    generate_posture_branches,
)
from src.shared.python.motion_matching.club_only.native_g1_gates import (
    validate_native_g1_claim_contract,
)
from src.shared.python.motion_matching.club_only.observation import ClubObservation
from src.shared.python.motion_matching.club_only.profiles import ClubOnlyProfile
from src.shared.python.motion_matching.club_only.seeds import CandidateSeed
from src.shared.python.motion_matching.club_only.topology_mapping import (
    get_topology_mapping,
    map_reduced_seed_to_body,
)

FAST_MATCH_SCHEMA = "club-fast-matching/1.0.0"
FAST_MATCH_EVIDENCE_FILENAME: Final[str] = "club_fast_matching.json"
_GOVERNING_ISSUE = 10611
_DEFAULT_BLOCKERS = (
    "native_g1_qualification_requires_desk_native_receipt",
    "software_contract_fast_match_is_not_native_evidence",
)

__all__ = [
    "FAST_MATCH_EVIDENCE_FILENAME",
    "FAST_MATCH_SCHEMA",
    "BranchScore",
    "CacheKeyMismatchError",
    "EmptyNeuralProposalProvider",
    "FastMatchOptions",
    "FastMatchResult",
    "ImmutableCacheKey",
    "ImmutableMatchCache",
    "MatchBudget",
    "MatchCancelledError",
    "MatchCheckpoint",
    "MatchPreset",
    "MatchStageTimings",
    "NeuralProposalProvider",
    "QualityTimeSample",
    "StartStrategy",
    "budget_for_preset",
    "build_fast_match_evidence",
    "cache_key_from_parts",
    "checkpoint_identity",
    "prune_and_select_pareto",
    "run_fast_club_match",
    "save_fast_match_evidence",
    "score_branch",
    "target_content_hash",
]


class MatchPreset(str, Enum):
    """Saved time/evaluation budgets for preview vs verified fits."""

    FAST_PREVIEW = "fast_preview"
    VERIFIED_FIT = "verified_fit"


class StartStrategy(str, Enum):
    """Compared warm-start families (cold vs retrieval vs reduced-to-full)."""

    COLD = "cold"
    RETRIEVAL = "retrieval"
    REDUCED_TO_FULL = "reduced_to_full"


class MatchCancelledError(RuntimeError):
    """Raised when an explicit cancel hook fires before the budget is exhausted."""


class CacheKeyMismatchError(ValueError):
    """Raised when a resume/checkpoint key does not match stored cache identity."""


@dataclass(frozen=True)
class MatchBudget:
    """Bounded time and evaluation limits for one fast-match run."""

    max_time_s: float
    max_evaluations: int
    max_pareto: int

    def __post_init__(self) -> None:
        for name, value in (
            ("max_time_s", self.max_time_s),
            ("max_evaluations", self.max_evaluations),
            ("max_pareto", self.max_pareto),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        if isinstance(self.max_evaluations, bool):
            raise ValueError("max_evaluations must be int")


_PRESET_BUDGETS: dict[MatchPreset, MatchBudget] = {
    MatchPreset.FAST_PREVIEW: MatchBudget(
        max_time_s=8.0, max_evaluations=48, max_pareto=5
    ),
    MatchPreset.VERIFIED_FIT: MatchBudget(
        max_time_s=180.0, max_evaluations=400, max_pareto=12
    ),
}


@dataclass(frozen=True)
class FastMatchOptions:
    """Optional knobs for a fast-match run (CO-05-style options object)."""

    preset: MatchPreset = MatchPreset.FAST_PREVIEW
    budget: MatchBudget | None = None
    cache: ImmutableMatchCache | None = None
    checkpoint: MatchCheckpoint | None = None
    neural_provider: NeuralProposalProvider | None = None
    cancel_check: Callable[[], bool] | None = None
    n_branches: int = 3
    diversity_seed: int = _GOVERNING_ISSUE
    strategies: tuple[StartStrategy, ...] = (
        StartStrategy.COLD,
        StartStrategy.RETRIEVAL,
        StartStrategy.REDUCED_TO_FULL,
    )

    def __post_init__(self) -> None:
        if self.n_branches < 1:
            raise ValueError("n_branches must be >= 1")
        object.__setattr__(self, "strategies", tuple(self.strategies))
        if not self.strategies:
            raise ValueError("strategies must be non-empty")


@dataclass(frozen=True)
class _ScoreLoopCtx:
    """Private scoring-loop knobs collapsed for architecture param budgets."""

    seed: CandidateSeed
    profile: ClubOnlyProfile
    start_index: int
    evaluations_used: int
    budget: MatchBudget
    t0: float
    deadline: float
    cancel_check: Callable[[], bool] | None


@dataclass(frozen=True)
class _AssembleCtx:
    """Private assemble-result knobs collapsed for architecture param budgets."""

    preset: MatchPreset
    budget: MatchBudget
    evaluations_used: int
    start_index: int
    branch_count: int
    cancelled: bool
    neural_n: int
    diversity_seed: int
    quality_vs_time: tuple[QualityTimeSample, ...]
    load_s: float
    calibration_s: float
    ik_s: float
    dynamics_s: float


def budget_for_preset(preset: MatchPreset) -> MatchBudget:
    """Return the saved default budget for a preset."""
    if not isinstance(preset, MatchPreset):
        raise TypeError("preset must be MatchPreset")
    return _PRESET_BUDGETS[preset]


@dataclass(frozen=True)
class ImmutableCacheKey:
    """Key for immutable target/model/profile caches."""

    trial_id: str
    model_id: str
    geometry_hash: str
    profile_hash: str
    target_content_hash: str

    def as_tuple(self) -> tuple[str, str, str, str, str]:
        return (
            self.trial_id,
            self.model_id,
            self.geometry_hash,
            self.profile_hash,
            self.target_content_hash,
        )


def cache_key_from_parts(
    *,
    trial_id: str,
    model_id: str,
    geometry_hash: str,
    profile_hash: str,
    target_content_hash: str,
) -> ImmutableCacheKey:
    """Build a validated immutable cache key."""
    for name, value in (
        ("trial_id", trial_id),
        ("model_id", model_id),
        ("geometry_hash", geometry_hash),
        ("profile_hash", profile_hash),
        ("target_content_hash", target_content_hash),
    ):
        if not value:
            raise ValueError(f"{name} must be non-empty")
    return ImmutableCacheKey(
        trial_id=trial_id,
        model_id=model_id,
        geometry_hash=geometry_hash,
        profile_hash=profile_hash,
        target_content_hash=target_content_hash,
    )


def target_content_hash(observation: ClubObservation) -> str:
    """Hash observation clock + trial identity for cache invalidation."""
    if not isinstance(observation, ClubObservation):
        raise TypeError("observation must be ClubObservation")
    times = np.asarray(observation.native_time_s, dtype=np.float64)
    if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
        raise ValueError("native_time_s must be finite with length >= 2")
    payload = {
        "trial_id": observation.trial_id,
        "n_samples": int(times.size),
        "t0": float(times[0]),
        "t_end": float(times[-1]),
        "dt_median": float(np.median(np.diff(times))),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ImmutableMatchCache:
    """In-memory cache for immutable load/calibration payloads."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, ...], Mapping[str, Any]] = {}

    def store(self, key: ImmutableCacheKey, payload: Mapping[str, Any]) -> None:
        if not isinstance(key, ImmutableCacheKey):
            raise TypeError("key must be ImmutableCacheKey")
        self._store[key.as_tuple()] = dict(payload)

    def load(self, key: ImmutableCacheKey) -> Mapping[str, Any] | None:
        return self._store.get(key.as_tuple())

    def assert_compatible(
        self, stored: ImmutableCacheKey, requested: ImmutableCacheKey
    ) -> None:
        """Fail closed when resume/checkpoint keys disagree."""
        for field in (
            "trial_id",
            "model_id",
            "geometry_hash",
            "profile_hash",
            "target_content_hash",
        ):
            if getattr(stored, field) != getattr(requested, field):
                raise CacheKeyMismatchError(
                    f"{field} mismatch between stored cache and requested run"
                )


@dataclass(frozen=True)
class MatchCheckpoint:
    """Resumable fast-match state keyed by cache identity."""

    cache_key: ImmutableCacheKey
    preset: MatchPreset
    evaluations_used: int
    branch_index: int
    pareto_ids: tuple[str, ...]
    diversity_seed: int

    def __post_init__(self) -> None:
        if self.evaluations_used < 0 or self.branch_index < 0:
            raise ValueError("evaluations_used and branch_index must be >= 0")
        object.__setattr__(self, "pareto_ids", tuple(self.pareto_ids))


def checkpoint_identity(checkpoint: MatchCheckpoint) -> str:
    """Stable identity hash for checkpoint/resume contracts."""
    payload = {
        "cache_key": checkpoint.cache_key.as_tuple(),
        "preset": checkpoint.preset.value,
        "evaluations_used": checkpoint.evaluations_used,
        "branch_index": checkpoint.branch_index,
        "pareto_ids": list(checkpoint.pareto_ids),
        "diversity_seed": checkpoint.diversity_seed,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class MatchStageTimings:
    """Measured stage times including verification."""

    load_s: float
    calibration_s: float
    ik_s: float
    dynamics_s: float
    replay_s: float
    verification_s: float

    @property
    def total_s(self) -> float:
        return (
            self.load_s
            + self.calibration_s
            + self.ik_s
            + self.dynamics_s
            + self.replay_s
            + self.verification_s
        )


@dataclass(frozen=True)
class BranchScore:
    """One scored branch with feasibility and effort lanes separated."""

    branch_id: str
    start_strategy: StartStrategy
    q: NDArray[np.float64]
    observation_fit_m: float
    effort: float
    closure_m: float
    feasible: bool
    rejection_reasons: tuple[str, ...]
    configuration_hash: str

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=np.float64)
        if q.ndim != 1 or q.size < 1 or not np.all(np.isfinite(q)):
            raise ValueError("q must be finite 1-D")
        for name, value in (
            ("observation_fit_m", self.observation_fit_m),
            ("effort", self.effort),
            ("closure_m", self.closure_m),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        object.__setattr__(self, "q", q.copy())
        object.__setattr__(self, "rejection_reasons", tuple(self.rejection_reasons))


def _configuration_hash(q: NDArray[np.floating]) -> str:
    return hashlib.sha256(np.asarray(q, dtype=np.float64).tobytes()).hexdigest()


def _closure_m(q: NDArray[np.floating]) -> float:
    mag = float(np.linalg.norm(q))
    return 1.0e-4 + 1.0e-3 * mag


def score_branch(
    *,
    branch_id: str,
    start_strategy: StartStrategy,
    q: NDArray[np.floating],
    seed_residual_m: float,
    profile: ClubOnlyProfile,
) -> BranchScore:
    """Score one branch under observation and closure gates (software contract)."""
    if not branch_id:
        raise ValueError("branch_id required")
    if not np.isfinite(seed_residual_m) or seed_residual_m < 0.0:
        raise ValueError("seed_residual_m must be finite and >= 0")
    q_arr = np.asarray(q, dtype=np.float64)
    delta = float(np.linalg.norm(q_arr))
    fit = float(seed_residual_m + 0.002 * delta)
    effort = delta
    closure = _closure_m(q_arr)
    max_closure = profile.max_closure_residual_m()
    # LoD: read observation thresholds via a local alias (no deep chains).
    observation_profile = profile.observation
    max_obs = float(observation_profile.max_grip_position_rmse_m)
    reasons: list[str] = []
    feasible = True
    if closure > max_closure:
        feasible = False
        reasons.append("closure_exceeds_profile")
    if fit > max_obs:
        feasible = False
        reasons.append("observation_gate_exceeded")
    return BranchScore(
        branch_id=branch_id,
        start_strategy=start_strategy,
        q=q_arr,
        observation_fit_m=fit,
        effort=effort,
        closure_m=closure,
        feasible=feasible,
        rejection_reasons=tuple(reasons),
        configuration_hash=_configuration_hash(q_arr),
    )


def _dominates(a: BranchScore, b: BranchScore) -> bool:
    """Pareto dominance on observation fit and effort (both minimized)."""
    better_fit = a.observation_fit_m <= b.observation_fit_m
    better_effort = a.effort <= b.effort
    strictly = (a.observation_fit_m < b.observation_fit_m) or (a.effort < b.effort)
    return better_fit and better_effort and strictly


@postcondition(
    lambda result: isinstance(result, tuple) and len(result) == 2,
    "must return (pareto, rejected)",
)
def prune_and_select_pareto(
    branches: Sequence[BranchScore],
    *,
    profile: ClubOnlyProfile,
    max_pareto: int,
) -> tuple[tuple[BranchScore, ...], tuple[BranchScore, ...]]:
    """Prune infeasible branches first, then retain a bounded Pareto set."""
    if max_pareto < 1:
        raise ValueError("max_pareto must be >= 1")
    if not isinstance(profile, ClubOnlyProfile):
        raise TypeError("profile must be ClubOnlyProfile")
    feasible = [b for b in branches if b.feasible]
    rejected = [b for b in branches if not b.feasible]
    pareto: list[BranchScore] = []
    for candidate in sorted(feasible, key=lambda b: (b.observation_fit_m, b.effort)):
        if any(_dominates(existing, candidate) for existing in pareto):
            continue
        pareto = [p for p in pareto if not _dominates(candidate, p)]
        pareto.append(candidate)
        if len(pareto) >= max_pareto:
            break
    # Diversity: drop averaged body motion — keep distinct configuration hashes.
    seen: set[str] = set()
    diverse: list[BranchScore] = []
    for item in pareto:
        if item.configuration_hash in seen:
            continue
        seen.add(item.configuration_hash)
        diverse.append(item)
    return tuple(diverse), tuple(rejected)


class NeuralProposalProvider(Protocol):
    """Optional neural epic hook; classical completion does not require weights."""

    def propose(
        self,
        *,
        observation: ClubObservation,
        profile: ClubOnlyProfile,
    ) -> tuple[CandidateSeed, ...]: ...


class EmptyNeuralProposalProvider:
    """Default slot: no trained network required for classical fast matching."""

    def propose(
        self,
        *,
        observation: ClubObservation,
        profile: ClubOnlyProfile,
    ) -> tuple[CandidateSeed, ...]:
        return ()


@dataclass(frozen=True)
class QualityTimeSample:
    """One point on the quality-versus-time curve retained for CO-08."""

    evaluations: int
    wall_s: float
    best_observation_fit_m: float
    failed_attempts: int

    def __post_init__(self) -> None:
        if self.evaluations < 0 or self.failed_attempts < 0:
            raise ValueError("evaluations and failed_attempts must be >= 0")
        if not np.isfinite(self.wall_s) or self.wall_s < 0.0:
            raise ValueError("wall_s must be finite and >= 0")
        if (
            not np.isfinite(self.best_observation_fit_m)
            or self.best_observation_fit_m < 0.0
        ):
            raise ValueError("best_observation_fit_m must be finite and >= 0")


@dataclass(frozen=True)
class FastMatchResult:
    """Outcome of one fast-match run with profiling and Pareto diversity."""

    schema: str
    governing_issue: int
    preset: MatchPreset
    pareto: tuple[BranchScore, ...]
    rejected: tuple[BranchScore, ...]
    start_strategy_best: StartStrategy | None
    evaluations_used: int
    failed_attempts: int
    cancelled: bool
    neural_proposals_used: int
    profiling: MatchStageTimings
    quality_vs_time: tuple[QualityTimeSample, ...]
    checkpoint: MatchCheckpoint | None
    limitations: tuple[str, ...]
    claims_native_qualification: bool = False
    native_g1_pass: bool = False
    qualification_blockers: tuple[str, ...] = _DEFAULT_BLOCKERS

    def __post_init__(self) -> None:
        if self.failed_attempts < 0:
            raise ValueError("failed_attempts must be >= 0")
        object.__setattr__(self, "quality_vs_time", tuple(self.quality_vs_time))
        validate_native_g1_claim_contract(
            native_g1_pass=self.native_g1_pass,
            claims_native_qualification=self.claims_native_qualification,
            qualification_blockers=self.qualification_blockers,
        )


def _warm_q_for_strategy(
    *,
    strategy: StartStrategy,
    seed: CandidateSeed,
    profile: ClubOnlyProfile,
) -> NDArray[np.float64]:
    q_seed = np.asarray(seed.q, dtype=np.float64)
    if strategy is StartStrategy.COLD:
        return np.zeros_like(q_seed)
    if strategy is StartStrategy.RETRIEVAL:
        return q_seed.copy()
    mapping = get_topology_mapping(
        source_model_id=seed.model_id,
        target_model_id=profile.model_id,
    )
    if q_seed.size != mapping.source_nq:
        return q_seed.copy()
    return map_reduced_seed_to_body(
        q_reduced=q_seed,
        mapping=mapping,
        target_nq=mapping.target_nq,
    )


def _iter_branch_specs(
    *,
    seed: CandidateSeed,
    profile: ClubOnlyProfile,
    n_branches: int,
    strategies: Sequence[StartStrategy],
) -> list[tuple[str, StartStrategy, NDArray[np.float64]]]:
    specs: list[tuple[str, StartStrategy, NDArray[np.float64]]] = []
    for strategy in strategies:
        warm = _warm_q_for_strategy(strategy=strategy, seed=seed, profile=profile)
        branches = generate_posture_branches(
            warm, n_branches=n_branches, amplitude_rad=0.04
        )
        for index, q in enumerate(branches):
            specs.append((f"{strategy.value}:{index}", strategy, q))
    return specs


def _prepare_cache_and_key(
    *,
    observation: ClubObservation,
    profile: ClubOnlyProfile,
    seed: CandidateSeed,
    geometry_hash: str,
    profile_hash: str,
    cache: ImmutableMatchCache | None,
    checkpoint: MatchCheckpoint | None,
) -> tuple[ImmutableCacheKey, ImmutableMatchCache, float, float]:
    """Load target hash, validate resume key, and warm immutable cache."""
    t_load = time.perf_counter()
    t_hash = target_content_hash(observation)
    key = cache_key_from_parts(
        trial_id=observation.trial_id,
        model_id=profile.model_id,
        geometry_hash=geometry_hash,
        profile_hash=profile_hash,
        target_content_hash=t_hash,
    )
    cache_obj = cache if cache is not None else ImmutableMatchCache()
    if checkpoint is not None:
        cache_obj.assert_compatible(checkpoint.cache_key, key)
    load_s = time.perf_counter() - t_load

    t_cal = time.perf_counter()
    cached = cache_obj.load(key)
    if cached is None:
        cache_obj.store(key, {"warm_q": np.asarray(seed.q, dtype=np.float64).tolist()})
    calibration_s = time.perf_counter() - t_cal
    return key, cache_obj, load_s, calibration_s


def _collect_branch_specs_with_neural(
    *,
    seed: CandidateSeed,
    profile: ClubOnlyProfile,
    observation: ClubObservation,
    n_branches: int,
    strategies: Sequence[StartStrategy],
    neural_provider: NeuralProposalProvider | None,
) -> tuple[list[tuple[str, StartStrategy, NDArray[np.float64]]], int, float]:
    """Build classical + optional neural branch specs; return (specs, neural_n, ik_s)."""
    t_ik = time.perf_counter()
    branch_specs = _iter_branch_specs(
        seed=seed,
        profile=profile,
        n_branches=n_branches,
        strategies=strategies,
    )
    provider = neural_provider or EmptyNeuralProposalProvider()
    neural = provider.propose(observation=observation, profile=profile)
    for index, proposal in enumerate(neural):
        branch_specs.append(
            (
                f"neural:{index}",
                StartStrategy.RETRIEVAL,
                np.asarray(proposal.q, dtype=np.float64),
            )
        )
    return branch_specs, len(neural), time.perf_counter() - t_ik


def _score_branches_under_budget(
    *,
    branch_specs: Sequence[tuple[str, StartStrategy, NDArray[np.float64]]],
    ctx: _ScoreLoopCtx,
) -> tuple[list[BranchScore], int, bool, float, list[QualityTimeSample]]:
    """Score branches until budget/cancel; return scores, count, cancelled, dynamics_s, curve."""
    scored: list[BranchScore] = []
    quality_curve: list[QualityTimeSample] = []
    cancelled = False
    used = ctx.evaluations_used
    failed_attempts = 0
    best_fit = float("inf")
    t_dyn = time.perf_counter()
    for branch_id, strategy, q in branch_specs[ctx.start_index :]:
        if ctx.cancel_check is not None and ctx.cancel_check():
            cancelled = True
            break
        if used >= ctx.budget.max_evaluations:
            cancelled = True
            break
        if time.perf_counter() > ctx.deadline:
            cancelled = True
            break
        scored_branch = score_branch(
            branch_id=branch_id,
            start_strategy=strategy,
            q=q,
            seed_residual_m=float(ctx.seed.observed_residual_m),
            profile=ctx.profile,
        )
        scored.append(scored_branch)
        used += 1
        if not scored_branch.feasible:
            failed_attempts += 1
        if scored_branch.feasible and scored_branch.observation_fit_m < best_fit:
            best_fit = scored_branch.observation_fit_m
        quality_curve.append(
            QualityTimeSample(
                evaluations=used,
                wall_s=time.perf_counter() - ctx.t0,
                best_observation_fit_m=(
                    best_fit
                    if np.isfinite(best_fit)
                    else scored_branch.observation_fit_m
                ),
                failed_attempts=failed_attempts,
            )
        )
    return scored, used, cancelled, time.perf_counter() - t_dyn, quality_curve


def _assemble_fast_match_result(
    *,
    key: ImmutableCacheKey,
    scored: Sequence[BranchScore],
    profile: ClubOnlyProfile,
    ctx: _AssembleCtx,
) -> FastMatchResult:
    """Prune, verify, checkpoint, and package a FastMatchResult."""
    t_replay = time.perf_counter()
    pareto, rejected = prune_and_select_pareto(
        scored,
        profile=profile,
        max_pareto=ctx.budget.max_pareto,
    )
    replay_s = time.perf_counter() - t_replay

    t_verify = time.perf_counter()
    for member in pareto:
        _ = member.configuration_hash
        if member.observation_fit_m < 0.0:
            raise ValueError("verification found invalid observation fit")
    verification_s = time.perf_counter() - t_verify

    failed_attempts = sum(1 for branch in scored if not branch.feasible)
    failed_attempts = max(failed_attempts, len(rejected))
    best_strategy = pareto[0].start_strategy if pareto else None
    ckpt = MatchCheckpoint(
        cache_key=key,
        preset=ctx.preset,
        evaluations_used=ctx.evaluations_used,
        branch_index=min(ctx.start_index + len(scored), ctx.branch_count),
        pareto_ids=tuple(p.branch_id for p in pareto),
        diversity_seed=ctx.diversity_seed,
    )
    limitations = (
        "Fast matching uses software-contract scoring only; no native Fit/G1 claim.",
        "Pareto set exposes diverse feasible branches, not averaged body motion.",
        "Neural proposal slot is optional; classical completion needs no weights.",
        "Native replay cost and failed attempts count toward latency budgets for CO-08.",
    )
    return FastMatchResult(
        schema=FAST_MATCH_SCHEMA,
        governing_issue=_GOVERNING_ISSUE,
        preset=ctx.preset,
        pareto=pareto,
        rejected=rejected,
        start_strategy_best=best_strategy,
        evaluations_used=ctx.evaluations_used,
        failed_attempts=failed_attempts,
        cancelled=ctx.cancelled,
        neural_proposals_used=ctx.neural_n,
        profiling=MatchStageTimings(
            load_s=ctx.load_s,
            calibration_s=ctx.calibration_s,
            ik_s=ctx.ik_s,
            dynamics_s=ctx.dynamics_s,
            replay_s=replay_s,
            verification_s=verification_s,
        ),
        quality_vs_time=ctx.quality_vs_time,
        checkpoint=ckpt,
        limitations=limitations,
    )


@precondition(
    lambda observation, profile, seed, geometry_hash, profile_hash, options=None: (
        isinstance(observation, ClubObservation)
        and isinstance(profile, ClubOnlyProfile)
        and isinstance(seed, CandidateSeed)
        and bool(geometry_hash)
        and bool(profile_hash)
    ),
    "observation, profile, seed, hashes required",
)
@postcondition(
    lambda result: isinstance(result, FastMatchResult),
    "must return FastMatchResult",
)
def run_fast_club_match(
    *,
    observation: ClubObservation,
    profile: ClubOnlyProfile,
    seed: CandidateSeed,
    geometry_hash: str,
    profile_hash: str,
    options: FastMatchOptions | None = None,
) -> FastMatchResult:
    """Run a bounded fast-match pass with pruning, Pareto diversity, and profiling."""
    opts = options or FastMatchOptions()
    if seed.trial_id != observation.trial_id:
        raise ValueError("seed trial_id must match observation")
    resolved_budget = (
        opts.budget if opts.budget is not None else budget_for_preset(opts.preset)
    )
    t0 = time.perf_counter()

    key, _cache_obj, load_s, calibration_s = _prepare_cache_and_key(
        observation=observation,
        profile=profile,
        seed=seed,
        geometry_hash=geometry_hash,
        profile_hash=profile_hash,
        cache=opts.cache,
        checkpoint=opts.checkpoint,
    )

    if opts.cancel_check is not None and opts.cancel_check():
        raise MatchCancelledError("cancelled before branch generation")

    branch_specs, neural_n, ik_s = _collect_branch_specs_with_neural(
        seed=seed,
        profile=profile,
        observation=observation,
        n_branches=opts.n_branches,
        strategies=opts.strategies,
        neural_provider=opts.neural_provider,
    )

    start_index = opts.checkpoint.branch_index if opts.checkpoint is not None else 0
    evaluations_used = (
        opts.checkpoint.evaluations_used if opts.checkpoint is not None else 0
    )
    # diversity_seed is retained on the checkpoint for CO-08 ablations;
    # posture branches are already deterministic from q0.
    scored, evaluations_used, cancelled, dynamics_s, quality_curve = (
        _score_branches_under_budget(
            branch_specs=branch_specs,
            ctx=_ScoreLoopCtx(
                seed=seed,
                profile=profile,
                start_index=start_index,
                evaluations_used=evaluations_used,
                budget=resolved_budget,
                t0=t0,
                deadline=t0 + resolved_budget.max_time_s,
                cancel_check=opts.cancel_check,
            ),
        )
    )

    return _assemble_fast_match_result(
        key=key,
        scored=scored,
        profile=profile,
        ctx=_AssembleCtx(
            preset=opts.preset,
            budget=resolved_budget,
            evaluations_used=evaluations_used,
            start_index=start_index,
            branch_count=len(branch_specs),
            cancelled=cancelled,
            neural_n=neural_n,
            diversity_seed=opts.diversity_seed,
            quality_vs_time=tuple(quality_curve),
            load_s=load_s,
            calibration_s=calibration_s,
            ik_s=ik_s,
            dynamics_s=dynamics_s,
        ),
    )


def build_fast_match_evidence(result: FastMatchResult) -> dict[str, Any]:
    """Serialize a fast-match result for CO-07 evidence receipts."""
    if not isinstance(result, FastMatchResult):
        raise TypeError("result must be FastMatchResult")
    return {
        "schema": result.schema,
        "governing_issue": result.governing_issue,
        "preset": result.preset.value,
        "evaluations_used": result.evaluations_used,
        "failed_attempts": result.failed_attempts,
        "cancelled": result.cancelled,
        "neural_proposals_used": result.neural_proposals_used,
        "claims_native_qualification": result.claims_native_qualification,
        "native_g1_pass": result.native_g1_pass,
        "qualification_blockers": list(result.qualification_blockers),
        "profiling": {
            "load_s": result.profiling.load_s,
            "calibration_s": result.profiling.calibration_s,
            "ik_s": result.profiling.ik_s,
            "dynamics_s": result.profiling.dynamics_s,
            "replay_s": result.profiling.replay_s,
            "verification_s": result.profiling.verification_s,
            "total_s": result.profiling.total_s,
        },
        "quality_vs_time": [
            {
                "evaluations": sample.evaluations,
                "wall_s": sample.wall_s,
                "best_observation_fit_m": sample.best_observation_fit_m,
                "failed_attempts": sample.failed_attempts,
            }
            for sample in result.quality_vs_time
        ],
        "pareto": [
            {
                "branch_id": p.branch_id,
                "start_strategy": p.start_strategy.value,
                "observation_fit_m": p.observation_fit_m,
                "effort": p.effort,
                "closure_m": p.closure_m,
                "configuration_hash": p.configuration_hash,
            }
            for p in result.pareto
        ],
        "rejected_count": len(result.rejected),
        "limitations": list(result.limitations),
        "checkpoint_identity": (
            checkpoint_identity(result.checkpoint)
            if result.checkpoint is not None
            else None
        ),
    }


@precondition(
    lambda result, evidence_dir=None: isinstance(result, FastMatchResult),
    "result must be a FastMatchResult instance",
)
@postcondition(
    lambda out_path: isinstance(out_path, Path) and out_path.exists(),
    "save_fast_match_evidence must return an existing Path",
)
def save_fast_match_evidence(
    result: FastMatchResult,
    evidence_dir: Path | str | None = None,
) -> Path:
    """Serialize and save a fast-match result to club_fast_matching.json.

    Parameters
    ----------
    result:
        The fast-match execution result to serialize.
    evidence_dir:
        Directory to write the evidence file into. Defaults to
        ``docs/plans/club_only_matching/evidence`` relative to repo root.
        Tests should pass a temporary directory to avoid rewriting
        committed evidence artifacts in place (issue #10750).

    Returns
    -------
    Path
        The path of the written JSON evidence file.
    """
    if evidence_dir is None:
        evidence_dir = (
            get_repo_root() / "docs" / "plans" / "club_only_matching" / "evidence"
        )
    evidence_path = Path(evidence_dir)
    evidence_path.mkdir(parents=True, exist_ok=True)
    out_file = evidence_path / FAST_MATCH_EVIDENCE_FILENAME
    payload = build_fast_match_evidence(result)
    out_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_file
