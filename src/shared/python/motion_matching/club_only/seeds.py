"""Reusable candidate seed contracts and four-trial starting-guess reports (CO-03 #10607).

Kinematic previews only — no torque or physiological inference claims.
Geometry/profile content hashes invalidate the seed cache fail-closed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.club_only.hand_geometry import (
    GolferHandedness,
    resolve_hand_frame_offsets,
)
from src.shared.python.motion_matching.club_only.observation import (
    ClubObservation,
    build_calibrated_observation_fixture,
    require_strictly_increasing_timestamps,
)
from src.shared.python.motion_matching.club_only.profiles import (
    ClubOnlyProfile,
    get_club_only_profile,
)
from src.shared.python.motion_matching.club_only.workbook_identity import (
    CANONICAL_TRIAL_SHEETS,
)

if TYPE_CHECKING:
    from src.shared.python.motion_matching.club_only.retrieval import LibraryEntry

SEED_SCHEMA = "club-starting-guesses/1.0.0"
# Seed sources produced by a real CO-03 retrieval or IK step; ``ui_synthetic`` is not one.
VERIFIED_SEED_SOURCES = frozenset({"retrieval", "constrained_ik"})
# Hand-made seeds (GUI preview, CO-05 preview); never evidence of a fit.
SYNTHETIC_SEED_SOURCES = frozenset({"ui_synthetic", "synthetic"})
_GOVERNING_ISSUE = 10607

__all__ = [
    "SEED_SCHEMA",
    "SYNTHETIC_SEED_SOURCES",
    "VERIFIED_SEED_SOURCES",
    "CandidateSeed",
    "SeedCache",
    "StartingGuessReport",
    "build_starting_guess_report",
    "evidence_payload",
    "geometry_content_hash",
    "profile_content_hash",
    "starting_guess_evidence_payload",
]


@dataclass(frozen=True)
class CandidateSeed:
    """One reusable starting guess with observed residual and feasibility notes."""

    seed_id: str
    trial_id: str
    model_id: str
    source: str
    q: np.ndarray
    body_configuration_hash: str
    observed_residual_m: float
    prior_score: float
    feasibility_reasons: tuple[str, ...]
    timestamps_s: np.ndarray
    geometry_hash: str
    profile_hash: str
    body_is_prior: bool = False
    placement: Any = None
    claims_torque: bool = False
    claims_physiological_inference: bool = False
    is_kinematic_preview: bool = True
    backend_kind: str | None = None

    def __post_init__(self) -> None:
        if not self.seed_id or not self.trial_id or not self.model_id:
            raise ValueError("seed_id, trial_id, and model_id required")
        if self.source not in VERIFIED_SEED_SOURCES | SYNTHETIC_SEED_SOURCES:
            raise ValueError(f"unsupported seed source: {self.source!r}")
        q = np.asarray(self.q, dtype=np.float64)
        if q.ndim != 1 or q.size < 1 or not np.all(np.isfinite(q)):
            raise ValueError("q must be a finite non-empty 1-D array")
        times = require_strictly_increasing_timestamps(self.timestamps_s)
        if not np.isfinite(self.observed_residual_m) or self.observed_residual_m < 0.0:
            raise ValueError("observed_residual_m must be finite and >= 0")
        if not np.isfinite(self.prior_score):
            raise ValueError("prior_score must be finite")
        if not self.body_configuration_hash:
            raise ValueError("body_configuration_hash required")
        if not self.geometry_hash or not self.profile_hash:
            raise ValueError("geometry_hash and profile_hash required")
        if self.claims_torque or self.claims_physiological_inference:
            raise ValueError("CO-03 seeds cannot claim torque or physiology")
        if not self.is_kinematic_preview:
            raise ValueError("CO-03 seeds must remain kinematic previews")
        object.__setattr__(self, "q", q.copy())
        object.__setattr__(self, "timestamps_s", times.copy())
        object.__setattr__(self, "feasibility_reasons", tuple(self.feasibility_reasons))

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "trial_id": self.trial_id,
            "model_id": self.model_id,
            "source": self.source,
            "body_configuration_hash": self.body_configuration_hash,
            "observed_residual_m": self.observed_residual_m,
            "prior_score": self.prior_score,
            "feasibility_reasons": list(self.feasibility_reasons),
            "geometry_hash": self.geometry_hash,
            "profile_hash": self.profile_hash,
            "body_is_prior": self.body_is_prior,
            "claims_torque": self.claims_torque,
            "claims_physiological_inference": self.claims_physiological_inference,
            "is_kinematic_preview": self.is_kinematic_preview,
            "backend_kind": self.backend_kind,
            "q_dim": int(self.q.size),
            "n_timestamps": int(self.timestamps_s.size),
        }


class SeedCache:
    """In-memory seed cache keyed by trial/model/source/geometry/profile."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str, str, str, str], CandidateSeed] = {}

    def put(self, seed: CandidateSeed) -> None:
        if not isinstance(seed, CandidateSeed):
            raise TypeError("seed must be CandidateSeed")
        key = (
            seed.trial_id,
            seed.model_id,
            seed.source,
            seed.geometry_hash,
            seed.profile_hash,
        )
        self._store[key] = seed

    def get(
        self,
        trial_id: str,
        model_id: str,
        source: str,
        geometry_hash: str,
        profile_hash: str,
    ) -> CandidateSeed | None:
        return self._store.get(
            (trial_id, model_id, source, geometry_hash, profile_hash)
        )


@dataclass(frozen=True)
class StartingGuessReport:
    """Bounded retrieval + constrained-IK baselines for all four club trials."""

    schema: str
    governing_issue: int
    model_id: str
    handedness: str
    trials: Mapping[str, dict[str, Any]]
    limitations: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.schema != SEED_SCHEMA:
            raise ValueError(f"schema must be {SEED_SCHEMA!r}")
        if self.governing_issue != _GOVERNING_ISSUE:
            raise ValueError(f"governing_issue must be {_GOVERNING_ISSUE}")
        missing = set(CANONICAL_TRIAL_SHEETS) - set(self.trials)
        if missing:
            raise ValueError(f"report missing trials: {sorted(missing)}")


def geometry_content_hash(
    *,
    club_type: str,
    catalog_length_m: float,
    tool_to_model_residual_m: float,
) -> str:
    """Hash geometry inputs that invalidate cached seeds when changed."""
    if not club_type:
        raise ValueError("club_type must be non-empty")
    if not np.isfinite(catalog_length_m) or catalog_length_m <= 0.0:
        raise ValueError("catalog_length_m must be finite and > 0")
    if not np.isfinite(tool_to_model_residual_m):
        raise ValueError("tool_to_model_residual_m must be finite")
    payload = {
        "club_type": club_type,
        "catalog_length_m": catalog_length_m,
        "tool_to_model_residual_m": tool_to_model_residual_m,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def profile_content_hash(profile: ClubOnlyProfile) -> str:
    """Hash a frozen club-only profile for seed-cache invalidation."""
    if not isinstance(profile, ClubOnlyProfile):
        raise TypeError("profile must be ClubOnlyProfile")
    raw = json.dumps(profile.as_dict(), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def _library_for_observation(
    obs: ClubObservation,
    profile: ClubOnlyProfile,
    geometry_hash: str,
    other_observations: Sequence[ClubObservation] | None = None,
) -> tuple[LibraryEntry, ...]:
    """Build a small prior library from other trials' descriptors only (CO-03 #10607, #10960).

    Never self-copies the query observation; returns an empty library if no other-trial
    source is available, allowing retrieval to fail closed.
    """
    from src.shared.python.motion_matching.club_only.retrieval import (
        LibraryEntry,
        RigidPlacement,
        build_observable_descriptor,
    )

    if other_observations is not None:
        candidate_sources = [
            o for o in other_observations if o.trial_id != obs.trial_id
        ]
    elif obs.trial_id in CANONICAL_TRIAL_SHEETS:
        candidate_sources = [
            build_calibrated_observation_fixture(t)
            for t in CANONICAL_TRIAL_SHEETS
            if t != obs.trial_id
        ]
    else:
        candidate_sources = []

    if not candidate_sources:
        return ()

    entries: list[LibraryEntry] = []
    amplitudes = (0.0, 0.03, -0.02)
    for index, source_obs in enumerate(candidate_sources):
        if source_obs.native_time_s.shape != obs.native_time_s.shape:
            continue
        if not np.allclose(source_obs.native_time_s, obs.native_time_s):
            continue
        desc = build_observable_descriptor(
            source_obs,
            model_id=profile.model_id,
            geometry_hash=geometry_hash,
        )
        amp = amplitudes[index % len(amplitudes)]
        q0 = np.array([amp, 0.05 - amp, -0.02 + 0.5 * amp, 0.01 * index])
        entries.append(
            LibraryEntry(
                reference_id=f"tour-prior-{source_obs.trial_id}-{index}",
                descriptor=desc,
                body_q0=q0,
                rigid_placement=RigidPlacement.identity(),
                source_clock_times_s=source_obs.native_time_s.copy(),
                body_is_prior=True,
            )
        )
    return tuple(entries)


@precondition(
    lambda model_id="full_body_pinocchio", handedness=GolferHandedness.RIGHT, max_seeds_per_source=3: (
        isinstance(model_id, str)
        and bool(model_id)
        and isinstance(handedness, GolferHandedness)
        and int(max_seeds_per_source) >= 1
    ),
    "model_id, handedness, max_seeds_per_source>=1 required",
)
@postcondition(
    lambda result: isinstance(result, StartingGuessReport),
    "must return StartingGuessReport",
)
def build_starting_guess_report(
    *,
    model_id: str = "full_body_pinocchio",
    handedness: GolferHandedness = GolferHandedness.RIGHT,
    max_seeds_per_source: int = 3,
) -> StartingGuessReport:
    """Build retrieval and constrained-IK baselines for all four club trials."""
    from src.shared.python.motion_matching.club_only.constrained_ik import (
        IkSolverKind,
        run_constrained_ik_seeds,
    )
    from src.shared.python.motion_matching.club_only.retrieval import (
        retrieve_starting_seeds,
    )

    profile = get_club_only_profile(model_id)
    p_hash = profile_content_hash(profile)
    hand_offsets = resolve_hand_frame_offsets(model_id=model_id, handedness=handedness)
    trials: dict[str, dict[str, Any]] = {}
    for trial_id in CANONICAL_TRIAL_SHEETS:
        obs = build_calibrated_observation_fixture(trial_id)
        g_hash = geometry_content_hash(
            club_type=obs.club_type,
            catalog_length_m=obs.catalog_length_m,
            tool_to_model_residual_m=0.0,
        )
        library = _library_for_observation(obs, profile, g_hash)
        retrieval = retrieve_starting_seeds(
            observation=obs,
            profile=profile,
            library=library,
            geometry_hash=g_hash,
            profile_hash=p_hash,
            max_seeds=max_seeds_per_source,
        )
        ik = run_constrained_ik_seeds(
            observation=obs,
            profile=profile,
            hand_offsets=hand_offsets,
            geometry_hash=g_hash,
            profile_hash=p_hash,
            backend=IkSolverKind.PINK_NATIVE,
            n_branches=max_seeds_per_source,
        )
        if not retrieval:
            raise ValueError(f"failed to produce retrieval baselines for {trial_id}")
        ik_baseline: dict[str, Any] = {
            "seed_ids": [s.seed_id for s in ik.seeds],
            "is_kinematic_preview": True,
            "backend": ik.backend.value,
            "seeds": [s.as_dict() for s in ik.seeds],
        }
        if ik.failure_reason is not None:
            ik_baseline["failure_reason"] = ik.failure_reason.value
        trials[trial_id] = {
            "trial_id": trial_id,
            "geometry_hash": g_hash,
            "profile_hash": p_hash,
            "hand_offsets": hand_offsets.as_dict(),
            "claims_torque": False,
            "claims_physiological_inference": False,
            "baselines": {
                "retrieval": {
                    "seed_ids": [s.seed_id for s in retrieval],
                    "is_kinematic_preview": True,
                    "seeds": [s.as_dict() for s in retrieval],
                },
                "constrained_ik": ik_baseline,
            },
        }

    return StartingGuessReport(
        schema=SEED_SCHEMA,
        governing_issue=_GOVERNING_ISSUE,
        model_id=model_id,
        handedness=handedness.value,
        trials=trials,
        limitations=(
            "Seeds are kinematic previews only; scientific acceptance awaits CO-06.",
            "Retrieved body configurations remain priors, not measured labels.",
            "No torque or physiological inference is claimed.",
            "Pink vs DLS capabilities are distinct; unsupported constraints fail closed.",
        ),
    )


def evidence_payload(report: StartingGuessReport) -> dict[str, Any]:
    """Serialize a starting-guess report into the CO-03 evidence receipt shape."""
    if not isinstance(report, StartingGuessReport):
        raise TypeError("report must be StartingGuessReport")
    return {
        "schema": report.schema,
        "governing_issue": report.governing_issue,
        "model_id": report.model_id,
        "handedness": report.handedness,
        "trials": dict(report.trials),
        "limitations": list(report.limitations),
        "notes": [
            "Retrieval applies one rigid placement and preserves the native clock.",
            "Constrained IK seeds use reviewed hand-frame offsets and hard constraints.",
            "Kinematic preview is not physical acceptance.",
        ],
    }


starting_guess_evidence_payload = evidence_payload
