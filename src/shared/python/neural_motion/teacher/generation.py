"""Teacher episode generation campaign with resume and rejection accounting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.dataset_tools.canonical import (
    CANONICAL_JOINTS,
    COEFFICIENT_LETTERS,
    N_COEFFS,
    N_JOINTS,
)
from src.shared.python.neural_motion.episodes import EPISODE_STORE_SCHEMA, EpisodeRecord
from src.shared.python.neural_motion.episodes.store import EpisodeStore
from src.shared.python.neural_motion.json_io import write_sorted_json

from .backends import TeacherRolloutBackend
from .perturb import perturbation_for_attempt
from .types import (
    TEACHER_CAMPAIGN_SCHEMA,
    TeacherAnchor,
    TeacherGenerationSpec,
    TeacherRolloutRequest,
    TeacherRolloutResult,
)

__all__ = [
    "RejectedRolloutStore",
    "TeacherGenerationCampaign",
    "TeacherGenerationState",
    "TeacherStageReceipt",
    "load_generation_state",
    "save_generation_state",
]

_REJECTED_SCHEMA = "neural-teacher-rejected/1.0.0"


@dataclass(frozen=True)
class TeacherGenerationState:
    """Resumable campaign progress (attempt keys only — no array payload)."""

    campaign_id: str
    completed_attempts: int
    attempt_keys: tuple[str, ...]
    simulation_cost_units: float

    def __post_init__(self) -> None:
        if not self.campaign_id:
            raise ValueError("campaign_id required")
        if self.completed_attempts < 0:
            raise ValueError("completed_attempts must be >= 0")
        if (
            not np.isfinite(self.simulation_cost_units)
            or self.simulation_cost_units < 0
        ):
            raise ValueError("simulation_cost_units must be finite and >= 0")


@dataclass(frozen=True)
class TeacherStageReceipt:
    """Per-stage accounting for accepted, rejected and duplicate skips."""

    stage_index: int
    target_count: int
    accepted: int
    rejected: int
    skipped_duplicate: int
    simulation_cost_units: float

    @property
    def attempted(self) -> int:
        return self.accepted + self.rejected + self.skipped_duplicate


def save_generation_state(path: Path, state: TeacherGenerationState) -> None:
    write_sorted_json(
        path,
        {
            "schema": TEACHER_CAMPAIGN_SCHEMA,
            "campaign_id": state.campaign_id,
            "completed_attempts": state.completed_attempts,
            "attempt_keys": list(state.attempt_keys),
            "simulation_cost_units": state.simulation_cost_units,
        },
    )


def load_generation_state(path: Path) -> TeacherGenerationState | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return TeacherGenerationState(
        campaign_id=str(data.get("campaign_id", "")),
        completed_attempts=int(data.get("completed_attempts", 0)),
        attempt_keys=tuple(data.get("attempt_keys", ())),
        simulation_cost_units=float(data.get("simulation_cost_units", 0.0)),
    )


class RejectedRolloutStore:
    """Append-only rejected / infeasible rollout ledger."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._path = self._root / "rejected_rollouts.jsonl"

    def append(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def count(self) -> int:
        if not self._path.exists():
            return 0
        return sum(1 for _ in self._path.open(encoding="utf-8"))


class TeacherGenerationCampaign:
    """Generate feasible teacher episodes into an :class:`EpisodeStore`."""

    def __init__(
        self,
        spec: TeacherGenerationSpec,
        *,
        backend: TeacherRolloutBackend,
        anchors: tuple[TeacherAnchor, ...] | None = None,
    ) -> None:
        self._spec = spec
        self._backend = backend
        self._anchors = anchors or _default_pilot_anchors()
        self._store = EpisodeStore(spec.store_root)
        self._rejected = RejectedRolloutStore(spec.rejected_root)
        loaded = load_generation_state(spec.state_path)
        if loaded is None:
            self._state = TeacherGenerationState(
                campaign_id=spec.campaign_id,
                completed_attempts=0,
                attempt_keys=(),
                simulation_cost_units=0.0,
            )
        else:
            self._state = loaded
            if self._state.campaign_id != spec.campaign_id:
                raise ValueError("campaign_id mismatch for resume state")
        self._known_keys = set(self._state.attempt_keys)
        self._ledger_entries: list[dict[str, Any]] = []
        if spec.ledger_path.exists():
            existing = json.loads(spec.ledger_path.read_text(encoding="utf-8"))
            self._ledger_entries = list(existing.get("entries", []))

    def run_stage(self, *, target_count: int, stage_index: int) -> TeacherStageReceipt:
        if target_count <= 0:
            raise ValueError("target_count must be positive")
        if stage_index < 0 or stage_index >= len(self._spec.nested_stages):
            raise ValueError("stage_index out of range for nested_stages")
        accepted = rejected = skipped_duplicate = 0
        cost_total = self._state.simulation_cost_units
        attempt = self._state.completed_attempts
        attempts_this_stage = 0
        max_attempts = max(target_count * 50, 50)
        while accepted < target_count and attempts_this_stage < max_attempts:
            attempts_this_stage += 1
            anchor = self._anchors[attempt % len(self._anchors)]
            perturbation = perturbation_for_attempt(
                master_seed=self._spec.master_seed,
                stage_index=stage_index,
                attempt_index=attempt,
                dim=4,
            )
            request = TeacherRolloutRequest(
                anchor=anchor,
                attempt_index=attempt,
                stage_index=stage_index,
                master_seed=self._spec.master_seed,
                perturbation=perturbation,
            )
            key = request.attempt_key()
            attempt += 1
            if key in self._known_keys:
                skipped_duplicate += 1
                continue
            self._known_keys.add(key)
            outcome = self._backend.rollout(request)
            cost_total += outcome.simulation_cost_units
            if not outcome.feasible:
                rejected += 1
                self._rejected.append(_rejected_payload(request, outcome))
                self._append_ledger(request, outcome, episode_id=None)
                continue
            episode = _episode_from_outcome(request, outcome, spec=self._spec)
            written = self._store.write_episode(episode)
            accepted += 1
            self._append_ledger(request, outcome, episode_id=written.episode_id)
        self._state = TeacherGenerationState(
            campaign_id=self._spec.campaign_id,
            completed_attempts=attempt,
            attempt_keys=tuple(sorted(self._known_keys)),
            simulation_cost_units=cost_total,
        )
        save_generation_state(self._spec.state_path, self._state)
        self._flush_ledger()
        return TeacherStageReceipt(
            stage_index=stage_index,
            target_count=target_count,
            accepted=accepted,
            rejected=rejected,
            skipped_duplicate=skipped_duplicate,
            simulation_cost_units=cost_total,
        )

    def _append_ledger(
        self,
        request: TeacherRolloutRequest,
        outcome: TeacherRolloutResult,
        *,
        episode_id: str | None,
    ) -> None:
        self._ledger_entries.append(
            {
                "attempt_key": request.attempt_key(),
                "anchor_id": request.anchor.anchor_id,
                "feasible": outcome.feasible,
                "teacher_objective": outcome.teacher_objective,
                "convergence_iterations": outcome.convergence_iterations,
                "independent_replay_digest": outcome.independent_replay_digest,
                "simulation_cost_units": outcome.simulation_cost_units,
                "episode_id": episode_id,
                "rejection_reason": outcome.rejection_reason,
            }
        )

    def _flush_ledger(self) -> None:
        write_sorted_json(
            self._spec.ledger_path,
            {
                "schema": TEACHER_CAMPAIGN_SCHEMA,
                "campaign_id": self._spec.campaign_id,
                "model_id": self._spec.model_id,
                "entries": self._ledger_entries,
            },
        )


def _default_pilot_anchors() -> tuple[TeacherAnchor, ...]:
    rng = np.random.default_rng(0)
    q0 = rng.normal(size=(N_JOINTS,))
    coeffs = rng.normal(size=(N_COEFFS,))
    return (
        TeacherAnchor(
            anchor_id="co03:TW_wiffle:retrieval",
            trial_id="TW_wiffle",
            model_id="driven_double_pendulum",
            source="co03_retrieval",
            q0=q0,
            coefficients=coeffs,
            geometry_stratum="std_driver",
            contact_stratum="no_contact",
            club_stratum="driver",
        ),
    )


def _rejected_payload(
    request: TeacherRolloutRequest,
    outcome: TeacherRolloutResult,
) -> dict[str, Any]:
    return {
        "schema": _REJECTED_SCHEMA,
        "attempt_key": request.attempt_key(),
        "anchor_id": request.anchor.anchor_id,
        "reason": outcome.rejection_reason or "infeasible",
        "teacher_objective": outcome.teacher_objective,
        "simulation_cost_units": outcome.simulation_cost_units,
    }


def _episode_from_outcome(
    request: TeacherRolloutRequest,
    outcome: TeacherRolloutResult,
    *,
    spec: TeacherGenerationSpec,
) -> EpisodeRecord:
    availability = dict(outcome.channel_availability or {})
    _reject_zero_dynamic_channels(outcome, availability)
    trial_id = f"teacher_{request.attempt_key()}"
    family_id = f"fam_{request.anchor.anchor_id}"
    ancestry = (
        "teacher:nm04",
        f"anchor:{request.anchor.anchor_id}",
        f"co03_trial:{request.anchor.trial_id}",
        "simulated_body:conditional",
    )
    assert outcome.q is not None
    assert outcome.sample_times_s is not None
    return EpisodeRecord(
        trial_id=trial_id,
        family_id=family_id,
        model_id=spec.model_id,
        control_basis="joint_torque",
        units="SI",
        joint_names=CANONICAL_JOINTS,
        coefficient_letters=COEFFICIENT_LETTERS,
        schema_version=EPISODE_STORE_SCHEMA,
        sample_times_s=outcome.sample_times_s,
        q=outcome.q,
        v=outcome.v,
        u=outcome.u,
        a_native=outcome.a_native,
        q_next=outcome.q_next,
        channel_availability=availability,
        ancestry=ancestry,
        geometry_stratum=request.anchor.geometry_stratum,
        contact_stratum=request.anchor.contact_stratum,
        club_stratum=request.anchor.club_stratum,
        coefficients=request.anchor.coefficients,
        source_schema="teacher-rollout/1.0.0",
    )


def _reject_zero_dynamic_channels(
    outcome: TeacherRolloutResult,
    availability: dict[str, str],
) -> None:
    """Unavailable channels stay None; available dynamics cannot be all-zero arrays."""
    for name, values in (
        ("a_native", outcome.a_native),
        ("v", outcome.v),
        ("u", outcome.u),
    ):
        if availability.get(name) != "available":
            continue
        if values is None:
            raise ValueError(f"{name} marked available but missing")
        arr = np.asarray(values, dtype=np.float64)
        if arr.size > 0 and not np.any(np.abs(arr) > 0.0):
            raise ValueError(
                f"{name} cannot pass as available measurement when all-zero"
            )
