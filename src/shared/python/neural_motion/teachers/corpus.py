"""Nested teacher corpus builder with resume and rejection accounting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.python.neural_motion.episodes.store import EpisodeStore
from src.shared.python.neural_motion.json_io import write_sorted_json

from .generation import TeacherEpisodeGenerator
from .types import (
    PerturbationKind,
    RejectionLedger,
    TeacherSpec,
)

__all__ = ["NestedTeacherCorpus", "StageRunResult"]


@dataclass(frozen=True)
class StageRunResult:
    """Counters for one corpus stage run."""

    accepted_count: int
    rejected_count: int
    duplicate_seed_count: int
    quarantined_count: int
    nested_reuse_count: int
    rejection_ledger: RejectionLedger
    spent_cost: float


class NestedTeacherCorpus:
    """Build nested 100/500/2_000-style stages into an EpisodeStore.

    Checkpointing records completed ``(stage, seed)`` pairs so a partial batch
    can resume without rewriting immutable episodes. Rejected and quarantined
    rollouts land in separate stores when provided.
    """

    def __init__(
        self,
        *,
        store: EpisodeStore,
        generator: TeacherEpisodeGenerator,
        checkpoint_path: str | Path,
        stage_sizes: tuple[int, ...],
        compute_cap_cost: float,
        rejected_store: EpisodeStore | None = None,
        quarantine_store: EpisodeStore | None = None,
        default_perturbation: PerturbationKind = PerturbationKind.NEAR_BASELINE,
    ) -> None:
        if not stage_sizes or any(size <= 0 for size in stage_sizes):
            raise ValueError("stage_sizes must be positive integers")
        if compute_cap_cost <= 0.0:
            raise ValueError("compute_cap_cost must be positive")
        self.store = store
        self.generator = generator
        self.checkpoint_path = Path(checkpoint_path)
        self.stage_sizes = stage_sizes
        self.compute_cap_cost = float(compute_cap_cost)
        self.rejected_store = rejected_store
        self.quarantine_store = quarantine_store
        self._default_perturbation = default_perturbation
        self._state = self._load_checkpoint()

    def record_seed_done(self, *, stage_index: int, seed: int, episode_id: str) -> None:
        """Mark a seed completed (used when episodes are written externally)."""
        key = self._seed_key(stage_index, seed)
        self._state["completed"][key] = episode_id
        stage_key = str(stage_index)
        ids = list(self._state["stage_episode_ids"].get(stage_key, []))
        if episode_id not in ids:
            ids.append(episode_id)
        self._state["stage_episode_ids"][stage_key] = ids
        self._save_checkpoint()

    def learning_curve_subsets(self) -> tuple[frozenset[str], ...]:
        """Return nested episode-id subsets (stage i ⊆ stage i+1)."""
        subsets: list[frozenset[str]] = []
        accumulated: set[str] = set()
        for index in range(len(self.stage_sizes)):
            stage_ids = set(self._state["stage_episode_ids"].get(str(index), []))
            accumulated |= stage_ids
            subsets.append(frozenset(accumulated))
        return tuple(subsets)

    def run_stage(
        self,
        *,
        stage_index: int,
        seeds: tuple[int, ...],
        max_new: int | None = None,
        family_id: str = "teacher_fam",
    ) -> StageRunResult:
        """Generate/accept episodes for ``seeds`` up to the stage budget."""
        if stage_index < 0 or stage_index >= len(self.stage_sizes):
            raise ValueError("stage_index out of range")
        target = self.stage_sizes[stage_index]
        ledger = RejectionLedger.empty()
        accepted = 0
        rejected = 0
        duplicates = 0
        quarantined = 0
        nested_reuse = 0
        spent = 0.0
        new_count = 0

        stage_ids = list(self._state["stage_episode_ids"].get(str(stage_index), []))

        for seed in seeds:
            key = self._seed_key(stage_index, seed)
            if key in self._state["completed"]:
                # Count duplicates even when the stage budget is already full.
                duplicates += 1
                continue
            if len(set(stage_ids)) >= target and max_new is None:
                # Stage filled; remaining novel seeds ignored unless max_new forces work.
                break
            if max_new is not None and new_count >= max_new:
                break
            if spent >= self.compute_cap_cost:
                ledger = ledger.append(seed=seed, reason="budget_exceeded", cost=0.0)
                break

            reused_id = self._prior_episode_id(stage_index, seed)
            if reused_id is not None:
                nested_reuse += 1
                stage_ids.append(reused_id)
                self._state["completed"][key] = reused_id
                continue

            spec = TeacherSpec(
                seed=seed,
                family_id=f"{family_id}_{seed}",
                model_id="mock.driven_double_pendulum",
                perturbation=self._default_perturbation,
                geometry_stratum="std_driver",
                contact_stratum="no_contact",
                club_stratum="driver",
                duration_s=0.3,
                ancestry=("baseline:tour_driver", f"seed:{seed}"),
                requested_channels=("q", "v", "u", "a_native"),
            )
            outcome = self.generator.generate(spec)
            spent += 1.0 if outcome.feasible else float(outcome.rejection_cost)

            if outcome.corpus_role == "quarantine":
                quarantined += 1
                rejected += 1
                ledger = ledger.append(
                    seed=seed, reason=outcome.reason, cost=outcome.rejection_cost
                )
                if self.quarantine_store is not None and outcome.episode is not None:
                    self.quarantine_store.write_episode(outcome.episode)
                continue

            if not outcome.feasible or outcome.episode is None:
                rejected += 1
                ledger = ledger.append(
                    seed=seed, reason=outcome.reason, cost=outcome.rejection_cost
                )
                if self.rejected_store is not None:
                    # Rejected path has no episode payload by contract.
                    pass
                continue

            written = self.store.write_episode(outcome.episode)
            stage_ids.append(written.episode_id)
            self._state["completed"][key] = written.episode_id
            accepted += 1
            new_count += 1

        self._state["stage_episode_ids"][str(stage_index)] = list(
            dict.fromkeys(stage_ids)
        )
        self._save_checkpoint()
        return StageRunResult(
            accepted_count=accepted,
            rejected_count=rejected,
            duplicate_seed_count=duplicates,
            quarantined_count=quarantined,
            nested_reuse_count=nested_reuse,
            rejection_ledger=ledger,
            spent_cost=spent,
        )

    def _prior_episode_id(self, stage_index: int, seed: int) -> str | None:
        for prior in range(stage_index):
            prior_key = self._seed_key(prior, seed)
            if prior_key in self._state["completed"]:
                return str(self._state["completed"][prior_key])
        return None

    def _seed_key(self, stage_index: int, seed: int) -> str:
        return f"{stage_index}:{seed}"

    def _load_checkpoint(self) -> dict[str, Any]:
        if not self.checkpoint_path.exists():
            return {"completed": {}, "stage_episode_ids": {}}
        data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        return {
            "completed": dict(data.get("completed", {})),
            "stage_episode_ids": {
                str(k): list(v) for k, v in data.get("stage_episode_ids", {}).items()
            },
        }

    def _save_checkpoint(self) -> None:
        write_sorted_json(self.checkpoint_path, self._state)
