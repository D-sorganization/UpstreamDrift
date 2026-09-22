"""Active-learning candidate selection without consuming test labels."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.shared.python.neural_motion.episodes.record import EpisodeRecord
from src.shared.python.neural_motion.episodes.splits import FamilySplitPlan

from .types import AcquisitionMode

__all__ = ["AcquisitionBatch", "select_acquisition_candidates"]

_FORBIDDEN_SPLITS = frozenset({"test", "real_data_eval", "eval_held_out"})


@dataclass(frozen=True)
class AcquisitionBatch:
    """Selected training-pool trials for the next acquisition batch."""

    mode: AcquisitionMode
    selected_trial_ids: tuple[str, ...]
    content_digest: str


def select_acquisition_candidates(
    episodes: tuple[EpisodeRecord, ...] | list[EpisodeRecord],
    *,
    split_plan: FamilySplitPlan,
    scores: dict[str, float],
    k: int,
    mode: AcquisitionMode,
    rng_seed: int,
) -> AcquisitionBatch:
    """Pick up to ``k`` trials from train/val only (never test / eval buckets)."""
    if k <= 0:
        raise ValueError("k must be positive")
    eligible: list[tuple[str, float]] = []
    for episode in episodes:
        split = split_plan.split_of(episode.trial_id)
        if split in _FORBIDDEN_SPLITS:
            continue
        if episode.trial_id not in scores:
            raise KeyError(f"missing score for trial {episode.trial_id!r}")
        eligible.append((episode.trial_id, float(scores[episode.trial_id])))
    if not eligible:
        raise ValueError("no eligible trials outside forbidden splits")

    if mode is AcquisitionMode.RANDOM_CONTROL:
        ordered = _stable_order(eligible, seed=rng_seed)
        selected = tuple(trial for trial, _score in ordered[:k])
    elif mode is AcquisitionMode.UNCERTAINTY:
        ordered = sorted(eligible, key=lambda item: item[1], reverse=True)
        selected = tuple(trial for trial, _score in ordered[:k])
    elif mode is AcquisitionMode.DISAGREEMENT:
        ordered = sorted(eligible, key=lambda item: abs(item[1]), reverse=True)
        selected = tuple(trial for trial, _score in ordered[:k])
    else:
        raise ValueError(f"unsupported acquisition mode: {mode}")

    digest = hashlib.sha256(
        f"{mode.value}:{rng_seed}:{','.join(selected)}".encode()
    ).hexdigest()
    return AcquisitionBatch(
        mode=mode, selected_trial_ids=selected, content_digest=digest
    )


def _stable_order(
    items: list[tuple[str, float]],
    *,
    seed: int,
) -> list[tuple[str, float]]:
    keyed = [
        (hashlib.sha256(f"{seed}:{trial}".encode()).hexdigest(), trial, score)
        for trial, score in items
    ]
    keyed.sort()
    return [(trial, score) for _digest, trial, score in keyed]
