"""Thin task views over immutable episodes (NM-03 #10618)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .record import EpisodeRecord

__all__ = [
    "FeasibilityView",
    "InstantaneousDynamicsView",
    "ObservationMaskView",
    "SequenceMatchingView",
]


@dataclass(frozen=True)
class InstantaneousDynamicsView:
    """Same-time dynamics identity channels (q, v, u, a)."""

    trial_id: str
    sample_times_s: np.ndarray
    q: np.ndarray
    v: np.ndarray | None
    u: np.ndarray | None
    a_native: np.ndarray | None

    @classmethod
    def from_episode(cls, episode: EpisodeRecord) -> InstantaneousDynamicsView:
        return cls(
            trial_id=episode.trial_id,
            sample_times_s=np.asarray(episode.sample_times_s, dtype=np.float64),
            q=np.asarray(episode.q, dtype=np.float64),
            v=None if episode.v is None else np.asarray(episode.v, dtype=np.float64),
            u=None if episode.u is None else np.asarray(episode.u, dtype=np.float64),
            a_native=(
                None
                if episode.a_native is None
                else np.asarray(episode.a_native, dtype=np.float64)
            ),
        )


@dataclass(frozen=True)
class SequenceMatchingView:
    """Sliding windows for sequence-matching tasks (derived, not stored raw)."""

    trial_id: str
    window: int
    windows: np.ndarray
    sample_times_s: np.ndarray

    @classmethod
    def from_episode(
        cls, episode: EpisodeRecord, *, window: int
    ) -> SequenceMatchingView:
        if window < 1:
            raise ValueError("window must be >= 1")
        q = np.asarray(episode.q, dtype=np.float64)
        if q.shape[0] < window:
            raise ValueError("episode shorter than window")
        stacked = np.stack(
            [q[i : i + window] for i in range(q.shape[0] - window + 1)],
            axis=0,
        )
        times = np.asarray(episode.sample_times_s, dtype=np.float64)
        return cls(
            trial_id=episode.trial_id,
            window=window,
            windows=stacked,
            sample_times_s=times[window - 1 :],
        )


@dataclass(frozen=True)
class ObservationMaskView:
    """Explicit observation vs hidden channel mask (never invent zeros)."""

    trial_id: str
    observed_mask: Mapping[str, bool]
    observed: Mapping[str, np.ndarray | None]

    @classmethod
    def from_episode(
        cls,
        episode: EpisodeRecord,
        *,
        observed: Sequence[str],
        hidden: Sequence[str],
    ) -> ObservationMaskView:
        observed_set = set(observed)
        hidden_set = set(hidden)
        if observed_set & hidden_set:
            raise ValueError("observed and hidden channels must be disjoint")
        mask = {name: name in observed_set for name in (*observed, *hidden)}
        arrays: dict[str, np.ndarray | None] = {}
        for name in (*observed, *hidden):
            value = getattr(episode, name, None)
            if name in observed_set:
                if value is None:
                    raise ValueError(
                        f"observed channel {name!r} is unavailable; "
                        "refusing to invent zeros"
                    )
                arrays[name] = np.asarray(value, dtype=np.float64)
            else:
                arrays[name] = None
        return cls(
            trial_id=episode.trial_id,
            observed_mask=mask,
            observed=arrays,
        )


@dataclass(frozen=True)
class FeasibilityView:
    """Feasibility classification label retained with a reason string."""

    trial_id: str
    feasible: bool
    reason: str

    @classmethod
    def from_episode(
        cls,
        episode: EpisodeRecord,
        *,
        feasible: bool,
        reason: str,
    ) -> FeasibilityView:
        if not reason.strip():
            raise ValueError("feasibility reason must be non-empty")
        return cls(trial_id=episode.trial_id, feasible=feasible, reason=reason)
