"""Train-only normalisation with immutable resume (NM-03 #10618)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .record import EpisodeRecord

__all__ = ["TrainOnlyNormalizer"]


@dataclass(frozen=True)
class TrainOnlyNormalizer:
    """Fit channel mean/std on train episodes only; resume is immutable."""

    channels: tuple[str, ...]
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]
    stats_digest: str
    _frozen: bool = True

    @classmethod
    def fit(
        cls,
        episodes: Sequence[EpisodeRecord],
        *,
        channels: Sequence[str],
    ) -> TrainOnlyNormalizer:
        if not episodes:
            raise ValueError("episodes must be non-empty for normalisation fit")
        if not channels:
            raise ValueError("channels must be non-empty")
        channel_tuple = tuple(channels)
        sums: dict[str, np.ndarray] = {}
        sq: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        for episode in episodes:
            for name in channel_tuple:
                if episode.channel_availability.get(name) != "available":
                    continue
                values = getattr(episode, name, None)
                if values is None:
                    continue
                arr = np.asarray(values, dtype=np.float64)
                if name not in sums:
                    sums[name] = np.zeros(arr.shape[1], dtype=np.float64)
                    sq[name] = np.zeros(arr.shape[1], dtype=np.float64)
                    counts[name] = 0
                sums[name] = sums[name] + arr.sum(axis=0)
                sq[name] = sq[name] + np.square(arr).sum(axis=0)
                counts[name] += int(arr.shape[0])
        mean: dict[str, np.ndarray] = {}
        std: dict[str, np.ndarray] = {}
        for name in channel_tuple:
            if name not in counts or counts[name] == 0:
                raise ValueError(f"no available samples to fit channel {name!r}")
            n = float(counts[name])
            mu = sums[name] / n
            var = np.maximum(sq[name] / n - np.square(mu), 0.0)
            mean[name] = mu
            std[name] = np.sqrt(var) + 1e-8
        digest = _stats_digest(channel_tuple, mean, std)
        return cls(
            channels=channel_tuple,
            mean=mean,
            std=std,
            stats_digest=digest,
            _frozen=True,
        )

    def transform(self, episode: EpisodeRecord) -> EpisodeRecord:
        """Apply train stats; retain unavailable masks (do not invent zeros)."""

        def _scaled(name: str, current: np.ndarray | None) -> np.ndarray | None:
            if name not in self.channels:
                return current
            if episode.channel_availability.get(name) != "available":
                return None
            if current is None:
                return None
            arr = np.asarray(current, dtype=np.float64)
            return (arr - self.mean[name]) / self.std[name]

        return EpisodeRecord(
            trial_id=episode.trial_id,
            family_id=episode.family_id,
            model_id=episode.model_id,
            control_basis=episode.control_basis,
            units=episode.units,
            joint_names=episode.joint_names,
            coefficient_letters=episode.coefficient_letters,
            schema_version=episode.schema_version,
            sample_times_s=episode.sample_times_s,
            q=_scaled("q", episode.q),  # type: ignore[arg-type]
            v=_scaled("v", episode.v),
            u=_scaled("u", episode.u),
            a_native=_scaled("a_native", episode.a_native),
            q_next=episode.q_next,
            channel_availability=dict(episode.channel_availability),
            ancestry=episode.ancestry,
            geometry_stratum=episode.geometry_stratum,
            contact_stratum=episode.contact_stratum,
            club_stratum=episode.club_stratum,
            coefficients=episode.coefficients,
            source_schema=episode.source_schema,
        )

    def refit(self, episodes: Sequence[EpisodeRecord]) -> TrainOnlyNormalizer:
        """Resume is immutable — refuse mutation of fitted stats."""
        raise ValueError(
            "TrainOnlyNormalizer resume is immutable; refuse refit of frozen stats "
            f"(digest={self.stats_digest[:12]})"
        )


def _stats_digest(
    channels: tuple[str, ...],
    mean: dict[str, np.ndarray],
    std: dict[str, np.ndarray],
) -> str:
    payload = {
        "channels": list(channels),
        "mean": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in mean.items()},
        "std": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in std.items()},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
