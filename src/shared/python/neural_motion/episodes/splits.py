"""Family-level splits, alias detection and held-out strata (NM-03 #10618)."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.shared.python.neural_motion.json_io import SortedJsonWritableMixin

from .record import EPISODE_STORE_SCHEMA, EpisodeRecord

__all__ = [
    "FamilySplitPlan",
    "build_family_splits",
    "detect_source_aliases",
]


@dataclass(frozen=True)
class FamilySplitPlan(SortedJsonWritableMixin):
    """Deterministic split assignment keyed by generation family."""

    schema: str
    splits: dict[str, tuple[str, ...]]
    family_split: dict[str, str]
    trial_split: dict[str, str]
    content_digest: str
    held_out_strata: Mapping[str, tuple[str, ...]]

    def split_of(self, trial_id: str) -> str:
        try:
            return self.trial_split[trial_id]
        except KeyError as exc:
            raise KeyError(f"unknown trial_id {trial_id!r}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "splits": {name: list(ids) for name, ids in sorted(self.splits.items())},
            "family_split": dict(sorted(self.family_split.items())),
            "trial_split": dict(sorted(self.trial_split.items())),
            "content_digest": self.content_digest,
            "held_out_strata": {
                key: list(values)
                for key, values in sorted(self.held_out_strata.items())
            },
        }


def detect_source_aliases(
    episodes: Sequence[EpisodeRecord],
) -> dict[str, tuple[str, ...]]:
    """Group trial ids that share content hash or workbook ancestry aliases."""
    by_hash: dict[str, list[str]] = defaultdict(list)
    by_lineage: dict[str, list[str]] = defaultdict(list)
    for episode in episodes:
        by_hash[episode.content_payload_digest()].append(episode.trial_id)
        for ancestor in episode.ancestry:
            if ancestor.startswith("workbook:") or ancestor == "dataset_copy":
                # Bind copies to the first workbook ancestor when present.
                workbook = next(
                    (a for a in episode.ancestry if a.startswith("workbook:")),
                    ancestor,
                )
                by_lineage[workbook].append(episode.trial_id)
    groups: dict[str, tuple[str, ...]] = {}
    for digest, trials in by_hash.items():
        unique = tuple(sorted(set(trials)))
        if len(unique) > 1:
            groups[f"content:{digest[:12]}"] = unique
    for lineage, trials in by_lineage.items():
        unique = tuple(sorted(set(trials)))
        if len(unique) > 1:
            groups[f"lineage:{lineage}"] = unique
    return groups


def build_family_splits(
    episodes: Sequence[EpisodeRecord],
    *,
    ratios: Mapping[str, float],
    seed: int,
    held_out_strata: Mapping[str, Sequence[str]] | None = None,
) -> FamilySplitPlan:
    """Split by complete family before any window/augmentation slicing.

    Near-duplicates and augmentations that share ``family_id`` always land in
    the same split. Geometry / contact / club held-out strata and a reserved
    ``real_data_eval`` bucket are assigned before random family allocation.
    """
    if not episodes:
        raise ValueError("episodes must be non-empty")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1.0")
    for name, value in ratios.items():
        if value < 0.0:
            raise ValueError(f"ratio {name} must be non-negative")

    held = {key: tuple(values) for key, values in (held_out_strata or {}).items()}
    families: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for episode in episodes:
        families[episode.family_id].append(episode)

    family_split: dict[str, str] = {}
    trial_split: dict[str, str] = {}
    buckets: dict[str, list[str]] = {name: [] for name in ratios}
    buckets.setdefault("eval_held_out", [])
    buckets.setdefault("real_data_eval", [])

    random_families: list[str] = []
    for family_id, members in sorted(families.items()):
        if _is_held_out_family(members, held):
            family_split[family_id] = "eval_held_out"
            for member in members:
                trial_split[member.trial_id] = "eval_held_out"
                buckets["eval_held_out"].append(member.trial_id)
            continue
        if _is_real_data_family(members):
            family_split[family_id] = "real_data_eval"
            for member in members:
                trial_split[member.trial_id] = "real_data_eval"
                buckets["real_data_eval"].append(member.trial_id)
            continue
        random_families.append(family_id)

    ordered = _stable_shuffle(random_families, seed=seed)
    assignments = _assign_by_ratios(ordered, ratios)
    for family_id, split_name in assignments.items():
        family_split[family_id] = split_name
        for member in families[family_id]:
            trial_split[member.trial_id] = split_name
            buckets[split_name].append(member.trial_id)

    splits = {name: tuple(sorted(set(ids))) for name, ids in buckets.items()}
    payload = {
        "schema": EPISODE_STORE_SCHEMA,
        "family_split": family_split,
        "trial_split": trial_split,
        "splits": {k: list(v) for k, v in splits.items()},
        "held_out_strata": {k: list(v) for k, v in held.items()},
        "seed": seed,
        "ratios": dict(ratios),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return FamilySplitPlan(
        schema=EPISODE_STORE_SCHEMA,
        splits=splits,
        family_split=family_split,
        trial_split=trial_split,
        content_digest=digest,
        held_out_strata=held,
    )


def _is_held_out_family(
    members: Sequence[EpisodeRecord],
    held: Mapping[str, Sequence[str]],
) -> bool:
    geom = set(held.get("geometry", ()))
    contact = set(held.get("contact", ()))
    club = set(held.get("club", ()))
    for member in members:
        if member.geometry_stratum in geom:
            return True
        if member.contact_stratum in contact:
            return True
        if member.club_stratum in club:
            return True
    return False


def _is_real_data_family(members: Sequence[EpisodeRecord]) -> bool:
    for member in members:
        if any(a.startswith("workbook:") for a in member.ancestry):
            return True
        if any(a.startswith("c3d:") for a in member.ancestry):
            return True
    return False


def _stable_shuffle(items: Sequence[str], *, seed: int) -> list[str]:
    """Deterministic order from seed without depending on global RNG state."""
    keyed = [
        (hashlib.sha256(f"{seed}:{item}".encode()).hexdigest(), item) for item in items
    ]
    keyed.sort()
    return [item for _, item in keyed]


def _assign_by_ratios(
    families: Sequence[str],
    ratios: Mapping[str, float],
) -> dict[str, str]:
    """Allocate families with the largest-remainder method (no empty-force)."""
    if not families:
        return {}
    names = [name for name in ("train", "val", "test") if name in ratios] + [
        name for name in ratios if name not in {"train", "val", "test"}
    ]
    n_families = len(families)
    exact = [float(ratios[name]) * n_families for name in names]
    floors = [int(value) for value in exact]
    remainder = n_families - sum(floors)
    # Prefer larger fractional parts; break ties by declared name order.
    frac_order = sorted(
        range(len(names)),
        key=lambda idx: (exact[idx] - floors[idx], -idx),
        reverse=True,
    )
    for idx in frac_order[:remainder]:
        floors[idx] += 1

    assignment: dict[str, str] = {}
    cursor = 0
    for name, take in zip(names, floors, strict=True):
        for family_id in families[cursor : cursor + take]:
            assignment[family_id] = name
        cursor += take
    for family_id in families[cursor:]:
        assignment[family_id] = names[0]
    return assignment
