"""Trial-level feature/target matrices from EpisodeStore (NM-05)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.shared.python.neural_motion.episodes import (
    EpisodeRecord,
    EpisodeStore,
    FamilySplitPlan,
)

from .types import DynamicsTaskKind, InverseLabelConditioning

__all__ = ["TrialMatrixBundle", "build_trial_matrices", "read_episode_by_trial"]


@dataclass(frozen=True, slots=True)
class TrialMatrixBundle:
    """Row-per-timestep matrices with trial ids (no row-level shuffle)."""

    features: np.ndarray
    targets: np.ndarray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    trial_ids: tuple[str, ...]
    active_dofs: tuple[int, ...]
    sample_mask: np.ndarray


def build_trial_matrices(
    store: EpisodeStore,
    plan: FamilySplitPlan,
    *,
    task: DynamicsTaskKind,
    split: str,
    active_dofs: Sequence[int],
    conditioning: InverseLabelConditioning | None = None,
) -> TrialMatrixBundle:
    """Build features/targets for one family split.

    Design by Contract:
    - Splits are trial-level (from ``FamilySplitPlan``); never mix families.
    - Identity channels ``q``/``v`` are never scored as dynamics targets.
    - Inverse tasks require ``conditioning``; unavailable torque raises.
    - ``active_dofs`` must be non-empty and within joint dimension.
    """

    if not isinstance(task, DynamicsTaskKind):
        raise TypeError("task must be a DynamicsTaskKind")
    if task is DynamicsTaskKind.INVERSE_CONTROL and conditioning is None:
        raise ValueError(
            "inverse_control requires InverseLabelConditioning "
            "(contact/actuation/allocation); refuse ambiguous torque labels"
        )
    dofs = tuple(int(i) for i in active_dofs)
    if not dofs:
        raise ValueError("active_dofs must be non-empty")
    if any(i < 0 for i in dofs):
        raise ValueError("active_dofs must be non-negative")

    trial_ids = plan.ids_for(split)
    if not trial_ids:
        n_dof = len(dofs)
        if task is DynamicsTaskKind.INVERSE_CONTROL:
            feature_names = tuple(
                f"{ch}[{i}]" for ch in ("q", "v", "a_native") for i in dofs
            )
            target_names = tuple(f"u[{i}]" for i in dofs)
        else:
            feature_names = tuple(f"{ch}[{i}]" for ch in ("q", "v", "u") for i in dofs)
            if task is DynamicsTaskKind.FORWARD_ACCELERATION:
                target_names = tuple(f"a_native[{i}]" for i in dofs)
            else:
                target_names = tuple(f"q_next[{i}]" for i in dofs)
        n_feat = 3 * n_dof
        return TrialMatrixBundle(
            features=np.zeros((0, n_feat), dtype=np.float64),
            targets=np.zeros((0, n_dof), dtype=np.float64),
            feature_names=feature_names,
            target_names=target_names,
            trial_ids=(),
            active_dofs=dofs,
            sample_mask=np.zeros(0, dtype=bool),
        )

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    kept_trials: list[str] = []
    masks: list[np.ndarray] = []
    names_f: tuple[str, ...] = ()
    names_t: tuple[str, ...] = ()

    for trial_id in trial_ids:
        episode = read_episode_by_trial(store, trial_id)
        feat, targ, names_f, names_t, mask = _episode_arrays(
            episode, task=task, active_dofs=dofs
        )
        features.append(feat)
        targets.append(targ)
        masks.append(mask)
        kept_trials.extend([trial_id] * feat.shape[0])

    return TrialMatrixBundle(
        features=np.concatenate(features, axis=0),
        targets=np.concatenate(targets, axis=0),
        feature_names=names_f,
        target_names=names_t,
        trial_ids=tuple(kept_trials),
        active_dofs=dofs,
        sample_mask=np.concatenate(masks, axis=0),
    )


def read_episode_by_trial(store: EpisodeStore, trial_id: str) -> EpisodeRecord:
    for eid in store.iter_episode_ids():
        episode = store.read_episode(eid, lazy=False)
        if episode.trial_id == trial_id:
            return episode
    raise KeyError(f"trial_id {trial_id!r} not found in EpisodeStore")


def _state_control_features(
    q: np.ndarray,
    v: object,
    u: object,
    active_dofs: tuple[int, ...],
) -> tuple[list[np.ndarray], tuple[str, ...]]:
    """Assemble ``(q, v, u)`` active-DOF feature columns for forward tasks."""
    feat_parts = [
        q[:, active_dofs],
        np.asarray(v)[:, active_dofs],
        np.asarray(u)[:, active_dofs],
    ]
    feature_names = tuple(f"{ch}[{i}]" for ch in ("q", "v", "u") for i in active_dofs)
    return feat_parts, feature_names


def _episode_arrays(
    episode: EpisodeRecord,
    *,
    task: DynamicsTaskKind,
    active_dofs: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], np.ndarray]:
    q = np.asarray(episode.q, dtype=np.float64)
    n_t, n_j = q.shape
    if any(i >= n_j for i in active_dofs):
        raise ValueError("active_dofs exceed joint dimension")

    v = episode.v
    u = episode.u
    a_native = episode.a_native
    q_next = episode.q_next

    if task is DynamicsTaskKind.INVERSE_CONTROL:
        if episode.channel_availability.get("u") != "available" or u is None:
            raise ValueError(
                "inverse_control torque channel is unavailable; "
                "refuse silent zero mask as measured torque labels"
            )
        if v is None or a_native is None:
            raise ValueError("inverse_control requires v and a_native")
        feat_parts = [
            q[:, active_dofs],
            np.asarray(v)[:, active_dofs],
            np.asarray(a_native)[:, active_dofs],
        ]
        feature_names = tuple(
            f"{ch}[{i}]" for ch in ("q", "v", "a_native") for i in active_dofs
        )
        targ = np.asarray(u, dtype=np.float64)[:, active_dofs]
        target_names = tuple(f"u[{i}]" for i in active_dofs)
    elif task is DynamicsTaskKind.FORWARD_ACCELERATION:
        if v is None or u is None or a_native is None:
            raise ValueError("forward_acceleration requires v, u and a_native")
        feat_parts, feature_names = _state_control_features(q, v, u, active_dofs)
        targ = np.asarray(a_native, dtype=np.float64)[:, active_dofs]
        target_names = tuple(f"a_native[{i}]" for i in active_dofs)
    else:
        if v is None or u is None or q_next is None:
            raise ValueError("forward_next_state requires v, u and q_next")
        feat_parts, feature_names = _state_control_features(q, v, u, active_dofs)
        targ = np.asarray(q_next, dtype=np.float64)[:, active_dofs]
        target_names = tuple(f"q_next[{i}]" for i in active_dofs)

    for name in target_names:
        base = name.split("[", 1)[0]
        if base in {"q", "v"}:
            raise ValueError(
                f"identity channel {base!r} cannot be scored as dynamics skill"
            )

    features = np.concatenate(feat_parts, axis=1)
    if not bool(np.all(np.isfinite(features))) or not bool(np.all(np.isfinite(targ))):
        raise ValueError("features/targets must be finite")
    mask = np.ones(n_t, dtype=bool)
    return features, targ, feature_names, target_names, mask
