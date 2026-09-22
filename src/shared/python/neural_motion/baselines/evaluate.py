"""Evaluation helpers for NM-05 dynamics baselines."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.shared.python.neural_motion.episodes import EpisodeStore

__all__ = ["evaluate_rollout_vs_onestep", "identity_channel_leakage_score"]


def identity_channel_leakage_score(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
) -> dict[str, bool]:
    """Detect whether identity channels are scored as dynamics skill."""
    feat = {str(n) for n in feature_names}
    targ = {str(n) for n in target_names}
    q_scored = any(n == "q" or n.startswith("q[") for n in targ)
    v_scored = any(n == "v" or n.startswith("v[") for n in targ)
    q_in_feat = any(n == "q" or n.startswith("q[") for n in feat)
    return {
        "q_in_features_and_scored_as_target": bool(q_in_feat and q_scored),
        "v_scored_as_target": bool(v_scored),
        "features_finite": bool(np.all(np.isfinite(features))),
        "targets_finite": bool(np.all(np.isfinite(targets))),
    }


def evaluate_rollout_vs_onestep(
    *,
    store: EpisodeStore,
    trial_ids: Sequence[str],
    predictor: str,
    horizon: int,
    active_dofs: Sequence[int],
) -> dict[str, object]:
    """Compare open-loop multi-step drift against one-step prediction error."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if predictor != "identity_hold":
        raise ValueError(f"unsupported predictor {predictor!r}")

    dofs = tuple(int(i) for i in active_dofs)
    one_step_errors: list[float] = []
    rollout_errors: list[float] = []

    for trial_id in trial_ids:
        episode = _read_by_trial(store, trial_id)
        q = np.asarray(episode.q, dtype=np.float64)[:, dofs]
        q_next = episode.q_next
        if q_next is None:
            raise ValueError("q_next required for rollout comparison")
        qn = np.asarray(q_next, dtype=np.float64)[:, dofs]
        one_step_errors.append(float(np.mean((qn - q) ** 2)))
        pred = q.copy()
        errs = []
        steps = min(horizon, q.shape[0] - 1)
        for t in range(steps):
            pred[t + 1] = pred[t]
            errs.append(float(np.mean((pred[t + 1] - q[t + 1]) ** 2)))
        rollout_errors.append(float(np.mean(errs)) if errs else 0.0)

    one_step = float(np.mean(one_step_errors)) if one_step_errors else 0.0
    rollout = float(np.mean(rollout_errors)) if rollout_errors else 0.0
    return {
        "one_step_mse": one_step,
        "rollout_mse": rollout,
        "horizon": int(horizon),
        "native_qualified": False,
        "limitations": ("software_contract", "identity_hold_predictor"),
    }


def _read_by_trial(store: EpisodeStore, trial_id: str):
    for eid in store.iter_episode_ids():
        episode = store.read_episode(eid, lazy=False)
        if episode.trial_id == trial_id:
            return episode
    raise KeyError(f"trial_id {trial_id!r} not found")
