"""Classical and small neural dynamics baselines (NM-05 #10620).

Schema: ``neural-dynamics-baselines/1.0.0``.

Reuses NM-03 ``EpisodeStore`` / ``FamilySplitPlan`` / ``TrainOnlyNormalizer`` and
NM-04 teacher corpus registration seams. Benchmarks analytical / nearest-
neighbor / ridge baselines before an optional small MLP. Software-contract
fixtures only — never invent native training success.
"""

from __future__ import annotations

from .classical import ClassicalMethod, fit_classical, predict_classical
from .dataset import TrialMatrixBundle, build_trial_matrices, read_episode_by_trial
from .evaluate import evaluate_rollout_vs_onestep, identity_channel_leakage_score
from .neural import optional_torch_available, torch_is_available, train_small_mlp
from .train import DynamicsBaselineTrainer, PilotCheckpointCard
from .types import (
    BASELINE_SCHEMA,
    DynamicsTaskKind,
    InverseLabelConditioning,
)

__all__ = [
    "BASELINE_SCHEMA",
    "ClassicalMethod",
    "DynamicsBaselineTrainer",
    "DynamicsTaskKind",
    "InverseLabelConditioning",
    "PilotCheckpointCard",
    "TrialMatrixBundle",
    "build_trial_matrices",
    "evaluate_rollout_vs_onestep",
    "fit_classical",
    "identity_channel_leakage_score",
    "optional_torch_available",
    "predict_classical",
    "read_episode_by_trial",
    "torch_is_available",
    "train_small_mlp",
]
