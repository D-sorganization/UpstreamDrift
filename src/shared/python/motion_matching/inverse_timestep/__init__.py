"""Per-timestep inverse-dynamics model.

A switch from full-trial inverse models (cVAE in :mod:`..inverse.cvae`,
deterministic regressor in :mod:`..inverse.regressor`) — both of which
plateau at the mean-prediction baseline on the random-sweep dataset
because the trajectory -> 189-coefficient mapping is under-determined —
to a per-timestep framing: each timestep is an independent sample of
``(q, qd, qdd)`` -> ``tau``.

Pipeline:
    1. Load the compact dataset.
    2. Filter timesteps by realistic clubhead speed (default 50-150 mph).
    3. Train an MLP ``(q, qd, qdd) -> tau`` per timestep, masking joint
       indices whose ``tau`` is NaN (joints not exposed by the dump).

Public API:
    realistic_speed_mask, filter_timesteps_by_speed,
    TimestepInverseDynamics, TimestepInverseConfig,
    train_timestep_inverse, TimestepTrainingResult,
    TimestepEpochMetrics, predict_torques.
"""

from __future__ import annotations

from .filter import (
    filter_timesteps_by_speed,
    realistic_speed_mask,
)
from .model import (
    DEFAULT_HIDDEN,
    DEFAULT_INPUT_DIM,
    DEFAULT_N_BLOCKS,
    DEFAULT_OUTPUT_DIM,
    TimestepInverseConfig,
    TimestepInverseDynamics,
)
from .predict import (
    predict_torques,
)
from .training import (
    TimestepEpochMetrics,
    TimestepTrainingResult,
    train_timestep_inverse,
)

__all__ = [
    "DEFAULT_HIDDEN",
    "DEFAULT_INPUT_DIM",
    "DEFAULT_N_BLOCKS",
    "DEFAULT_OUTPUT_DIM",
    "TimestepEpochMetrics",
    "TimestepInverseConfig",
    "TimestepInverseDynamics",
    "TimestepTrainingResult",
    "filter_timesteps_by_speed",
    "predict_torques",
    "realistic_speed_mask",
    "train_timestep_inverse",
]
