"""Option-3 inverse cVAE: trajectory -> torque-coefficient posterior.

Public API (per GH issue #4076):
    SwingInverseCVAE          -- model class (1D-conv encoder + Gaussian heads).
    CVAEConfig                -- frozen architectural config dataclass.
    EncoderOutput             -- posterior + prior parameter bundle.
    kl_divergence             -- closed-form KL between diagonal Gaussians.
    train_inverse_cvae        -- training loop, returns TrainingResult.
    TrainingConfig            -- training hyperparameter dataclass.
    TrainingResult            -- frozen outcome dataclass.
    EpochMetrics              -- per-epoch loss summary.
    predict_coefficients      -- sample N coefficient vectors for one target.
    predict_coefficients_from_checkpoint -- load + sample one-shot.
    load_inverse_cvae         -- restore a SwingInverseCVAE from a checkpoint.
    CoefficientPredictions    -- frozen predict result.
"""

from .cvae import (
    COEFFICIENT_LETTER_BOUNDS,
    DEFAULT_COEFFICIENT_DIM,
    DEFAULT_LATENT_DIM,
    DEFAULT_N_JOINTS,
    DEFAULT_TRAJECTORY_CHANNELS,
    CVAEConfig,
    EncoderOutput,
    SwingInverseCVAE,
    build_coefficient_bound_vector,
    kl_divergence,
    parameter_count,
)
from .predict import (
    CoefficientPredictions,
    load_inverse_cvae,
    predict_coefficients,
    predict_coefficients_from_checkpoint,
)
from .training import (
    EpochMetrics,
    TrainingConfig,
    TrainingResult,
    train_inverse_cvae,
)

__all__ = [
    "COEFFICIENT_LETTER_BOUNDS",
    "CVAEConfig",
    "CoefficientPredictions",
    "DEFAULT_COEFFICIENT_DIM",
    "DEFAULT_LATENT_DIM",
    "DEFAULT_N_JOINTS",
    "DEFAULT_TRAJECTORY_CHANNELS",
    "EncoderOutput",
    "EpochMetrics",
    "SwingInverseCVAE",
    "TrainingConfig",
    "TrainingResult",
    "build_coefficient_bound_vector",
    "kl_divergence",
    "load_inverse_cvae",
    "parameter_count",
    "predict_coefficients",
    "predict_coefficients_from_checkpoint",
    "train_inverse_cvae",
]
