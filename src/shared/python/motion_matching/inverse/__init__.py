"""Option-3 inverse models: trajectory -> torque coefficients.

The cVAE (``SwingInverseCVAE``) is preserved here for future research. The
production inverse model is the deterministic :class:`InverseRegressor`,
introduced after the cVAE exhibited a hard reconstruction plateau on the
real compact dataset.

Public API:

cVAE (research):
    SwingInverseCVAE, CVAEConfig, EncoderOutput, kl_divergence,
    train_inverse_cvae, TrainingConfig, TrainingResult, EpochMetrics,
    predict_coefficients, predict_coefficients_from_checkpoint,
    load_inverse_cvae, CoefficientPredictions.

Regressor (production):
    InverseRegressor, RegressorConfig, train_inverse_regressor,
    RegressorTrainingResult, predict_coefficients_regressor,
    load_inverse_regressor, predict_coefficients_regressor_from_checkpoint.

Common:
    parameter_count, build_coefficient_bound_vector, COEFFICIENT_LETTER_BOUNDS,
    DEFAULT_COEFFICIENT_DIM, DEFAULT_LATENT_DIM, DEFAULT_N_JOINTS,
    DEFAULT_TRAJECTORY_CHANNELS.
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
    kl_divergence_per_dim,
    parameter_count,
)
from .collapse import (
    CVAE_PLATEAU_EVIDENCE,
    ModeCollapseDiagnostic,
    diagnose_mode_collapse,
)
from .masked_proposal import (
    MaskedControlProposal,
    MaskedObservation,
    MaskedProposalConfig,
    ProposalMode,
    ProposalOutput,
)
from .predict import (
    CoefficientPredictions,
    load_inverse_cvae,
    predict_coefficients,
    predict_coefficients_from_checkpoint,
)
from .proposal_training import (
    ProposalTrainingConfig,
    ProposalTrainingResult,
    train_masked_control_proposals,
)
from .regressor import (
    InverseRegressor,
    RegressorConfig,
)
from .regressor_predict import (
    load_inverse_regressor,
    predict_coefficients_regressor,
    predict_coefficients_regressor_from_checkpoint,
)
from .regressor_training import (
    EpochMetrics as RegressorEpochMetrics,
)
from .regressor_training import (
    RegressorTrainingResult,
    train_inverse_regressor,
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
    "CVAE_PLATEAU_EVIDENCE",
    "CoefficientPredictions",
    "DEFAULT_COEFFICIENT_DIM",
    "DEFAULT_LATENT_DIM",
    "DEFAULT_N_JOINTS",
    "DEFAULT_TRAJECTORY_CHANNELS",
    "EncoderOutput",
    "EpochMetrics",
    "InverseRegressor",
    "MaskedControlProposal",
    "MaskedObservation",
    "MaskedProposalConfig",
    "ModeCollapseDiagnostic",
    "ProposalMode",
    "ProposalOutput",
    "ProposalTrainingConfig",
    "ProposalTrainingResult",
    "RegressorConfig",
    "RegressorEpochMetrics",
    "RegressorTrainingResult",
    "SwingInverseCVAE",
    "TrainingConfig",
    "TrainingResult",
    "build_coefficient_bound_vector",
    "diagnose_mode_collapse",
    "kl_divergence",
    "kl_divergence_per_dim",
    "load_inverse_cvae",
    "load_inverse_regressor",
    "parameter_count",
    "predict_coefficients",
    "predict_coefficients_from_checkpoint",
    "predict_coefficients_regressor",
    "predict_coefficients_regressor_from_checkpoint",
    "train_inverse_cvae",
    "train_inverse_regressor",
    "train_masked_control_proposals",
]
