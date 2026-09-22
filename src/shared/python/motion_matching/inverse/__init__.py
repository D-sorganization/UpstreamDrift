"""Option-3 inverse models: trajectory -> torque coefficients.

The cVAE (``SwingInverseCVAE``) is preserved here for future research. The
production inverse model is the deterministic :class:`InverseRegressor`,
introduced after the cVAE exhibited a hard reconstruction plateau on the
real compact dataset.

NM-06 (#10621) exports masked proposal surfaces that must import without
``torch`` so the unit lane can collect software-contract tests. Torch-heavy
cVAE / regressor / training symbols stay lazy via ``__getattr__``.

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

NM-06 proposals (torch-optional at import):
    MaskedControlProposal, MaskedObservation, MaskedProposalConfig,
    ProposalMode, ProposalOutput, ProposalTrainingConfig,
    ProposalTrainingResult, train_masked_control_proposals,
    CVAE_PLATEAU_EVIDENCE, diagnose_mode_collapse.

Common:
    parameter_count, build_coefficient_bound_vector, COEFFICIENT_LETTER_BOUNDS,
    DEFAULT_COEFFICIENT_DIM, DEFAULT_LATENT_DIM, DEFAULT_N_JOINTS,
    DEFAULT_TRAJECTORY_CHANNELS.
"""

from __future__ import annotations

import importlib
from typing import Any

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
from .proposal_training import (
    ProposalTrainingConfig,
    ProposalTrainingResult,
    train_masked_control_proposals,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "COEFFICIENT_LETTER_BOUNDS": (".cvae", "COEFFICIENT_LETTER_BOUNDS"),
    "CVAEConfig": (".cvae", "CVAEConfig"),
    "CoefficientPredictions": (".predict", "CoefficientPredictions"),
    "DEFAULT_COEFFICIENT_DIM": (".cvae", "DEFAULT_COEFFICIENT_DIM"),
    "DEFAULT_LATENT_DIM": (".cvae", "DEFAULT_LATENT_DIM"),
    "DEFAULT_N_JOINTS": (".cvae", "DEFAULT_N_JOINTS"),
    "DEFAULT_TRAJECTORY_CHANNELS": (".cvae", "DEFAULT_TRAJECTORY_CHANNELS"),
    "EncoderOutput": (".cvae", "EncoderOutput"),
    "EpochMetrics": (".training", "EpochMetrics"),
    "InverseRegressor": (".regressor", "InverseRegressor"),
    "RegressorConfig": (".regressor", "RegressorConfig"),
    "RegressorEpochMetrics": (".regressor_training", "EpochMetrics"),
    "RegressorTrainingResult": (".regressor_training", "RegressorTrainingResult"),
    "SwingInverseCVAE": (".cvae", "SwingInverseCVAE"),
    "TrainingConfig": (".training", "TrainingConfig"),
    "TrainingResult": (".training", "TrainingResult"),
    "build_coefficient_bound_vector": (".cvae", "build_coefficient_bound_vector"),
    "kl_divergence": (".cvae", "kl_divergence"),
    "kl_divergence_per_dim": (".cvae", "kl_divergence_per_dim"),
    "load_inverse_cvae": (".predict", "load_inverse_cvae"),
    "load_inverse_regressor": (".regressor_predict", "load_inverse_regressor"),
    "parameter_count": (".cvae", "parameter_count"),
    "predict_coefficients": (".predict", "predict_coefficients"),
    "predict_coefficients_from_checkpoint": (
        ".predict",
        "predict_coefficients_from_checkpoint",
    ),
    "predict_coefficients_regressor": (
        ".regressor_predict",
        "predict_coefficients_regressor",
    ),
    "predict_coefficients_regressor_from_checkpoint": (
        ".regressor_predict",
        "predict_coefficients_regressor_from_checkpoint",
    ),
    "train_inverse_cvae": (".training", "train_inverse_cvae"),
    "train_inverse_regressor": (".regressor_training", "train_inverse_regressor"),
}

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


def __getattr__(name: str) -> Any:
    """Load torch-backed inverse symbols on demand (NM-06 unit-lane safe)."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
