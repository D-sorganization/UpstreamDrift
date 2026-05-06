"""Option-3 inverse CVAE: kinematics -> torque coefficients.

Public API:
    CVAEConfig          -- frozen dataclass of architectural hyperparameters.
    EncoderOutput       -- frozen dataclass holding posterior (mu, log_var, z).
    SwingInverseCVAE    -- the model class (encoder + decoder + reparam).
    TrainInverseConfig  -- training-loop hyperparameters (#033).
    TrainedInverseCVAE  -- canonical handle returned by :func:`train_inverse_cvae`.
    train_inverse_cvae  -- ELBO + work-regularised training loop with KL anneal.
    InverseFitResult    -- result of a rejection-sampling inference call.
    ValidationReport    -- single-sample round-trip validation summary.
    predict_coefficients -- sample-and-validate inference (#034).
    round_trip_validate  -- helper for round-trip RMSE checks.

See ``src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/
option3_inverse_nn/INTERFACES.md`` for the full design contract.
"""

from ._validate import ValidationReport, round_trip_validate
from .cvae import CVAEConfig, EncoderOutput, SwingInverseCVAE
from .predict import InverseFitResult, predict_coefficients
from .train import TrainedInverseCVAE, TrainInverseConfig, train_inverse_cvae

__all__ = [
    "CVAEConfig",
    "EncoderOutput",
    "InverseFitResult",
    "SwingInverseCVAE",
    "TrainInverseConfig",
    "TrainedInverseCVAE",
    "ValidationReport",
    "predict_coefficients",
    "round_trip_validate",
    "train_inverse_cvae",
]
