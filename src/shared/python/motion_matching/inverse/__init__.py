"""Option-3 inverse CVAE: kinematics -> torque coefficients.

Public API:
    CVAEConfig          -- frozen dataclass of architectural hyperparameters.
    EncoderOutput       -- frozen dataclass holding posterior (mu, log_var, z).
    SwingInverseCVAE    -- the model class (encoder + decoder + reparam).
    TrainInverseConfig  -- training-loop hyperparameters (#033).
    TrainedInverseCVAE  -- handle returned by :func:`train_inverse_cvae`.
    train_inverse_cvae  -- ELBO + work-regularised training loop with KL anneal.

Inference with rejection sampling (#034) lives in a sibling module.

See ``src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/
option3_inverse_nn/INTERFACES.md`` for the full design contract.
"""

from .cvae import CVAEConfig, EncoderOutput, SwingInverseCVAE
from .train import TrainedInverseCVAE, TrainInverseConfig, train_inverse_cvae

__all__ = [
    "CVAEConfig",
    "EncoderOutput",
    "SwingInverseCVAE",
    "TrainInverseConfig",
    "TrainedInverseCVAE",
    "train_inverse_cvae",
]
