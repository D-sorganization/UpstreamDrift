"""Option-3 inverse CVAE: kinematics -> torque coefficients.

Public API:
    CVAEConfig       -- frozen dataclass of hyperparameters.
    EncoderOutput    -- frozen dataclass holding posterior (mu, log_var, z).
    SwingInverseCVAE -- the model class (encoder + decoder + reparam + sampling).

Training (#033) and inference with rejection sampling (#034) live in
sibling modules and are out of scope for this package.

See ``src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/
option3_inverse_nn/INTERFACES.md`` for the full design contract.
"""

from .cvae import CVAEConfig, EncoderOutput, SwingInverseCVAE

__all__ = [
    "CVAEConfig",
    "EncoderOutput",
    "SwingInverseCVAE",
]
