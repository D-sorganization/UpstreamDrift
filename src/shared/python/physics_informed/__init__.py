"""Physics-Informed Neural Network (PINN) hybrid architecture.

Phases 1-3 of the PINNs epic (#5419):

Phase 1 -- hybrid architecture:

- :class:`RigidCore`: Pinocchio RNEA rigid-body inverse-dynamics torque.
- :class:`MlpResidual`: JAX/Equinox MLP predicting unmodelled torque residuals.
- :class:`HybridPINN`: Summed predictor ``tau = tau_rigid + tau_residual``.

Phase 2 -- three-component loss function:

- :class:`LossWeights`: frozen dataclass with DbC-validated positive weights.
- :func:`data_loss`: MSE between predicted and actual motion kinematics.
- :func:`physics_loss`: penalises joint-limit violations and energy anomalies.
- :func:`contact_loss`: penalises non-zero torques during non-contact phases.
- :func:`total_loss`: weighted sum of the three component losses.

Phase 3 -- shadow model and mode factory:

- :class:`SwingPhase`: Enum for the three golf-swing phases.
- :class:`ShadowReport`: Dataclass holding peak residual torques per phase.
- :class:`ShadowModel`: Observation-mode runner; records peaks without
  modifying simulation state.
- :class:`PhysicsMode`: Enum for model-creation modes.
- :func:`create_model`: Factory returning RigidCore, MlpResidual, or
  HybridPINN based on the requested :class:`PhysicsMode`.

Optional dependencies
---------------------
Pinocchio:
    ``pip install upstream-drift[pinocchio]``

JAX + Equinox:
    ``pip install upstream-drift[physics_informed]``

Both sets of dependencies are optional; the package imports cleanly without
them, and classes/functions that require missing extras raise
:class:`ImportError` at call time.
"""

from src.shared.python.physics_informed.hybrid_model import HybridPINN
from src.shared.python.physics_informed.loss import (
    LossWeights,
    contact_loss,
    data_loss,
    physics_loss,
    total_loss,
)
from src.shared.python.physics_informed.mlp_residual import MlpResidual
from src.shared.python.physics_informed.mode import PhysicsMode, create_model
from src.shared.python.physics_informed.rigid_core import RigidCore
from src.shared.python.physics_informed.shadow_model import (
    ShadowModel,
    ShadowReport,
    SwingPhase,
)

__all__ = [
    "HybridPINN",
    "LossWeights",
    "MlpResidual",
    "PhysicsMode",
    "RigidCore",
    "ShadowModel",
    "ShadowReport",
    "SwingPhase",
    "contact_loss",
    "create_model",
    "data_loss",
    "physics_loss",
    "total_loss",
]
