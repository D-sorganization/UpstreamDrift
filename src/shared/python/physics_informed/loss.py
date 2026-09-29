"""Three-component PINN loss function for hybrid physics-informed models.

Phase 2 of the PINNs epic (#5419). Provides:

- :class:`LossWeights`: frozen dataclass with DbC validation of positive weights.
- :func:`data_loss`: MSE between predicted and actual motion kinematics.
- :func:`physics_loss`: penalises joint-limit violations and energy anomalies.
- :func:`contact_loss`: penalises non-zero torques during non-contact phases.
- :func:`total_loss`: weighted sum of the three component losses.

JAX is an *optional* dependency:

    pip install upstream-drift[physics_informed]

If JAX is not installed the module imports cleanly, but calling the loss
functions will raise :class:`ImportError` at runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

try:
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:  # pragma: no cover
    HAS_JAX = False
    jnp = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import jax


@dataclass(frozen=True)
class LossWeights:
    """Weights for the three loss components.

    All weights must be strictly positive (DbC precondition).

    Attributes:
        data_weight:    Weight applied to :func:`data_loss`.
        physics_weight: Weight applied to :func:`physics_loss`.
        contact_weight: Weight applied to :func:`contact_loss`.

    Raises:
        ValueError: If any weight is not strictly positive.
    """

    data_weight: float
    physics_weight: float
    contact_weight: float

    def __post_init__(self) -> None:
        """DbC: all weights must be strictly positive."""
        for name, val in [
            ("data_weight", self.data_weight),
            ("physics_weight", self.physics_weight),
            ("contact_weight", self.contact_weight),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")


def data_loss(
    predicted_motion: jax.Array,
    actual_kinematics: jax.Array,
) -> jax.Array:
    """Compute mean squared error between predicted and actual motion.

    Args:
        predicted_motion:  Model-predicted motion array, shape ``(n,)``.
        actual_kinematics: Ground-truth motion array, same shape as
            ``predicted_motion``.

    Returns:
        Scalar MSE loss value.

    Raises:
        ImportError: If JAX is not installed.
    """
    if not HAS_JAX:  # pragma: no cover
        raise ImportError(
            "jax is required for data_loss; install with: "
            "pip install upstream-drift[physics_informed]"
        )
    # ⚡ Bolt: jnp.vdot is significantly faster than jnp.mean(diff ** 2) for MSE calculation
    diff = predicted_motion - actual_kinematics
    return jnp.vdot(diff, diff) / diff.size


def physics_loss(
    torques: jax.Array,
    joint_limits: jax.Array,
    energy_delta: jax.Array,
) -> jax.Array:
    """Penalise joint-limit violations and energy anomalies.

    Args:
        torques:      Torque vector of shape ``(n_joints,)``.
        joint_limits: Limit array of shape ``(n_joints, 2)`` where column 0
            contains minimum values and column 1 contains maximum values.
        energy_delta: Scalar representing the energy change between consecutive
            frames.  Physical motion should have near-zero energy delta.

    Returns:
        Scalar penalty: sum of out-of-bound violations plus the absolute
        energy delta.

    Raises:
        ImportError: If JAX is not installed.
    """
    if not HAS_JAX:  # pragma: no cover
        raise ImportError(
            "jax is required for physics_loss; install with: "
            "pip install upstream-drift[physics_informed]"
        )
    upper_violations = jnp.maximum(0.0, torques - joint_limits[:, 1])
    lower_violations = jnp.maximum(0.0, joint_limits[:, 0] - torques)
    violations = upper_violations + lower_violations
    return jnp.sum(violations) + jnp.abs(energy_delta)


def contact_loss(
    impact_phase_torques: jax.Array,
    *,
    stiffness_coeff: float,
) -> jax.Array:
    """Penalise non-zero torques during non-contact phases.

    During actual contact phases this function should be called with a
    zero-filled array so it contributes 0 to the total loss.

    Args:
        impact_phase_torques: Torques measured (or predicted) during the
            phase under test.  Shape ``(n_joints,)``.  Pass zeros for
            genuine contact frames so no penalty is incurred.
        stiffness_coeff:      Scaling factor for the contact penalty.

    Returns:
        Scalar penalty: ``stiffness_coeff * sum(impact_phase_torques ** 2)``.

    Raises:
        ImportError: If JAX is not installed.
    """
    if not HAS_JAX:  # pragma: no cover
        raise ImportError(
            "jax is required for contact_loss; install with: "
            "pip install upstream-drift[physics_informed]"
        )
    # ⚡ Bolt: jnp.vdot is significantly faster than jnp.sum(x**2)
    return stiffness_coeff * jnp.vdot(impact_phase_torques, impact_phase_torques)


def total_loss(
    weights: LossWeights,
    predicted_motion: jax.Array,
    actual_kinematics: jax.Array,
    torques: jax.Array,
    joint_limits: jax.Array,
    energy_delta: jax.Array,
    impact_phase_torques: jax.Array,
    stiffness_coeff: float,
) -> jax.Array:
    """Compute the weighted sum of data, physics and contact losses.

    Args:
        weights:              Weight coefficients for each loss component.
        predicted_motion:     Model-predicted motion array.
        actual_kinematics:    Ground-truth motion array.
        torques:              Torque vector of shape ``(n_joints,)``.
        joint_limits:         Limit array of shape ``(n_joints, 2)``.
        energy_delta:         Scalar energy-change penalty term.
        impact_phase_torques: Torques during the phase under test.
        stiffness_coeff:      Scaling factor passed to :func:`contact_loss`.

    Returns:
        Scalar total loss:
        ``weights.data_weight * data_loss
        + weights.physics_weight * physics_loss
        + weights.contact_weight * contact_loss``.

    Raises:
        ImportError: If JAX is not installed.
    """
    if not HAS_JAX:  # pragma: no cover
        raise ImportError(
            "jax is required for total_loss; install with: "
            "pip install upstream-drift[physics_informed]"
        )
    d_loss = data_loss(predicted_motion, actual_kinematics)
    p_loss = physics_loss(torques, joint_limits, energy_delta)
    c_loss = contact_loss(impact_phase_torques, stiffness_coeff=stiffness_coeff)

    result = (
        weights.data_weight * d_loss
        + weights.physics_weight * p_loss
        + weights.contact_weight * c_loss
    )

    logger.debug(
        "total_loss: data=%.4f physics=%.4f contact=%.4f total=%.4f",
        float(d_loss),
        float(p_loss),
        float(c_loss),
        float(result),
    )

    return result
