"""Hybrid Physics-Informed Neural Network combining rigid-body and MLP residual.

This module provides :class:`HybridPINN`, which combines:

1. A :class:`~.rigid_core.RigidCore` — Pinocchio RNEA inverse-dynamics torque.
2. An :class:`~.mlp_residual.MlpResidual` — JAX/Equinox MLP predicting the
   torque residual not captured by the rigid-body model.

Total predicted torque is:

.. math::

    \\hat{\\tau} = \\tau_{\\text{rigid}}(q, \\dot{q}, \\ddot{q})
                 + \\tau_{\\text{MLP}}\\bigl([q;\\, \\dot{q};\\, \\ddot{q}]\\bigr)

The MLP receives the concatenation of ``(q, dq, ddq)`` as its input, so
``input_dim`` should be set to ``nq + 2 * nv`` when constructing
:class:`~.mlp_residual.MlpResidual`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.shared.python.physics_informed.mlp_residual import MlpResidual
    from src.shared.python.physics_informed.rigid_core import RigidCore

logger = logging.getLogger(__name__)


class HybridPINN:
    """Sum of rigid-body torque and MLP residual torque.

    Implements the hybrid PINN architecture for golf-swing biomechanics.
    The model predicts the total joint torques by combining the analytical
    rigid-body inverse-dynamics solution with a learned residual that captures
    unmodelled effects (muscle co-contraction, soft-tissue deformation, etc.).

    **MLP input convention**: the network receives the concatenation
    ``[q, dq, ddq]``, where ``q`` has length ``nq`` and ``dq``, ``ddq``
    each have length ``nv``.  Therefore ``mlp_residual.input_dim`` must equal
    ``nq + 2 * nv``.

    Parameters
    ----------
    rigid_core:
        Configured :class:`~.rigid_core.RigidCore` instance.
    mlp_residual:
        Configured :class:`~.mlp_residual.MlpResidual` instance.
    """

    def __init__(
        self,
        rigid_core: RigidCore,
        mlp_residual: MlpResidual,
    ) -> None:
        """Compose rigid-body core and MLP residual into a hybrid predictor.

        DbC preconditions:
        - ``rigid_core`` must be a non-None :class:`RigidCore`.
        - ``mlp_residual`` must be a non-None :class:`MlpResidual`.

        Args:
            rigid_core:    Pinocchio-backed rigid-body torque calculator.
            mlp_residual:  JAX/Equinox MLP residual torque predictor.

        Raises:
            ValueError: If either argument is ``None``.
        """
        if rigid_core is None:
            raise ValueError("rigid_core must not be None")
        if mlp_residual is None:
            raise ValueError("mlp_residual must not be None")

        self._rigid = rigid_core
        self._mlp = mlp_residual

        logger.debug(
            "HybridPINN created: nq=%d nv=%d mlp_output_dim=%d",
            rigid_core.nq,
            rigid_core.nv,
            mlp_residual.output_dim,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
    ) -> np.ndarray:
        """Return total torque = rigid + residual.

        The MLP receives ``concat([q, dq, ddq])`` as its input feature vector.

        DbC preconditions:
        - ``q``, ``dq``, ``ddq`` must each be 1-D arrays.
        - ``dq`` and ``ddq`` must have the same length.

        DbC postcondition:
        - Result shape matches the rigid-body torque output shape ``(nv,)``.

        Args:
            q:   Configuration vector, shape ``(nq,)``.
            dq:  Velocity vector, shape ``(nv,)``.
            ddq: Acceleration vector, shape ``(nv,)``.

        Returns:
            Total torque ``τ_rigid + τ_residual``, shape ``(nv,)``.

        Raises:
            ValueError: If input shapes are incompatible or the rigid-body and
                MLP output shapes disagree.
        """
        q_arr = np.asarray(q, dtype=np.float64)
        dq_arr = np.asarray(dq, dtype=np.float64)
        ddq_arr = np.asarray(ddq, dtype=np.float64)

        self._validate_input_shapes(q_arr, dq_arr, ddq_arr)

        tau_rigid = self._rigid.compute_torques(q_arr, dq_arr, ddq_arr)
        tau_residual = self._compute_residual(q_arr, dq_arr, ddq_arr)

        self._validate_output_shapes(tau_rigid, tau_residual)

        result: np.ndarray = tau_rigid + tau_residual
        logger.debug(
            "HybridPINN.predict: rigid_norm=%.4f residual_norm=%.4f",
            float(np.linalg.norm(tau_rigid)),
            float(np.linalg.norm(tau_residual)),
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_residual(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
    ) -> np.ndarray:
        """Concatenate kinematic state and run MLP forward pass.

        Args:
            q:   Configuration vector.
            dq:  Velocity vector.
            ddq: Acceleration vector.

        Returns:
            Residual torque as a NumPy float64 array.
        """
        x = np.concatenate([q, dq, ddq])
        mlp_out = self._mlp(x)
        # Convert from JAX array (if applicable) to numpy
        return np.asarray(mlp_out, dtype=np.float64)

    def _validate_input_shapes(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
    ) -> None:
        """Raise ValueError if input arrays have incompatible shapes.

        Args:
            q:   Configuration vector.
            dq:  Velocity vector.
            ddq: Acceleration vector.

        Raises:
            ValueError: On shape mismatch.
        """
        if q.ndim != 1:
            raise ValueError(f"q must be a 1-D array; got shape {q.shape}")
        if dq.ndim != 1:
            raise ValueError(f"dq must be a 1-D array; got shape {dq.shape}")
        if ddq.ndim != 1:
            raise ValueError(f"ddq must be a 1-D array; got shape {ddq.shape}")
        if dq.shape != ddq.shape:
            raise ValueError(
                f"dq and ddq must have the same shape; "
                f"got dq={dq.shape}, ddq={ddq.shape}"
            )
        # Validate against the rigid-body model's expected dimensions
        nq = self._rigid.nq
        nv = self._rigid.nv
        if q.shape[0] != nq:
            raise ValueError(f"q shape mismatch: expected ({nq},), got {q.shape}")
        if dq.shape[0] != nv:
            raise ValueError(f"dq shape mismatch: expected ({nv},), got {dq.shape}")
        if ddq.shape[0] != nv:
            raise ValueError(f"ddq shape mismatch: expected ({nv},), got {ddq.shape}")

    @staticmethod
    def _validate_output_shapes(
        tau_rigid: np.ndarray,
        tau_residual: np.ndarray,
    ) -> None:
        """Raise ValueError if rigid and MLP output shapes disagree.

        Args:
            tau_rigid:    Torque from the rigid-body model.
            tau_residual: Torque from the MLP.

        Raises:
            ValueError: If the arrays have different shapes.
        """
        if tau_rigid.shape != tau_residual.shape:
            raise ValueError(
                f"Rigid-body and MLP output shape mismatch: "
                f"rigid={tau_rigid.shape}, residual={tau_residual.shape}. "
                f"Ensure mlp_residual.output_dim == rigid_core.nv."
            )
