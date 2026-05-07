"""Polynomial torque controller for OpenSim forward dynamics (issue #4120).

Implements a custom OpenSim Controller that applies polynomial torque profiles
to coordinate actuators. The polynomial degree matches Pinocchio (#4118),
with 7 coefficients per joint: tau(t) = sum_{k=0}^{6} theta[7*j + k] * t^k.

Public API:
    PolynomialTorqueController -- OpenSim Controller subclass
    evaluate_torque_polynomial -- utility to evaluate a single joint's torque
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Try to import OpenSim; skip if unavailable
try:
    import opensim

    OPENSIM_AVAILABLE = True
except ImportError:
    OPENSIM_AVAILABLE = False

__all__ = [
    "PolynomialTorqueController",
    "evaluate_torque_polynomial",
]


def evaluate_torque_polynomial(t: float, coeffs: NDArray[np.float64]) -> float:
    """Evaluate polynomial tau(t) = sum_k a_k * t^k for a single joint.

    Args:
        t: Time (seconds)
        coeffs: (7,) array [a0, a1, ..., a6]

    Returns:
        tau(t) scalar

    Raises:
        ValueError: If coeffs is not length 7
    """
    if len(coeffs) != 7:
        raise ValueError(f"coeffs must be length 7, got {len(coeffs)}")
    result = 0.0
    for k, a_k in enumerate(coeffs):
        result += a_k * (t**k)
    return result


# Define base class conditionally based on OpenSim availability
if OPENSIM_AVAILABLE:
    _ControllerBase = opensim.Controller  # type: ignore[misc]
else:
    _ControllerBase = object


class PolynomialTorqueController(_ControllerBase):  # type: ignore[misc,valid-type]
    """OpenSim Controller applying polynomial torque to coordinate actuators.

    This controller evaluates polynomial torque profiles tau_j(t) for each
    joint and writes them into the model's controls vector at each integration step.
    The controller is picklable (for multiprocess optimization) by storing
    coefficients as a numpy array.

    Attributes:
        _theta: (n_joints * 7,) numpy array of polynomial coefficients
        _n_joints: Number of joints
    """

    def __init__(
        self, theta: NDArray[np.float64] | None = None, n_joints: int | None = None
    ) -> None:
        """Initialize the polynomial torque controller.

        Args:
            theta: (n_joints * 7,) polynomial coefficients. If None, defaults
                   to zero coefficients for n_joints.
            n_joints: Number of joints. Required if theta is None.

        Raises:
            ValueError: If both theta and n_joints are None, or if
                        theta length is not divisible by 7.
        """
        super().__init__()

        if theta is not None:
            self._theta = np.asarray(theta, dtype=np.float64).copy()
            if self._theta.ndim != 1:
                raise ValueError(f"theta must be 1-D, got shape {self._theta.shape}")
            if len(self._theta) % 7 != 0:
                raise ValueError(
                    f"theta length must be divisible by 7, got {len(self._theta)}"
                )
            self._n_joints = len(self._theta) // 7
        elif n_joints is not None:
            self._theta = np.zeros(n_joints * 7, dtype=np.float64)
            self._n_joints = n_joints
        else:
            raise ValueError("Either theta or n_joints must be provided")

    def set_theta(self, theta: NDArray[np.float64]) -> None:
        """Update polynomial coefficients.

        Args:
            theta: (n_joints * 7,) coefficient vector

        Raises:
            ValueError: If theta is invalid or wrong length
        """
        theta_arr = np.asarray(theta, dtype=np.float64)
        if theta_arr.ndim != 1:
            raise ValueError(f"theta must be 1-D, got shape {theta_arr.shape}")
        if len(theta_arr) != len(self._theta):
            raise ValueError(
                f"theta length mismatch: expected {len(self._theta)}, "
                f"got {len(theta_arr)}"
            )
        if not np.all(np.isfinite(theta_arr)):
            raise ValueError("theta must contain only finite values")
        self._theta = theta_arr.copy()

    def get_theta(self) -> NDArray[np.float64]:
        """Return current polynomial coefficients.

        Returns:
            (n_joints * 7,) numpy array
        """
        return self._theta.copy()

    def tau_at(self, t: float, joint_idx: int) -> float:
        """Evaluate torque for a single joint at time t.

        Utility for tests and diagnostics (does not require state).

        Args:
            t: Time (seconds)
            joint_idx: Joint index (0-indexed)

        Returns:
            tau_j(t) scalar

        Raises:
            ValueError: If joint_idx is out of bounds
        """
        if not (0 <= joint_idx < self._n_joints):
            raise ValueError(
                f"joint_idx {joint_idx} out of bounds [0, {self._n_joints})"
            )
        coeffs = self._theta[joint_idx * 7 : (joint_idx + 1) * 7]
        return evaluate_torque_polynomial(t, coeffs)

    def computeControls(
        self,
        s: Any,
        controls: Any,  # type: ignore[no-untyped-def]
    ) -> None:
        """Compute and set controls for each coordinate actuator.

        Called by OpenSim's integrator at each step. Evaluates polynomial
        torques and writes them into the controls vector.

        Args:
            s: SimTK State (OpenSim wrapper around C++ state)
            controls: OpenSim Vector (writable reference to model's controls)

        Side effects:
            Modifies `controls` in place with evaluated torques.
        """
        # Get current simulation time from state
        t = s.getTime()

        # Evaluate and write torques for each joint
        for j in range(self._n_joints):
            tau_j = self.tau_at(t, j)
            controls.set(j, tau_j)

    def __getstate__(self) -> dict[str, Any]:
        """Support pickling by returning theta as numpy array.

        Returns:
            Dictionary with pickled state
        """
        return {
            "theta": self._theta.copy(),
            "n_joints": self._n_joints,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickling state.

        Args:
            state: Dictionary from __getstate__
        """
        self._theta = state["theta"].copy()
        self._n_joints = state["n_joints"]
