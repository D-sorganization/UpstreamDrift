from __future__ import annotations

import numpy as np

from src.shared.python.core.contracts import precondition


class KalmanFilter:
    """Simple linear Kalman filter implementation.

    Supports n-dimensional state and measurement vectors.
    """

    @precondition(
        lambda self, dim_x, dim_z, F=None, H=None, Q=None, R=None, P=None, x=None: (
            dim_x > 0
        ),
        "State dimension must be positive",
    )
    @precondition(
        lambda self, dim_x, dim_z, F=None, H=None, Q=None, R=None, P=None, x=None: (
            dim_z > 0
        ),
        "Measurement dimension must be positive",
    )
    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        F: np.ndarray | None = None,
        H: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        P: np.ndarray | None = None,
        x: np.ndarray | None = None,
    ) -> None:
        """Initialize Kalman Filter.

        Args:
            dim_x: State dimension
            dim_z: Measurement dimension
            F: State transition matrix (dim_x, dim_x)
            H: Measurement function (dim_z, dim_x)
            Q: Process noise covariance (dim_x, dim_x)
            R: Measurement noise covariance (dim_z, dim_z)
            P: Error covariance matrix (dim_x, dim_x)
            x: Initial state (dim_x,)
        """
        if not (dim_x is not None):
            raise ValueError("dim_x must be provided")
        if not (dim_x is not None):
            raise ValueError("dim_x must be provided")
        self.dim_x = dim_x
        self.dim_z = dim_z

        # State transition matrix
        self.F = F if F is not None else np.eye(dim_x)

        # Measurement function
        self.H = H if H is not None else np.zeros((dim_z, dim_x))

        # Process noise covariance
        self.Q = Q if Q is not None else np.eye(dim_x)

        # Measurement noise covariance
        self.R = R if R is not None else np.eye(dim_z)

        # Error covariance
        self.P = P if P is not None else np.eye(dim_x)

        # State
        self.x = x if x is not None else np.zeros(dim_x)

    def predict(self) -> None:
        """Predict next state (prior)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        """Update state with measurement (posterior).

        Args:
            z: Measurement vector
        """
        # System uncertainty
        if not (z is not None):
            raise ValueError("z must be provided")
        if not (z is not None):
            raise ValueError("z must be provided")
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        # Use solve instead of inv for stability
        try:
            K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(self.dim_z))
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if S is singular
            K = self.P @ self.H.T @ np.linalg.pinv(S)

        # Residual
        y = z - self.H @ self.x

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I_mat = np.eye(self.dim_x)
        self.P = (I_mat - K @ self.H) @ self.P
