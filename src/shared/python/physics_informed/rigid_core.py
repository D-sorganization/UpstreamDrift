"""Rigid-body core using Pinocchio for inverse-dynamics torque computation.

This module provides :class:`RigidCore`, which wraps the Pinocchio library to
compute the standard rigid-body (RNEA) inverse-dynamics torque vector for a
given kinematic state ``(q, dq, ddq)``.

Pinocchio is an *optional* dependency (``pip install upstream-drift[pinocchio]``).
If it is not installed the module still imports cleanly, but instantiating
:class:`RigidCore` will raise :class:`ImportError`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pinocchio as pin

    _PINOCCHIO_AVAILABLE = True
except ImportError:
    _PINOCCHIO_AVAILABLE = False
    pin = None  # type: ignore[assignment]


class RigidCore:
    """Wrap Pinocchio to compute the standard rigid-body torque vector.

    The torque vector is obtained by running the Recursive Newton-Euler
    Algorithm (RNEA / inverse dynamics):

    .. math::

        \\tau = \\text{RNEA}(q, \\dot{q}, \\ddot{q})

    Parameters
    ----------
    urdf_path:
        Absolute or relative path to a URDF file.  The path is validated
        before any Pinocchio call is made.

    Raises
    ------
    ImportError
        If Pinocchio is not installed.
    ValueError
        If ``urdf_path`` does not point to an existing file.
    """

    def __init__(
        self,
        urdf_path: str | None = None,
        *,
        model: Any = None,
        data: Any = None,
    ) -> None:
        """Load URDF into a Pinocchio model or wrap an existing model.

        DbC preconditions:
        - Either ``urdf_path`` or ``model`` must be provided.
        - If ``urdf_path`` is given, it must be a non-empty string pointing to an existing file.
        - If ``model`` is given, it must not be None.

        Args:
            urdf_path: Path to a URDF file.
            model:     Optional existing Pinocchio model (or duck-typed equivalent).
            data:      Optional Pinocchio data container for the model.

        Raises:
            ImportError: If Pinocchio is not installed when loading from URDF.
            ValueError: If neither argument is provided or urdf_path is invalid.
        """
        if model is not None:
            self._model = model
            if data is not None:
                self._data = data
            elif hasattr(model, "createData"):
                self._data = model.createData()
            else:
                self._data = None
            logger.info(
                "RigidCore loaded from in-memory model: %s (nq=%d, nv=%d)",
                getattr(model, "name", "custom"),
                getattr(model, "nq", 0),
                getattr(model, "nv", 0),
            )
            return

        if not _PINOCCHIO_AVAILABLE:
            raise ImportError(
                "pinocchio not available; install with: "
                "pip install upstream-drift[pinocchio]"
            )

        if not urdf_path:
            raise ValueError("urdf_path must be a non-empty string; got empty string")

        if not os.path.isfile(urdf_path):
            raise ValueError(
                f"urdf_path does not point to an existing file: {urdf_path!r}"
            )

        logger.debug("Loading Pinocchio model from %s", urdf_path)
        self._model = pin.buildModelFromUrdf(urdf_path)
        self._data = self._model.createData()
        logger.info(
            "RigidCore loaded: %s  (nq=%d, nv=%d)",
            self._model.name,
            self._model.nq,
            self._model.nv,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def nq(self) -> int:
        """Configuration-space dimension of the loaded model."""
        return int(self._model.nq)

    @property
    def nv(self) -> int:
        """Velocity/torque-space dimension of the loaded model."""
        return int(self._model.nv)

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute_torques(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
    ) -> np.ndarray:
        """Return the inverse-dynamics torque vector via Pinocchio RNEA.

        DbC preconditions:
        - ``q`` must have shape ``(nq,)``.
        - ``dq`` and ``ddq`` must have shape ``(nv,)``.

        DbC postcondition:
        - Returns a finite 1-D array of shape ``(nv,)``.

        Args:
            q:   Configuration vector, shape ``(nq,)``.
            dq:  Velocity vector, shape ``(nv,)``.
            ddq: Acceleration vector, shape ``(nv,)``.

        Returns:
            Torque vector ``τ`` of shape ``(nv,)``.

        Raises:
            ValueError: If any input has the wrong shape.
        """
        q_arr = np.asarray(q, dtype=np.float64)
        dq_arr = np.asarray(dq, dtype=np.float64)
        ddq_arr = np.asarray(ddq, dtype=np.float64)

        self._validate_shapes(q_arr, dq_arr, ddq_arr)

        current_pin = pin
        if current_pin is None:
            import sys

            current_pin = sys.modules.get("pinocchio")

        logger.debug(
            "compute_torques: current_pin=%s, hasattr model rnea=%s",
            current_pin,
            hasattr(self._model, "rnea"),
        )
        if hasattr(self._model, "rnea"):
            tau = self._model.rnea(self._model, self._data, q_arr, dq_arr, ddq_arr)
        elif hasattr(self._model, "compute_torques"):
            tau = self._model.compute_torques(q_arr, dq_arr, ddq_arr)
        elif current_pin is not None and hasattr(current_pin, "rnea"):
            tau = current_pin.rnea(self._model, self._data, q_arr, dq_arr, ddq_arr)
        else:
            raise ImportError(
                "pinocchio not available; install with: "
                "pip install upstream-drift[pinocchio]"
            )
        result: np.ndarray = np.atleast_1d(
            np.asarray(tau, dtype=np.float64).reshape(-1)
        )

        assert result.shape == (self.nv,), (
            f"Postcondition violated: expected shape ({self.nv},), got {result.shape}"
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_shapes(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
    ) -> None:
        """Raise ValueError if input arrays have unexpected shapes."""
        if q.shape != (self.nq,):
            raise ValueError(f"q must have shape ({self.nq},); got {q.shape}")
        if dq.shape != (self.nv,):
            raise ValueError(f"dq must have shape ({self.nv},); got {dq.shape}")
        if ddq.shape != (self.nv,):
            raise ValueError(f"ddq must have shape ({self.nv},); got {ddq.shape}")
