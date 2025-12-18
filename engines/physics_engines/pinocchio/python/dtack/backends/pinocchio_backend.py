"""Pinocchio backend wrapper for dynamics computations."""

from __future__ import annotations

import logging
import typing
from pathlib import Path

import numpy as np  # noqa: TID253
import numpy.typing as npt  # noqa: TID253

try:
    import pinocchio as pin

    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False

    # Define dummy pin module to allow import without pinocchio
    # Define dummy pin module to allow import without pinocchio
    class DummyPin:
        """Dummy class to prevent NameError when Pinocchio is missing.

        Any attribute or method access raises ImportError with a clear message.
        """

        def __getattr__(self, name: str) -> typing.Any:  # noqa: ANN401
            """Raise ImportError on any attribute access."""
            msg = (
                f"Pinocchio is required for '{name}', but is not installed. "
                "Install with: pip install pin"
            )
            raise ImportError(msg)

        class ReferenceFrame:
            """Dummy ReferenceFrame enum."""

            LOCAL_WORLD_ALIGNED = 0

            def __getattr__(self, name: str) -> typing.Any:  # noqa: ANN401
                """Raise ImportError on any attribute access."""
                msg = (
                    f"Pinocchio ReferenceFrame is required for '{name}', "
                    "but is not installed. "
                    "Install with: pip install pin"
                )
                raise ImportError(msg)

        class SE3:
            """Dummy SE3 class."""

            def __getattr__(self, name: str) -> typing.Any:  # noqa: ANN401
                """Raise ImportError on any attribute access."""
                msg = (
                    f"Pinocchio SE3 is required for '{name}', but is not installed. "
                    "Install with: pip install pin"
                )
                raise ImportError(msg)

    pin = DummyPin()
logger = logging.getLogger(__name__)


class PinocchioBackend:
    """Pinocchio backend for forward/inverse dynamics and kinematics.

    This backend provides:
    - URDF model loading
    - Forward dynamics (ABA)
    - Inverse dynamics (RNEA)
    - Mass matrix computation (CRBA)
    - Jacobian computation
    - Frame placement updates
    """

    def __init__(self, model_path: Path | str) -> None:
        """Initialize Pinocchio backend.

        Args:
            model_path: Path to URDF file or canonical YAML specification

        Raises:
            ImportError: If Pinocchio is not installed
            FileNotFoundError: If model file does not exist
        """
        if not PINOCCHIO_AVAILABLE:
            msg = (
                "Pinocchio is required but not installed. Install with: pip install pin"
            )
            raise ImportError(msg)

        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        # For now, assume URDF. Later we'll add YAML parser
        if model_path_obj.suffix == ".urdf":
            self.model, self.collision_model, self.visual_model = (
                pin.buildModelsFromUrdf(str(model_path_obj), "")
            )
        else:
            msg = f"Unsupported model format: {model_path_obj.suffix}"
            raise ValueError(msg)

        self.data = self.model.createData()
        self.collision_data = self.collision_model.createData()
        self.visual_data = self.visual_model.createData()

        logger.info(
            "Loaded Pinocchio model: %d DOF, %d velocity DOF",
            self.model.nq,
            self.model.nv,
        )

    def compute_inverse_dynamics(
        self,
        q: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        a: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute inverse dynamics (RNEA).

        Args:
            q: Joint positions [nq]
            v: Joint velocities [nv]
            a: Joint accelerations [nv]

        Returns:
            Joint torques [nv]
        """
        q_arr = np.asarray(q, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        a_arr = np.asarray(a, dtype=np.float64)

        return pin.rnea(self.model, self.data, q_arr, v_arr, a_arr)

    def compute_forward_dynamics(
        self,
        q: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        tau: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute forward dynamics (ABA).

        Args:
            q: Joint positions [nq]
            v: Joint velocities [nv]
            tau: Joint torques [nv]

        Returns:
            Joint accelerations [nv]
        """
        q_arr = np.asarray(q, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        tau_arr = np.asarray(tau, dtype=np.float64)

        return pin.aba(self.model, self.data, q_arr, v_arr, tau_arr)

    def compute_mass_matrix(
        self, q: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute mass matrix (CRBA).

        Args:
            q: Joint positions [nq]

        Returns:
            Mass matrix [nv x nv]
        """
        q_arr = np.asarray(q, dtype=np.float64)
        return pin.crba(self.model, self.data, q_arr)

    def compute_bias_forces(
        self,
        q: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute bias forces (gravity + Coriolis + centrifugal).

        Args:
            q: Joint positions [nq]
            v: Joint velocities [nv]

        Returns:
            Bias forces [nv]
        """
        q_arr = np.asarray(q, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)

        # Optimization: Use pin.nle directly.
        # This avoids:
        # 1. Redundant computeGeneralizedGravity (rnea computes it internally)
        # 2. Allocating np.zeros(self.model.nv) for 'a' in rnea
        return pin.nle(self.model, self.data, q_arr, v_arr)

    def compute_frame_jacobian(
        self,
        q: npt.NDArray[np.float64],
        frame_id: int | str,
        reference_frame: int = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    ) -> npt.NDArray[np.float64]:
        """Compute frame Jacobian.

        Args:
            q: Joint positions [nq]
            frame_id: Frame ID (int) or frame name (str)
            reference_frame: Reference frame for Jacobian

        Returns:
            Jacobian matrix [6 x nv]
        """
        q_arr = np.asarray(q, dtype=np.float64)

        if isinstance(frame_id, str):
            frame_id = self.model.getFrameId(frame_id)

        pin.forwardKinematics(self.model, self.data, q_arr)
        pin.updateFramePlacements(self.model, self.data)
        return pin.computeFrameJacobian(
            self.model, self.data, q_arr, frame_id, reference_frame
        )

    def forward_kinematics(self, q: npt.NDArray[np.float64]) -> list[pin.SE3]:
        """Compute forward kinematics for all frames.

        Args:
            q: Joint positions [nq]

        Returns:
            List of frame placements
        """
        q_arr = np.asarray(q, dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, q_arr)
        pin.updateFramePlacements(self.model, self.data)
        return [self.data.oMf[i] for i in range(len(self.model.frames))]
