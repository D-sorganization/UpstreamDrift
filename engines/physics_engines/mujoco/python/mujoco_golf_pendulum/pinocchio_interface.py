"""Pinocchio interface for advanced dynamics algorithms.

This module provides a bridge between MuJoCo and Pinocchio, enabling:
- Fast inverse dynamics (RNEA)
- Forward dynamics (ABA)
- Analytical Jacobians and derivatives
- Mass matrix computation
- Trajectory optimization support

Pinocchio is used for analytical dynamics computations while MuJoCo handles
simulation, contacts, and constraints.

Usage:
    >>> from mujoco_golf_pendulum.pinocchio_interface import PinocchioWrapper
    >>> wrapper = PinocchioWrapper(model, data)
    >>> torques = wrapper.compute_inverse_dynamics(q, v, a)
    >>> jacobian = wrapper.compute_end_effector_jacobian(q, "club_head")
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from .urdf_io import export_model_to_urdf

logger = logging.getLogger(__name__)

# Constants
FREEJOINT_DOF_COUNT: int = 7  # DOF count for freejoint (3 pos + 4 quat)

# Try to import Pinocchio
try:
    import pinocchio as pin

    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False
    logger.warning(
        "Pinocchio not available. Install with: pip install pin",
    )


class PinocchioWrapper:
    """Wrapper for Pinocchio dynamics computations with MuJoCo models.

    This class maintains both MuJoCo and Pinocchio representations of the same
    model, allowing you to use Pinocchio's fast analytical algorithms while
    leveraging MuJoCo for simulation.

    Attributes:
        model: MuJoCo model
        data: MuJoCo data
        pin_model: Pinocchio model (built from MuJoCo model)
        pin_data: Pinocchio data
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        use_mjcf_parser: bool = True,
    ) -> None:
        """Initialize Pinocchio wrapper.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            use_mjcf_parser: If True, use Pinocchio's MJCF parser (recommended).
                If False, convert via URDF (slower but more compatible).

        Raises:
            ImportError: If Pinocchio is not installed
        """
        if not PINOCCHIO_AVAILABLE:
            msg = (
                "Pinocchio is required but not installed. "
                "Install with: pip install pin"
            )
            raise ImportError(msg)

        self.model = model
        self.data = data

        # Build Pinocchio model from MuJoCo
        if use_mjcf_parser:
            self.pin_model, self.pin_data = self._build_from_mjcf()
        else:
            self.pin_model, self.pin_data = self._build_from_urdf()

        logger.info(
            "Initialized Pinocchio wrapper: %d DOF, %d velocity DOF",
            self.pin_model.nq,
            self.pin_model.nv,
        )

    def _build_from_mjcf(self) -> tuple:
        """Build Pinocchio model directly from MuJoCo MJCF.

        This uses Pinocchio's built-in MJCF parser, which is the fastest
        and most accurate method.

        Returns:
            Tuple of (pin_model, pin_data)
        """
        # Export MuJoCo model to XML string

        # For now, we'll use URDF conversion as intermediate step
        # In the future, can use Pinocchio's direct MJCF support
        return self._build_from_urdf()

    def _build_from_urdf(self) -> tuple:
        """Build Pinocchio model via URDF conversion.

        This converts MuJoCo model to URDF, then loads in Pinocchio.

        Returns:
            Tuple of (pin_model, pin_data)
        """
        # Export to temporary URDF file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            urdf_path = Path(f.name)

        try:
            # Export MuJoCo model to URDF
            export_model_to_urdf(self.model, urdf_path)

            # Load in Pinocchio
            pin_model = pin.buildModelFromUrdf(str(urdf_path))
            pin_data = pin_model.createData()

            return pin_model, pin_data

        finally:
            # Clean up temporary file
            if urdf_path.exists():
                urdf_path.unlink()

    def sync_mujoco_to_pinocchio(self) -> None:
        """Synchronize state from MuJoCo to Pinocchio.

        This updates Pinocchio's configuration and velocity to match MuJoCo.
        """
        # Convert MuJoCo quaternions to Pinocchio format if needed
        q = self._mujoco_q_to_pinocchio_q(self.data.qpos)
        v = self.data.qvel.copy()

        # Update Pinocchio forward kinematics
        pin.forwardKinematics(self.pin_model, self.pin_data, q, v)
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

    def sync_pinocchio_to_mujoco(self) -> None:
        """Synchronize state from Pinocchio to MuJoCo.

        This updates MuJoCo's configuration to match Pinocchio.
        Note: Pinocchio doesn't store configuration state, so this method
        maintains MuJoCo's current state. For proper synchronization,
        use sync_mujoco_to_pinocchio after updating MuJoCo state.
        """
        # Pinocchio's data object doesn't store q - it's passed as parameter
        # Since we're syncing TO MuJoCo, we keep MuJoCo's current state
        # The configuration should already be synchronized via sync_mujoco_to_pinocchio
        # This method exists for API completeness but is a no-op

    def _mujoco_q_to_pinocchio_q(self, q_mj: np.ndarray) -> np.ndarray:
        """Convert MuJoCo configuration to Pinocchio format.

        Handles quaternion conventions (MuJoCo: w,x,y,z vs Pinocchio: x,y,z,w).
        """
        q_pin = q_mj.copy()

        # MuJoCo uses freejoint for 7-DOF (3 pos + 4 quat)
        # Pinocchio uses SE3 for 7-DOF with quaternion [x,y,z,w]
        # Check if model has freejoints (quaternions) and convert if needed
        if self.model.nq >= FREEJOINT_DOF_COUNT:
            # Check if we have a freejoint (first joint might be freejoint)
            # For freejoint: first 3 are position, next 4 are quaternion [w,x,y,z]
            # Convert to Pinocchio format [x,y,z,w]
            for i in range(self.model.njnt):
                if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    # Find the qpos address for this freejoint
                    qpos_addr = self.model.jnt_qposadr[i]
                    if qpos_addr + 6 < len(q_pin):
                        # Convert quat from MuJoCo [w,x,y,z] to Pinocchio [x,y,z,w]
                        w, x, y, z = (
                            q_pin[qpos_addr + 3],
                            q_pin[qpos_addr + 4],
                            q_pin[qpos_addr + 5],
                            q_pin[qpos_addr + 6],
                        )
                        q_pin[qpos_addr + 3 : qpos_addr + 7] = [x, y, z, w]
                        # Continue to process all freejoints (don't break)

        return q_pin

    def _pinocchio_q_to_mujoco_q(self, q_pin: np.ndarray) -> np.ndarray:
        """Convert Pinocchio configuration to MuJoCo format.

        Handles quaternion conventions (Pinocchio: x,y,z,w vs MuJoCo: w,x,y,z).
        """
        q_mj = q_pin.copy()

        # Convert back from Pinocchio [x, y, z, w] to MuJoCo [w, x, y, z]
        # Check if model has freejoints (quaternions) and convert if needed
        if self.model.nq >= FREEJOINT_DOF_COUNT:
            # Iterate through all joints to find freejoints (same as forward conversion)
            for i in range(self.model.njnt):
                if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    # Find the qpos address for this freejoint
                    qpos_addr = self.model.jnt_qposadr[i]
                    if qpos_addr + 6 < len(q_mj):
                        # Convert quat from Pinocchio [x,y,z,w] to MuJoCo [w,x,y,z]
                        x, y, z, w = (
                            q_mj[qpos_addr + 3],
                            q_mj[qpos_addr + 4],
                            q_mj[qpos_addr + 5],
                            q_mj[qpos_addr + 6],
                        )
                        q_mj[qpos_addr + 3 : qpos_addr + 7] = [w, x, y, z]
                        # Continue to process all freejoints (don't break)

        return q_mj

    def compute_inverse_dynamics(
        self,
        q: np.ndarray | None = None,
        v: np.ndarray | None = None,
        a: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute inverse dynamics (RNEA) using Pinocchio.

        Computes joint torques required to achieve desired accelerations.

        Args:
            q: Joint positions (if None, uses current MuJoCo state)
            v: Joint velocities (if None, uses current MuJoCo state)
            a: Joint accelerations (required)

        Returns:
            Joint torques [nq]

        Example:
            >>> wrapper = PinocchioWrapper(model, data)
            >>> # Option 1: Use None to auto-convert from MuJoCo state
            >>> a = np.zeros(model.nv)
            >>> torques = wrapper.compute_inverse_dynamics(None, None, a)
            >>> # Option 2: Manually convert quaternions if passing q explicitly
            >>> q = wrapper._mujoco_q_to_pinocchio_q(data.qpos.copy())
            >>> v = data.qvel.copy()
            >>> torques = wrapper.compute_inverse_dynamics(q, v, a)
        """
        if a is None:
            msg = "Accelerations 'a' are required for inverse dynamics"
            raise ValueError(msg)

        # Use current state if not provided
        if q is None:
            q = self._mujoco_q_to_pinocchio_q(self.data.qpos)
        if v is None:
            v = self.data.qvel.copy()

        # Compute inverse dynamics (RNEA)
        return pin.rnea(self.pin_model, self.pin_data, q, v, a)

    def compute_forward_dynamics(
        self,
        q: np.ndarray | None = None,
        v: np.ndarray | None = None,
        tau: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute forward dynamics (ABA) using Pinocchio.

        Computes joint accelerations from applied torques.

        Args:
            q: Joint positions (if None, uses current MuJoCo state)
            v: Joint velocities (if None, uses current MuJoCo state)
            tau: Joint torques (if None, uses zero torques)

        Returns:
            Joint accelerations [nv]

        Example:
            >>> wrapper = PinocchioWrapper(model, data)
            >>> # Option 1: Use None to auto-convert from MuJoCo state
            >>> tau = np.zeros(model.nu)
            >>> a = wrapper.compute_forward_dynamics(None, None, tau)
            >>> # Option 2: Manually convert quaternions if passing q explicitly
            >>> q = wrapper._mujoco_q_to_pinocchio_q(data.qpos.copy())
            >>> v = data.qvel.copy()
            >>> a = wrapper.compute_forward_dynamics(q, v, tau)
        """
        # Use current state if not provided
        if q is None:
            q = self._mujoco_q_to_pinocchio_q(self.data.qpos)
        if v is None:
            v = self.data.qvel.copy()
        if tau is None:
            tau = np.zeros(self.pin_model.nv)

        # Compute forward dynamics (ABA)
        return pin.aba(self.pin_model, self.pin_data, q, v, tau)

    def compute_mass_matrix(
        self,
        q: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute mass matrix using Pinocchio.

        Args:
            q: Joint positions (if None, uses current MuJoCo state)

        Returns:
            Mass matrix [nv x nv]

        Example:
            >>> wrapper = PinocchioWrapper(model, data)
            >>> M = wrapper.compute_mass_matrix()
        """
        # Use current state if not provided
        if q is None:
            q = self._mujoco_q_to_pinocchio_q(self.data.qpos)

        # Compute mass matrix (CRBA)
        return pin.crba(self.pin_model, self.pin_data, q)

    def compute_coriolis_matrix(
        self,
        q: np.ndarray | None = None,
        v: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute Coriolis matrix using Pinocchio.

        Args:
            q: Joint positions (if None, uses current MuJoCo state)
            v: Joint velocities (if None, uses current MuJoCo state)

        Returns:
            Coriolis matrix [nv x nv]

        Example:
            >>> wrapper = PinocchioWrapper(model, data)
            >>> C = wrapper.compute_coriolis_matrix()
        """
        # Use current state if not provided
        if q is None:
            q = self._mujoco_q_to_pinocchio_q(self.data.qpos)
        if v is None:
            v = self.data.qvel.copy()

        # Compute Coriolis matrix
        return pin.computeCoriolisMatrix(self.pin_model, self.pin_data, q, v)

    def compute_gravity_vector(
        self,
        q: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute gravity vector using Pinocchio.

        Args:
            q: Joint positions (if None, uses current MuJoCo state)

        Returns:
            Gravity vector [nv]

        Example:
            >>> wrapper = PinocchioWrapper(model, data)
            >>> g = wrapper.compute_gravity_vector()
        """
        # Use current state if not provided
        if q is None:
            q = self._mujoco_q_to_pinocchio_q(self.data.qpos)

        # Compute gravity vector
        return pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)

    def compute_end_effector_jacobian(
        self,
        frame_name: str,
        q: np.ndarray | None = None,
        *,
        local: bool = True,
    ) -> np.ndarray:
        """Compute end-effector Jacobian using Pinocchio.

        Args:
            frame_name: Name of the end-effector frame (e.g., "club_head")
            q: Joint positions (if None, uses current MuJoCo state)
            local: If True, returns local Jacobian. If False, returns world Jacobian.

        Returns:
            Jacobian matrix [6 x nv] (linear and angular)

        Example:
            >>> wrapper = PinocchioWrapper(model, data)
            >>> J = wrapper.compute_end_effector_jacobian("club_head")
        """
        # Use current state if not provided
        if q is None:
            q = self._mujoco_q_to_pinocchio_q(self.data.qpos)

        # Find frame ID
        frame_id = self.pin_model.getFrameId(frame_name)
        if frame_id == len(self.pin_model.frames):
            msg = f"Frame '{frame_name}' not found in Pinocchio model"
            raise ValueError(msg)

        # Update forward kinematics
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        # Compute frame Jacobian
        if local:
            jacobian = pin.getFrameJacobian(
                self.pin_model,
                self.pin_data,
                frame_id,
                pin.LOCAL,
            )
        else:
            jacobian = pin.getFrameJacobian(
                self.pin_model,
                self.pin_data,
                frame_id,
                pin.WORLD,
            )

        return jacobian

    def compute_dynamics_derivatives(
        self,
        q: np.ndarray | None = None,
        v: np.ndarray | None = None,
        tau: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute analytical derivatives of dynamics.

        Computes ∂f/∂q, ∂f/∂v, ∂f/∂τ, ∂f/∂u (for control).

        Args:
            q: Joint positions (if None, uses current MuJoCo state)
            v: Joint velocities (if None, uses current MuJoCo state)
            tau: Joint torques (if None, uses zero torques)

        Returns:
            Tuple of (df_dq, df_dv, df_dtau, df_du) derivatives

        Example:
            >>> wrapper = PinocchioWrapper(model, data)
            >>> df_dq, df_dv, df_dtau, df_du = wrapper.compute_dynamics_derivatives()
        """
        # Use current state if not provided
        if q is None:
            q = self._mujoco_q_to_pinocchio_q(self.data.qpos)
        if v is None:
            v = self.data.qvel.copy()
        if tau is None:
            tau = np.zeros(self.pin_model.nv)

        # Compute derivatives using Pinocchio
        pin.computeABADerivatives(
            self.pin_model,
            self.pin_data,
            q,
            v,
            tau,
        )

        df_dq = self.pin_data.ddq_dq.copy()
        df_dv = self.pin_data.ddq_dv.copy()
        df_dtau = self.pin_data.Minv.copy()  # ∂a/∂τ = M^-1

        # For control derivatives, assume identity mapping (tau = u)
        df_du = df_dtau.copy()

        return df_dq, df_dv, df_dtau, df_du

    def compute_kinetic_energy(
        self,
        q: np.ndarray | None = None,
        v: np.ndarray | None = None,
    ) -> float:
        """Compute kinetic energy using Pinocchio.

        Args:
            q: Joint positions (if None, uses current MuJoCo state)
            v: Joint velocities (if None, uses current MuJoCo state)

        Returns:
            Kinetic energy [J]

        Example:
            >>> wrapper = PinocchioWrapper(model, data)
            >>> KE = wrapper.compute_kinetic_energy()
        """
        # Use current state if not provided
        if q is None:
            q = self._mujoco_q_to_pinocchio_q(self.data.qpos)
        if v is None:
            v = self.data.qvel.copy()

        # Compute kinetic energy: KE = 0.5 * v^T * M * v
        m_matrix = self.compute_mass_matrix(q)
        ke = 0.5 * v.T @ m_matrix @ v

        return float(ke)

    def compute_potential_energy(
        self,
        q: np.ndarray | None = None,
    ) -> float:
        """Compute potential energy using Pinocchio.

        Args:
            q: Joint positions (if None, uses current MuJoCo state)

        Returns:
            Potential energy [J]

        Example:
            >>> wrapper = PinocchioWrapper(model, data)
            >>> PE = wrapper.compute_potential_energy()
        """
        # Use current state if not provided
        if q is None:
            q = self._mujoco_q_to_pinocchio_q(self.data.qpos)

        # Compute potential energy using Pinocchio's built-in function
        # Pinocchio provides computePotentialEnergy for accurate computation
        pe = pin.computePotentialEnergy(self.pin_model, self.pin_data, q)

        return float(pe)


def create_pinocchio_wrapper(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> PinocchioWrapper:
    """Convenience function to create Pinocchio wrapper.

    Args:
        model: MuJoCo model
        data: MuJoCo data

    Returns:
        PinocchioWrapper instance

    Example:
        >>> import mujoco
        >>> from mujoco_golf_pendulum.pinocchio_interface import (
        ...     create_pinocchio_wrapper,
        ... )
        >>> model = mujoco.MjModel.from_xml_string(xml_string)
        >>> data = mujoco.MjData(model)
        >>> wrapper = create_pinocchio_wrapper(model, data)
    """
    return PinocchioWrapper(model, data)
