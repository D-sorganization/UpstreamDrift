# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""OpenSim Physics Engine implementation.

Refactored to use shared engine availability module (DRY principle).
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from src.engines.tiers import warn_if_experimental
from src.shared.python.core.contracts import (
    check_finite,
    postcondition,
    precondition,
)
from src.shared.python.engine_core.engine_availability import OPENSIM_AVAILABLE
from src.shared.python.engine_core.base_physics_engine import BasePhysicsEngine
from src.shared.python.logging_pkg.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

# Import OpenSim if available
if OPENSIM_AVAILABLE:
    import opensim
else:
    opensim = None  # type: ignore[assignment]
    logger.warning(
        "OpenSim python package not found. OpenSimPhysicsEngine will not function fully."
    )


class OpenSimPhysicsEngine(BasePhysicsEngine):
    """OpenSim Physics Engine Implementation.

    Inherits checkpoint save/restore (Checkpointable contract), path validation,
    and model name tracking from BasePhysicsEngine.
    """

    def __init__(self) -> None:
        super().__init__()
        warn_if_experimental("opensim", "OpenSim")
        self._model = None
        self._state = None
        self._manager = None
        self._model_path = ""
        self._time_step = 0.01

        if opensim is None:
            logger.error("OpenSim library is not installed.")

    @property
    def engine_type(self) -> str:
        """Get engine type identifier (Checkpointable contract)."""
        return "opensim"

    @property
    def model_name(self) -> str:
        """Return the OpenSim model name."""
        if self._model:
            return self._model.getName()
        return "OpenSim_NoModel"

    @property
    def is_initialized(self) -> bool:
        """Check if the engine has a loaded model."""
        return self._model is not None and self._state is not None

    def _load_from_path_impl(self, path: str) -> None:
        """Engine-specific path loading (called by BasePhysicsEngine)."""
        if opensim is None:
            raise ImportError("OpenSim library not installed")

        self._model = opensim.Model(path)
        self._model_path = path
        if self._model is None:
            raise ValueError("Failed to create OpenSim Model object")
        # Initialize the system and state
        self._state = self._model.initSystem()
        self._manager = opensim.Manager(self._model)
        logger.info("Loaded OpenSim model from %s", path)

    def _load_from_string_impl(self, content: str, extension: str | None) -> None:
        """Engine-specific string loading (called by BasePhysicsEngine)."""
        import tempfile

        suffix = f".{extension}" if extension else ".osim"
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            self._load_from_path_impl(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except (RuntimeError, ValueError, OSError) as cleanup_error:
                    logger.warning(
                        "Failed to remove temporary file %s: %s",
                        tmp_path,
                        cleanup_error,
                    )

    def load_from_path(self, path: str) -> None:
        """Load an OpenSim model from a file path."""
        if self.is_initialized:
            raise RuntimeError(
                "Engine already has a loaded model. Re-loading is not supported."
            )

        if opensim is None:
            raise ImportError("OpenSim library not installed")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            self._load_from_path_impl(path)
            self._is_initialized = True
        except ImportError as e:
            logger.error("Failed to load OpenSim model: %s", e)
            raise

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Load model from XML string using a temporary file."""
        if opensim is None:
            raise ImportError("OpenSim library not installed")

        try:
            self._load_from_string_impl(content, extension)
            self._is_initialized = True
        except (PermissionError, OSError) as e:
            logger.error("Failed to load OpenSim model from string: %s", e)
            raise

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    def reset(self) -> None:
        """Reinitialize the model state and equilibrate muscles."""
        if self._model and self._state:
            # Re-initialize the system to defaults
            self._state = self._model.initializeState()
            self._model.equilibrateMuscles(self._state)
            self._manager.setSessionTime(0.0)
            self._manager.setIntegrator(opensim.RungeKuttaMersonIntegrator(self._model))

    @precondition(
        lambda self, dt=None: self.is_initialized, "Engine must be initialized"
    )
    def step(self, dt: float | None = None) -> None:
        """Integrate the simulation forward by one time step."""
        if not self._model or not self._state:
            return

        step_size = dt if dt is not None else self._time_step
        current_time = self._state.getTime()

        # Integrate to new time
        self._manager.setInitialTime(current_time)
        self._manager.setFinalTime(current_time + step_size)

        # Integrate
        self._manager.integrate(current_time + step_size)

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    def forward(self) -> None:
        """Realize the model to the dynamics stage."""
        if self._model and self._state:
            self._model.realizeDynamics(self._state)

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return coordinate positions and speeds as numpy arrays."""
        if not self._model or not self._state:
            return np.array([]), np.array([])

        # Q (Coordinates)
        n_q = self._model.getNumCoordinates()
        q_vec = self._state.getQ()
        q = np.array([q_vec.get(i) for i in range(n_q)])

        # U (Speeds)
        n_u = self._model.getNumSpeeds()
        u_vec = self._state.getU()
        v = np.array([u_vec.get(i) for i in range(n_u)])

        return q, v

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set coordinate positions and speeds on the model state."""
        if q is None:
            raise ValueError("q must be provided")
        if not self._model or not self._state:
            return

        # Set Q
        n_q = self._model.getNumCoordinates()
        if len(q) == n_q:
            q_vec = opensim.Vector(n_q)
            for i in range(n_q):
                q_vec.set(i, float(q[i]))
            self._state.setQ(q_vec)

        # Set U
        n_u = self._model.getNumSpeeds()
        if len(v) == n_u:
            u_vec = opensim.Vector(n_u)
            for i in range(n_u):
                u_vec.set(i, float(v[i]))
            self._state.setU(u_vec)

        self._model.realizeVelocity(self._state)

    def set_control(self, u: np.ndarray) -> None:
        """Set controls for the model."""
        if u is None:
            raise ValueError("u must be provided")
        if not self._model or not self._state:
            return

        try:
            # Get writable reference to controls
            controls = self._model.updControls(self._state)
            if len(u) == controls.size():
                for i in range(len(u)):
                    controls.set(i, float(u[i]))
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Failed to set OpenSim controls: {e}")

    def get_time(self) -> float:
        """Return the current simulation time in seconds."""
        if self._state:
            return self._state.getTime()
        return 0.0

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Mass matrix must contain finite values")
    def compute_mass_matrix(self) -> np.ndarray:
        """Compute the joint-space mass matrix via MatterSubsystem."""
        if not self._model or not self._state:
            return np.array([])

        matter = self._model.getMatterSubsystem()
        n_u = self._model.getNumSpeeds()
        m_mat = opensim.Matrix()
        # Ensure state is realizable to Position
        self._model.realizePosition(self._state)
        matter.calcM(self._state, m_mat)

        # Convert opensim Matrix to numpy
        res = np.zeros((n_u, n_u))
        for r in range(n_u):
            for c in range(n_u):
                res[r, c] = m_mat.get(r, c)
        return res

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Bias forces must contain finite values")
    def compute_bias_forces(self) -> np.ndarray:
        """Compute C(q,u) + G(q).

        Uses inverse dynamics with zero acceleration to get bias forces.
        """
        if not self._model or not self._state:
            return np.array([])

        try:
            # Bias forces = ID(q, v, 0) = M*0 + C + g = C + g
            n_u = self._model.getNumSpeeds()
            zero_acc = np.zeros(n_u)
            bias = self.compute_inverse_dynamics(zero_acc)
            return bias
        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(f"Failed to compute bias forces: {e}")
            return np.array([])

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Gravity forces must contain finite values")
    def compute_gravity_forces(self) -> np.ndarray:
        """Compute gravity forces g(q).

        Sets velocities to zero temporarily, then computes bias (which becomes pure gravity).
        """
        if not self._model or not self._state:
            return np.array([])

        try:
            # Save current velocities
            _, v_saved = self.get_state()

            # Set velocities to zero
            n_u = self._model.getNumSpeeds()
            zero_vel = np.zeros(n_u)
            q_current, _ = self.get_state()
            self.set_state(q_current, zero_vel)

            # With v=0, bias forces become pure gravity
            gravity = self.compute_bias_forces()

            # Restore velocities
            self.set_state(q_current, v_saved)

            return gravity
        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(f"Failed to compute gravity forces: {e}")
            return np.array([])

    @precondition(lambda self, qacc: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Inverse dynamics torques must contain finite values")
    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Compute required torques for the given joint accelerations."""
        if qacc is None:
            raise ValueError("qacc must be provided")
        if not self._model or not self._state:
            return np.array([])

        # Use an InverseDynamicsSolver
        n_u = self._model.getNumSpeeds()

        if len(qacc) != n_u:
            return np.array([])

        udot = opensim.Vector(n_u)
        for i in range(n_u):
            udot.set(i, float(qacc[i]))

        # We need realized Acceleration
        # But we can't just 'set' acceleration in state for ID. Use Solver.
        # ID Solver takes (model, state, udot) -> tau

        try:
            # Ensure state realized to Velocity
            self._model.realizeVelocity(self._state)

            solver = opensim.InverseDynamicsSolver(self._model)
            # Some versions use solve(state, udot, applied_loads, tau_out)
            # applied_loads can be empty
            # tau = solver.solve(self._state, udot) # if wrapper is friendly

            # Standard C++ wrapping often returns Vector
            tau = solver.solve(self._state, udot)

            res = np.zeros(n_u)
            for i in range(n_u):
                res[i] = tau.get(i)

            return res
        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(f"OpenSim ID failed: {e}")
            return np.array([])

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Compute spatial Jacobian for a body in OpenSim.

        Uses SimTK SimbodyMatterSubsystem to compute body Jacobian matrices.

        Args:
            body_name: Name of the body in the model.

        Returns:
            Dictionary with:
            - 'linear': Position Jacobian (3 × nv) [m/rad or m/m]
            - 'angular': Rotation Jacobian (3 × nv) [rad/rad or rad/m]
            - 'spatial': Combined [angular; linear] (6 × nv)
        """
        if body_name is None:
            raise ValueError("body_name must be provided")
        if not self._model or not self._state or opensim is None:
            return None

        try:
            # Get the body
            body_set = self._model.getBodySet()
            body = body_set.get(body_name)

            # Realize to position stage
            self._model.realizePosition(self._state)

            # Number of generalized coordinates (nq for positions, nv for velocities)
            nq = self._state.getNQ()
            nv = self._state.getNU()

            # Build Jacobian using finite differences (OpenSim doesn't expose
            # direct Jacobian computation as easily as MuJoCo)
            # For each generalized coordinate, compute d(body_position)/dq
            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))

            # Central differences with a macroscopic angular step are more stable
            # for OpenSim coordinates than sqrt(eps) forward differences.
            eps = 1e-4

            # Store original state
            q_orig = np.zeros(nq)
            for i in range(nq):
                q_orig[i] = self._state.getQ()[i]

            for i in range(nv):
                # Perturb coordinate i symmetrically.
                local_eps = eps * max(1.0, abs(q_orig[i]))
                q_plus = q_orig.copy()
                q_minus = q_orig.copy()
                q_plus[i] += local_eps
                q_minus[i] -= local_eps

                # Set positively perturbed state.
                for j in range(nq):
                    self._state.updQ()[j] = q_plus[j]
                self._model.realizePosition(self._state)

                transform_plus = body.getTransformInGround(self._state)
                pos_plus = np.array(
                    [
                        transform_plus.p()[0],
                        transform_plus.p()[1],
                        transform_plus.p()[2],
                    ]
                )
                rotation_plus = transform_plus.R()

                # Set negatively perturbed state.
                for j in range(nq):
                    self._state.updQ()[j] = q_minus[j]
                self._model.realizePosition(self._state)

                transform_minus = body.getTransformInGround(self._state)
                pos_minus = np.array(
                    [
                        transform_minus.p()[0],
                        transform_minus.p()[1],
                        transform_minus.p()[2],
                    ]
                )
                rotation_minus = transform_minus.R()

                # Position and angular Jacobian columns.
                jacp[:, i] = (pos_plus - pos_minus) / (2.0 * local_eps)
                jacr[:, i] = self._rotation_difference(
                    rotation_minus, rotation_plus
                ) / (2.0 * local_eps)

            # Restore original state
            for i in range(nq):
                self._state.updQ()[i] = q_orig[i]
            self._model.realizePosition(self._state)

            return {
                "linear": jacp,
                "angular": jacr,
                "spatial": np.vstack([jacr, jacp]),  # [Angular; Linear] convention
            }

        except ImportError as e:
            logger.error(f"Failed to compute Jacobian for '{body_name}': {e}")
            return None

    def _rotation_difference(self, R0: Any, R1: Any) -> np.ndarray:
        """Compute rotation difference as angular velocity vector.

        Args:
            R0: Initial rotation (SimTK Rotation)
            R1: Final rotation (SimTK Rotation)

        Returns:
            Angular velocity approximation (3,) [rad]
        """
        try:
            # Convert rotations to matrices
            mat0 = np.array(
                [
                    [R0[0][0], R0[0][1], R0[0][2]],
                    [R0[1][0], R0[1][1], R0[1][2]],
                    [R0[2][0], R0[2][1], R0[2][2]],
                ]
            )
            mat1 = np.array(
                [
                    [R1[0][0], R1[0][1], R1[0][2]],
                    [R1[1][0], R1[1][1], R1[1][2]],
                    [R1[2][0], R1[2][1], R1[2][2]],
                ]
            )

            # Compute relative rotation
            mat_diff = mat0.T @ mat1

            # Extract axis-angle from rotation matrix
            # Using Rodrigues formula inverse
            trace = np.trace(mat_diff)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))

            if abs(angle) < 1e-10:
                return np.zeros(3)

            # Axis from skew-symmetric part
            axis = np.array(
                [
                    mat_diff[2, 1] - mat_diff[1, 2],
                    mat_diff[0, 2] - mat_diff[2, 0],
                    mat_diff[1, 0] - mat_diff[0, 1],
                ]
            )
            axis_arr = np.asarray(axis, dtype=float).reshape(-1)
            axis_norm = float(0.0 if axis_arr.size == 0 else math.hypot(*axis_arr))
            if axis_norm > 1e-10:
                axis = axis / axis_norm

            return np.asarray(axis * angle)

        except ImportError:
            return np.zeros(3)

    # -------- Section F: Drift-Control Decomposition --------

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Drift acceleration must contain finite values")
    def compute_drift_acceleration(self) -> np.ndarray:
        """Compute passive (drift) acceleration with zero control inputs.

        Section F Implementation: Computes acceleration with all muscle activations
        and control forces set to zero.

        Returns:
            q_ddot_drift: Drift acceleration vector (nv,) [rad/s² or m/s²]
        """
        if not self._model or not self._state:
            logger.warning("Model or state not initialized")
            return np.array([])

        # Get mass matrix and bias forces
        M = self.compute_mass_matrix()
        bias = self.compute_bias_forces()

        # Drift acceleration = -M^-1 * bias
        # (bias includes Coriolis + gravity with zero muscle forces)
        a_drift = -np.linalg.solve(M, bias)

        return a_drift

    @precondition(lambda self, tau: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Control acceleration must contain finite values")
    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Compute control-attributed acceleration from applied torques/muscles.

        Section F Implementation: Computes M(q)^-1 * tau to isolate control component.

        Args:
            tau: Applied generalized forces (nv,) [N·m or N]

        Returns:
            q_ddot_control: Control acceleration vector (nv,) [rad/s² or m/s²]
        """
        if tau is None:
            raise ValueError("tau must be provided")
        if not self._model or not self._state:
            logger.warning("Model or state not initialized")
            return np.array([])

        # Get mass matrix
        M = self.compute_mass_matrix()

        # Control component: M^-1 * tau
        a_control = np.linalg.solve(M, tau)

        return a_control

    # -------- Section J: Muscle Model Integration --------

    def get_muscle_analyzer(self) -> Any | None:
        """Get muscle analyzer for biomechanical analysis.

        Section J: Provides access to muscle-specific analysis capabilities.

        Returns:
            OpenSimMuscleAnalyzer instance or None if model/state not ready
        """
        if not self._model or not self._state:
            logger.warning("Cannot create muscle analyzer - model not initialized")
            return None

        try:
            from .muscle_analysis import OpenSimMuscleAnalyzer

            return OpenSimMuscleAnalyzer(self._model, self._state)
        except ImportError as e:
            logger.error(f"Failed to import muscle analyzer: {e}")
            return None

    def create_grip_model(self) -> Any | None:
        """Create grip modeling interface.

        Section J1: Provides grip wrapping geometry and force analysis.

        Returns:
            OpenSimGripModel instance or None if model not ready
        """
        if not self._model:
            logger.warning("Cannot create grip model - model not initialized")
            return None

        try:
            from .muscle_analysis import OpenSimGripModel

            return OpenSimGripModel(self._model)
        except ImportError as e:
            logger.error(f"Failed to import grip model: {e}")
            return None

    def compute_muscle_induced_accelerations(self) -> dict[str, np.ndarray]:
        """Compute acceleration contributions from each muscle.

        Section J Requirement: Muscle contribution to joint accelerations.

        Returns:
            Dictionary mapping muscle names to induced accelerations [rad/s²]
        """
        analyzer = self.get_muscle_analyzer()
        if analyzer is None:
            return {}

        return dict(analyzer.compute_muscle_induced_accelerations())

    def compute_iaa_decomposition(self) -> dict[str, np.ndarray]:
        """Update the IAA decomposition to separate active vs. passive muscle contributions."""
        analyzer = self.get_muscle_analyzer()
        if not analyzer or not self.is_initialized:
            return {}

        assert self._model is not None  # guaranteed by is_initialized check above

        # M * a = tau  =>  a = M^-1 * tau
        M = self.compute_mass_matrix()

        cond = np.linalg.cond(M)
        if cond > 1e8:
            lambda_reg = 1e-6 * np.trace(M) / M.shape[0]
            M_solve = M + lambda_reg * np.eye(M.shape[0])
        else:
            M_solve = M

        # Gravity and Velocity
        gravity = self.compute_gravity_forces()
        bias = self.compute_bias_forces()
        velocity_forces = bias - gravity

        gravity_accel = np.linalg.solve(M_solve, gravity)
        velocity_accel = -np.linalg.solve(
            M_solve, velocity_forces
        )  # Since bias = C*v + G

        # Muscle Active vs Passive
        active_forces = analyzer.get_muscle_forces()
        passive_forces = analyzer.get_passive_muscle_forces()
        moment_arms = analyzer.get_moment_arms()

        n_u = self._model.getNumSpeeds()  # type: ignore
        active_tau = np.zeros(n_u)
        passive_tau = np.zeros(n_u)

        for muscle_name in active_forces:
            if muscle_name in moment_arms:
                moment_arm_values = list(moment_arms[muscle_name].values())
                for coord_idx, r in enumerate(moment_arm_values):
                    if coord_idx < n_u:
                        active_tau[coord_idx] += active_forces[muscle_name] * r
                        passive_tau[coord_idx] += (
                            passive_forces.get(muscle_name, 0.0) * r
                        )

        active_muscle_accel = np.linalg.solve(M_solve, active_tau)
        passive_muscle_accel = np.linalg.solve(M_solve, passive_tau)

        external_accel = np.zeros(n_u)
        total_accel = (
            gravity_accel
            + velocity_accel
            + active_muscle_accel
            + passive_muscle_accel
            + external_accel
        )

        return {
            "gravity": gravity_accel,
            "velocity": velocity_accel,
            "active_muscle": active_muscle_accel,
            "passive_muscle": passive_muscle_accel,
            "external": external_accel,
            "total": total_accel,
        }

    def analyze_muscle_contributions(self) -> Any | None:
        """Full muscle contribution analysis.

        Section J Requirement: Comprehensive muscle reports (forces, moments, power).

        Returns:
            MuscleAnalysis object with all muscle metrics
        """
        analyzer = self.get_muscle_analyzer()
        if analyzer is None:
            logger.warning("Cannot analyze muscles - analyzer not available")
            return None

        return analyzer.analyze_all()

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Zero-Torque Counterfactual (ZTCF) - Guideline G1.

        Compute acceleration with applied torques set to zero, preserving current state.
        This isolates drift (gravity + Coriolis + constraints) from control effects.

        Args:
            q: Joint positions (n_v,)
            v: Joint velocities (n_v,)

        Returns:
            q̈_ZTCF: Acceleration under zero applied torque (n_v,)
        """
        if q is None:
            raise ValueError("q must be provided")
        if not self._model or not self._state:
            return np.array([])

        try:
            # Save current state and controls
            q_saved, v_saved = self.get_state()
            controls_saved = opensim.Vector(self._model.updControls(self._state))

            # Set desired state
            self.set_state(q, v)

            # Set zero control
            n_controls = self._model.getNumControls()
            zero_controls = np.zeros(n_controls)
            self.set_control(zero_controls)

            # Compute forward dynamics
            # Note: realizeDynamics computes accelerations (udot) in the state
            self._model.realizeDynamics(self._state)

            # Extract accelerations
            n_u = self._model.getNumSpeeds()
            udot = self._state.getUDot()
            a_ztcf = np.array([udot.get(i) for i in range(n_u)])

            # Restore state and controls
            self._model.updControls(self._state).update(controls_saved)

            self.set_state(q_saved, v_saved)

            return a_ztcf

        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(f"Failed to compute ZTCF: {e}")
            return np.array([])

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Zero-Velocity Counterfactual (ZVCF) - Guideline G2.

        Compute acceleration with joint velocities set to zero, preserving configuration
        and controls. This isolates configuration-dependent effects (gravity, constraints).

        Args:
            q: Joint positions (n_v,)

        Returns:
            q̈_ZVCF: Acceleration with v=0 (n_v,)
        """
        if q is None:
            raise ValueError("q must be provided")
        if not self._model or not self._state:
            return np.array([])

        try:
            # Save current state
            q_saved, v_saved = self.get_state()

            # Set state with zero velocity
            n_u = self._model.getNumSpeeds()
            self.set_state(q, np.zeros(n_u))

            # Controls are preserved in state
            # Compute forward dynamics
            self._model.realizeDynamics(self._state)

            # Extract accelerations
            udot = self._state.getUDot()
            a_zvcf = np.array([udot.get(i) for i in range(n_u)])

            # Restore state
            self.set_state(q_saved, v_saved)

            return a_zvcf

        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(f"Failed to compute ZVCF: {e}")
            return np.array([])
