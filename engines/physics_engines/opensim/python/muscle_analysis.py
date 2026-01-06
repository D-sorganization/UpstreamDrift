"""OpenSim Muscle Analysis and Grip Modeling Extensions.

Section J: OpenSim-Class Biomechanics Features
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    import opensim
except ImportError:
    opensim = None
    logger.warning("OpenSim not installed - muscle analysis unavailable")

# Constants for muscle analysis
NINETY_DEGREES_RAD = 1.5708  # π/2 radians for 90° rotation
MIN_PHYSIOLOGICAL_GRIP_N = 50.0  # Minimum physiological grip force per hand [N]
MAX_PHYSIOLOGICAL_GRIP_N = 200.0  # Maximum physiological grip force per hand [N]


@dataclass
class MuscleAnalysis:
    """Section J: OpenSim muscle analysis results.

    Attributes:
        muscle_forces: Dictionary mapping muscle names to forces [N]
        moment_arms: Dictionary mapping muscle names to moment arms [m]
        activation_levels: Dictionary mapping muscle names to activation [0-1]
        muscle_lengths: Dictionary mapping muscle names to lengths [m]
        total_muscle_torque: Net torque from all muscles [N·m]
    """

    muscle_forces: dict[str, float]
    moment_arms: dict[str, dict[str, float]]  # muscle_name -> {coord_name: moment_arm}
    activation_levels: dict[str, float]
    muscle_lengths: dict[str, float]
    total_muscle_torque: np.ndarray


class OpenSimMuscleAnalyzer:
    """Section J: OpenSim muscle model analysis and control.

    Provides muscle-specific analysis capabilities including:
    - Hill-type muscle force computation
    - Moment arm analysis
    - Activation → force → torque pipeline
    - Muscle contribution to joint accelerations
    """

    def __init__(self, model: opensim.Model, state: opensim.State):
        """Initialize muscle analyzer.

        Args:
            model: OpenSim model with muscles
            state: Current state of the simulation
        """
        self.model = model
        self.state = state
        self.muscle_set = model.getMuscles()
        self.n_muscles = self.muscle_set.getSize()

    def get_muscle_forces(self) -> dict[str, float]:
        """Compute current muscle forces for all muscles.

        Section J Requirement: Muscle force computation using Hill-type model.

        Returns:
            Dictionary mapping muscle names to forces [N]
        """
        if opensim is None:
            return {}

        forces = {}
        self.model.realizeDynamics(self.state)

        for i in range(self.n_muscles):
            muscle = opensim.Muscle.safeDownCast(self.muscle_set.get(i))
            if muscle:
                name = muscle.getName()
                # Get active fiber force (Hill-type model output)
                force = muscle.getActiveFiberForce(self.state)
                forces[name] = float(force)

        return forces

    def get_moment_arms(
        self, coordinate_name: str | None = None
    ) -> dict[str, dict[str, float]]:
        """Compute muscle moment arms about coordinates.

        Section J Requirement: Moment arm analysis for torque computation.

        Args:
            coordinate_name: Specific coordinate to analyze (None = all)

        Returns:
            Nested dictionary: muscle_name -> {coord_name: moment_arm [m]}
        """
        if opensim is None:
            return {}

        moment_arms = {}
        coords = self.model.getCoordinateSet()

        for i in range(self.n_muscles):
            muscle = opensim.Muscle.safeDownCast(self.muscle_set.get(i))
            if muscle:
                muscle_name = muscle.getName()
                moment_arms[muscle_name] = {}

                # Compute moment arm for each coordinate
                for j in range(coords.getSize()):
                    coord = coords.get(j)
                    coord_name = coord.getName()

                    if coordinate_name and coord_name != coordinate_name:
                        continue

                    try:
                        # Moment arm = dL/dq (change in muscle length per unit coordinate change)
                        moment_arm = muscle.computeMomentArm(self.state, coord)
                        moment_arms[muscle_name][coord_name] = float(moment_arm)
                    except Exception as e:
                        logger.debug(
                            f"Could not compute moment arm for {muscle_name} about {coord_name}: {e}"
                        )
                        moment_arms[muscle_name][coord_name] = 0.0

        return moment_arms

    def get_activation_levels(self) -> dict[str, float]:
        """Get current muscle activation levels.

        Section J Requirement: Activation tracking for neural control analysis.

        Returns:
            Dictionary mapping muscle names to activation [0-1]
        """
        if opensim is None:
            return {}

        activations = {}
        self.model.realizeDynamics(self.state)

        for i in range(self.n_muscles):
            muscle = opensim.Muscle.safeDownCast(self.muscle_set.get(i))
            if muscle:
                name = muscle.getName()
                activation = muscle.getActivation(self.state)
                activations[name] = float(activation)

        return activations

    def set_activation_levels(self, activations: dict[str, float]) -> None:
        """Set muscle activation levels.

        Args:
            activations: Dictionary mapping muscle names to desired activation [0-1]
        """
        if opensim is None:
            return

        for muscle_name, activation in activations.items():
            try:
                muscle = opensim.Muscle.safeDownCast(self.muscle_set.get(muscle_name))
                if muscle:
                    # Clamp to [0, 1]
                    activation_clamped = max(0.0, min(1.0, activation))
                    muscle.setActivation(self.state, activation_clamped)
            except Exception as e:
                logger.warning(f"Could not set activation for {muscle_name}: {e}")

    def compute_muscle_joint_torques(self) -> dict[str, np.ndarray]:
        """Compute joint torques generated by each muscle.

        Section J Requirement: Activation → force → joint torque mapping.

        Returns:
            Dictionary mapping muscle names to torque vectors [N·m]
        """
        if opensim is None:
            return {}

        forces = self.get_muscle_forces()
        moment_arms = self.get_moment_arms()

        n_coords = self.model.getNumCoordinates()
        muscle_torques = {}

        for muscle_name, force in forces.items():
            torques = np.zeros(n_coords)

            if muscle_name in moment_arms:
                coord_idx = 0
                for _ in moment_arms[muscle_name].values():
                    torques[coord_idx] = (
                        force * list(moment_arms[muscle_name].values())[coord_idx]
                    )
                    coord_idx += 1

            muscle_torques[muscle_name] = torques

        return muscle_torques

    def compute_muscle_induced_accelerations(self) -> dict[str, np.ndarray]:
        """Compute induced accelerations from each muscle.

        Section J Requirement: Muscle contribution to joint accelerations (induced acceleration).

        Returns:
            Dictionary mapping muscle names to induced accelerations [rad/s²]
        """
        if opensim is None:
            return {}

        # Get mass matrix
        matter = self.model.getMatterSubsystem()
        n_u = self.model.getNumSpeeds()
        m_mat = opensim.Matrix()
        self.model.realizePosition(self.state)
        matter.calcM(self.state, m_mat)

        # Convert to numpy
        M = np.zeros((n_u, n_u))
        for r in range(n_u):
            for c in range(n_u):
                M[r, c] = m_mat.get(r, c)

        # Get muscle torques
        muscle_torques = self.compute_muscle_joint_torques()

        # Compute induced acceleration: a = M^-1 * tau
        induced_accelerations = {}
        for muscle_name, tau in muscle_torques.items():
            # Pad or trim to match size
            tau_full = np.zeros(n_u)
            tau_full[: min(len(tau), n_u)] = tau[: min(len(tau), n_u)]

            a_induced = np.linalg.solve(M, tau_full)
            induced_accelerations[muscle_name] = a_induced

        return induced_accelerations

    def analyze_all(self) -> MuscleAnalysis:
        """Comprehensive muscle analysis.

        Section J Requirement: Complete muscle contribution reports.

        Returns:
            MuscleAnalysis object with all computed quantities
        """
        forces = self.get_muscle_forces()
        moment_arms = self.get_moment_arms()
        activations = self.get_activation_levels()

        # Compute muscle lengths
        lengths = {}
        if opensim:
            self.model.realizeDynamics(self.state)
            for i in range(self.n_muscles):
                muscle = opensim.Muscle.safeDownCast(self.muscle_set.get(i))
                if muscle:
                    name = muscle.getName()
                    lengths[name] = float(muscle.getLength(self.state))

        # Compute total muscle torque contribution
        torques = self.compute_muscle_joint_torques()
        total_torque = np.zeros(self.model.getNumCoordinates())
        for tau_vec in torques.values():
            total_torque[: len(tau_vec)] += tau_vec

        return MuscleAnalysis(
            muscle_forces=forces,
            moment_arms=moment_arms,
            activation_levels=activations,
            muscle_lengths=lengths,
            total_muscle_torque=total_torque,
        )


class OpenSimGripModel:
    """Section J1: OpenSim grip modeling via wrapping geometry.

    Models hand-grip interface using:
    - Wrapping surfaces (cylinder/ellipsoid around grip)
    - Via-point constraints for key grip locations
    - Muscle routing through contact points
    """

    def __init__(self, model: opensim.Model):
        """Initialize grip model.

        Args:
            model: OpenSim model (should have grip body and hand muscles)
        """
        self.model = model

    def add_cylindrical_wrap(
        self,
        muscle_name: str,
        grip_body_name: str,
        radius: float,
        length: float,
        location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Add cylindrical wrapping surface for grip.

        Section J1 Requirement: Wrapping geometry for muscle routing.

        Args:
            muscle_name: Name of muscle to wrap
            grip_body_name: Name of grip body
            radius: Wrap cylinder radius [m] (typically shaft radius + hand thickness)
            length: Wrap cylinder length [m]
            location: (x, y, z) location in grip body frame [m]
        """
        if opensim is None:
            logger.warning("OpenSim not installed - cannot add wrap")
            return

        try:
            # Get the grip body
            grip_body = self.model.getBodySet().get(grip_body_name)

            # Create wrap cylinder
            wrap_cylinder = opensim.WrapCylinder()
            wrap_cylinder.setName(f"{muscle_name}_grip_wrap")
            wrap_cylinder.set_radius(radius)
            wrap_cylinder.set_length(length)

            # Set location in body frame
            wrap_cylinder.set_translation(
                opensim.Vec3(location[0], location[1], location[2])
            )

            # Rotation: typically align cylinder with shaft axis (e.g., along Y)
            wrap_cylinder.set_xyz_body_rotation(
                opensim.Vec3(0, NINETY_DEGREES_RAD, 0)
            )  # 90° about Y

            # Add to body
            wrap_obj_set = grip_body.getWrapObjectSet()
            wrap_obj_set.cloneAndAppend(wrap_cylinder)

            logger.info(f"Added cylindrical wrap for {muscle_name} on {grip_body_name}")

        except Exception as e:
            logger.error(f"Failed to add wrap geometry: {e}")

    def compute_grip_constraint_forces(
        self, state: opensim.State
    ) -> dict[str, np.ndarray]:
        """Compute constraint reaction forces at grip via-points.

        Section J1 Requirement: Constraint forces at grip attachment points.

        Args:
            state: Current simulation state

        Returns:
            Dictionary mapping constraint names to reaction forces [N]
        """
        if opensim is None:
            return {}

        # This requires accessing SimTK constraint forces
        # Placeholder implementation - full implementation needs SimTK API
        logger.warning("Grip constraint force computation: Placeholder implementation")
        return {}

    def analyze_grip_forces(
        self, state: opensim.State, analyzer: OpenSimMuscleAnalyzer
    ) -> dict[str, float]:
        """Analyze total grip force from all hand muscles.

        Section J1 Validation: Grip force magnitude [N] within physiological range.

        Args:
            state: Current simulation state
            analyzer: Muscle analyzer for force computation

        Returns:
            Dictionary with grip analysis metrics
        """
        # Get forces from grip-related muscles
        muscle_forces = analyzer.get_muscle_forces()

        # Filter for grip muscles (typically hand/finger muscles)
        grip_muscle_names = [
            name
            for name in muscle_forces.keys()
            if any(
                keyword in name.lower()
                for keyword in ["flexor", "extensor", "grip", "hand"]
            )
        ]

        total_grip_force = sum(
            muscle_forces.get(name, 0.0) for name in grip_muscle_names
        )

        return {
            "total_grip_force_N": total_grip_force,
            "n_grip_muscles": len(grip_muscle_names),
            "grip_muscles": grip_muscle_names,
            "within_physiological_range": MIN_PHYSIOLOGICAL_GRIP_N
            <= total_grip_force
            <= MAX_PHYSIOLOGICAL_GRIP_N,  # Per hand
        }
