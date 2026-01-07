"""Multi-muscle coordination for antagonist muscle groups.

This module manages multiple muscles acting on the same joint(s), including:
- Antagonist muscle pairs (e.g., biceps/triceps, flexors/extensors)
- Synergist muscles (multiple muscles contributing to same motion)
- Moment arm computation from muscle routing geometry

Critical for golf biomechanics:
- Shoulder: Deltoid, rotator cuff (4 muscles), pectoralis
- Elbow: Biceps brachii, brachialis, triceps
- Wrist: Flexor carpi radialis, extensor carpi radialis
- Trunk: Abdominals, erector spinae, obliques

Reference:
- Zajac (1993), "Muscle coordination of movement: a perspective"
- Delp et al. (2007), "OpenSim: Open-source software to create and analyze musculoskeletal models"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from shared.python.hill_muscle import HillMuscleModel

logger = logging.getLogger(__name__)


@dataclass
class MuscleAttachment:
    """Muscle attachment point (origin or insertion).

    Attributes:
        body_name: Name of body segment (e.g., "humerus", "radius")
        position: Attachment point in body frame [m] (3×1 vector)
    """

    body_name: str
    position: np.ndarray  # [m] (3,)


@dataclass
class MusclePath:
    """Muscle routing path from origin to insertion.

    Attributes:
        origin: Origin attachment point
        insertion: Insertion attachment point
        via_points: Intermediate via points for wrapping (optional)
    """

    origin: MuscleAttachment
    insertion: MuscleAttachment
    via_points: list[MuscleAttachment] | None = None


class MuscleGroup:
    """Collection of muscles acting on common joint(s).

    Manages antagonist pairs, synergists, and computes net joint torque.

    Example:
        >>> # Create elbow flexor group
        >>> biceps = HillMuscleModel(biceps_params)
        >>> brachialis = HillMuscleModel(brachialis_params)
        >>>
        >>> flexors = MuscleGroup("elbow_flexors")
        >>> flexors.add_muscle("biceps", biceps, moment_arm=0.04)  # 4 cm
        >>> flexors.add_muscle("brachialis", brachialis, moment_arm=0.03)  # 3 cm
        >>>
        >>> # Compute total flexor torque
        >>> tau_flex = flexors.compute_net_torque(activations, states)
    """

    def __init__(self, name: str):
        """Initialize muscle group.

        Args:
            name: Group identifier (e.g., "elbow_flexors", "shoulder_abductors")
        """
        self.name = name
        self.muscles: dict[str, HillMuscleModel] = {}
        self.moment_arms: dict[str, float] = {}  # [m]
        self.paths: dict[str, MusclePath] = {}

    def add_muscle(
        self,
        muscle_name: str,
        muscle: HillMuscleModel,
        moment_arm: float,
        path: MusclePath | None = None,
    ) -> None:
        """Add muscle to group.

        Args:
            muscle_name: Muscle identifier (e.g., "biceps_brachii")
            muscle: HillMuscleModel instance
            moment_arm: Muscle moment arm [m] (∂l_MT/∂q)
            path: Muscle routing path (optional, for geometry)

        Example:
            >>> group.add_muscle("biceps", biceps_model, moment_arm=0.04)
        """
        self.muscles[muscle_name] = muscle
        self.moment_arms[muscle_name] = moment_arm
        if path is not None:
            self.paths[muscle_name] = path

        logger.debug(
            f"Added muscle '{muscle_name}' to group '{self.name}': "
            f"r = {moment_arm*100:.2f} cm"
        )

    def compute_net_torque(
        self,
        activations: dict[str, float],
        muscle_states: dict[str, tuple[float, float]],
    ) -> float:
        """Compute net joint torque from all muscles.

        Sum of individual muscle contributions:
            τ_net = Σ(F_i · r_i)

        Args:
            activations: Muscle activations {muscle_name: activation}
            muscle_states: Fiber states {muscle_name: (l_CE, v_CE)}

        Returns:
            Net joint torque [N·m]

        Example:
            >>> activations = {"biceps": 0.8, "brachialis": 0.6}
            >>> states = {"biceps": (0.12, 0.0), "brachialis": (0.10, 0.0)}
            >>> tau = group.compute_net_torque(activations, states)
        """
        tau_net = 0.0

        for muscle_name, muscle in self.muscles.items():
            # Get muscle state
            if muscle_name not in muscle_states:
                logger.warning(
                    f"Muscle '{muscle_name}' missing state. Assuming zero force."
                )
                continue

            l_CE, v_CE = muscle_states[muscle_name]
            activation = activations.get(muscle_name, 0.0)

            # Build temporary muscle state for force computation
            from shared.python.hill_muscle import MuscleState

            # Approximate l_MT from l_CE (simplified, ignores tendon compliance)
            l_MT_approx = l_CE + muscle.params.l_slack

            state = MuscleState(
                l_MT=l_MT_approx,
                v_MT=0.0,  # Simplified
                activation=activation,
                l_CE=l_CE,
                v_CE=v_CE,
            )

            # Compute muscle force
            F_muscle = muscle.compute_muscle_force(state)

            # Apply moment arm: τ = F · r
            r = self.moment_arms[muscle_name]
            tau_muscle = F_muscle * r

            tau_net += tau_muscle

            logger.debug(
                f"  {muscle_name}: F={F_muscle:.1f}N, r={r*100:.2f}cm → τ={tau_muscle:.2f}N·m"
            )

        return float(tau_net)


class AntagonistPair:
    """Manages antagonist muscle pair (agonist vs. antagonist).

    Example: Biceps (flexor) vs. Triceps (extensor) at elbow.

    This models the biomechanical principle of reciprocal inhibition:
    - Agonist activation → movement in desired direction
    - Antagonist co-activation → joint stiffness modulation

    Example:
        >>> pair = AntagonistPair("elbow", flexor_group, extensor_group)
        >>> tau_net = pair.compute_net_torque(flexor_act, extensor_act, states)
        >>> # Positive = flexion, negative = extension
    """

    def __init__(
        self,
        joint_name: str,
        agonist_group: MuscleGroup,
        antagonist_group: MuscleGroup,
    ):
        """Initialize antagonist pair.

        Args:
            joint_name: Joint identifier (e.g., "elbow", "shoulder")
            agonist_group: Muscle group for primary motion
            antagonist_group: Opposing muscle group
        """
        self.joint_name = joint_name
        self.agonist = agonist_group
        self.antagonist = antagonist_group

    def compute_net_torque(
        self,
        agonist_activations: dict[str, float],
        antagonist_activations: dict[str, float],
        muscle_states: dict[str, tuple[float, float]],
    ) -> float:
        """Compute net joint torque (agonist - antagonist).

        Args:
            agonist_activations: Agonist muscle activations
            antagonist_activations: Antagonist muscle activations
            muscle_states: Combined fiber states for all muscles

        Returns:
            Net joint torque [N·m] (positive = agonist direction)

        Example:
            >>> tau = pair.compute_net_torque(
            ...     {"biceps": 0.8},  # Flexors active
            ...     {"triceps": 0.2},  # Extensors co-contracting
            ...     states
            ... )
            >>> # Result: positive (flexion) but reduced by co-contraction
        """
        tau_agonist = self.agonist.compute_net_torque(
            agonist_activations, muscle_states
        )
        tau_antagonist = self.antagonist.compute_net_torque(
            antagonist_activations, muscle_states
        )

        # Net torque (agonist - antagonist)
        tau_net = tau_agonist - tau_antagonist

        logger.debug(
            f"{self.joint_name}: τ_agonist={tau_agonist:.2f}N·m, "
            f"τ_antagonist={tau_antagonist:.2f}N·m → "
            f"τ_net={tau_net:.2f}N·m"
        )

        return float(tau_net)

    def compute_joint_stiffness(
        self,
        agonist_activations: dict[str, float],
        antagonist_activations: dict[str, float],
    ) -> float:
        """Compute effective joint stiffness from co-contraction.

        Joint stiffness K ≈ K_agonist + K_antagonist

        Higher co-contraction → higher stiffness → more stable joint.

        Args:
            agonist_activations: Agonist muscle activations
            antagonist_activations: Antagonist muscle activations

        Returns:
            Approximate joint stiffness [N·m/rad]

        Note:
            This is a simplified estimate. Actual stiffness depends on
            muscle force-length derivatives and moment arm geometry.
        """
        # Sum activations as proxy for stiffness
        # Real version would compute ∂F/∂l for each muscle
        total_activation = sum(agonist_activations.values()) + sum(
            antagonist_activations.values()
        )

        # Approximate stiffness (empirical scaling)
        # Typical range: 10-100 N·m/rad for elbow
        K_approx = total_activation * 50.0  # [N·m/rad] (heuristic)

        return float(K_approx)


# Example: Create elbow muscle system
def create_elbow_muscle_system() -> AntagonistPair:
    """Create anatomically realistic elbow muscle system.

    Returns:
        AntagonistPair for elbow (flexors vs. extensors)

    Example:
        >>> elbow = create_elbow_muscle_system()
        >>> tau = elbow.compute_net_torque(flexor_act, extensor_act, states)
    """
    from shared.python.hill_muscle import HillMuscleModel, MuscleParameters

    # Biceps brachii (primary flexor)
    biceps_params = MuscleParameters(
        F_max=700.0,  # N (Holzbaur et al. 2005)
        l_opt=0.116,  # m (11.6 cm)
        l_slack=0.267,  # m (26.7 cm)
        v_max=1.16,  # m/s (10 · l_opt/s)
        pennation_angle=0.0,  # rad (approximately parallel fibers)
    )
    biceps = HillMuscleModel(biceps_params)

    # Brachialis (deep flexor)
    brachialis_params = MuscleParameters(
        F_max=987.0,  # N (stronger than biceps!)
        l_opt=0.097,  # m
        l_slack=0.054,  # m
        v_max=0.97,  # m/s
        pennation_angle=0.0,
    )
    brachialis = HillMuscleModel(brachialis_params)

    # Triceps brachii (primary extensor)
    triceps_params = MuscleParameters(
        F_max=798.0,  # N (long head)
        l_opt=0.134,  # m
        l_slack=0.143,  # m
        v_max=1.34,  # m/s
        pennation_angle=np.radians(12),  # 12° pennation
    )
    triceps = HillMuscleModel(triceps_params)

    # Create muscle groups
    flexors = MuscleGroup("elbow_flexors")
    flexors.add_muscle("biceps_brachii", biceps, moment_arm=0.040)  # 4.0 cm
    flexors.add_muscle("brachialis", brachialis, moment_arm=0.030)  # 3.0 cm

    extensors = MuscleGroup("elbow_extensors")
    extensors.add_muscle("triceps_brachii", triceps, moment_arm=0.025)  # 2.5 cm

    # Create antagonist pair
    elbow_pair = AntagonistPair("elbow", flexors, extensors)

    logger.info(
        f"Created elbow muscle system:\\n"
        f"  Flexors: biceps ({biceps_params.F_max:.0f}N), "
        f"brachialis ({brachialis_params.F_max:.0f}N)\\n"
        f"  Extensors: triceps ({triceps_params.F_max:.0f}N)"
    )

    return elbow_pair


# Example usage / validation
if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Muscle Coordination Test")
    print("=" * 60)

    # Create elbow system
    elbow = create_elbow_muscle_system()

    # Test case: Flexion with co-contraction
    print("\\nTest: Elbow flexion with antagonist co-contraction")

    # Activations
    flexor_act = {
        "biceps_brachii": 0.8,  # 80% biceps activation
        "brachialis": 0.6,  # 60% brachialis activation
    }

    extensor_act = {
        "triceps_brachii": 0.2,  # 20% triceps (co-contraction for stability)
    }

    # Muscle states (at optimal lengths for simplicity)
    states = {
        "biceps_brachii": (0.116, 0.0),  # (l_CE, v_CE)
        "brachialis": (0.097, 0.0),
        "triceps_brachii": (0.134, 0.0),
    }

    # Compute net torque
    tau_net = elbow.compute_net_torque(flexor_act, extensor_act, states)

    print(f"\\nFlexor activations: {flexor_act}")
    print(f"Extensor activations: {extensor_act}")
    print(f"\\nNet elbow torque: {tau_net:.2f} N·m")
    print("  (Positive = flexion)")

    # Compute joint stiffness
    K = elbow.compute_joint_stiffness(flexor_act, extensor_act)
    print(f"\\nEstimated joint stiffness: {K:.1f} N·m/rad")
    print("  (Higher co-contraction → higher stiffness)")

    print("\\n" + "=" * 60)
    print("✓ Multi-muscle test complete")
    print("=" * 60)
