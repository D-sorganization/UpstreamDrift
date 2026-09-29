"""Multi-muscle coordination and antagonist pairs.

This module models groups of muscles working together (synergists) or
against each other (antagonists) to produce joint torque.

Key concepts:
- Agonist: Muscle creating torque in the desired direction
- Antagonist: Muscle opposing the torque (provides stability/stiffness)
- Co-contraction: Simultaneous activation of both to increase joint stiffness

Reference:
- Hogan (1984), "Adaptive Control of Mechanical Impedance by Co-activation of Antagonist Muscles"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.core.contracts import ensure, require
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from src.shared.python.biomechanics.hill_muscle import HillMuscleModel

logger = get_logger(__name__)

try:
    import upstream_muscle

    HAS_RUST_BACKEND = True
except ImportError:
    HAS_RUST_BACKEND = False


def _to_rust_muscle(muscle: Any) -> Any:
    """Return the ``upstream_muscle`` equivalent of *muscle*, or None if it has none.

    Only an ``upstream_muscle.HillMuscleModel`` or a Python ``HillMuscleModel``
    whose ``compute_force`` is not overridden can be mirrored in Rust; anything
    else (subclasses with custom physics, test doubles) keeps the group on the
    pure-Python path so both paths always compute the same thing (#10946).
    """
    if not HAS_RUST_BACKEND:
        return None
    if isinstance(muscle, upstream_muscle.HillMuscleModel):
        return muscle
    from src.shared.python.biomechanics.hill_muscle import HillMuscleModel

    if not isinstance(muscle, HillMuscleModel) or (
        type(muscle).compute_force is not HillMuscleModel.compute_force
    ):
        return None
    p = muscle.params
    rust_params = upstream_muscle.MuscleParameters(
        float(p.F_max),
        float(p.l_opt),
        float(p.l_slack),
        float(p.v_max),
        float(p.pennation_angle),
        float(p.damping),
    )
    return upstream_muscle.HillMuscleModel(
        rust_params, float(muscle._force_length_width)
    )


def _validate_activations(
    activations: dict[str, float], arg_name: str = "activations"
) -> None:
    """Validate activation dictionary preconditions.

    Design by Contract:
        Preconditions:
            - activations must be provided
            - all activation values in [0, 1] and finite
    """
    if activations is None:
        raise ValueError(f"{arg_name} must be provided")
    for mname, act_val in activations.items():
        require(
            act_val is not None and np.isfinite(act_val) and 0.0 <= act_val <= 1.0,
            f"activation for '{mname}' must be in [0, 1]",
            act_val,
        )


@dataclass
class MuscleAttachment:
    """Defines how a muscle attaches to a joint (moment arm)."""

    muscle_name: str
    moment_arm: float  # [m] Positive = flexion, Negative = extension
    # In reality, moment arm varies with angle r(θ). Simplified here as constant.


class MuscleGroup:
    """A group of muscles acting on a single joint."""

    def __init__(self, name: str, enable_rust: bool = True) -> None:
        """Initialize muscle group.

        Args:
            name: Group name (e.g., "Elbow Flexors")
            enable_rust: Whether to enable Rust backend acceleration if available.
        """
        if name is None:
            raise ValueError("name must be provided")
        self.name = name
        self.enable_rust = enable_rust
        self.muscles: dict[str, HillMuscleModel] = {}
        self.attachments: dict[str, MuscleAttachment] = {}

        self._rust_backend = None
        if enable_rust and HAS_RUST_BACKEND:
            self._rust_backend = upstream_muscle.MuscleGroup(name)

    def add_muscle(self, name: str, muscle: HillMuscleModel, moment_arm: float) -> None:
        """Add a muscle to the group.

        Design by Contract:
            Preconditions:
                - name must be non-empty
                - moment_arm must be non-zero (zero moment arm produces no torque)

        Args:
            name: Muscle identifier
            muscle: HillMuscleModel instance
            moment_arm: Moment arm [m] (+ for flexion, - for extension)
        """
        if name is None:
            raise ValueError("name must be provided")
        require(bool(name), "muscle name must be non-empty", name)
        require(moment_arm != 0.0, "moment_arm must be non-zero", moment_arm)
        self.muscles[name] = muscle
        self.attachments[name] = MuscleAttachment(name, moment_arm)

        if self._rust_backend is not None:
            rust_muscle = _to_rust_muscle(muscle)
            if rust_muscle is not None:
                self._rust_backend.add_muscle(name, rust_muscle, moment_arm)
            else:
                self._rust_backend = None

    def compute_net_torque(
        self,
        activations: dict[str, float],
        muscle_states: dict[str, tuple[float, float]],
    ) -> float:
        """Compute net torque generated by the group.

        Design by Contract:
            Preconditions:
                - all activation values in [0, 1]
            Postconditions:
                - result is finite

        Args:
            activations: Dict of {muscle_name: activation}
            muscle_states: Dict of {muscle_name: (l_CE, v_CE)}

        Returns:
            Net joint torque [N·m]
        """
        _validate_activations(activations, "activations")
        if muscle_states is None:
            raise ValueError("muscle_states must be provided")

        if self._rust_backend is not None:
            try:
                result = float(
                    self._rust_backend.compute_net_torque(activations, muscle_states)
                )
                ensure(np.isfinite(result), "net torque must be finite", result)
                return result
            except RuntimeError as e:
                logger.warning(
                    f"Rust MuscleGroup compute_net_torque failed, falling back to Python: {e}"
                )

        net_torque = 0.0

        for name, muscle in self.muscles.items():
            if name not in activations:
                continue

            # Get state
            l_CE, v_CE = muscle_states.get(name, (muscle.params.l_opt, 0.0))

            # Create temporary state object for force computation
            from src.shared.python.biomechanics.hill_muscle import MuscleState

            state = MuscleState(
                activation=activations[name],
                l_CE=l_CE,
                v_CE=v_CE,
                l_MT=0.0,  # Not used for force computation in this simplified call
            )

            # Compute force
            force = muscle.compute_force(state)

            # Add to torque (tau = r × F)
            r = self.attachments[name].moment_arm
            torque = r * force
            net_torque += torque

        result = float(net_torque)
        ensure(np.isfinite(result), "net torque must be finite", result)
        return result


class AntagonistPair:
    """A pair of agonist/antagonist muscle groups (e.g., Biceps/Triceps)."""

    def __init__(
        self,
        agonist: MuscleGroup,
        antagonist: MuscleGroup,
        enable_rust: bool = True,
    ) -> None:
        """Initialize antagonist pair.

        Args:
            agonist: MuscleGroup for positive torque (Flexors)
            antagonist: MuscleGroup for negative torque (Extensors)
            enable_rust: Whether to enable Rust backend acceleration if available.
        """
        if agonist is None:
            raise ValueError("agonist must be provided")
        if antagonist is None:
            raise ValueError("antagonist must be provided")
        self.agonist = agonist
        self.antagonist = antagonist
        self.enable_rust = enable_rust

        self._rust_backend = None
        if (
            enable_rust
            and HAS_RUST_BACKEND
            and getattr(agonist, "_rust_backend", None) is not None
            and getattr(antagonist, "_rust_backend", None) is not None
        ):
            self._rust_backend = upstream_muscle.AntagonistPair(
                agonist._rust_backend, antagonist._rust_backend
            )

    def _current_rust_pair(self) -> Any:
        """Rust pair built from the groups' current backends, or None if either is Python-only."""
        ag_rb = self.agonist._rust_backend
        ant_rb = self.antagonist._rust_backend
        if ag_rb is None or ant_rb is None:
            return None
        return upstream_muscle.AntagonistPair(ag_rb, ant_rb)

    def compute_net_torque(
        self,
        agonist_activations: dict[str, float],
        antagonist_activations: dict[str, float],
        muscle_states: dict[str, tuple[float, float]],
    ) -> float:
        """Compute net torque from both groups.

        Design by Contract:
            Postcondition: result is finite

        Args:
            agonist_activations: Activations for agonist muscles
            antagonist_activations: Activations for antagonist muscles
            muscle_states: Shared state dictionary

        Returns:
            Net torque [N·m]
        """
        _validate_activations(agonist_activations, "agonist_activations")
        _validate_activations(antagonist_activations, "antagonist_activations")
        if muscle_states is None:
            raise ValueError("muscle_states must be provided")

        if self._rust_backend is not None:
            # Groups may have gained muscles (or dropped to Python) since construction.
            self._rust_backend = self._current_rust_pair()
        if self._rust_backend is not None:
            try:
                result = float(
                    self._rust_backend.compute_net_torque(
                        agonist_activations, antagonist_activations, muscle_states
                    )
                )
                ensure(
                    np.isfinite(result),
                    "antagonist pair net torque must be finite",
                    result,
                )
                return result
            except RuntimeError as e:
                logger.warning(
                    f"Rust AntagonistPair compute_net_torque failed, falling back to Python: {e}"
                )

        tau_agonist = self.agonist.compute_net_torque(
            agonist_activations, muscle_states
        )
        tau_antagonist = self.antagonist.compute_net_torque(
            antagonist_activations, muscle_states
        )

        # Antagonist moment arms are typically negative, so torque adds up correctly
        # if defined that way. Here we assume compute_net_torque handles signs via moment arms.

        result = tau_agonist + tau_antagonist
        ensure(np.isfinite(result), "antagonist pair net torque must be finite", result)
        return result

    @property
    def muscle_names(self) -> list[str]:
        """Return all muscle names (agonist + antagonist) without chain traversal."""
        return list(self.agonist.muscles.keys()) + list(self.antagonist.muscles.keys())


def create_elbow_muscle_system(enable_rust: bool = True) -> AntagonistPair:
    """Factory function to create a simplified elbow muscle system.

    Args:
        enable_rust: Whether to enable Rust backend acceleration if available.

    Returns:
        AntagonistPair with Biceps (flexor) and Triceps (extensor)
    """
    from src.shared.python.biomechanics.hill_muscle import (
        HillMuscleModel,
        MuscleParameters,
    )

    # Flexors (Biceps)
    flexors = MuscleGroup("Elbow Flexors", enable_rust=enable_rust)
    biceps_params = MuscleParameters(F_max=1000.0, l_opt=0.15, l_slack=0.20)
    flexors.add_muscle("biceps", HillMuscleModel(biceps_params), moment_arm=0.04)

    # Brachialis (synergist)
    brachialis_params = MuscleParameters(F_max=800.0, l_opt=0.12, l_slack=0.10)
    flexors.add_muscle(
        "brachialis", HillMuscleModel(brachialis_params), moment_arm=0.03
    )

    # Extensors (Triceps)
    extensors = MuscleGroup("Elbow Extensors", enable_rust=enable_rust)
    triceps_params = MuscleParameters(F_max=1200.0, l_opt=0.18, l_slack=0.22)
    extensors.add_muscle("triceps", HillMuscleModel(triceps_params), moment_arm=-0.035)

    return AntagonistPair(flexors, extensors, enable_rust=enable_rust)


# Example usage
if __name__ == "__main__":
    elbow = create_elbow_muscle_system()

    # Test co-contraction
    flexor_act = {"biceps": 0.5, "brachialis": 0.5}
    extensor_act = {"triceps": 0.2}

    # Assume isometric state at optimal lengths
    states = {"biceps": (0.15, 0.0), "brachialis": (0.12, 0.0), "triceps": (0.18, 0.0)}

    tau_net = elbow.compute_net_torque(flexor_act, extensor_act, states)

    # Estimate stiffness (simplified: stiffness proportional to force)
    # K ≈ sum(F_i / l_opt_i * r_i^2)

    # Just printing results
    logger.info("=" * 60)
    logger.info("Multi-Muscle Coordination Test")
    logger.info("=" * 60)
    logger.info("\\nTest: Elbow flexion with antagonist co-contraction")
    logger.info(f"\\nFlexor activations: {flexor_act}")
    logger.info(f"Extensor activations: {extensor_act}")

    logger.info(f"\\nNet elbow torque: {tau_net:.2f} N·m")
    logger.info("  (Positive = flexion)")

    # Simple stiffness proxy
    K = (1000 * 0.5 + 800 * 0.5 + 1200 * 0.2) * 0.04  # Rough scaling
    logger.info(f"\\nEstimated joint stiffness: {K:.1f} N·m/rad")
    logger.info("  (Higher co-contraction → higher stiffness)")

    logger.info("\\n" + "=" * 60)
    logger.info("✓ Multi-muscle test complete")
    logger.info("=" * 60)
