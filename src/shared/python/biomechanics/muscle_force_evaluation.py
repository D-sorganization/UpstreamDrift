"""Shared scalar force evaluation for the analytic quasistatic Hill muscle kernel.

Activation dynamics and excitation-command semantics belong to callers. This
function accepts an already-computed activation and assumes zero fiber velocity;
it does not consume OpenSim or other native-engine state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from src.shared.python.biomechanics.hill_muscle import HillMuscleModel
from src.shared.python.biomechanics.muscle_equilibrium import EquilibriumSolver


@dataclass(frozen=True, slots=True)
class QuasistaticMuscleForceEvaluation:
    """Immutable scalar force and geometry observations in SI units."""

    fiber_length_m: float
    tendon_length_m: float
    active_fiber_along_fiber_n: float
    passive_fiber_along_fiber_n: float
    damping_fiber_along_fiber_n: float
    active_tendon_axis_n: float
    passive_tendon_axis_n: float
    damping_tendon_axis_n: float
    fiber_tendon_axis_n: float
    tendon_force_n: float
    force_equilibrium_residual_n: float


def evaluate_quasistatic_muscle_force(
    muscle: HillMuscleModel,
    muscle_tendon_length_m: float,
    activation: float,
) -> QuasistaticMuscleForceEvaluation:
    """Evaluate Hill force components at zero fiber velocity.

    The returned residual is observed, not thresholded: callers decide whether
    it meets their declared equilibrium tolerance. Solver failures propagate.
    """
    if not isinstance(muscle, HillMuscleModel):
        raise TypeError("muscle must be a HillMuscleModel")
    length = float(muscle_tendon_length_m)
    level = float(activation)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("muscle-tendon length must be finite and positive")
    if not math.isfinite(level) or not 0.0 <= level <= 1.0:
        raise ValueError("activation must be finite and lie in [0, 1]")

    fiber_length = float(EquilibriumSolver(muscle).solve_fiber_length(length, level))
    if not math.isfinite(fiber_length) or fiber_length <= 0.0:
        raise ValueError("equilibrium produced an invalid fiber length")
    cosine = math.cos(muscle.params.pennation_angle)
    tendon_length = length - fiber_length * cosine
    if not math.isfinite(tendon_length) or tendon_length <= 0.0:
        raise ValueError("equilibrium produced a non-positive tendon length")

    normalized_fiber_length = fiber_length / muscle.params.l_opt
    active_along_fiber = float(
        muscle.params.F_max
        * level
        * muscle.force_length_active(normalized_fiber_length)
        * muscle.force_velocity(0.0)
    )
    passive_along_fiber = float(
        muscle.params.F_max * muscle.force_length_passive(normalized_fiber_length)
    )
    # The declared kernel is quasistatic, so velocity-dependent damping is zero.
    damping_along_fiber = 0.0
    active_tendon_axis = active_along_fiber * cosine
    passive_tendon_axis = passive_along_fiber * cosine
    damping_tendon_axis = damping_along_fiber * cosine
    fiber_tendon_axis = active_tendon_axis + passive_tendon_axis + damping_tendon_axis
    tendon_force = float(
        muscle.params.F_max * muscle.tendon_force(tendon_length / muscle.params.l_slack)
    )
    residual = fiber_tendon_axis - tendon_force
    values = (
        active_along_fiber,
        passive_along_fiber,
        damping_along_fiber,
        active_tendon_axis,
        passive_tendon_axis,
        damping_tendon_axis,
        fiber_tendon_axis,
        tendon_force,
        residual,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("force evaluation produced a non-finite component")

    return QuasistaticMuscleForceEvaluation(
        fiber_length_m=fiber_length,
        tendon_length_m=tendon_length,
        active_fiber_along_fiber_n=active_along_fiber,
        passive_fiber_along_fiber_n=passive_along_fiber,
        damping_fiber_along_fiber_n=damping_along_fiber,
        active_tendon_axis_n=active_tendon_axis,
        passive_tendon_axis_n=passive_tendon_axis,
        damping_tendon_axis_n=damping_tendon_axis,
        fiber_tendon_axis_n=fiber_tendon_axis,
        tendon_force_n=tendon_force,
        force_equilibrium_residual_n=residual,
    )
