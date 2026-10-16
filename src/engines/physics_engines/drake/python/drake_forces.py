"""External-force construction helpers for the Drake engine (#12980).

``MultibodyPlant`` has no force-factory method; the pydrake spelling is the
``MultibodyForces(plant)`` constructor.  ``CalcInverseDynamics`` returns
``M*vdot + C*v - tau_ext`` and does not include gravity on its own, so the
generalized gravity force must be supplied as the external force to obtain
the engine-neutral ``M*vdot + C*v + g(q)`` (bias at ``vdot = 0``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from pydrake.multibody.tree import MultibodyForces
except ImportError:  # pragma: no cover - exercised only without pydrake
    MultibodyForces = None  # type: ignore[assignment,misc]


def gravity_external_forces(plant: Any, context: Any) -> Any:
    """Return ``MultibodyForces`` holding the generalized gravity force.

    Args:
        plant: A finalized pydrake ``MultibodyPlant``.
        context: The plant context holding the state to evaluate.

    Returns:
        A ``MultibodyForces`` whose generalized forces equal
        ``CalcGravityGeneralizedForces(context)`` (``-g(q)`` in the
        ``M*a + C*v + g = tau`` convention).

    Raises:
        ImportError: If pydrake is not installed.
        ValueError: If ``plant`` or ``context`` is ``None``.
    """
    if plant is None or context is None:
        raise ValueError("plant and context must be provided")
    if MultibodyForces is None:
        raise ImportError("pydrake is required to build MultibodyForces")
    forces = MultibodyForces(plant)
    forces.mutable_generalized_forces()[:] = np.asarray(
        plant.CalcGravityGeneralizedForces(context)
    )
    return forces
