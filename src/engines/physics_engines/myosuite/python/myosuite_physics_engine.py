# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""MyoSuite Physics Engine Implementation.



Wraps MyoSuite (OpenAI Gym-based) environments into the PhysicsEngine protocol.

Documentation: https://myosuite.readthedocs.io/en/latest/



Refactored to use shared engine availability module (DRY principle).

"""

from __future__ import annotations

# Path: src/engines/physics_engines/myosuite/python/myosuite_physics_engine.py -> need 6 parents
from src.engines.tiers import warn_if_experimental
from src.shared.python.engine_core.base_physics_engine import BasePhysicsEngine  # noqa: E402
from src.shared.python.logging_pkg.logging_config import get_logger  # noqa: E402

from ._drift_control import DriftControlMixin
from ._dynamics import DynamicsMixin
from ._engine_init import EngineInitMixin
from ._muscle_interface import MuscleInterfaceMixin
from ._simulation_core import SimulationCoreMixin

logger = get_logger(__name__)


class MyoSuitePhysicsEngine(
    EngineInitMixin,
    SimulationCoreMixin,
    DynamicsMixin,
    DriftControlMixin,
    MuscleInterfaceMixin,
    BasePhysicsEngine,
):
    """MyoSuite Engine Wrapper.



    Treats 'model paths' as Gym Environment IDs (e.g. 'myoElbowPose1D6MRandom-v0').

    Accesses underlying MuJoCo simulation for dynamics where possible.

    Inherits checkpoint save/restore (Checkpointable contract) from BasePhysicsEngine.

    """

    def __init__(self) -> None:
        """Initialize."""
        warn_if_experimental("myosuite", "MyoSuite")
        BasePhysicsEngine.__init__(self)
        EngineInitMixin.__init__(self)

    @property
    def engine_type(self) -> str:
        """Get engine type identifier (Checkpointable contract)."""
        return "myosuite"


def main() -> None:
    """Entry point for standalone execution."""
    logger.error("MyoSuite does not have a standalone GUI. Use the web UI.")
    import sys

    sys.exit(1)


if __name__ == "__main__":
    main()
