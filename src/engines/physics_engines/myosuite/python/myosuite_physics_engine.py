"""MyoSuite Physics Engine Implementation.



Wraps MyoSuite (OpenAI Gym-based) environments into the PhysicsEngine protocol.

Documentation: https://myosuite.readthedocs.io/en/latest/



Refactored to use shared engine availability module (DRY principle).

"""

from __future__ import annotations

from src.shared.python.engine_core.interfaces import PhysicsEngine  # noqa: E402
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
    PhysicsEngine,
):
    """MyoSuite Engine Wrapper.



    Treats 'model paths' as Gym Environment IDs (e.g. 'myoElbowPose1D6MRandom-v0').

    Accesses underlying MuJoCo simulation for dynamics where possible.

    """

    def __init__(self) -> None:
        """Initialize."""
        EngineInitMixin.__init__(self)


def main() -> None:
    """Entry point for standalone execution."""
    logger.error("MyoSuite does not have a standalone GUI. Use the web UI.")
    import sys

    sys.exit(1)


if __name__ == "__main__":
    main()
