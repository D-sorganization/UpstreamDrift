"""Per-engine :class:`LiveKinematicsService` registry.

Each engine ships a service module (``drake.py``, ``mujoco.py`` etc.)
that lazily imports its engine wheel.  The
:data:`KINEMATICS_SERVICE_REGISTRY` maps an engine name to a zero-arg
factory that returns either a real-engine service (if the wheel is
installed) or a :class:`MockKinematicsService` configured for that
engine's name.

The fallback decision is made *at factory invocation time*, not at
import time, so test environments can mutate ``sys.modules`` to
simulate a wheel being present or absent.

Subtask 3 of EPIC #4895 (issue #4898).
"""

from __future__ import annotations

from collections.abc import Callable

from src.shared.python.pose_interchange.live_kinematics import (
    LiveKinematicsService,
)
from src.shared.python.pose_interchange.services._mock import (
    MockKinematicsService,
)
from src.shared.python.pose_interchange.services.drake import (
    create_drake_service,
)
from src.shared.python.pose_interchange.services.mujoco import (
    create_mujoco_service,
)
from src.shared.python.pose_interchange.services.opensim import (
    create_opensim_service,
)
from src.shared.python.pose_interchange.services.pinocchio import (
    create_pinocchio_service,
)
from src.shared.python.pose_interchange.services.simscape import (
    create_simscape_service,
)
from src.shared.python.pose_interchange.services.myosuite import (
    create_myosuite_service,
)

KINEMATICS_SERVICE_REGISTRY: dict[str, Callable[[], LiveKinematicsService]] = {
    "drake": create_drake_service,
    "mujoco": create_mujoco_service,
    "pinocchio": create_pinocchio_service,
    "opensim": create_opensim_service,
    "simscape": create_simscape_service,
    "myosuite": create_myosuite_service,
}
"""Engine name -> zero-arg factory for a :class:`LiveKinematicsService`.

Each factory returns a real-engine service when the corresponding
engine wheel is importable, and a :class:`MockKinematicsService`
configured with the right ``engine_name`` otherwise.  Calling the
factory has no module-level side effects.
"""

__all__ = [
    "KINEMATICS_SERVICE_REGISTRY",
    "MockKinematicsService",
]
