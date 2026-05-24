"""Per-engine :class:`PoseConventionAdapter` implementations.

One adapter per supported engine. Adapters live behind the same
:class:`PoseConventionAdapter` Protocol so client code can treat them
uniformly.

The :data:`ADAPTER_REGISTRY` mapping is the canonical lookup table —
keys are stable engine identifiers (``"drake"``, ``"mujoco"``,
``"myosuite"``, ``"pinocchio"``, ``"opensim"``, ``"simscape"``) and
values are adapter **classes** (instantiate per-call to keep adapters
cheap and stateless).
"""

from __future__ import annotations

from src.shared.python.pose_interchange.adapters.drake import DrakeAdapter
from src.shared.python.pose_interchange.adapters.mujoco import MujocoAdapter
from src.shared.python.pose_interchange.adapters.myosuite import MyoSuiteAdapter
from src.shared.python.pose_interchange.adapters.opensim import OpenSimAdapter
from src.shared.python.pose_interchange.adapters.pinocchio import PinocchioAdapter
from src.shared.python.pose_interchange.adapters.simscape import SimscapeAdapter
from src.shared.python.pose_interchange.protocol import PoseConventionAdapter

ADAPTER_REGISTRY: dict[str, type[PoseConventionAdapter]] = {
    DrakeAdapter.engine_name: DrakeAdapter,
    MujocoAdapter.engine_name: MujocoAdapter,
    MyoSuiteAdapter.engine_name: MyoSuiteAdapter,
    PinocchioAdapter.engine_name: PinocchioAdapter,
    OpenSimAdapter.engine_name: OpenSimAdapter,
    SimscapeAdapter.engine_name: SimscapeAdapter,
}

__all__ = [
    "ADAPTER_REGISTRY",
    "DrakeAdapter",
    "MujocoAdapter",
    "MyoSuiteAdapter",
    "OpenSimAdapter",
    "PinocchioAdapter",
    "SimscapeAdapter",
]
